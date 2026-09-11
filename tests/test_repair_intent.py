"""Synthetic adversaries for the revised intent filter.

Each adversary preserves the *global* statistics the old Block A/B filter looked at and is
caught by a local, signed or event-level check — or, for the smoothing case, is a desirable
repair the old jitter-retention rule would have rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from snmr.repair_intent import (
    CheckStatus,
    Decision,
    IntentTolerances,
    IntervalDecision,
    check_amplitude,
    check_energy,
    check_jitter_not_increased,
    coverage,
    decide,
    evaluate_intent,
    global_xcorr_lag,
)

FPS = 50.0
T = 300
TIMES = np.arange(T) / FPS
TOL = IntentTolerances()


def _walking_anchor(phase_shift: float = 0.0, amplitude: float = 0.08) -> np.ndarray:
    """A foot-like anchor: forward travel with a 1.5 Hz oscillation and a vertical lift."""
    t = TIMES
    x = 0.8 * t + amplitude * np.sin(2 * np.pi * 1.5 * t + phase_shift)
    y = 0.02 * np.cos(2 * np.pi * 1.5 * t + phase_shift)
    z = 0.04 * np.maximum(np.sin(2 * np.pi * 1.5 * t + phase_shift), 0.0)
    return np.stack([x, y, z], axis=-1)


def _reference(n_anchors: int = 2) -> np.ndarray:
    return np.stack([_walking_anchor(phase_shift=np.pi * i) for i in range(n_anchors)], axis=1)


def _names(report):
    return {c.name: c.status for c in report.checks}


# --- sanity: identity is accepted ---------------------------------------------------------


def test_identity_candidate_is_accepted():
    ref = _reference()
    report = evaluate_intent(ref, ref.copy(), TIMES, tolerances=TOL)
    assert report.accepted, report.rejection_reasons
    assert report.diagnostics["global_xcorr_lag_frames"] == 0


def test_identity_passes_when_a_larger_event_sits_inside_the_search_window():
    """Two reference events 7 frames apart, the second larger: an identical candidate must be
    matched peak-to-peak (shift 0), not captured by the larger neighbour (a real bug seen on
    walk1_subject5 window 0, right foot, frames 123/130)."""
    t = TIMES
    z = np.zeros(T)
    for centre, amp in ((120, 0.02), (127, 0.05)):
        z += amp * np.exp(-((np.arange(T) - centre) ** 2) / (2 * 1.5 ** 2))
    ref = np.stack([0.3 * t, np.zeros(T), z], axis=-1)[:, None, :]
    report = evaluate_intent(ref, ref.copy(), TIMES, tolerances=TOL)
    events = next(c for c in report.checks if c.name == "events")
    assert events.passed, events.detail["failures"]
    assert all(m["shift"] == 0 for m in events.detail["matched"])


def test_tolerances_are_hash_bound_and_flag_calibration():
    assert "provisional" in TOL.calibration_status
    assert TOL.sha256 != IntentTolerances(amplitude_ratio_min=0.95).sha256
    with pytest.raises(ValueError):
        IntentTolerances(jitter_ratio_max=0.9)  # would demand jitter retention
    with pytest.raises(ValueError):
        IntentTolerances(amplitude_ratio_min=1.1)


# --- adversary 1: amplitude halving -------------------------------------------------------


def test_amplitude_halving_is_rejected():
    ref = _reference()
    cand = ref[:1] + 0.5 * (ref - ref[:1])  # every displacement halved, same grid and duration
    report = evaluate_intent(ref, cand, TIMES, tolerances=TOL)
    assert not report.accepted
    assert "amplitude" in report.rejection_reasons
    assert "signed_travel" in report.rejection_reasons


def test_halving_only_the_oscillation_of_a_travelling_anchor_is_caught_locally():
    """RMS amplitude is dominated by travel, so a global amplitude ratio barely moves; the
    windowed speed / event checks are what catch the attenuated gait oscillation."""
    ref = _reference()
    travel = 0.8 * TIMES[:, None, None] * np.array([1.0, 0.0, 0.0])
    cand = travel + 0.5 * (ref - travel)
    report = evaluate_intent(ref, cand, TIMES, tolerances=TOL)
    assert not report.accepted
    assert {"events", "local_direction_speed", "energy"} & set(report.rejection_reasons)


# --- adversary 2: slow-down by an interior time warp, same duration -----------------------


def test_interior_slowdown_is_rejected_by_local_and_event_checks_not_by_global_lag():
    ref = _reference(1)
    t = TIMES
    # phi(t): fixed endpoints, slow in the middle third, fast elsewhere.
    warp = 0.12 * np.sin(2 * np.pi * t / t[-1])
    phi = t + warp
    cand = np.stack([np.interp(phi, t, ref[:, 0, k]) for k in range(3)], axis=-1)[:, None, :]
    report = evaluate_intent(ref, cand, TIMES, tolerances=TOL)
    assert not report.accepted
    assert {"local_direction_speed", "events"} & set(report.rejection_reasons)
    # A global lag test would not have caught it.
    assert abs(report.diagnostics["global_xcorr_lag_frames"]) <= 1


# --- adversary 3: direction reversal preserving amplitude, energy and jitter ---------------


def test_direction_reversal_passes_global_stats_but_fails_signed_checks():
    ref = _reference(2)
    cand = ref.copy()
    # Mirror anchor 1's oscillation about its travel line: same amplitude/energy/jitter,
    # opposite local direction of the oscillatory component.
    travel = 0.8 * TIMES[:, None] * np.array([1.0, 0.0, 0.0])
    cand[:, 1, :] = travel - (ref[:, 1, :] - travel)
    cand[:, 1, 2] = ref[:, 1, 2]  # keep the lift (non-negative) so z amplitude matches
    labels = ["a0", "a1"]
    assert check_amplitude(ref, cand, TOL, labels).passed
    assert check_energy(ref, cand, TIMES, TOL, labels).passed
    assert check_jitter_not_increased(ref, cand, TIMES, TOL, labels).passed
    report = evaluate_intent(ref, cand, TIMES, tolerances=TOL, anchor_labels=labels)
    assert not report.accepted
    assert {"local_direction_speed", "events"} & set(report.rejection_reasons)


# --- adversary 4: removed foot-strike -----------------------------------------------------


def test_removed_foot_strike_is_rejected_by_contact_sequence_and_events():
    ref = _reference(1)
    contact = (ref[:, :, 2] <= 1e-6)  # stance when the lift is zero -> (T, 1)
    cand_contact = contact.copy()
    runs = [(s, e) for s, e in _runs(contact[:, 0])]
    assert len(runs) >= 3
    s, e = runs[1]
    cand_contact[s:e, 0] = False  # delete one stance phase from the labels
    # And smooth the corresponding lift so the strike event disappears kinematically.
    cand = ref.copy()
    lo, hi = max(s - 12, 0), min(e + 12, T)
    cand[lo:hi, 0, 2] = np.interp(np.arange(lo, hi), [lo, hi - 1], [ref[lo, 0, 2], ref[hi - 1, 0, 2]]) + 0.02
    report = evaluate_intent(ref, cand, TIMES, tolerances=TOL, reference_contact=contact, candidate_contact=cand_contact)
    assert not report.accepted
    assert "contact_sequence" in report.rejection_reasons
    assert check_amplitude(ref, cand, TOL, ["a0"]).passed  # amplitude alone would not see it


def _runs(mask):
    out, start = [], None
    for i, v in enumerate(mask.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


# --- adversary 5: shifted transition with zero global lag ---------------------------------


def test_local_phase_shift_has_zero_global_lag_but_fails_events():
    ref = _reference(1)
    t = TIMES
    # Shift only the middle third by +8 frames using a smooth bump time map (fixed endpoints).
    bump = np.exp(-((t - t[-1] / 2) ** 2) / (2 * 0.6 ** 2))
    phi = t - (8 / FPS) * bump
    cand = np.stack([np.interp(phi, t, ref[:, 0, k]) for k in range(3)], axis=-1)[:, None, :]
    assert global_xcorr_lag(ref, cand) == 0
    report = evaluate_intent(ref, cand, TIMES, tolerances=TOL)
    assert not report.accepted
    assert "events" in report.rejection_reasons or "local_direction_speed" in report.rejection_reasons


# --- desirable repair: jitter removal must be ACCEPTED -----------------------------------


def test_jitter_removal_is_accepted_and_old_retention_rule_would_reject():
    rng = np.random.default_rng(0)
    clean = _reference(2)
    noisy = clean + rng.normal(scale=0.002, size=clean.shape)  # 2 mm white noise
    # The proposal is the noisy one; the candidate is the clean one (a smoothing repair).
    # Events are defined on the clean source (as the method defines them on the human source),
    # otherwise the proposal's own noise spikes would masquerade as task events.
    report = evaluate_intent(noisy, clean, TIMES, tolerances=TOL, event_source=clean)
    assert report.accepted, report.rejection_reasons
    jitter = next(c for c in report.checks if c.name == "jitter_not_increased")
    ratios = [r for r in jitter.detail["ratios"] if r is not None]
    assert max(ratios) < 0.95  # the old rule (retain >= 95 % of jitter) would have rejected it


# --- near-zero denominators --------------------------------------------------------------


def test_static_reference_anchor_uses_absolute_floor_not_a_ratio():
    ref = _reference(2)
    ref[:, 1, :] = ref[0, 1, :]  # anchor 1 perfectly static
    cand = ref.copy()
    cand[:, 1, 0] += 0.003 * np.sin(2 * np.pi * TIMES)  # 3 mm wobble
    report = evaluate_intent(ref, cand, TIMES, tolerances=TOL)
    amp = next(c for c in report.checks if c.name == "amplitude")
    assert amp.detail["ratios"][1] is None and "a1" not in " ".join(amp.detail["failures"])
    assert amp.passed  # below the 5 mm floor
    cand[:, 1, 0] += 0.05 * np.sin(2 * np.pi * TIMES)  # now a real 5 cm motion on a static anchor
    report = evaluate_intent(ref, cand, TIMES, tolerances=TOL)
    assert not report.accepted and "amplitude" in report.rejection_reasons


def test_inputs_are_validated():
    ref = _reference(1)
    with pytest.raises(ValueError):
        evaluate_intent(ref, ref[:-1], TIMES[:-1])  # grid mismatch is an error, never resampled
    with pytest.raises(ValueError):
        evaluate_intent(ref, ref, TIMES, event_source=ref[:-1])
    with pytest.raises(ValueError):
        evaluate_intent(ref, ref, TIMES[::-1])
    bad = ref.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_intent(ref, bad, TIMES)


# --- decisions and coverage --------------------------------------------------------------


def test_rejected_candidate_with_invalid_baseline_abstains():
    d = decide(False, ["amplitude"], False, ["joint_limits"])
    assert d.decision is Decision.ABSTAIN
    assert decide(False, ["events"], True, []).decision is Decision.FALLBACK_BASELINE
    assert decide(True, [], False, ["joint_limits"]).decision is Decision.ACCEPT_CANDIDATE


def test_coverage_keeps_every_interval_in_the_denominator():
    decisions = [
        decide(True, [], True, []),
        decide(False, ["events"], True, []),
        decide(False, ["amplitude", "events"], False, ["penetration"]),
    ]
    cov = coverage(decisions)
    assert cov["n_intervals"] == 3
    assert cov["counts"] == {"accept_candidate": 1, "fallback_baseline": 1, "abstain": 1}
    assert cov["fractions"]["abstain"] == pytest.approx(1 / 3)
    assert cov["candidate_rejection_reason_counts"] == {"events": 2, "amplitude": 1}


def test_missing_contact_labels_are_explicit_not_silent():
    ref = _reference(1)
    report = evaluate_intent(ref, ref.copy(), TIMES, tolerances=TOL)
    assert "contact_sequence" in report.not_evaluable and report.accepted
    strict = evaluate_intent(ref, ref.copy(), TIMES, tolerances=TOL, require_contact=True)
    assert not strict.accepted and strict.diagnostics["contact_required_but_missing"]
    as_dict = strict.to_dict()
    assert as_dict["not_evaluable"] == ["contact_sequence"]
