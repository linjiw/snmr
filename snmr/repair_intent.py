"""Intent-preservation filter for bounded reference corrections (revised, 2026-09-10).

This replaces the four-block filter in
``docs/program/METHOD_SPEC_TRANSFERABLE_CORRECTIONS_2026-09-10.md`` §2, whose non-regression
rule demanded that a candidate retain at least 95 % of the proposal's *jitter* (so a desirable
smoothing repair was rejected) and whose only timing check was a global cross-correlation lag
(which a local phase shift leaves at zero).

Design
------
* **Desirable and undesirable statistics are separated.**  Amplitude, travel, direction,
  contact sequence and event timing are things a good repair *preserves*.  Jitter is something
  a good repair may *reduce*; only an increase is penalised.
* **Local, signed checks** complement the global amplitude/energy ratios: windowed signed
  displacement direction, windowed speed, per-event timing/direction/magnitude, and the
  per-foot stance sequence.  Each has an explicit near-zero-denominator rule.
* **Every metric here is part of the method.**  Anything used to accept or reject candidates
  cannot also serve as an untouched evaluation metric; keep independent measures for that.
* **Explicit statuses.**  A candidate is ``accepted`` or ``rejected`` (with reasons), or a
  check is ``not_evaluable`` (e.g. no contact labels).  The interval-level *decision* is
  ``accept_candidate``, ``fallback_baseline`` (candidate rejected, baseline valid) or
  ``abstain`` (candidate rejected and baseline invalid).  Coverage is reported over all three.
* **Tolerances are data.**  :class:`IntentTolerances` is hash-bound and carries a calibration
  status.  The defaults below are *provisional* and must be calibrated on development data
  and frozen (with the historical failures kept as regression cases) before they can gate
  anything that is reported.

Inputs are plain arrays in a common task frame (``(T, A, 3)`` anchor positions plus
timestamps); the adapter from ``HumanMotionSpec`` / ``RobotFKTrajectory`` is deliberately
separate so this logic can be tested on synthetic adversaries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUABLE = "not_evaluable"


class Decision(str, Enum):
    ACCEPT_CANDIDATE = "accept_candidate"
    FALLBACK_BASELINE = "fallback_baseline"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class IntentTolerances:
    """All thresholds of the filter.  ``calibration_status`` must say where they came from."""

    # global per-anchor statistics (candidate / reference ratios)
    amplitude_ratio_min: float = 0.90
    amplitude_ratio_max: float = 1.25
    energy_ratio_min: float = 0.80
    energy_ratio_max: float = 1.50
    jitter_ratio_max: float = 1.25  # one-sided: reductions are always allowed
    # signed travel (net displacement) per anchor
    travel_cosine_min: float = 0.90
    travel_ratio_min: float = 0.85
    travel_ratio_max: float = 1.20
    # local windowed checks
    local_window_frames: int = 10
    local_cosine_min: float = 0.80
    local_speed_ratio_min: float = 0.70
    local_speed_ratio_max: float = 1.40
    local_pass_fraction_min: float = 0.90
    # events on the reference (acceleration-magnitude peaks)
    event_percentile: float = 95.0
    event_min_separation_frames: int = 6
    event_search_frames: int = 8
    event_shift_max_frames: int = 4
    event_peak_ratio_min: float = 0.75
    event_direction_cosine_min: float = 0.70
    # contact sequence
    contact_onset_shift_max_frames: int = 4
    # near-zero denominators (absolute floors in metres / metres-per-second)
    motion_floor_m: float = 0.005
    speed_floor_m_s: float = 0.02
    accel_floor_m_s2: float = 0.05
    calibration_status: str = "provisional: not calibrated on development data; not frozen"

    def __post_init__(self) -> None:
        if not 0 < self.amplitude_ratio_min <= 1.0 <= self.amplitude_ratio_max:
            raise ValueError("amplitude band must bracket 1.0")
        if not 0 < self.energy_ratio_min <= 1.0 <= self.energy_ratio_max:
            raise ValueError("energy band must bracket 1.0")
        if self.jitter_ratio_max < 1.0:
            raise ValueError("jitter_ratio_max must be >= 1.0 (reductions are always allowed)")
        if not 0 < self.local_pass_fraction_min <= 1.0:
            raise ValueError("local_pass_fraction_min must be in (0, 1]")
        for name in ("local_window_frames", "event_min_separation_frames", "event_search_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


PROVISIONAL_TOLERANCES = IntentTolerances()


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASSED


@dataclass(frozen=True)
class IntentReport:
    accepted: bool
    checks: tuple[CheckResult, ...]
    tolerances_sha256: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if c.status is CheckStatus.FAILED)

    @property
    def not_evaluable(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if c.status is CheckStatus.NOT_EVALUABLE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "not_evaluable": list(self.not_evaluable),
            "checks": [{"name": c.name, "status": c.status.value, "detail": _jsonable(c.detail)} for c in self.checks],
            "tolerances_sha256": self.tolerances_sha256,
            "diagnostics": _jsonable(self.diagnostics),
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# --- primitives ---------------------------------------------------------------------------


def _validate(reference: np.ndarray, candidate: np.ndarray, timestamps: np.ndarray) -> None:
    if reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("anchor trajectories must be (T, A, 3)")
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must share (T, A, 3); the grid is fixed")
    if timestamps.shape != (reference.shape[0],):
        raise ValueError("timestamps must be (T,)")
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all() and np.isfinite(timestamps).all()):
        raise ValueError("inputs contain NaN/Inf")
    if reference.shape[0] < 4:
        raise ValueError("need at least four frames")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")


def _amplitude(x: np.ndarray) -> np.ndarray:  # (A,)
    centered = x - x.mean(axis=0, keepdims=True)
    return np.sqrt((centered ** 2).sum(-1).mean(0))


def _velocity(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=0) / np.diff(t)[:, None, None]


def _acceleration(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    v = _velocity(x, t)
    dt = 0.5 * (np.diff(t)[1:] + np.diff(t)[:-1])
    return np.diff(v, axis=0) / dt[:, None, None]


def _ratio_check(name: str, cand: np.ndarray, ref: np.ndarray, lo: float | None, hi: float | None, floor: float, labels: Sequence[str]) -> CheckResult:
    """Per-anchor ratio band with an explicit near-zero-denominator rule.

    Where the reference statistic is below ``floor`` the ratio is not formed; instead the
    candidate statistic must itself be below ``floor`` (a static anchor must stay static).
    """
    ratios, failures, static = [], [], []
    for i, label in enumerate(labels):
        if ref[i] <= floor:
            static.append(label)
            if cand[i] > floor:
                failures.append(f"{label}: reference below floor ({ref[i]:.4g} <= {floor}) but candidate moves ({cand[i]:.4g})")
            ratios.append(None)
            continue
        r = float(cand[i] / ref[i])
        ratios.append(r)
        if lo is not None and r < lo:
            failures.append(f"{label}: ratio {r:.3f} < {lo}")
        if hi is not None and r > hi:
            failures.append(f"{label}: ratio {r:.3f} > {hi}")
    return CheckResult(
        name,
        CheckStatus.FAILED if failures else CheckStatus.PASSED,
        {"ratios": ratios, "failures": failures, "static_anchors": static, "band": [lo, hi]},
    )


def _peaks(signal: np.ndarray, threshold: float, min_separation: int) -> list[int]:
    """Indices of local maxima above ``threshold`` separated by at least ``min_separation``."""
    idx = [i for i in range(1, signal.size - 1) if signal[i] >= threshold and signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]]
    kept: list[int] = []
    for i in sorted(idx, key=lambda j: -signal[j]):
        if all(abs(i - k) >= min_separation for k in kept):
            kept.append(i)
    return sorted(kept)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def global_xcorr_lag(reference: np.ndarray, candidate: np.ndarray, max_lag: int | None = None) -> int:
    """Argmax cross-correlation lag of the summed centered anchor signals (diagnostic only).

    Retained to document that a zero global lag does not exclude local phase shifts
    (``test_local_phase_shift_has_zero_global_lag``); it is never a pass criterion here.
    """
    r = reference.reshape(reference.shape[0], -1)
    c = candidate.reshape(candidate.shape[0], -1)
    r = r - r.mean(0)
    c = c - c.mean(0)
    t = r.shape[0]
    max_lag = t // 4 if max_lag is None else max_lag
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            score = float((r[lag:] * c[: t - lag]).sum())
        else:
            score = float((r[: t + lag] * c[-lag:]).sum())
        if score > best:
            best, best_lag = score, lag
    return best_lag


# --- the checks ---------------------------------------------------------------------------


def check_amplitude(reference, candidate, tol: IntentTolerances, labels) -> CheckResult:
    return _ratio_check("amplitude", _amplitude(candidate), _amplitude(reference), tol.amplitude_ratio_min, tol.amplitude_ratio_max, tol.motion_floor_m, labels)


def check_energy(reference, candidate, timestamps, tol: IntentTolerances, labels) -> CheckResult:
    ref = (_velocity(reference, timestamps) ** 2).sum(-1).mean(0)
    cand = (_velocity(candidate, timestamps) ** 2).sum(-1).mean(0)
    return _ratio_check("energy", cand, ref, tol.energy_ratio_min, tol.energy_ratio_max, tol.speed_floor_m_s ** 2, labels)


def check_jitter_not_increased(reference, candidate, timestamps, tol: IntentTolerances, labels) -> CheckResult:
    """One-sided: the candidate may be smoother than the reference, never rougher."""
    ref = (_acceleration(reference, timestamps) ** 2).sum(-1).mean(0)
    cand = (_acceleration(candidate, timestamps) ** 2).sum(-1).mean(0)
    return _ratio_check("jitter_not_increased", cand, ref, None, tol.jitter_ratio_max, tol.accel_floor_m_s2 ** 2, labels)


def check_signed_travel(reference, candidate, tol: IntentTolerances, labels) -> CheckResult:
    """Net displacement vector: direction (cosine) and magnitude ratio, per anchor."""
    ref = reference[-1] - reference[0]
    cand = candidate[-1] - candidate[0]
    failures, detail = [], []
    for i, label in enumerate(labels):
        rn, cn = float(np.linalg.norm(ref[i])), float(np.linalg.norm(cand[i]))
        if rn <= tol.motion_floor_m:
            detail.append({"anchor": label, "static": True, "candidate_norm": cn})
            if cn > tol.motion_floor_m * 2:
                failures.append(f"{label}: reference has no net travel but candidate travels {cn:.4g} m")
            continue
        cos = float(np.dot(ref[i], cand[i]) / (rn * max(cn, 1e-12)))
        ratio = cn / rn
        detail.append({"anchor": label, "cosine": cos, "ratio": ratio})
        if cos < tol.travel_cosine_min:
            failures.append(f"{label}: travel direction cosine {cos:.3f} < {tol.travel_cosine_min}")
        if not tol.travel_ratio_min <= ratio <= tol.travel_ratio_max:
            failures.append(f"{label}: travel ratio {ratio:.3f} outside [{tol.travel_ratio_min}, {tol.travel_ratio_max}]")
    return CheckResult("signed_travel", CheckStatus.FAILED if failures else CheckStatus.PASSED, {"anchors": detail, "failures": failures})


def check_local_direction_and_speed(reference, candidate, timestamps, tol: IntentTolerances, labels) -> CheckResult:
    """Windowed signed displacement: cosine and speed ratio per (window, anchor)."""
    w = tol.local_window_frames
    t = reference.shape[0]
    if t <= w:
        return CheckResult("local_direction_speed", CheckStatus.NOT_EVALUABLE, {"reason": f"interval shorter than one window ({t} <= {w})"})
    evaluated = passed = 0
    failures = []
    for start in range(0, t - w, max(w // 2, 1)):
        end = start + w
        dt = float(timestamps[end] - timestamps[start])
        for i, label in enumerate(labels):
            rd = reference[end, i] - reference[start, i]
            cd = candidate[end, i] - candidate[start, i]
            rs = float(np.linalg.norm(rd)) / dt
            if rs <= tol.speed_floor_m_s:
                continue  # reference nearly static in this window: no direction to preserve
            cs = float(np.linalg.norm(cd)) / dt
            evaluated += 1
            cos = float(np.dot(rd, cd) / (np.linalg.norm(rd) * max(np.linalg.norm(cd), 1e-12)))
            ratio = cs / rs
            ok = cos >= tol.local_cosine_min and tol.local_speed_ratio_min <= ratio <= tol.local_speed_ratio_max
            passed += ok
            if not ok and len(failures) < 20:
                failures.append({"window": [start, end], "anchor": label, "cosine": cos, "speed_ratio": ratio})
    if evaluated == 0:
        return CheckResult("local_direction_speed", CheckStatus.NOT_EVALUABLE, {"reason": "reference static in every window"})
    fraction = passed / evaluated
    status = CheckStatus.PASSED if fraction >= tol.local_pass_fraction_min else CheckStatus.FAILED
    return CheckResult("local_direction_speed", status, {"pass_fraction": fraction, "evaluated": evaluated, "first_failures": failures})


def reference_events(reference: np.ndarray, timestamps: np.ndarray, tol: IntentTolerances) -> list[dict[str, Any]]:
    """Acceleration-magnitude peaks per anchor above the clip's own percentile."""
    acc = _acceleration(reference, timestamps)  # (T-2, A, 3)
    mag = np.linalg.norm(acc, axis=-1)
    events = []
    for a in range(reference.shape[1]):
        threshold = max(float(np.percentile(mag[:, a], tol.event_percentile)), tol.accel_floor_m_s2)
        for idx in _peaks(mag[:, a], threshold, tol.event_min_separation_frames):
            events.append({"anchor": a, "frame": idx + 1, "peak": float(mag[idx, a]), "direction": acc[idx, a] / max(float(mag[idx, a]), 1e-12)})
    return events


def check_events(reference, candidate, timestamps, tol: IntentTolerances, labels, events: Sequence[dict[str, Any]] | None = None) -> CheckResult:
    """Each reference event must exist in the candidate near the same frame, with the same
    direction and comparable magnitude; and the candidate must not add events nearby."""
    events = reference_events(reference, timestamps, tol) if events is None else list(events)
    if not events:
        return CheckResult("events", CheckStatus.NOT_EVALUABLE, {"reason": "reference has no events above threshold"})
    acc_c = _acceleration(candidate, timestamps)
    mag_c = np.linalg.norm(acc_c, axis=-1)
    failures, matched = [], []
    for ev in events:
        a, f = ev["anchor"], ev["frame"]
        lo, hi = max(f - 1 - tol.event_search_frames, 0), min(f - 1 + tol.event_search_frames + 1, mag_c.shape[0])
        window = mag_c[lo:hi, a]
        # Match the reference event to the NEAREST local maximum of the candidate inside the
        # search window (ties broken by magnitude), not to the window argmax: a larger
        # neighbouring event inside the window must not capture the match, or an identical
        # candidate would be reported as shifted.  Fall back to the argmax when the window has
        # no interior local maximum (an attenuated or removed event).
        local = [
            i for i in range(1, window.size - 1)
            if window[i] >= window[i - 1] and window[i] >= window[i + 1]
        ]
        if local:
            j = min(local, key=lambda i: (abs(i + lo + 1 - f), -window[i])) + lo
        else:
            j = int(np.argmax(window)) + lo
        peak_ratio = float(mag_c[j, a] / max(ev["peak"], 1e-12))
        shift = (j + 1) - f
        direction_cos = float(np.dot(ev["direction"], acc_c[j, a] / max(float(mag_c[j, a]), 1e-12)))
        record = {"anchor": labels[a], "frame": f, "shift": shift, "peak_ratio": peak_ratio, "direction_cosine": direction_cos}
        matched.append(record)
        if peak_ratio < tol.event_peak_ratio_min:
            failures.append(f"{labels[a]}@{f}: event attenuated or removed (peak ratio {peak_ratio:.2f})")
        elif abs(shift) > tol.event_shift_max_frames:
            failures.append(f"{labels[a]}@{f}: event shifted by {shift} frames (> {tol.event_shift_max_frames})")
        if direction_cos < tol.event_direction_cosine_min:
            failures.append(f"{labels[a]}@{f}: event direction cosine {direction_cos:.2f} < {tol.event_direction_cosine_min}")
    # Added events: candidate peaks above the reference threshold with no reference event nearby.
    added = []
    for a in range(reference.shape[1]):
        threshold = max(float(np.percentile(np.linalg.norm(_acceleration(reference, timestamps), axis=-1)[:, a], tol.event_percentile)), tol.accel_floor_m_s2)
        for idx in _peaks(mag_c[:, a], threshold, tol.event_min_separation_frames):
            if not any(e["anchor"] == a and abs(e["frame"] - (idx + 1)) <= tol.event_search_frames for e in events):
                added.append({"anchor": labels[a], "frame": idx + 1, "peak": float(mag_c[idx, a])})
    if added:
        failures.append(f"{len(added)} event(s) added by the candidate")
    return CheckResult("events", CheckStatus.FAILED if failures else CheckStatus.PASSED, {"matched": matched, "added": added, "failures": failures, "n_events": len(events)})


def check_contact_sequence(reference_contact: np.ndarray | None, candidate_contact: np.ndarray | None, tol: IntentTolerances) -> CheckResult:
    """Per-foot stance runs must match in number and order, with bounded onset/offset shifts."""
    if reference_contact is None or candidate_contact is None:
        return CheckResult("contact_sequence", CheckStatus.NOT_EVALUABLE, {"reason": "no contact labels supplied"})
    ref = np.asarray(reference_contact, dtype=bool)
    cand = np.asarray(candidate_contact, dtype=bool)
    if ref.shape != cand.shape or ref.ndim != 2:
        raise ValueError("contact masks must share a (T, F) shape")
    if not ref.any():
        return CheckResult("contact_sequence", CheckStatus.NOT_EVALUABLE, {"reason": "reference has no stance labels"})
    failures, detail = [], []
    for f in range(ref.shape[1]):
        r_runs, c_runs = _runs(ref[:, f]), _runs(cand[:, f])
        detail.append({"foot": f, "reference_runs": len(r_runs), "candidate_runs": len(c_runs)})
        if len(r_runs) != len(c_runs):
            failures.append(f"foot {f}: {len(r_runs)} reference stance runs vs {len(c_runs)} candidate runs")
            continue
        for (rs, re), (cs, ce) in zip(r_runs, c_runs):
            if abs(rs - cs) > tol.contact_onset_shift_max_frames or abs(re - ce) > tol.contact_onset_shift_max_frames:
                failures.append(f"foot {f}: stance run [{rs},{re}) moved to [{cs},{ce})")
    return CheckResult("contact_sequence", CheckStatus.FAILED if failures else CheckStatus.PASSED, {"feet": detail, "failures": failures})


# --- the filter ---------------------------------------------------------------------------


def evaluate_intent(
    reference: np.ndarray,
    candidate: np.ndarray,
    timestamps: np.ndarray,
    *,
    tolerances: IntentTolerances = PROVISIONAL_TOLERANCES,
    anchor_labels: Sequence[str] | None = None,
    reference_contact: np.ndarray | None = None,
    candidate_contact: np.ndarray | None = None,
    events: Sequence[dict[str, Any]] | None = None,
    event_source: np.ndarray | None = None,
    require_contact: bool = False,
) -> IntentReport:
    """Run every check of a candidate against its reference on a fixed frame grid.

    ``accepted`` requires every evaluable check to pass.  A ``not_evaluable`` check does not
    reject by itself unless ``require_contact`` demands the contact-sequence check.  The
    global cross-correlation lag is attached as a diagnostic only.

    Events are defined on ``event_source`` when given (the method spec defines them on the
    *human source*, so that a noisy proposal's own acceleration spikes are not mistaken for
    task events), else on ``reference``; explicit ``events`` override both.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    _validate(reference, candidate, timestamps)
    labels = list(anchor_labels) if anchor_labels is not None else [f"anchor{i}" for i in range(reference.shape[1])]
    if len(labels) != reference.shape[1]:
        raise ValueError("anchor_labels must match the anchor count")
    if events is None and event_source is not None:
        event_source = np.asarray(event_source, dtype=np.float64)
        if event_source.shape != reference.shape:
            raise ValueError("event_source must share the reference's (T, A, 3) grid")
        events = reference_events(event_source, timestamps, tolerances)
    checks = (
        check_amplitude(reference, candidate, tolerances, labels),
        check_energy(reference, candidate, timestamps, tolerances, labels),
        check_jitter_not_increased(reference, candidate, timestamps, tolerances, labels),
        check_signed_travel(reference, candidate, tolerances, labels),
        check_local_direction_and_speed(reference, candidate, timestamps, tolerances, labels),
        check_events(reference, candidate, timestamps, tolerances, labels, events),
        check_contact_sequence(reference_contact, candidate_contact, tolerances),
    )
    failed = any(c.status is CheckStatus.FAILED for c in checks)
    contact_missing = require_contact and checks[-1].status is CheckStatus.NOT_EVALUABLE
    return IntentReport(
        accepted=not failed and not contact_missing,
        checks=checks,
        tolerances_sha256=tolerances.sha256,
        diagnostics={
            "global_xcorr_lag_frames": global_xcorr_lag(reference, candidate),
            "global_xcorr_lag_note": "diagnostic only; zero lag does not exclude local phase shifts",
            "contact_required_but_missing": contact_missing,
        },
    )


@dataclass(frozen=True)
class IntervalDecision:
    decision: Decision
    candidate_accepted: bool
    baseline_valid: bool
    candidate_reasons: tuple[str, ...]
    baseline_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "candidate_accepted": self.candidate_accepted,
            "baseline_valid": self.baseline_valid,
            "candidate_reasons": list(self.candidate_reasons),
            "baseline_reasons": list(self.baseline_reasons),
        }


def decide(candidate_accepted: bool, candidate_reasons: Sequence[str], baseline_valid: bool, baseline_reasons: Sequence[str]) -> IntervalDecision:
    """Combine the candidate's filter outcome with the baseline's own validity.

    A rejected candidate falls back to the baseline only when the baseline is itself valid
    (joint limits, penetration, its own intent envelope).  Otherwise the interval is an
    explicit abstention — never a claimed guaranteed output.
    """
    if candidate_accepted:
        decision = Decision.ACCEPT_CANDIDATE
    elif baseline_valid:
        decision = Decision.FALLBACK_BASELINE
    else:
        decision = Decision.ABSTAIN
    return IntervalDecision(decision, candidate_accepted, baseline_valid, tuple(candidate_reasons), tuple(baseline_reasons))


def coverage(decisions: Sequence[IntervalDecision]) -> dict[str, Any]:
    """Counts and fractions of accepted / fallback / abstained intervals (all in the denominator)."""
    n = len(decisions)
    counts = {d.value: sum(x.decision is d for x in decisions) for d in Decision}
    reasons: dict[str, int] = {}
    for x in decisions:
        if x.decision is not Decision.ACCEPT_CANDIDATE:
            for r in x.candidate_reasons:
                reasons[r] = reasons.get(r, 0) + 1
    return {
        "n_intervals": n,
        "counts": counts,
        "fractions": {k: (v / n if n else None) for k, v in counts.items()},
        "candidate_rejection_reason_counts": reasons,
    }
