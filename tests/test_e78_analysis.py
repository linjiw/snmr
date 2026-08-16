"""CPU tests for scripts/analyze_e78_dropout.py on synthetic paired reports."""

import importlib.util
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_e78", ROOT / "scripts" / "analyze_e78_dropout.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _report(done, surv, starts):
    return {
        "completed": [bool(x) for x in done],
        "survival_s": [float(x) for x in surv],
        "start_steps": [int(x) for x in starts],
        "motion_ids": [int(s % 2) for s in starts],
    }


def _write(tmp_path, arm, label, rep):
    name = f"{arm}_eval.json" if label is None else f"{arm}_eval_{label}.json"
    (tmp_path / name).write_text(json.dumps(rep))


def test_matched_subset_and_paired_bootstrap_detect_a_planted_effect(tmp_path):
    rng = np.random.default_rng(0)
    n = 1024
    starts = np.repeat(np.arange(64), 16)
    # Clean: treatment slightly worse (0.85) than reference (0.92) -> should NOT be laundered.
    tc = rng.random(n) < 0.85
    rc = rng.random(n) < 0.92
    # Under dropout: reference collapses (0.30), treatment holds (0.70).
    td = tc & (rng.random(n) < 0.70 / 0.85)
    rd = rc & (rng.random(n) < 0.30 / 0.92)
    t_dir, r_dir = tmp_path / "t", tmp_path / "r"
    t_dir.mkdir(); r_dir.mkdir()
    label = "maskall_hold_f0.3_s5-25"
    _write(t_dir, "d_prior_explicit_snmr", None, _report(tc, tc * 10.0, starts))
    _write(r_dir, "c_prior_explicit", None, _report(rc, rc * 10.0, starts))
    _write(t_dir, "d_prior_explicit_snmr", label, _report(td, td * 10.0, starts))
    _write(r_dir, "c_prior_explicit", label, _report(rd, rd * 10.0, starts))

    t_clean = [mod.load_report(t_dir / "d_prior_explicit_snmr_eval.json")]
    r_clean = [mod.load_report(r_dir / "c_prior_explicit_eval.json")]
    clean = mod.analyze(t_clean, r_clean, t_clean, r_clean)
    assert clean["paired_diff"] < 0 and clean["paired_diff_ci95"][1] < 0  # clean regression visible

    t = [mod.load_report(t_dir / f"d_prior_explicit_snmr_eval_{label}.json")]
    r = [mod.load_report(r_dir / f"c_prior_explicit_eval_{label}.json")]
    deg = mod.analyze(t, r, t_clean, r_clean)
    assert deg["paired_diff"] > 0.3 and deg["paired_diff_ci95"][0] > 0.2
    ms = deg["matched_subset"]
    assert ms["n"] > 0 and ms["treatment"] > ms["reference"]
    assert ms["treatment_only"] > ms["reference_only"]
    assert mod.discover_severities(t_dir, "d_prior_explicit_snmr") == {
        label: t_dir / f"d_prior_explicit_snmr_eval_{label}.json"
    }


def test_unpaired_reports_are_rejected():
    a = _report([1, 0], [10, 3], [0, 1])
    b = _report([1, 0], [10, 3], [0, 2])
    try:
        mod.paired_arrays([a], [b])
    except ValueError as exc:
        assert "not paired" in str(exc)
    else:
        raise AssertionError("pairing invariant not enforced")
