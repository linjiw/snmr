"""Regression tests for the E72 phase-sensitivity macro renderer.

The load-bearing claim is that degradation grows with |delta|, because a static clip label cannot
be misaligned in time.  These tests exist to make sure the renderer refuses to emit macros that
would state that claim when the data does not support it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.render_e72_phase_values import (
    DETECTABILITY_FLOOR,
    SEEDS,
    collect_arm,
    latex_macros,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_e72_phase_values.py"
TIME_NULL = 0.5621744791666666
REPEATS = 3


def _report(completion: float) -> dict:
    return {
        "arm": "a_prior_snmr",
        "eval_z": "prior",
        "destroy_zcmd": "none",
        "noise_cmd": 0.0,
        "noise_zret": 0.0,
        "noise_proprio": 0.0,
        "num_rollouts": 1024,
        "evaluation_seed": 404,
        "completion_rate": completion,
        "mean_survival_s": 8.5,
        "ambiguity_precheck": "/frozen/e70_ambiguity_precheck.json",
    }


# Means chosen to mirror the realized experiment: monotone in |delta|, static arms collapsed.
ARM_MEANS = {
    "control": 0.7539,
    "shift_m0250": 0.6497,
    "shift_p0250": 0.6546,
    "shift_p0500": 0.5947,
    "first_frame": 0.0016,
    "clip_mean": 0.5822,
}


def _students(tmp_path: Path, means: dict[str, float] | None = None) -> Path:
    root = tmp_path / "students"
    for arm, mean in (means or ARM_MEANS).items():
        for seed in SEEDS:
            for repeat in range(1, REPEATS + 1):
                directory = root / arm / f"seed{seed}_snmr" / f"repeat{repeat}"
                directory.mkdir(parents=True)
                # Small deterministic jitter so stdev() is defined and non-zero.
                jitter = (repeat - 2) * 0.001 + seed * 0.0005
                (directory / "a_prior_snmr_eval_ambiguity.json").write_text(
                    json.dumps(_report(max(0.0, mean + jitter)))
                )
    return root


def test_macros_report_the_phase_ladder_and_stamp_every_input(tmp_path: Path) -> None:
    rendered = latex_macros(_students(tmp_path), time_completion=TIME_NULL)
    assert r"\newcommand{\EPhaseSeeds}{3}" in rendered
    assert r"\newcommand{\EPhaseRepeats}{3}" in rendered
    assert r"\newcommand{\EPhaseControlCompletion}{0.754}" in rendered
    assert r"\newcommand{\EPhaseShiftPlusHalfCompletion}{0.595}" in rendered
    assert r"\newcommand{\EPhaseShiftPlusHalfLostPercent}{83}" in rendered
    assert r"\newcommand{\EPhaseFirstFrameCompletion}{0.002}" in rendered
    assert "% inputs_sha256=" in rendered
    # 6 arms x 3 seeds x 3 repeats
    assert rendered.count("% input_sha256[") == 6 * len(SEEDS) * REPEATS


def test_half_second_arm_must_lose_more_than_the_quarter_second_arms(tmp_path: Path) -> None:
    """Non-monotone degradation must not be rendered as a phase ladder."""
    means = dict(ARM_MEANS, shift_p0500=0.70)  # loses LESS than the quarter-second arms
    with pytest.raises(ValueError, match="not monotone"):
        latex_macros(_students(tmp_path, means), time_completion=TIME_NULL)


def test_effect_below_the_registered_floor_fails_closed(tmp_path: Path) -> None:
    means = dict(ARM_MEANS, shift_m0250=ARM_MEANS["control"] - DETECTABILITY_FLOOR / 2)
    with pytest.raises(ValueError, match="within evaluation noise"):
        latex_macros(_students(tmp_path, means), time_completion=TIME_NULL)


def test_control_at_or_below_the_clock_null_fails_closed(tmp_path: Path) -> None:
    """Without a positive advantage the 'fraction of the advantage lost' ladder is meaningless."""
    means = dict(ARM_MEANS, control=0.50)
    with pytest.raises(ValueError, match="must exceed the clock null"):
        latex_macros(_students(tmp_path, means), time_completion=TIME_NULL)


def test_unreplicated_arm_fails_closed(tmp_path: Path) -> None:
    root = _students(tmp_path)
    for repeat in root.glob("shift_p0500/seed1_snmr/repeat*"):
        for child in repeat.iterdir():
            child.unlink()
        repeat.rmdir()
    with pytest.raises(ValueError, match="no replicated cells"):
        collect_arm(root, "shift_p0500")


def test_destruction_report_is_rejected(tmp_path: Path) -> None:
    root = _students(tmp_path)
    path = root / "control" / "seed0_snmr" / "repeat1" / "a_prior_snmr_eval_ambiguity.json"
    payload = json.loads(path.read_text())
    payload["destroy_zcmd"] = "zero"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="destruction report"):
        collect_arm(root, "control")


def test_general_grid_report_is_rejected(tmp_path: Path) -> None:
    """Only ambiguity-grid evaluations may enter; the general grid is a different experiment."""
    root = _students(tmp_path)
    path = root / "control" / "seed0_snmr" / "repeat1" / "a_prior_snmr_eval_ambiguity.json"
    payload = json.loads(path.read_text())
    del payload["ambiguity_precheck"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ambiguity-grid"):
        collect_arm(root, "control")


def test_wrong_evaluation_grid_fails_closed(tmp_path: Path) -> None:
    root = _students(tmp_path)
    path = root / "clip_mean" / "seed2_snmr" / "repeat2" / "a_prior_snmr_eval_ambiguity.json"
    payload = json.loads(path.read_text())
    payload["evaluation_seed"] = 405
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen evaluation seed"):
        collect_arm(root, "clip_mean")


def test_mutated_input_changes_the_stamped_hash(tmp_path: Path) -> None:
    root = _students(tmp_path)
    before = dict(collect_arm(root, "control")[1])
    path = root / "control" / "seed0_snmr" / "repeat1" / "a_prior_snmr_eval_ambiguity.json"
    path.write_text(path.read_text() + "\n")
    after = dict(collect_arm(root, "control")[1])
    assert before["control/seed0/repeat1"] != after["control/seed0/repeat1"]


def test_direct_entrypoint_help_uses_repository_scripts_package(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--time-completion" in completed.stdout
