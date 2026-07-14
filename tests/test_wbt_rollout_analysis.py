import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_wbt_rollouts.py"
SPEC = importlib.util.spec_from_file_location("analyze_wbt_rollouts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
wbt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wbt)


def test_hierarchical_bootstrap_preserves_clear_noninferiority():
    shape = (3, 3, 3, 100)
    gmr_completion = np.full(shape, 0.90)
    snmr_completion = np.full(shape, 0.88)
    gmr_joint = np.full(shape, 0.20)
    snmr_joint = np.full(shape, 0.21)

    completion, joint = wbt._bootstrap_primary(
        gmr_completion,
        snmr_completion,
        gmr_joint,
        snmr_joint,
        replicates=50,
        seed=7,
    )

    assert np.allclose(completion, -0.02)
    assert np.allclose(joint, 0.05)


def test_validate_row_rejects_inconsistent_rollout(tmp_path):
    checkpoint = tmp_path / "model_00999.pt"
    checkpoint.touch()
    row = wbt.EvaluationRow(
        name="pilot_gmr_walk1_seed0_eval101",
        train_name="pilot_gmr_walk1_seed0",
        source="gmr",
        clip="walk1",
        training_seed=0,
        evaluation_seed=101,
        report_path=tmp_path / "report.json",
        checkpoint_path=checkpoint,
    )
    rollout = {
        "env_id": 0,
        "start_step": 0,
        "completed": True,
        "failed": False,
        "survival_steps": 10,
        "survival_s": 0.2,
        "metrics": {metric: 1.0 for metric in wbt.METRICS},
    }
    report = {
        "schema_version": 1,
        "passed": True,
        "seed": 101,
        "training_name": "pilot_gmr_walk1_seed0",
        "num_rollouts": 100,
        "horizon_steps": 500,
        "horizon_s": 10.0,
        "policy_dt": 0.02,
        "motion_file": "/motions/gmr/walk1_subject5_mj.npz",
        "motion_steps": 600,
        "completion_rate": 1.0,
        "mean_survival_s": 0.2,
        "rollouts": [
            {**rollout, "env_id": index, "start_step": index} for index in range(100)
        ],
    }

    errors = wbt._validate_row(row, report)

    assert any("completed rollout is inconsistent" in error for error in errors)

    starts = np.linspace(0, 99, 100).round().astype(int).tolist()
    for index, valid_rollout in enumerate(report["rollouts"]):
        valid_rollout["start_step"] = starts[index]
        valid_rollout["survival_steps"] = 500
        valid_rollout["survival_s"] = 10.0
    report["mean_survival_s"] = 10.0

    assert wbt._validate_row(row, report) == []
