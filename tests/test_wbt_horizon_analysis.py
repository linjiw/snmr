import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_wbt_horizon.py"
SPEC = importlib.util.spec_from_file_location("analyze_wbt_horizon", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
wbt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wbt)


def _summaries(
    pooled: tuple[float, float, float],
    per_clip: tuple[float, float, float],
):
    return {
        str(horizon): {
            "pooled_completion": pooled[index],
            "per_clip_completion": {
                clip: per_clip[index] for clip in wbt.CLIPS
            },
        }
        for index, horizon in enumerate(wbt.HORIZONS)
    }


def test_select_horizon_uses_earliest_inclusive_threshold():
    summaries = _summaries(
        pooled=(0.50, 0.80, 0.90),
        per_clip=(0.25, 0.80, 0.90),
    )

    assert wbt._select_horizon(summaries) == 2000


def test_select_horizon_requires_pooled_and_every_clip_threshold():
    summaries = _summaries(
        pooled=(0.49, 0.70, 0.80),
        per_clip=(0.40, 0.24, 0.30),
    )

    assert wbt._select_horizon(summaries) == 8000


def test_select_horizon_stops_when_8000_fails():
    summaries = _summaries(
        pooled=(0.10, 0.30, 0.49),
        per_clip=(0.30, 0.30, 0.30),
    )

    assert wbt._select_horizon(summaries) is None


def test_validate_report_rejects_missing_summary_without_crashing(tmp_path):
    motion_steps = 700
    starts = (
        np.linspace(
            0,
            motion_steps - wbt.HORIZON_STEPS - 1,
            wbt.ROLLOUTS_PER_EVALUATION,
        )
        .round()
        .astype(int)
        .tolist()
    )
    rollouts = [
        {
            "env_id": index,
            "start_step": starts[index],
            "completed": True,
            "failed": False,
            "survival_steps": wbt.HORIZON_STEPS,
            "survival_s": wbt.HORIZON_S,
            "metrics": {metric: 1.0 for metric in wbt.METRICS},
        }
        for index in range(wbt.ROLLOUTS_PER_EVALUATION)
    ]
    row = wbt.EvaluationRow(
        name=wbt._evaluation_name("walk1", 2000),
        train_name=wbt._training_name("walk1"),
        clip="walk1",
        total_iterations=2000,
        report_path=tmp_path / "report.json",
        checkpoint_path=tmp_path / "model_01999.pt",
        checkpoint_sha256="unused",
    )
    report = {
        "schema_version": 1,
        "passed": True,
        "seed": wbt.EVALUATION_SEED,
        "training_name": row.train_name,
        "num_rollouts": wbt.ROLLOUTS_PER_EVALUATION,
        "horizon_steps": wbt.HORIZON_STEPS,
        "horizon_s": wbt.HORIZON_S,
        "policy_dt": wbt.HORIZON_S / wbt.HORIZON_STEPS,
        "motion_file": "/motions/gmr/walk1_subject5_mj.npz",
        "motion_steps": motion_steps,
        "mean_survival_s": wbt.HORIZON_S,
        "rollouts": rollouts,
    }

    errors = wbt._validate_report(row, report)

    assert "completion_rate does not match rollout rows" in errors
