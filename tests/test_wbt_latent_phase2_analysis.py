from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_wbt_latent_phase2 import (  # noqa: E402
    ManifestRow,
    analyze,
    decide_absolute_floor,
    decide_baseline_relative,
)
from analyze_wbt_latent_pilot import HORIZON_S, HORIZON_STEPS, METRICS, ROLLOUTS  # noqa: E402


def _report(training_name, motion_file, seed, completion_rate, rmse):
    motion_steps = 13066
    starts = (
        np.linspace(0, motion_steps - HORIZON_STEPS - 1, ROLLOUTS)
        .round()
        .astype(int)
        .tolist()
    )
    rollouts = []
    for i in range(ROLLOUTS):
        completed = i < int(round(completion_rate * ROLLOUTS))
        survival_steps = HORIZON_STEPS if completed else HORIZON_STEPS // 2
        rollouts.append(
            {
                "env_id": i,
                "start_step": starts[i],
                "completed": completed,
                "failed": not completed,
                "survival_steps": survival_steps,
                "survival_s": survival_steps * HORIZON_S / HORIZON_STEPS,
                "metrics": {m: (rmse if m == "joint_position_rmse_rad" else 0.1) for m in METRICS},
            }
        )
    return {
        "schema_version": 1,
        "passed": True,
        "seed": seed,
        "training_name": training_name,
        "num_rollouts": ROLLOUTS,
        "horizon_steps": HORIZON_STEPS,
        "horizon_s": HORIZON_S,
        "policy_dt": HORIZON_S / HORIZON_STEPS,
        "motion_file": motion_file,
        "motion_steps": motion_steps,
        "completion_rate": float(np.mean([r["completed"] for r in rollouts])),
        "mean_survival_s": float(np.mean([r["survival_s"] for r in rollouts])),
        "rollouts": rollouts,
    }


def test_decision_rules():
    cells = [
        {"completion": 0.90, "completion_delta": 0.02, "survival_rel": 0.01, "joint_rmse_rel": -0.06},
        {"completion": 0.88, "completion_delta": -0.01, "survival_rel": 0.00, "joint_rmse_rel": -0.055},
        {"completion": 0.91, "completion_delta": 0.03, "survival_rel": 0.02, "joint_rmse_rel": -0.07},
    ]
    decision = decide_baseline_relative(cells)
    assert decision["replicated"] is True

    cells[1]["completion_delta"] = -0.06  # one floor violation kills it
    assert decide_baseline_relative(cells)["replicated"] is False

    floor_cells = [{"completion": 0.72}, {"completion": 0.71}, {"completion": 0.75}]
    assert decide_absolute_floor(floor_cells, 0.70)["replicated"] is True
    floor_cells[0]["completion"] = 0.65
    assert decide_absolute_floor(floor_cells, 0.70)["replicated"] is False


def test_analyze_pairs_cells_and_replicates(tmp_path):
    ref = tmp_path / "ref.npz"
    aug = tmp_path / "aug.npz"
    ref.write_bytes(b"")
    aug.write_bytes(b"")
    ckpt = tmp_path / "model_07999.pt"
    ckpt.write_bytes(b"x")

    rows = []
    for arm, motion, rmse_by_seed, completion_by_seed in (
        ("base", str(ref), {0: 0.25, 1: 0.26}, {0: 0.88, 1: 0.86}),
        ("s3", str(aug), {0: 0.235, 1: 0.244}, {0: 0.91, 1: 0.87}),
        ("l1", str(aug), {0: 0.27, 1: 0.28}, {0: 0.72, 1: 0.74}),
    ):
        for seed in (0, 1):
            name = f"{arm}_seed{seed}"
            report_path = tmp_path / f"{name}.json"
            report_path.write_text(
                json.dumps(
                    _report(name, motion, 404, completion_by_seed[seed], rmse_by_seed[seed])
                )
            )
            rows.append(ManifestRow(arm, seed, 404, name, ckpt, report_path))

    result = analyze(
        rows,
        {
            "base": {},
            "s3": {},
            "l1": {"min_completion": 0.70},
        },
        {"base": ref, "s3": aug, "l1": aug},
    )
    assert result["passed"], result["protocol_errors"]
    # s3: rmse_rel = 0.235/0.25-1 = -6% and 0.244/0.26-1 = -6.2% -> median passes
    assert result["arms"]["s3"]["decision"]["replicated"] is True
    assert result["arms"]["l1"]["decision"]["replicated"] is True
    assert result["verdict"] == "replicated:l1,s3"


def test_full_phase2_matrix_rejects_missing_and_duplicate_cells(tmp_path):
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"")
    ckpt = tmp_path / "model_07999.pt"
    ckpt.write_bytes(b"x")
    rows = []
    for arm, seeds in {
        "base": (0, 1, 2),
        "s3": (0, 1, 2),
        "l1": (0, 1, 2),
        "c3": (0,),
    }.items():
        for seed in seeds:
            for eval_seed in (404, 405):
                name = f"{arm}_s{seed}_e{eval_seed}"
                report_path = tmp_path / f"{name}.json"
                report_path.write_text(
                    json.dumps(
                        _report(name, str(motion), eval_seed, 0.8, 0.25)
                    )
                )
                rows.append(
                    ManifestRow(
                        arm, seed, eval_seed, name, ckpt, report_path
                    )
                )

    broken = rows[:-1] + [rows[0]]
    result = analyze(
        broken,
        {
            "base": {},
            "s3": {},
            "l1": {"min_completion": 0.70},
            "c3": {"screening": True},
        },
        {arm: motion for arm in ("base", "s3", "l1", "c3")},
    )

    assert not result["passed"]
    assert any(
        "duplicate manifest cells" in error
        for error in result["protocol_errors"]
    )
    assert any(
        "missing manifest cells" in error
        for error in result["protocol_errors"]
    )
