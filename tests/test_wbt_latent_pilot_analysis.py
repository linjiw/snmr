from __future__ import annotations

import numpy as np

from scripts import analyze_wbt_latent_pilot as pilot


def _report(tmp_path):
    motion_steps = 1200
    starts = (
        np.linspace(
            0,
            motion_steps - pilot.HORIZON_STEPS - 1,
            pilot.ROLLOUTS,
        )
        .round()
        .astype(int)
    )
    rollouts = []
    for env_id, start in enumerate(starts):
        rollouts.append(
            {
                "env_id": env_id,
                "start_step": int(start),
                "completed": True,
                "failed": False,
                "survival_steps": pilot.HORIZON_STEPS,
                "survival_s": pilot.HORIZON_S,
                "metrics": {metric: 0.25 for metric in pilot.METRICS},
            }
        )
    motion_file = tmp_path / "motion.npz"
    return {
        "schema_version": 1,
        "passed": True,
        "seed": pilot.EVALUATION_SEED,
        "training_name": "example",
        "num_rollouts": pilot.ROLLOUTS,
        "horizon_steps": pilot.HORIZON_STEPS,
        "horizon_s": pilot.HORIZON_S,
        "policy_dt": pilot.HORIZON_S / pilot.HORIZON_STEPS,
        "motion_steps": motion_steps,
        "motion_file": str(motion_file),
        "completion_rate": 1.0,
        "mean_survival_s": pilot.HORIZON_S,
        "rollouts": rollouts,
    }, motion_file


def test_official_report_validation_enforces_phase_grid(tmp_path):
    report, motion_file = _report(tmp_path)

    errors = pilot._validate_rollout_report(
        report,
        expected_training_name="example",
        expected_motion_file=motion_file,
    )
    assert errors == []

    report["rollouts"][2]["start_step"] += 1
    errors = pilot._validate_rollout_report(
        report,
        expected_training_name="example",
        expected_motion_file=motion_file,
    )
    assert "start_steps do not match the frozen phase-stratified grid" in errors


def test_promotion_requires_completion_floor_and_one_improvement():
    indices = np.tile(np.arange(pilot.ROLLOUTS), (100, 1))
    baseline = np.ones(pilot.ROLLOUTS)
    effects = {
        "completion": pilot._effect_summary(
            baseline, baseline - 0.04, indices
        ),
        "survival_s": pilot._effect_summary(
            baseline, baseline * 1.06, indices
        ),
        pilot.PRIMARY_JOINT_METRIC: pilot._effect_summary(
            baseline, baseline, indices
        ),
    }
    decision = pilot._promotion_decision(effects)
    assert decision["completion_floor_passed"]
    assert decision["improvement_checks"]["survival_at_least_5pct"]
    assert decision["eligible_for_replication"]

    effects["completion"] = pilot._effect_summary(
        baseline, baseline - 0.06, indices
    )
    decision = pilot._promotion_decision(effects)
    assert not decision["completion_floor_passed"]
    assert not decision["eligible_for_replication"]


def test_reference_validation_rejects_changed_gmr_field(tmp_path):
    reference = tmp_path / "reference.npz"
    augmented = tmp_path / "augmented.npz"
    joint_pos = np.zeros((4, 29), dtype=np.float32)
    np.savez(reference, joint_pos=joint_pos, fps=np.array([50]))
    np.savez(
        augmented,
        joint_pos=joint_pos + 1.0,
        fps=np.array([50]),
        latent_z=np.zeros((4, 128), dtype=np.float32),
    )

    _, errors = pilot._validate_augmented_reference(reference, augmented)

    assert any("standard GMR fields changed" in error for error in errors)
