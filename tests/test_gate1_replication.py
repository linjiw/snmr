import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_gate1_replication.py"
SPEC = importlib.util.spec_from_file_location("analyze_gate1_replication", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replication)


def _summary(speed, *, mpjpe=0.03, source_speed=0.3, jerk=600.0):
    return {
        "teacher_height_stance_speed_ms": speed,
        "source_contact_stance_speed_ms": source_speed,
        "mpjpe_m": mpjpe,
        "dof_jerk": jerk,
        "limit_violation_fraction": 0.0,
        "penetration_mean_m": 0.001,
        "penetration_fraction": 0.01,
        "per_clip_teacher_height_stance_speed_ms": {
            clip: speed for clip in replication.CLIPS
        },
        "evaluator_sha256": replication.EVALUATOR_SHA256,
        "checkpoint_sha256": "screen-checkpoint",
    }


def _write_screen(root, summaries):
    root.mkdir()
    arms = {}
    for family, arm in replication.SCREEN_ARMS.items():
        arms[arm] = {
            "passed": True,
            "errors": [],
            "summary": summaries[family],
        }
    payload = {
        "passed": True,
        "evaluator_sha256": replication.EVALUATOR_SHA256,
        "promoted_arms": [
            "c4_teacher_velocity_seed0",
            "c3_stance_seed0",
        ],
        "arms": arms,
    }
    (root / "analysis.json").write_text(json.dumps(payload))


def _write_replication_arm(root, family, seed, summary, *, evaluator_hash=None):
    arm = replication._arm_name(family, seed)
    arm_dir = root / arm
    arm_dir.mkdir()
    config = {
        **replication.BASE_CONFIG,
        **replication.OBJECTIVE_CONFIG[family],
        "seed": seed,
        "out": str(arm_dir),
    }
    manifest = {
        "status": "completed",
        "progress": {"completion_state": "completed", "step": 50000},
        "git": {"sha": replication.TRAIN_REVISION, "dirty": False},
        "config": config,
    }
    (arm_dir / "manifest.json").write_text(json.dumps(manifest))
    metrics = {
        key: value
        for key, value in summary.items()
        if key in replication.REQUIRED_METRICS
    }
    per_clip = {
        clip: {
            "teacher_height_stance_speed_ms": value,
            "teacher_height_contact_samples": 1.0,
        }
        for clip, value in summary[
            "per_clip_teacher_height_stance_speed_ms"
        ].items()
    }
    benchmark = {
        "_protocol": {
            "window_frames": 192,
            "windows_per_clip_max": 6,
            "clips": list(replication.CLIPS),
            "bootstrap": {"samples": 2000, "seed": 0},
            "checkpoint": {
                "step": 50000,
                "complete": True,
                "path": str(arm_dir / "ckpt.pt"),
                "sha256": f"{arm}-checkpoint",
            },
            "git": {"tracked_dirty": False},
            "evaluator_sha256": evaluator_hash or replication.EVALUATOR_SHA256,
        },
        "unitree_g1": {
            "num_windows": 42,
            "snmr": metrics,
            "per_clip": {"snmr": per_clip},
        },
    }
    (arm_dir / "benchmark.json").write_text(json.dumps(benchmark))


def _write_all_replication_arms(root, summaries):
    root.mkdir()
    for family in replication.FAMILIES:
        for seed in replication.REPLICATION_SEEDS:
            _write_replication_arm(root, family, seed, summaries[family][seed])


def test_gate1_replication_applies_three_seed_contract(tmp_path):
    summaries = {
        "c0": {
            0: _summary(0.4),
            1: _summary(0.42),
            2: _summary(0.38),
        },
        "c3_stance": {
            0: _summary(0.07, mpjpe=0.034),
            1: _summary(0.08, mpjpe=0.034),
            2: _summary(0.09, mpjpe=0.034),
        },
        "c4_teacher_velocity": {
            0: _summary(0.07, mpjpe=0.034),
            1: _summary(0.09, mpjpe=0.034),
            2: _summary(0.10, mpjpe=0.034),
        },
    }
    screen_root = tmp_path / "screen"
    replication_root = tmp_path / "replication"
    _write_screen(
        screen_root,
        {family: seed_summaries[0] for family, seed_summaries in summaries.items()},
    )
    _write_all_replication_arms(replication_root, summaries)

    report = replication.analyze_replication(replication_root, screen_root)

    assert report["passed"]
    assert report["gate1_passed"]
    assert report["gate1_passing_families"] == ["c3_stance"]
    assert report["candidate_decisions"]["c3_stance"]["endpoint_pass_seeds"] == [0, 1]
    assert not report["candidate_decisions"]["c4_teacher_velocity"]["passed"]


def test_gate1_replication_rejects_evaluator_mismatch(tmp_path):
    summaries = {
        family: {
            seed: _summary(0.4)
            for seed in replication.SEEDS
        }
        for family in replication.FAMILIES
    }
    screen_root = tmp_path / "screen"
    replication_root = tmp_path / "replication"
    _write_screen(
        screen_root,
        {family: seed_summaries[0] for family, seed_summaries in summaries.items()},
    )
    replication_root.mkdir()
    for family in replication.FAMILIES:
        for seed in replication.REPLICATION_SEEDS:
            _write_replication_arm(
                replication_root,
                family,
                seed,
                summaries[family][seed],
                evaluator_hash=(
                    "wrong"
                    if family == "c4_teacher_velocity" and seed == 2
                    else None
                ),
            )

    report = replication.analyze_replication(replication_root, screen_root)

    assert not report["passed"]
    errors = report["replication_arms"]["c4_teacher_velocity_seed2"]["errors"]
    assert any("evaluator_sha256" in error for error in errors)
