import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_gate1_screen.py"
SPEC = importlib.util.spec_from_file_location("analyze_gate1_screen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


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
            clip: speed for clip in screen.CLIPS
        },
    }


def _write_arm(root, arm, summary, *, evaluator_hash="eval-sha"):
    arm_dir = root / arm
    arm_dir.mkdir()
    config = {
        **screen.BASE_CONFIG,
        **screen.OBJECTIVE_CONFIG[arm],
        "out": str(arm_dir),
    }
    manifest = {
        "status": "completed",
        "progress": {"completion_state": "completed", "step": 50000},
        "git": {
            "sha": screen.TRAIN_REVISION,
            "dirty": False,
        },
        "config": config,
    }
    (arm_dir / "manifest.json").write_text(json.dumps(manifest))
    snmr = {
        key: value
        for key, value in summary.items()
        if key != "per_clip_teacher_height_stance_speed_ms"
    }
    per_clip = {
        clip: (
            {"teacher_height_contact_samples": 0.0}
            if value is None
            else {
                "teacher_height_stance_speed_ms": value,
                "teacher_height_contact_samples": 1.0,
            }
        )
        for clip, value in summary["per_clip_teacher_height_stance_speed_ms"].items()
    }
    benchmark = {
        "_protocol": {
            "window_frames": 192,
            "windows_per_clip_max": 6,
            "clips": list(screen.CLIPS),
            "bootstrap": {"samples": 2000, "seed": 0},
            "checkpoint": {
                "step": 50000,
                "complete": True,
                "path": str(arm_dir / "ckpt.pt"),
                "sha256": f"{arm}-ckpt",
            },
            "git": {"tracked_dirty": False},
            "evaluator_sha256": evaluator_hash,
        },
        "unitree_g1": {
            "num_windows": 42,
            "snmr": snmr,
            "per_clip": {"snmr": per_clip},
        },
    }
    (arm_dir / "benchmark.json").write_text(json.dumps(benchmark))


def test_gate1_screen_promotes_only_candidate_passing_all_guards(tmp_path):
    control = _summary(0.4)
    c1 = _summary(0.4)
    c3 = _summary(0.25, mpjpe=0.034, source_speed=0.31, jerk=700.0)
    c4 = _summary(0.2, mpjpe=0.04, source_speed=0.31, jerk=700.0)
    c4["per_clip_teacher_height_stance_speed_ms"][screen.CLIPS[0]] = 0.5
    c4["per_clip_teacher_height_stance_speed_ms"][screen.CLIPS[1]] = 0.5
    c4["per_clip_teacher_height_stance_speed_ms"][screen.CLIPS[2]] = 0.5
    for arm, summary in zip(screen.ARMS, (control, c1, c3, c4), strict=True):
        _write_arm(tmp_path, arm, summary)

    report = screen.analyze_screen(tmp_path)

    assert report["passed"]
    assert report["promoted_arms"] == ["c3_stance_seed0"]
    assert report["candidate_decisions"]["c3_stance_seed0"]["passed"]
    assert not report["candidate_decisions"]["c4_teacher_velocity_seed0"]["passed"]
    assert not report["candidate_decisions"]["c4_teacher_velocity_seed0"][
        "checks"
    ]["at_least_5_of_7_clips_improve"]


def test_gate1_screen_rejects_protocol_mismatch(tmp_path):
    for arm in screen.ARMS:
        _write_arm(
            tmp_path,
            arm,
            _summary(0.4),
            evaluator_hash="different" if arm == "c4_teacher_velocity_seed0" else "same",
        )

    report = screen.analyze_screen(tmp_path)

    assert not report["passed"]
    assert "evaluator hashes" in report["protocol_errors"][0]


def test_gate1_screen_treats_zero_support_clip_as_no_improvement(tmp_path):
    control = _summary(0.4)
    c1 = _summary(0.4)
    c3 = _summary(0.25, mpjpe=0.034, source_speed=0.31, jerk=700.0)
    c4 = _summary(0.4)
    unavailable_clip = screen.CLIPS[2]
    for summary in (control, c1, c3, c4):
        summary["per_clip_teacher_height_stance_speed_ms"][unavailable_clip] = None
    for arm, summary in zip(screen.ARMS, (control, c1, c3, c4), strict=True):
        _write_arm(tmp_path, arm, summary)

    report = screen.analyze_screen(tmp_path)

    assert report["passed"]
    decision = report["candidate_decisions"]["c3_stance_seed0"]
    assert decision["passed"]
    assert decision["clip_improvements"] == 6
    assert decision["comparable_clips"] == [
        clip for clip in screen.CLIPS if clip != unavailable_clip
    ]
    assert decision["unavailable_clips"] == [unavailable_clip]


def test_gate1_screen_rejects_malformed_per_clip_value(tmp_path):
    for arm in screen.ARMS:
        summary = _summary(0.4)
        if arm == "c3_stance_seed0":
            summary["per_clip_teacher_height_stance_speed_ms"][screen.CLIPS[0]] = "bad"
        _write_arm(tmp_path, arm, summary)

    report = screen.analyze_screen(tmp_path)

    assert not report["passed"]
    errors = report["arms"]["c3_stance_seed0"]["errors"]
    assert any("malformed or nonfinite" in error for error in errors)
