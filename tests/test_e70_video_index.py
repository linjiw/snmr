from __future__ import annotations

import hashlib
import pathlib

import pytest

from scripts.index_e70_video_captures import validate_capture_report


def _item() -> dict:
    return {
        "name": "snmr_walk1_subject1",
        "arm": "a_prior_snmr",
        "phase_only": False,
        "shuffle_latent": False,
        "destroy_zcmd": "none",
        "side": 0,
        "clip": "walk1_subject1",
        "start_step": 33,
        "checkpoint_sha256": "checkpoint-hash",
    }


def _report(tmp_path) -> dict:
    teacher_paths = [tmp_path / "teacher0.pt", tmp_path / "teacher1.pt"]
    motion_paths = [tmp_path / "motion0.npz", tmp_path / "motion1.npz"]
    for index, path in enumerate((*teacher_paths, *motion_paths)):
        path.write_bytes(f"fixture-{index}".encode())

    def digest(path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    runtime = pathlib.Path(__file__).parents[1] / "scripts" / "eval_e70_video.py"
    return {
        "protocol": "E70 exact simulation capture report v1",
        "capture_name": "snmr_walk1_subject1",
        "arm": "a_prior_snmr",
        "phase_only": False,
        "shuffle_latent": False,
        "destroy_zcmd": "none",
        "evaluation_seed": 404,
        "num_rollouts": 1,
        "simulator_num_envs": 1,
        "intervention_pool_size": 1,
        "exact_start": 33,
        "start_steps": [33],
        "motion_ids": [0],
        "clip": "walk1_subject1",
        "video_capture": True,
        "completed": [True],
        "survival_s": [10.0],
        "steps_executed": 500,
        "student_checkpoint_sha256": "checkpoint-hash",
        "teacher_ckpts": [str(path) for path in teacher_paths],
        "teacher_checkpoint_sha256": [digest(path) for path in teacher_paths],
        "motion_files": [str(path) for path in motion_paths],
        "motion_sha256": [digest(path) for path in motion_paths],
        "runtime": str(runtime),
        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "video_config": {
            "enabled": True,
            "width": 1920,
            "height": 1080,
            "playback_rate": 1.0,
            "record_env_id": 0,
            "vertical_fov": 45.0,
            "use_recording_thread": False,
            "camera": {
                "type": "cartesian",
                "offset": [2.0, 2.0, 1.0],
                "target_offset": [0.0, 0.0, 0.3],
                "smoothing": 0.95,
                "tracking_body_name": "pelvis",
            },
        },
    }


def test_validate_capture_report_accepts_exact_contract(tmp_path) -> None:
    validate_capture_report(_item(), _report(tmp_path))


@pytest.mark.parametrize("field,value", [("exact_start", 34), ("evaluation_seed", 0)])
def test_validate_capture_report_rejects_identity_change(tmp_path, field: str, value: object) -> None:
    report = _report(tmp_path)
    report[field] = value
    with pytest.raises(ValueError, match=field):
        validate_capture_report(_item(), report)


def test_validate_capture_report_rejects_low_resolution(tmp_path) -> None:
    report = _report(tmp_path)
    report["video_config"]["height"] = 360
    with pytest.raises(ValueError, match="height"):
        validate_capture_report(_item(), report)


def test_validate_capture_report_rejects_inconsistent_termination(tmp_path) -> None:
    report = _report(tmp_path)
    report["completed"] = [False]
    report["steps_executed"] = 240
    report["survival_s"] = [10.0]
    with pytest.raises(ValueError, match="survival"):
        validate_capture_report(_item(), report)


def test_validate_capture_report_allows_failure_on_final_step(tmp_path) -> None:
    report = _report(tmp_path)
    report["completed"] = [False]
    validate_capture_report(_item(), report)


def test_validate_capture_report_rejects_early_claimed_completion(tmp_path) -> None:
    report = _report(tmp_path)
    report["steps_executed"] = 499
    report["survival_s"] = [9.98]
    with pytest.raises(ValueError, match="completion disagrees"):
        validate_capture_report(_item(), report)


def test_validate_capture_report_rejects_camera_drift(tmp_path) -> None:
    report = _report(tmp_path)
    report["video_config"]["camera"]["offset"] = [5.0, 5.0, 3.0]
    with pytest.raises(ValueError, match="camera"):
        validate_capture_report(_item(), report)


def test_validate_capture_report_requires_population_for_marginal_intervention(tmp_path) -> None:
    item, report = _item(), _report(tmp_path)
    item["destroy_zcmd"] = report["destroy_zcmd"] = "marginal_random"
    report["simulator_num_envs"] = 1024
    report["intervention_pool_size"] = 1024
    validate_capture_report(item, report)

    report["intervention_pool_size"] = 1
    with pytest.raises(ValueError, match="intervention_pool_size"):
        validate_capture_report(item, report)
