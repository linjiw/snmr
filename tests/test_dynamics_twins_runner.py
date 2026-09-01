"""Fast runner-boundary regressions for the MuJoCo dynamics-twin worker."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest


pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_dynamics_twins as runner  # noqa: E402
from snmr.provenance import ArtifactSnapshot, MjcfBundleSnapshot  # noqa: E402
from snmr.robot_spec import (  # noqa: E402
    ControlSpec,
    FrameConvention,
    JointSpec,
    LinkSpec,
    RobotSpec,
    RuntimeSpec,
    SemanticManifest,
)
from snmr.verification import rollout_contract_hash  # noqa: E402


def _link(name: str, parent: str | None, xyz: tuple[float, float, float]) -> LinkSpec:
    return LinkSpec(
        name=name,
        parent=parent,
        local_position=xyz,
        local_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        mass=1.0,
        center_of_mass=(0.0, 0.0, 0.0),
        inertia_diagonal=(0.01, 0.01, 0.01),
        inertia_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def _robot_spec(asset_sha256: str) -> RobotSpec:
    spec = RobotSpec(
        schema_version="snmr.robot.v0.1",
        asset_sha256=asset_sha256,
        frames=FrameConvention(),
        semantics=SemanticManifest(root_link="pelvis"),
        control=ControlSpec(control_dt=0.02, latency_seconds=0.0),
        runtime=RuntimeSpec(simulation_dt=0.002),
        total_mass=2.0,
        standing_height=1.0,
        arm_span=None,
        links=(
            _link("pelvis", None, (0.0, 0.0, 0.0)),
            _link("left_hip_pitch_link", "pelvis", (0.0, 0.0, -0.5)),
        ),
        joints=(
            JointSpec(
                name="left_hip_pitch_joint",
                parent_link="pelvis",
                child_link="left_hip_pitch_link",
                joint_type="revolute",
                axis=(0.0, 1.0, 0.0),
                lower_limit=-100.0,
                upper_limit=100.0,
                velocity_limit=None,
                torque_limit=88.0,
                armature=0.01,
                damping=0.1,
                friction_loss=0.0,
                nominal_position=0.0,
                kp=40.179238471,
                kd=2.557889765,
            ),
        ),
    )
    spec.validate()
    return spec


def _captured_inputs(tmp_path: Path) -> tuple[ArtifactSnapshot, MjcfBundleSnapshot]:
    frames = 20
    joint_pos = np.zeros((frames, 8), dtype=np.float64)
    joint_pos[:, 3] = 1.0
    joint_pos[:, 7] = np.arange(frames, dtype=np.float64)
    joint_vel = np.zeros((frames, 7), dtype=np.float64)
    motion_path = tmp_path / "motion.npz"
    np.savez(
        motion_path,
        joint_names=np.asarray(["left_hip_pitch_joint"]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        fps=np.asarray([50.0]),
    )
    mjcf_path = tmp_path / "robot.xml"
    mjcf_path.write_text('<mujoco model="runner-test"><worldbody/></mujoco>')
    return ArtifactSnapshot.capture(motion_path), MjcfBundleSnapshot.capture(mjcf_path)


def _args(tmp_path: Path, *, start_frame: int = 10) -> argparse.Namespace:
    return argparse.Namespace(
        control_hz=50.0,
        effort_scales="0.5,1.0",
        seconds=0.08,
        start_frame=start_frame,
        saturation_threshold=0.10,
        torque_ratio_threshold=0.50,
        min_failure_frames=2,
        merge_gap_frames=0,
    )


def _revisions() -> dict[str, object]:
    result: dict[str, object] = {"provenance_schema_version": "snmr.provenance.v0.1"}
    for name, digest in (
        ("snmr", "1" * 40),
        ("newton", "2" * 40),
        ("isaac_lab", "3" * 40),
    ):
        result[f"{name}_commit"] = digest
        result[f"{name}_dirty"] = False
        result[f"{name}_repo_status"] = "available"
        result[f"{name}_repo_root"] = f"/{name}"
    return result


def _run_with_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rollout: dict[str, Any],
    *,
    start_frame: int = 10,
) -> dict[str, object]:
    motion_snapshot, asset_snapshot = _captured_inputs(tmp_path)
    spec = _robot_spec(asset_snapshot.entrypoint_snapshot.sha256)
    monkeypatch.setattr(runner.RobotSpec, "from_mjcf", lambda *args, **kwargs: spec)
    monkeypatch.setattr(runner, "replay", lambda *args, **kwargs: rollout)
    return runner._run_captured(
        _args(tmp_path, start_frame=start_frame),
        motion_path=motion_snapshot.source_path,
        mjcf_path=asset_snapshot.entrypoint_snapshot.source_path,
        runtime_motion=motion_snapshot.source_path,
        runtime_mjcf=asset_snapshot.entrypoint_snapshot.source_path,
        motion_snapshot=motion_snapshot,
        asset_snapshot=asset_snapshot,
        revisions=_revisions(),
    )


def _successful_rollout() -> dict[str, Any]:
    return {
        "diverged": False,
        "failure_frame": None,
        "nonfinite_state": False,
        "survived_fraction": 1.0,
        "survival_time_s": 0.08,
        "mean_dof_err_rad": 0.1,
        "mean_root_height_err_m": 0.01,
        "torque_saturation_fraction": 0.5,
        "max_requested_torque_ratio": 1.2,
        "trace": {
            "root_height_m": [0.9, 0.9, 0.9, 0.9],
            "torque_saturation_fraction": [0.0, 0.2, 0.2, 0.0],
        },
    }


def test_runner_offsets_window_failures_and_emits_shared_provenance_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    result = _run_with_stub(tmp_path, monkeypatch, _successful_rollout(), start_frame=10)

    report = result["reports"]["0.5"]
    saturation = [
        interval
        for interval in report["failure_intervals"]
        if interval["failure_type"] == "rollout_torque_saturation"
    ]
    assert [(item["start_frame"], item["end_frame"]) for item in saturation] == [(11, 12)]

    expected_contract = rollout_contract_hash(
        seconds=0.08,
        start_frame=10,
        fall_root_z_m=runner.FALL_ROOT_Z_M,
        root_xy_limit_m=runner.ROOT_XY_DEV_M,
        tilt_limit_rad=runner.TILT_LIMIT_RAD,
        saturation_threshold=0.10,
        min_failure_frames=2,
        merge_gap_frames=0,
    )
    assert result["config_hash"] == expected_contract == report["config_hash"]
    assert result["controller_hash"] == report["controller_hash"]
    assert report["backend_config_hash"] == result["backend_config_hash"]
    assert report["asset_bundle_sha256"] == result["asset_bundle_sha256"]
    assert result["input_artifacts"]["motion"]["sha256"] == result["motion_sha256"]
    assert result["command"] and result["command"][0]
    assert set(result["environment"]) == {"python", "numpy", "mujoco"}
    for name in ("snmr", "newton", "isaac_lab"):
        assert report[f"{name}_commit"] == result[f"{name}_commit"]
        assert report[f"{name}_dirty"] is False


def test_runner_rejects_nonfinite_required_rollout_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    rollout = _successful_rollout()
    rollout["survived_fraction"] = float("nan")
    with pytest.raises(ValueError, match="non-finite verification metric survival_fraction"):
        _run_with_stub(tmp_path, monkeypatch, rollout)


def test_runner_marks_nonfinite_divergence_failed_in_source_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    rollout = _successful_rollout()
    rollout.update({
        "diverged": True,
        "failure_frame": 10,
        "nonfinite_state": True,
        "survived_fraction": 0.0,
        "survival_time_s": 0.0,
        "mean_dof_err_rad": float("nan"),
        "mean_root_height_err_m": float("nan"),
    })
    result = _run_with_stub(tmp_path, monkeypatch, rollout, start_frame=10)
    report = result["reports"]["0.5"]
    assert report["passed"] is False
    divergence = [
        interval
        for interval in report["failure_intervals"]
        if interval["failure_type"] == "rollout_divergence"
    ]
    assert [(item["start_frame"], item["end_frame"]) for item in divergence] == [(10, 10)]
    metrics = dict(report["metrics"])
    assert "mean_dof_error_rad" not in metrics
    assert "mean_root_height_error_m" not in metrics


def test_runner_main_refuses_to_overwrite_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "existing.json"
    output.write_text("keep-me\n")

    def should_not_run(args: argparse.Namespace) -> dict[str, object]:
        raise AssertionError("runner must check output safety before reading inputs")

    monkeypatch.setattr(runner, "run", should_not_run)
    monkeypatch.setattr(sys, "argv", ["experiment_dynamics_twins.py", "--out", str(output)])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.main()
    assert output.read_text() == "keep-me\n"
