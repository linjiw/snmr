"""E50 export pipeline: synthetic recording -> segments + Stage-A metrics (no holosoma import)."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "export_e50_repaired_pairs.py"

mujoco = pytest.importorskip("mujoco")

from snmr import paths  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402


@pytest.fixture(scope="module")
def kin():
    return RobotKinematics(str(paths.g1_mjcf()))


def _make_reference(kin, frames: int) -> dict:
    """A static nominal-pose reference in WBT npz layout (joint_pos = qpos)."""
    dof = np.zeros((frames, kin.num_dof), dtype=np.float32)
    joint_pos = np.concatenate(
        [
            np.tile(np.array([0, 0, 0.76], dtype=np.float32), (frames, 1)),
            np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (frames, 1)),  # wxyz identity
            dof,
        ],
        axis=1,
    )
    return {"joint_pos": joint_pos, "fps": np.array([50])}


def _make_recording(kin, tmp_path: pathlib.Path, frames: int, envs: int) -> pathlib.Path:
    foot_names = [n for n in kin.body_names if "ankle_roll_link" in n]
    rng = np.random.default_rng(0)
    rec = {
        # constant per-env offset (not per-frame noise): per-frame jitter would read as
        # foot velocity at 50 Hz and contaminate the stance-speed readout
        "dof_pos": np.tile(
            rng.normal(0, 0.002, (1, envs, kin.num_dof)).astype(np.float32), (frames, 1, 1)
        ),
        "dof_vel": np.zeros((frames, envs, kin.num_dof), dtype=np.float32),
        "root_pos": np.tile(np.array([0, 0, 0.76], dtype=np.float32), (frames, envs, 1)),
        # recording layout is xyzw; identity = [0,0,0,1]
        "root_quat_xyzw": np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (frames, envs, 1)),
        "root_lin_vel": np.zeros((frames, envs, 3), dtype=np.float32),
        "root_ang_vel": np.zeros((frames, envs, 3), dtype=np.float32),
        "time_steps": np.tile(np.arange(frames, dtype=np.int64)[:, None], (1, envs)),
        "dones": np.zeros((frames, envs), dtype=bool),
        # both feet in firm contact throughout
        "feet_contact_force": np.full((frames, envs, len(foot_names)), 50.0, dtype=np.float32),
        "_metadata_json": np.array(
            json.dumps(
                {
                    "dt": 0.02,
                    "num_envs": envs,
                    "foot_body_names": foot_names,
                    "motion_file": "synthetic",
                    "motion_steps": frames,
                }
            )
        ),
    }
    path = tmp_path / "recording.npz"
    np.savez_compressed(path, **rec)
    return path


def test_export_static_pose_passes_gates(kin, tmp_path):
    frames, envs = 100, 3
    rec_path = _make_recording(kin, tmp_path, frames, envs)
    ref_path = tmp_path / "reference.npz"
    np.savez_compressed(ref_path, **_make_reference(kin, frames))
    out = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--recording", str(rec_path),
            "--reference", str(ref_path),
            "--out", str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads((out / "stage_a_metrics.json").read_text())
    # near-static rollout on a static reference: tiny MPJPE, no penetration, tiny stance speed
    assert metrics["num_accepted"] == envs
    assert metrics["mpjpe_mean_m"] < 0.02
    assert metrics["gates"]["H-A(ii)_penetration_approx_0"]
    assert metrics["gates"]["H-A(iii)_mpjpe_le_gate"]
    assert metrics["stance_speed_sim_ms"] < 0.05
    # settle-skip drops the first 25 recorded steps -> frames 25..99 covered
    assert metrics["reference_frame_coverage"] == pytest.approx((frames - 25) / frames)

    seg = np.load(out / "segments.npz")
    assert list(seg["accepted_env_ids"]) == [0, 1, 2]
    # settle-skip applied: recorded frames minus 25
    assert seg["env0_dof_pos"].shape == (frames - 25, kin.num_dof)
    # wxyz convention: w component first, ~1 for identity
    assert abs(seg["env0_root_quat_wxyz"][0, 0] - 1.0) < 1e-6


def test_export_rejects_high_error_env(kin, tmp_path):
    frames, envs = 80, 2
    rec_path = _make_recording(kin, tmp_path, frames, envs)
    # corrupt env 1: root shifted 30 cm -> MPJPE >> 5 cm gate
    with np.load(rec_path, allow_pickle=False) as data:
        rec = {k: np.asarray(data[k]) for k in data.files}
    rec["root_pos"][:, 1, 0] += 0.30
    np.savez_compressed(rec_path, **rec)

    ref_path = tmp_path / "reference.npz"
    np.savez_compressed(ref_path, **_make_reference(kin, frames))
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--recording", str(rec_path),
         "--reference", str(ref_path), "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    metrics = json.loads((out / "stage_a_metrics.json").read_text())
    assert metrics["num_accepted"] == 1
    rows = {r["env_id"]: r for r in metrics["per_env"]}
    assert rows[0]["mpjpe_gate_pass"] and not rows[1]["mpjpe_gate_pass"]
