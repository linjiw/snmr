"""N8 export correctness: MuJoCo-replay angular velocity must be world-frame (matches holosoma)."""

import sys
import pathlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))


def test_replay_angular_velocity_is_world_frame(g1_mjcf):
    """A base yawing at a constant world-z rate while tilted must yield body_ang_vel_w ≈ [0,0,rate]
    in the WORLD frame — not the body frame. Guards the q1*conj(q0) vs conj(q0)*q1 cross-product
    order (a review-confirmed bug: the flipped order returned body-frame angular velocity)."""
    import mujoco
    from export_wbt_npz import mujoco_replay

    fps, T, dt, rate = 50.0, 10, 1 / 50.0, 4.0
    m = mujoco.MjModel.from_xml_path(g1_mjcf)
    qpos = np.zeros((T, m.nq))
    qpos[:, 2] = 0.8
    tilt = R.from_rotvec([1.2, 0.0, 0.0])  # non-trivial tilt so body != world frame
    for t in range(T):
        q = (R.from_rotvec([0, 0, rate * t * dt]) * tilt).as_quat()  # xyzw
        qpos[t, 3:7] = np.r_[q[3], q[:3]]  # wxyz

    out = mujoco_replay(g1_mjcf, qpos, fps)
    pelvis = list(out["body_names"]).index("pelvis")
    w = out["body_ang_vel_w"][5, pelvis]
    assert abs(w[2] - rate) < 0.05, f"world yaw rate wrong: {w}"
    assert abs(w[0]) < 0.05 and abs(w[1]) < 0.05, f"leaked body-frame components: {w}"


def test_angular_velocity_centered_alignment(g1_mjcf):
    """ang_vel must be a CENTERED stencil aligned at frame t (like lin_vel), not a backward
    difference at t-0.5. On a reversing world-z rotation omega(t)=A·cos(2πft) the centered estimate
    tracks true(t) to <0.06 rad/s RMS; a backward difference would sit ~0.27 rad/s off."""
    import mujoco
    from export_wbt_npz import mujoco_replay

    fps, dt, T, f, A = 50.0, 1 / 50.0, 40, 2.0, 3.0
    m = mujoco.MjModel.from_xml_path(g1_mjcf)
    qpos = np.zeros((T, m.nq))
    qpos[:, 2] = 0.8
    ang = A / (2 * np.pi * f) * np.sin(2 * np.pi * f * np.arange(T) * dt)
    for t in range(T):
        q = R.from_rotvec([0, 0, ang[t]]).as_quat()
        qpos[t, 3:7] = np.r_[q[3], q[:3]]
    out = mujoco_replay(g1_mjcf, qpos, fps)
    pelvis = list(out["body_names"]).index("pelvis")
    w = out["body_ang_vel_w"][:, pelvis, 2]
    true_t = A * np.cos(2 * np.pi * f * np.arange(T) * dt)
    rms = np.sqrt(np.mean((w[3:-3] - true_t[3:-3]) ** 2))
    assert rms < 0.06, f"ang_vel not centered-aligned: RMS {rms}"


def test_resample_endpoints(g1_mjcf):
    """resample_qpos must preserve the first/last frames and unit root quats."""
    from export_wbt_npz import resample_qpos

    T = 30
    qpos = np.zeros((T, 36))
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0  # identity wxyz
    qpos[:, 7:] = np.linspace(0, 1, T)[:, None]  # ramp on dof
    out = resample_qpos(qpos, src_fps=30.0, dst_fps=50.0)
    assert np.allclose(out[0], qpos[0], atol=1e-6)
    assert np.allclose(out[-1, 7:], qpos[-1, 7:], atol=1e-3)
    qn = np.linalg.norm(out[:, 3:7], axis=1)
    assert np.allclose(qn, 1.0, atol=1e-5)
