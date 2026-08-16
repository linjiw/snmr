"""CPU tests for snmr/integration/goal_window.py (explicit future-goal window, E78 mGf / C-future)."""

import math
import types

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from snmr.integration.goal_window import (
    GOAL_DIM,
    GoalWindow,
    raw_goal_at_offsets,
    relative_orientation_6d,
    rotation_6d_from_quaternion,
)


def _quat_xyzw(rot):
    return torch.tensor(rot.as_quat(), dtype=torch.float64)  # scipy is xyzw


def test_rotation_6d_matches_first_two_matrix_columns():
    rng = np.random.default_rng(0)
    rots = R.random(16, random_state=1)
    got = rotation_6d_from_quaternion(_quat_xyzw(rots))
    want = torch.tensor(rots.as_matrix()[..., :2].reshape(16, -1), dtype=torch.float64)
    assert torch.allclose(got, want, atol=1e-12)


def test_relative_orientation_is_robot_inverse_times_ref():
    robot = R.random(8, random_state=2)
    ref = R.random(8, random_state=3)
    got = relative_orientation_6d(_quat_xyzw(robot), _quat_xyzw(ref))
    want = torch.tensor((robot.inv() * ref).as_matrix()[..., :2].reshape(8, -1), dtype=torch.float64)
    assert torch.allclose(got, want, atol=1e-12)


def _fake_motion_command(T=100, n_env=3, ref_body=2):
    motion = types.SimpleNamespace(
        joint_pos=torch.arange(T, dtype=torch.float64)[:, None].repeat(1, 29),
        joint_vel=-torch.arange(T, dtype=torch.float64)[:, None].repeat(1, 29),
        body_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(T, 5, 1),
        motion_end_idx=torch.tensor([T]),
    )
    # a yaw ramp on the ref body so the future orientation differs from the current one
    yaws = torch.linspace(0, math.pi / 2, T)
    motion.body_quat_w[:, ref_body] = torch.tensor(R.from_euler("z", yaws.numpy()).as_quat())
    return types.SimpleNamespace(
        motion=motion,
        ref_body_index=ref_body,
        time_steps=torch.tensor([0, 50, 97]),
        motion_ids=torch.zeros(n_env, dtype=torch.long),
        robot_ref_quat_w=torch.tensor(R.from_euler("z", [0.0, 0.1, 0.2]).as_quat()),
    )


def test_raw_goal_offsets_clamp_at_clip_end_and_use_current_robot_frame():
    mc = _fake_motion_command()
    g0, g5 = raw_goal_at_offsets(mc, (0, 5))
    assert g0.shape == (3, GOAL_DIM) and g5.shape == (3, GOAL_DIM)
    # joint columns come from frames t and t+5 (clamped to 99 for the env at 97)
    assert g0[:, 0].tolist() == [0.0, 50.0, 97.0]
    assert g5[:, 0].tolist() == [5.0, 55.0, 99.0]
    assert g5[:, 29].tolist() == [-5.0, -55.0, -99.0]
    # orientation block: robot yaw 0.1 at env 1, ref yaw at frame 55 -> rel yaw = yaw55 - 0.1
    yaw55 = float(torch.linspace(0, math.pi / 2, 100)[55])
    want = torch.tensor(R.from_euler("z", yaw55 - 0.1).as_matrix()[:, :2].reshape(-1))
    assert torch.allclose(g5[1, 58:], want.to(g5.dtype), atol=1e-6)


def test_goal_window_normalises_each_sample_with_the_goal_slice_stats():
    mc = _fake_motion_command()
    mean = torch.zeros(GOAL_DIM, dtype=torch.float64); mean[0] = 10.0
    std = torch.ones(GOAL_DIM, dtype=torch.float64) * 2.0
    gw = GoalWindow(mean, std, eps=0.0, offsets=(0, 5))
    assert gw.dim == 128
    out = gw(mc)
    assert out.shape == (3, 128)
    assert out[:, 0].tolist() == [(0 - 10) / 2, (50 - 10) / 2, (97 - 10) / 2]
    assert out[:, 64].tolist() == [(5 - 10) / 2, (55 - 10) / 2, (99 - 10) / 2]
    with pytest.raises(ValueError):
        GoalWindow(mean, std, 0.0, offsets=(5,))
