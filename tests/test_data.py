"""Data loading + graph feature extraction, validated against the real holosoma NPZ."""

import numpy as np
import torch

from snmr import rotation as rot
from snmr.data import (
    heading_quat_inverse,
    load_holosoma_wbt_npz,
    robot_node_static_features,
    robot_pose_features,
)
from snmr.robot_model import RobotKinematics
from snmr.skeleton import SkeletonGraph, smplx_body_skeleton


def test_load_real_npz_shapes(g1_train_npz):
    m = load_holosoma_wbt_npz(g1_train_npz)
    assert m.dof_pos.shape[1] == 29
    assert m.root_pos.shape[1] == 3 and m.root_quat.shape[1] == 4
    assert m.num_frames > 10
    assert m.fps == 50
    # root quats should be ~unit (wxyz on disk)
    norms = m.root_quat.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_qpos_layout_roundtrip(g1_train_npz):
    m = load_holosoma_wbt_npz(g1_train_npz)
    q = m.qpos()
    assert q.shape == (m.num_frames, 7 + 29)
    assert torch.allclose(q[:, 0:3], m.root_pos)
    assert torch.allclose(q[:, 3:7], m.root_quat)


def test_pose_features_heading_translation_invariant(g1_mjcf, g1_train_npz):
    rk = RobotKinematics(g1_mjcf)
    m = load_holosoma_wbt_npz(g1_train_npz)
    # take a short window
    from snmr.data import RobotMotion

    m = RobotMotion(m.root_pos[:20], m.root_quat[:20], m.dof_pos[:20], m.fps)
    feats_a = robot_pose_features(rk, m)

    # apply a global translation + yaw rotation; features should be (almost) unchanged
    zaxis = torch.zeros(m.num_frames, 3)
    zaxis[:, 2] = 1.0
    yaw = torch.full((m.num_frames,), 0.7)
    dq = rot.axis_angle_to_quat(zaxis, yaw)
    new_quat = rot.quat_mul(dq, m.root_quat)
    shift = torch.tensor([3.0, -2.0, 0.0])
    # rotate + translate the root position about origin in the plane
    new_pos = rot.quat_rotate(dq, m.root_pos) + shift
    m2 = RobotMotion(new_pos, new_quat, m.dof_pos, m.fps)
    feats_b = robot_pose_features(rk, m2)

    assert torch.allclose(feats_a, feats_b, atol=1e-4), (feats_a - feats_b).abs().max().item()


def test_world_local_root_roundtrip():
    """world -> human-heading-local -> world must be identity (used by the Phase-1 trainer)."""
    from snmr.data import local_root_to_world, world_root_to_local

    torch.manual_seed(3)
    T = 32
    anchor_pos = torch.randn(T, 3)
    anchor_quat = rot.quat_normalize(torch.randn(T, 4))
    root_pos = torch.randn(T, 3)
    root_quat = rot.quat_normalize(torch.randn(T, 4))

    lp, lq = world_root_to_local(anchor_pos, anchor_quat, root_pos, root_quat)
    wp, wq = local_root_to_world(anchor_pos, anchor_quat, lp, lq)
    assert torch.allclose(wp, root_pos, atol=1e-5)
    ang = rot.quat_geodesic_angle(wq, root_quat)
    assert torch.all(ang < 1e-4)

    # and the local pose must be invariant to a global yaw+planar-shift of both poses
    zaxis = torch.zeros(T, 3)
    zaxis[:, 2] = 1.0
    dq = rot.axis_angle_to_quat(zaxis, torch.full((T,), 1.1))
    shift = torch.tensor([4.0, -3.0, 0.0])
    a_pos2 = rot.quat_rotate(dq, anchor_pos) + shift
    a_quat2 = rot.quat_mul(dq, anchor_quat)
    r_pos2 = rot.quat_rotate(dq, root_pos) + shift
    r_quat2 = rot.quat_mul(dq, root_quat)
    lp2, lq2 = world_root_to_local(a_pos2, a_quat2, r_pos2, r_quat2)
    # z is absolute; a yaw about the origin changes z only if the rotation axis isn't vertical — it is.
    assert torch.allclose(lp, lp2, atol=1e-4), (lp - lp2).abs().max().item()
    ang2 = rot.quat_geodesic_angle(lq, lq2)
    assert torch.all(ang2 < 1e-3)


def test_heading_inverse_removes_yaw():
    zaxis = torch.tensor([[0.0, 0.0, 1.0]])
    q = rot.axis_angle_to_quat(zaxis, torch.tensor([1.234]))
    inv = heading_quat_inverse(q)
    net = rot.quat_mul(inv, q)
    # net should have no yaw: rotating +x stays in +x direction
    fwd = rot.quat_rotate(net, torch.tensor([[1.0, 0.0, 0.0]]))
    assert abs(fwd[0, 1].item()) < 1e-5


def test_static_features_shape(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    sf = robot_node_static_features(rk.graph)
    assert sf.shape == (rk.num_bodies, 8)
    # exactly 29 bodies should carry a dof flag
    assert int(sf[:, 6].sum().item()) == 29


def test_skeleton_from_robot_graph(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    sk = SkeletonGraph.from_robot_graph(rk.graph)
    assert sk.num_nodes == rk.num_bodies
    assert sk.is_end_effector.sum() > 0
    ei = sk.edge_index()
    assert ei.shape[0] == 2 and ei.shape[1] == 2 * (rk.num_bodies - 1)


def test_smplx_skeleton_topology():
    sk = smplx_body_skeleton()
    assert sk.num_nodes == 22
    # feet, hands, head are end effectors
    for name in ["left_foot", "right_foot", "left_wrist", "right_wrist", "head"]:
        assert sk.is_end_effector[sk.names.index(name)]
    # pelvis is root
    assert int(sk.parent_index[0]) == -1
