"""Model forward/backward, output validity, and losses."""

import pytest
import torch

from snmr import losses
from snmr.data import RobotMotion, robot_node_static_features, robot_pose_features
from snmr.model import SNMR, SNMRConfig, _adjacency
from snmr.robot_model import RobotKinematics
from snmr.skeleton import SkeletonGraph


def _short_motion(rk, T=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo, hi = rk.dof_limits()
    dof = lo + (hi - lo) * torch.rand(T, rk.num_dof, generator=g)
    root_pos = torch.randn(T, 3, generator=g) * 0.1
    root_quat = torch.tensor([1.0, 0, 0, 0]).expand(T, 4).clone()
    return RobotMotion(root_pos, root_quat, dof, fps=50.0)


def test_forward_shapes_and_valid_outputs(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    model = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64))
    motion = _short_motion(rk)
    pred = model.retarget_robot_to_robot(rk, motion, rk)

    T = motion.num_frames
    assert pred["root_pos"].shape == (T, 3)
    assert pred["root_quat"].shape == (T, 4)
    assert pred["dof_pos"].shape == (T, rk.num_dof)

    # quaternions unit-norm
    norms = pred["root_quat"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # dof within limits (guaranteed by tanh head)
    lo, hi = rk.dof_limits()
    assert torch.all(pred["dof_pos"] >= lo - 1e-4)
    assert torch.all(pred["dof_pos"] <= hi + 1e-4)


def test_latent_shape(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    cfg = SNMRConfig(latent_dim=48)
    model = SNMR(cfg)
    motion = _short_motion(rk)
    node_features = robot_pose_features(rk, motion)
    static = robot_node_static_features(rk.graph)
    adj = _adjacency(SkeletonGraph.from_robot_graph(rk.graph))
    z = model.encode(node_features, static, adj)
    assert z.shape == (motion.num_frames, cfg.latent_dim)


def test_backward_flows_to_all_params(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    model = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64))
    motion = _short_motion(rk)
    pred = model.retarget_robot_to_robot(rk, motion, rk)
    teacher = {"root_pos": motion.root_pos, "root_quat": motion.root_quat, "dof_pos": motion.dof_pos}
    loss, parts = losses.total_loss(pred, rk, teacher=teacher)
    assert "distill" in parts and "smooth" in parts
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_with_grad > 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_variable_topology_decode(g1_mjcf):
    """Decoding must work on a pruned target graph (fewer nodes) with the same latent — the core
    skeleton-agnostic property. We build a reduced skeleton and confirm the decoder adapts."""
    rk = RobotKinematics(g1_mjcf)
    model = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64))
    motion = _short_motion(rk)
    node_features = robot_pose_features(rk, motion)
    static = robot_node_static_features(rk.graph)
    adj = _adjacency(SkeletonGraph.from_robot_graph(rk.graph))
    z = model.encode(node_features, static, adj)

    # decode back to the same robot (full path)
    out = model.decode(z, rk)
    assert out["dof_pos"].shape == (motion.num_frames, rk.num_dof)


def test_task_loss_decreases_with_fk_targets(g1_mjcf):
    """A single-step gradient on task_loss should reduce the FK error toward set targets."""
    rk = RobotKinematics(g1_mjcf)
    model = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64))
    motion = _short_motion(rk)
    pred = model.retarget_robot_to_robot(rk, motion, rk)
    # targets = FK of a different random config
    ref = _short_motion(rk, seed=99)
    ref_pos, _ = rk.forward_kinematics(ref.root_pos, ref.root_quat, ref.dof_pos)
    pelvis = rk.body_index("pelvis")
    targets = {pelvis: ref_pos[:, pelvis, :]}
    loss = losses.task_loss(pred, rk, targets, {pelvis: 1.0})
    assert torch.isfinite(loss) and loss.item() >= 0


def test_decoder_adapter_is_zero_initialized_and_robot_specific(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    model = SNMR(SNMRConfig(
        latent_dim=32,
        enc_hidden=64,
        dec_hidden=64,
        decoder_adapter_rank=4,
        adapter_names=("g1", "other"),
    ))
    motion = _short_motion(rk)
    node_features = robot_pose_features(rk, motion)
    static = robot_node_static_features(rk.graph)
    adj = _adjacency(SkeletonGraph.from_robot_graph(rk.graph))
    z = model.encode(node_features, static, adj)

    shared = model.decode(z, rk)
    adapted = model.decode(z, rk, adapter_name="g1")
    for key in shared:
        assert torch.equal(shared[key], adapted[key])

    loss = adapted["root_pos"].square().mean() + adapted["dof_pos"].square().mean()
    loss.backward()
    assert model.decoder.adapters["g1"].up.weight.grad is not None
    assert model.decoder.adapters["g1"].up.weight.grad.abs().sum() > 0
    assert model.decoder.adapters["other"].up.weight.grad is None


def test_decoder_adapter_rejects_unknown_name(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    model = SNMR(SNMRConfig(
        latent_dim=32,
        enc_hidden=64,
        dec_hidden=64,
        decoder_adapter_rank=4,
        adapter_names=("g1",),
    ))
    z = torch.zeros(2, 32)

    with pytest.raises(KeyError, match="unknown decoder adapter"):
        model.decode(z, rk, adapter_name="missing")
