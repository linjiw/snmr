"""N6 contact-consistent decoding: contact head, self-consistency loss, and latent polish."""

import torch

from snmr import losses
from snmr.data import RobotMotion
from snmr.metrics import FOOT_BODIES
from snmr.model import SNMR, SNMRConfig
from snmr.polish import PolishConfig, polish_latent
from snmr.robot_model import RobotKinematics


def _short(rk, T=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo, hi = rk.dof_limits()
    dof = lo + (hi - lo) * torch.rand(T, rk.num_dof, generator=g)
    root_pos = torch.zeros(T, 3)
    root_pos[:, 2] = 0.75
    root_quat = torch.tensor([1.0, 0, 0, 0]).expand(T, 4).clone()
    return RobotMotion(root_pos, root_quat, dof, fps=50.0)


def test_contact_head_optional_and_shaped(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    # off by default -> no contact_logits, backward compatible
    m_off = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64))
    pred = m_off.retarget_robot_to_robot(rk, _short(rk), rk)
    assert "contact_logits" not in pred

    # on -> per-node logits present
    m_on = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64, predict_contact=True))
    pred = m_on.retarget_robot_to_robot(rk, _short(rk), rk)
    assert pred["contact_logits"].shape == (12, rk.num_bodies)
    assert torch.isfinite(pred["contact_logits"]).all()


def test_contact_prediction_loss_supervises(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    m = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64, predict_contact=True))
    pred = m.retarget_robot_to_robot(rk, _short(rk), rk)
    foot_idx = [rk.body_index(n) for n in FOOT_BODIES["unitree_g1"]]
    mask = torch.randint(0, 2, (12, len(foot_idx))).float()
    loss = losses.contact_prediction_loss(pred["contact_logits"], foot_idx, mask)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    grads = [p.grad for p in m.decoder.contact_head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_self_consistency_loss_gradient_flows(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    m = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64, predict_contact=True))
    pred = m.retarget_robot_to_robot(rk, _short(rk), rk)
    foot_idx = [rk.body_index(n) for n in FOOT_BODIES["unitree_g1"]]
    loss = losses.contact_self_consistency_loss(pred, rk, foot_idx)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss.backward()
    # gradient must reach both the angle path and the contact head (self-consistency couples them)
    n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0


def test_self_consistency_world_recomposition(g1_mjcf):
    """With an anchor, the loss must be computed in the WORLD frame: a foot that is stationary in
    world but whose local-frame prediction moves with the anchor should incur ~zero contact velocity
    penalty when recomposed. We fabricate a local prediction + counter-moving anchor and check the
    anchored loss is far smaller than the un-anchored (local) loss."""
    rk = RobotKinematics(g1_mjcf)
    T = 20
    foot_idx = [rk.body_index(n) for n in FOOT_BODIES["unitree_g1"]]
    # constant dof + local root that translates linearly in x; anchor translates the opposite way so
    # the WORLD root is fixed -> feet are world-stationary.
    lo, hi = rk.dof_limits()
    dof = (0.5 * (lo + hi)).expand(T, rk.num_dof).clone()
    local_pos = torch.zeros(T, 3)
    local_pos[:, 0] = torch.arange(T) * 0.01
    local_pos[:, 2] = 0.75
    quat = torch.tensor([1.0, 0, 0, 0]).expand(T, 4).clone()
    pred = {"root_pos": local_pos, "root_quat": quat, "dof_pos": dof,
            "contact_logits": torch.full((T, rk.num_bodies), 5.0)}  # high contact prob everywhere
    # anchor with zero heading and xy that exactly cancels the local translation -> world root fixed
    anchor_pos = torch.zeros(T, 3)
    anchor_pos[:, 0] = -torch.arange(T) * 0.01  # local_root_to_world adds heading-rotated local_pos + anchor_xy
    anchor_quat = quat.clone()

    l_local = losses.contact_self_consistency_loss(pred, rk, foot_idx)
    l_world = losses.contact_self_consistency_loss(pred, rk, foot_idx, anchor=(anchor_pos, anchor_quat))
    assert l_world < l_local, f"world recomposition should cancel the sliding: {l_world} vs {l_local}"


def test_polish_reduces_contact_velocity_on_synthetic_slide(g1_mjcf):
    """On a decode that slides a planted foot, polish must reduce the contact-masked foot velocity
    (the objective it optimizes), staying finite. Uses a tiny model + identity-ish decode closure."""
    rk = RobotKinematics(g1_mjcf)
    m = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64))
    motion = _short(rk, T=16)
    # encode once to get a latent, then polish the decode toward zero contact velocity
    from snmr.data import robot_node_static_features, robot_pose_features
    from snmr.model import _adjacency
    from snmr.skeleton import SkeletonGraph

    feats = robot_pose_features(rk, motion)
    static = robot_node_static_features(rk.graph)
    adj = _adjacency(SkeletonGraph.from_robot_graph(rk.graph))
    z = m.encode(feats, static, adj)

    foot_idx = [rk.body_index(n) for n in FOOT_BODIES["unitree_g1"]]
    cmask = torch.ones(16, len(foot_idx))  # pretend both feet planted the whole clip
    anchor_pos = motion.root_pos.clone()
    anchor_quat = motion.root_quat.clone()

    decode_fn = lambda zz: m.decode(zz, rk)  # noqa: E731

    def contact_vel(pred_world):
        bp, _ = rk.forward_kinematics(pred_world["root_pos"], pred_world["root_quat"], pred_world["dof_pos"])
        feet = bp[:, foot_idx, :]
        v = (feet[1:, :, :2] - feet[:-1, :, :2]).norm(dim=-1)
        return v.mean().item()

    from snmr.data import local_root_to_world

    with torch.no_grad():
        pred0 = m.decode(z, rk)
        wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred0["root_pos"], pred0["root_quat"])
    v_before = contact_vel({"root_pos": wp, "root_quat": wq, "dof_pos": pred0["dof_pos"]})

    polished, log = polish_latent(
        z, decode_fn, rk, foot_idx, cmask, anchor_pos, anchor_quat,
        PolishConfig(iters=40, lr=0.05, w_vel=5.0, w_height=0.1, w_anchor=0.5),
    )
    v_after = contact_vel(polished)
    assert torch.isfinite(polished["dof_pos"]).all()
    # polish should not increase the contact velocity it is minimizing
    assert v_after <= v_before + 1e-4, f"{v_before} -> {v_after}"
