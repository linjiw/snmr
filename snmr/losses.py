"""Training losses for SNMR (see NEURAL_RETARGETING_DESIGN.md §3.2).

All losses take a predicted robot configuration (root_pos, root_quat, dof_pos) and a target robot
:class:`RobotKinematics`, and return scalar tensors. Rotation losses use the 6D / matrix formulation
(smooth at zero angle) rather than the acos-based geodesic (which is only used as an eval metric).
"""

from __future__ import annotations

import torch

from . import rotation as rot
from .robot_model import RobotKinematics


def distill_loss(
    pred: dict[str, torch.Tensor],
    teacher_root_pos: torch.Tensor,
    teacher_root_quat: torch.Tensor,
    teacher_dof: torch.Tensor,
    rot_weight: float = 1.0,
) -> torch.Tensor:
    """Dense supervision from an optimization teacher (GMR / holosoma) in configuration space."""
    l_pos = torch.mean((pred["root_pos"] - teacher_root_pos) ** 2)
    l_dof = torch.mean((pred["dof_pos"] - teacher_dof) ** 2)
    m_pred = rot.quat_to_matrix(pred["root_quat"])
    m_tgt = rot.quat_to_matrix(teacher_root_quat)
    l_rot = torch.mean((m_pred - m_tgt) ** 2)
    return l_pos + l_dof + rot_weight * l_rot


def task_loss(
    pred: dict[str, torch.Tensor],
    target_kin: RobotKinematics,
    body_targets: dict[int, torch.Tensor],
    body_weights: dict[int, float],
) -> torch.Tensor:
    """Self-supervised task-space keypoint matching (GBC-style) through differentiable FK.

    Args:
        body_targets: {body_index -> (T, 3) desired world position}.
        body_weights: {body_index -> scalar weight}.
    """
    body_pos, _ = target_kin.forward_kinematics(pred["root_pos"], pred["root_quat"], pred["dof_pos"])
    total = body_pos.new_zeros(())
    wsum = 0.0
    for bidx, tgt in body_targets.items():
        w = body_weights.get(bidx, 1.0)
        total = total + w * torch.mean((body_pos[:, bidx, :] - tgt) ** 2)
        wsum += w
    return total / max(wsum, 1e-6)


def joint_limit_loss(pred: dict[str, torch.Tensor], target_kin: RobotKinematics) -> torch.Tensor:
    """Soft penalty for exceeding joint limits. Near-zero when the tanh-scaled head is used, but kept
    as a safety term (and to remain valid if the head parametrisation changes)."""
    lo, hi = target_kin.dof_limits()
    dof = pred["dof_pos"]
    over = torch.relu(dof - hi.to(dof.dtype))
    under = torch.relu(lo.to(dof.dtype) - dof)
    return torch.mean(over ** 2 + under ** 2)


def smoothness_loss(pred: dict[str, torch.Tensor]) -> torch.Tensor:
    """Acceleration + jerk penalty on dof and root position for RL-consumable smoothness."""
    dof = pred["dof_pos"]
    pos = pred["root_pos"]
    if dof.shape[0] < 3:
        return dof.new_zeros(())
    acc_dof = dof[2:] - 2 * dof[1:-1] + dof[:-2]
    acc_pos = pos[2:] - 2 * pos[1:-1] + pos[:-2]
    loss = torch.mean(acc_dof ** 2) + torch.mean(acc_pos ** 2)
    if dof.shape[0] >= 4:
        jerk_dof = dof[3:] - 3 * dof[2:-1] + 3 * dof[1:-2] - dof[:-3]
        loss = loss + 0.1 * torch.mean(jerk_dof ** 2)
    return loss


def foot_contact_loss(
    pred: dict[str, torch.Tensor],
    target_kin: RobotKinematics,
    foot_body_indices: list[int],
    contact_mask: torch.Tensor,
    ground_z: float = 0.0,
) -> torch.Tensor:
    """SAME-style contact loss on robot feet: penalise foot XY velocity during contact + ground
    penetration.

    Args:
        foot_body_indices: robot body indices to treat as feet.
        contact_mask: (T, len(foot_body_indices)) bool/float, 1 where that foot is in contact.
    """
    body_pos, _ = target_kin.forward_kinematics(pred["root_pos"], pred["root_quat"], pred["dof_pos"])
    feet = body_pos[:, foot_body_indices, :]  # (T, F, 3)
    T = feet.shape[0]
    loss = feet.new_zeros(())
    if T > 1:
        vel_xy = feet[1:, :, :2] - feet[:-1, :, :2]  # (T-1, F, 2)
        cm = contact_mask[1:].to(feet.dtype)         # align to velocity frames
        loss = loss + torch.mean(cm.unsqueeze(-1) * vel_xy ** 2)
    penetration = torch.relu(ground_z - feet[..., 2])
    loss = loss + torch.mean(penetration ** 2)
    return loss


def contact_prediction_loss(
    contact_logits: torch.Tensor,     # (T, N) per-node logits from the decoder
    foot_body_indices: list[int],
    teacher_contact_mask: torch.Tensor,  # (T, F) in {0,1}
) -> torch.Tensor:
    """Supervise the decoder's per-foot contact head against the teacher contact labels (BCE).

    Only the foot nodes are supervised; other nodes' logits are unconstrained (unused)."""
    foot_logits = contact_logits[:, foot_body_indices]  # (T, F)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        foot_logits, teacher_contact_mask.to(foot_logits.dtype)
    )


def contact_self_consistency_loss(
    pred: dict[str, torch.Tensor],
    target_kin: RobotKinematics,
    foot_body_indices: list[int],
    ground_z: float = 0.0,
    anchor: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """EDGE-style self-consistency: ‖(FK(x_{t+1})−FK(x_t))·σ(b̂_t)‖² masked by the network's *own*
    predicted contact probability. Encourages the model to both predict contact and keep the foot
    stationary where it predicts contact — removing the mm-level dof jitter at planted frames.

    A planted foot is stationary in the WORLD frame, not in the scaled-human-heading LOCAL frame the
    decoder predicts in (the local frame carries the negated per-frame anchor motion). Pass
    ``anchor=(anchor_pos, anchor_quat)`` to recompose the prediction to world before the velocity FK;
    without it the loss is computed in the local frame (only valid if the anchor is ~static).

    Requires ``pred['contact_logits']`` (decoder built with ``predict_contact=True``)."""
    assert "contact_logits" in pred, "model must be built with predict_contact=True"
    root_pos, root_quat = pred["root_pos"], pred["root_quat"]
    if anchor is not None:
        from .data import local_root_to_world

        root_pos, root_quat = local_root_to_world(anchor[0], anchor[1], root_pos, root_quat)
    body_pos, _ = target_kin.forward_kinematics(root_pos, root_quat, pred["dof_pos"])
    feet = body_pos[:, foot_body_indices, :]  # (T, F, 3)
    b = torch.sigmoid(pred["contact_logits"][:, foot_body_indices])  # (T, F) own prediction
    T = feet.shape[0]
    loss = feet.new_zeros(())
    if T > 1:
        vel = feet[1:, :, :] - feet[:-1, :, :]          # (T-1, F, 3) full 3D displacement
        loss = loss + torch.mean(b[1:].unsqueeze(-1) * vel ** 2)
    loss = loss + torch.mean(torch.relu(ground_z - feet[..., 2]) ** 2)  # penetration
    return loss


def latent_consistency_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """L_z: the same motion encoded from different embodiments must map to the same latent."""
    return torch.mean((z_a - z_b) ** 2)


def total_loss(
    pred: dict[str, torch.Tensor],
    target_kin: RobotKinematics,
    *,
    teacher: dict[str, torch.Tensor] | None = None,
    weights: dict[str, float] | None = None,
    body_targets: dict[int, torch.Tensor] | None = None,
    body_weights: dict[int, float] | None = None,
    foot_body_indices: list[int] | None = None,
    contact_mask: torch.Tensor | None = None,
    z_pair: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of active loss terms. Returns (scalar loss, per-term dict for logging)."""
    w = {
        "distill": 1.0,
        "task": 1.0,
        "limits": 0.1,
        "smooth": 0.01,
        "contact": 0.1,
        "latent": 1.0,
    }
    if weights:
        w.update(weights)

    terms: dict[str, torch.Tensor] = {}
    if teacher is not None:
        terms["distill"] = distill_loss(
            pred, teacher["root_pos"], teacher["root_quat"], teacher["dof_pos"]
        )
    if body_targets:
        terms["task"] = task_loss(pred, target_kin, body_targets, body_weights or {})
    terms["limits"] = joint_limit_loss(pred, target_kin)
    terms["smooth"] = smoothness_loss(pred)
    if foot_body_indices is not None and contact_mask is not None:
        terms["contact"] = foot_contact_loss(pred, target_kin, foot_body_indices, contact_mask)
    if z_pair is not None:
        terms["latent"] = latent_consistency_loss(z_pair[0], z_pair[1])

    total = pred["root_pos"].new_zeros(())
    for k, v in terms.items():
        total = total + w[k] * v
    return total, {k: float(v.detach()) for k, v in terms.items()}
