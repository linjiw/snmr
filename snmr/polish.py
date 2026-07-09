"""Inference-time latent polish for contact-consistent decoding (design doc §N6, step 2).

The Gate-G1 retargeter matches its teacher on MPJPE but has ~5x worse contact-phase foot skating,
which an error decomposition traced to mm-level frame-to-frame leg-angle jitter (dof-caused). The
motion-synthesis literature is unanimous that training losses alone do not fully close this gap;
the works that reach IK-teacher contact quality run a short optimization at *inference* using
predicted/teacher contacts (Villegas et al. 2021 "encoder-space optimization": contact accuracy
0.71->0.97; Holden et al. 2016 null-space; UnderPressure output-space, ~halves contact velocity).

We follow Villegas/Holden and optimize in the **latent** z (not qpos): gradient descent on a
contact-masked foot-velocity + foot-height energy, regularized to stay near the original latent so
the pose does not drift off the learned manifold. Requires only a trained checkpoint + a contact
mask (derived from the human feet at inference, or the teacher feet when benchmarking) — no
retraining. This module is decoder-agnostic: it takes an already-initialized latent and a decode
closure, so it works for both the Phase-1 and Phase-2 (multi-robot) models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import local_root_to_world
from .metrics import detect_contact
from .robot_model import RobotKinematics


@dataclass
class PolishConfig:
    iters: int = 30
    lr: float = 0.05
    w_vel: float = 1.0        # contact-masked foot XY velocity penalty (the skate term)
    w_height: float = 0.5     # keep contacting feet near the ground
    w_anchor: float = 5.0     # stay close to the original latent (manifold / pose-fidelity guard)
    ground_z: float = 0.0


def polish_latent(
    z0: torch.Tensor,
    decode_fn,
    kin: RobotKinematics,
    foot_body_indices: list[int],
    contact_mask: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    cfg: PolishConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    """Optimize the latent to remove contact-phase foot sliding, staying near the original pose.

    Args:
        z0:            (T, latent_dim) latent from the trained encoder (detached; we clone+optimize).
        decode_fn:     ``z -> {"root_pos","root_quat","dof_pos"}`` in the scaled-human-heading frame
                       (i.e. ``model.decode(z, kin)`` or the multi-robot decoder closure).
        kin:           target robot kinematics (differentiable FK).
        foot_body_indices: robot body indices treated as feet.
        contact_mask:  (T, F) bool/float, 1 where each foot should be planted (from human/teacher).
        anchor_pos/quat: (T, 3)/(T, 4) the scaled-human-root pose used to recompose world foot pos.
        cfg:           polish hyperparameters.

    Returns:
        (polished prediction dict in WORLD frame, per-term energy log).
    """
    cfg = cfg or PolishConfig()
    z_ref = z0.detach()                       # anchor target — no graph (avoids double-backward)
    z = z_ref.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=cfg.lr)
    cmask = contact_mask.to(z0.dtype)

    log: dict[str, float] = {}
    for _ in range(cfg.iters):
        opt.zero_grad()
        pred = decode_fn(z)
        # recompose to world so a planted foot is judged stationary in world coords (not the moving
        # anchor frame — the same subtlety the training-time contact loss handles).
        wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
        body_pos, _ = kin.forward_kinematics(wp, wq, pred["dof_pos"])
        feet = body_pos[:, foot_body_indices, :]  # (T, F, 3)

        vel = feet[1:, :, :2] - feet[:-1, :, :2]          # (T-1, F, 2)
        e_vel = (cmask[1:].unsqueeze(-1) * vel ** 2).mean()
        e_height = (cmask * torch.relu(feet[..., 2] - cfg.ground_z) ** 2).mean() \
            + torch.relu(cfg.ground_z - feet[..., 2]).pow(2).mean()  # penetration always penalised
        e_anchor = ((z - z_ref) ** 2).mean()
        energy = cfg.w_vel * e_vel + cfg.w_height * e_height + cfg.w_anchor * e_anchor
        energy.backward()
        opt.step()
        log = {"vel": float(e_vel.detach()), "height": float(e_height.detach()),
               "anchor": float(e_anchor.detach())}

    with torch.no_grad():
        pred = decode_fn(z)
        wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
    return {"root_pos": wp, "root_quat": wq, "dof_pos": pred["dof_pos"]}, log


def contact_from_human_feet(
    human_pos: torch.Tensor,
    human_names: list[str],
    robot_foot_count: int,
    contact_bodies: list[str] | None = None,
    fps: float = 30.0,
) -> torch.Tensor:
    """Derive a robot-foot contact mask from the human toe motion (deployable at inference — no
    teacher needed). Maps the human's per-toe contact onto the robot's ordered foot list; if the
    counts differ we broadcast/truncate (both sides are left/right ordered)."""
    from .human import LAFAN1_CONTACT_BODIES

    contact_bodies = contact_bodies or LAFAN1_CONTACT_BODIES
    idx = [human_names.index(b) for b in contact_bodies]
    feet = human_pos[:, idx, :]
    mask = detect_contact(feet, fps=fps)  # (T, len(contact_bodies))
    if mask.shape[1] == robot_foot_count:
        return mask
    if robot_foot_count <= mask.shape[1]:
        return mask[:, :robot_foot_count]
    reps = (robot_foot_count + mask.shape[1] - 1) // mask.shape[1]
    return mask.repeat(1, reps)[:, :robot_foot_count]
