"""Minimal training / fitting utilities for SNMR.

This module provides a single-motion, single-robot fitting loop used both for the overfit-a-batch
sanity check (Gate G1 smoke test) and as the skeleton of the full Phase-1 trainer. The full trainer
(multi-clip, multi-robot dataset, L_z consistency across embodiments) builds on ``fit_motion`` by
looping over clips/robots and adding the latent-consistency pairing; that scale-up needs a GPU and the
AMASS/LAFAN teacher dataset which are not present in this environment, so it is intentionally left as
a documented extension rather than dead code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import losses
from .data import RobotMotion
from .model import SNMR, SNMRConfig
from .robot_model import RobotKinematics


@dataclass
class FitResult:
    losses: list[float]
    final_pred: dict[str, torch.Tensor]


def fit_motion(
    model: SNMR,
    robot_kin: RobotKinematics,
    motion: RobotMotion,
    *,
    steps: int = 300,
    lr: float = 1e-3,
    loss_weights: dict[str, float] | None = None,
    verbose: bool = False,
) -> FitResult:
    """Fit ``model`` to reproduce ``motion`` on ``robot_kin`` via autoencoding + distillation.

    The teacher here is the motion itself (autoencoding): encode the robot motion, decode back to the
    same robot, and supervise with distillation + smoothness. Success (loss → ~0) proves the
    encode→latent→decode pipeline has the capacity and gradients to represent real motion — the
    prerequisite for Phase 1.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    teacher = {"root_pos": motion.root_pos, "root_quat": motion.root_quat, "dof_pos": motion.dof_pos}
    history: list[float] = []
    pred: dict[str, torch.Tensor] = {}
    for step in range(steps):
        opt.zero_grad()
        pred = model.retarget_robot_to_robot(robot_kin, motion, robot_kin)
        loss, parts = losses.total_loss(pred, robot_kin, teacher=teacher, weights=loss_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach()))
        if verbose and (step % max(1, steps // 10) == 0 or step == steps - 1):
            msg = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
            print(f"step {step:4d} loss={history[-1]:.5f}  {msg}")
    return FitResult(losses=history, final_pred=pred)


def default_model() -> SNMR:
    return SNMR(SNMRConfig())
