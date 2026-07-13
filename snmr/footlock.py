"""Inference-time foot-lock: hard contact cleanup for decoded robot motion (C5 primary mechanism).

The training-time contact loss did not reduce foot skate at any weight (E10a: 0.50–0.57 m/s vs
no-contact 0.26–0.39, teacher 0.05) — the distill objective, which pulls the dof toward the
teacher (whose small errors already skate), dominates and conflicts with it. The literature's
reliable fix (PFNN/MANN/motion-matching, Villegas ESO) is a post-hoc pass, so we implement it as
an inference cleanup rather than a loss:

  1. detect per-foot contact intervals on the DECODED world-frame motion (height+speed heuristic);
  2. within each contact interval, pin the foot to a single anchor position (the interval's median
     foot xy, contact-onset z);
  3. solve the leg dofs to hit the pinned foot target by a few damped Gauss-Newton steps on the
     differentiable FK (only that leg's hip/knee/ankle dofs move; the rest of the body is untouched).

This is a hard constraint at inference, so it removes contact-phase sliding by construction, at a
small MPJPE cost, and never fights the retargeter's training. Differentiable FK gives the Jacobian
via autograd — no analytic Jacobian needed.
"""

from __future__ import annotations

import torch

from .metrics import detect_contact
from .robot_model import RobotKinematics


def _leg_dof_indices(kin: RobotKinematics, foot_body: str) -> list[int]:
    """Dof indices on the kinematic path from root to ``foot_body`` (the only dofs foot-lock moves)."""
    names = kin.body_names
    parent = kin.graph.parent_index.tolist()
    chain = set()
    i = names.index(foot_body)
    while i >= 0:
        chain.add(i)
        i = parent[i]
    dof_body = kin.graph.dof_body_index.tolist()
    return [d for d, b in enumerate(dof_body) if b in chain]


def foot_lock(
    kin: RobotKinematics,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    dof_pos: torch.Tensor,
    foot_body_names: list[str],
    fps: float = 30.0,
    iters: int = 4,
    lr: float = 0.5,
    damping: float = 1e-2,
) -> torch.Tensor:
    """Return foot-locked dof_pos (T, D). Root pose is held fixed; only per-foot leg dofs are edited
    during that foot's contact intervals. Inputs are WORLD-frame (a planted foot is world-static)."""
    T = dof_pos.shape[0]
    dof = dof_pos.clone()
    foot_idx = [kin.body_index(n) for n in foot_body_names]

    with torch.no_grad():
        body_pos, _ = kin.forward_kinematics(root_pos, root_quat, dof)
        feet0 = body_pos[:, foot_idx, :]                 # (T, F, 3)
        contact = detect_contact(feet0, fps=fps)         # (T, F) bool

    for fi, foot in enumerate(foot_body_names):
        leg_dofs = _leg_dof_indices(kin, foot)
        leg_dofs_t = torch.tensor(leg_dofs, dtype=torch.long, device=dof.device)
        bidx = foot_idx[fi]
        # contiguous contact intervals for this foot
        for lo, hi in _intervals(contact[:, fi]):
            # target = median xy over the interval, min z (contact-onset ground height)
            with torch.no_grad():
                seg = feet0[lo:hi, fi, :]
                target = torch.empty(3, device=dof.device, dtype=dof.dtype)
                target[:2] = seg[:, :2].median(dim=0).values
                target[2] = seg[:, 2].min()
            for t in range(lo, hi):
                q = dof[t].clone().detach().requires_grad_(True)
                for _ in range(iters):
                    bp, _ = kin.forward_kinematics(root_pos[t], root_quat[t], q)
                    err = bp[bidx] - target
                    loss = (err ** 2).sum()
                    (grad,) = torch.autograd.grad(loss, q)
                    step = torch.zeros_like(q)
                    g = grad[leg_dofs_t]
                    step[leg_dofs_t] = g / (g.norm() + damping)
                    q = (q - lr * err.detach().norm() * step).detach().requires_grad_(True)
                dof[t] = q.detach()
    return dof


def _intervals(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Contiguous True runs in a 1-D bool mask as [lo, hi) index pairs."""
    m = mask.to(torch.int8).tolist()
    out, start = [], None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(m)))
    return out
