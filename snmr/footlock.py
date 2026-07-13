"""Inference-time foot-lock cleanup for decoded robot motion.

The training-time contact loss did not reduce foot skate at any weight (E10a: 0.50–0.57 m/s vs
no-contact 0.26–0.39, teacher 0.05) — the distill objective, which pulls the dof toward the
teacher (whose small errors already skate), dominates and conflicts with it. The literature's
reliable fix (PFNN/MANN/motion-matching, Villegas ESO) is a post-hoc pass, so we implement it as
an inference cleanup rather than a loss:

  1. detect per-foot contact intervals on the DECODED world-frame motion (height+speed heuristic);
  2. within each contact interval, pin the foot to a single anchor position (the interval's median
     foot xy, contact-onset z);
  3. solve the leg dofs to hit the pinned foot target by damped least squares on the differentiable
     FK Jacobian (only that leg's hip/knee/ankle dofs move; the rest of the body is untouched).

This is a bounded per-frame DLS heuristic, not the windowed constrained projection specified for
Gate 1 C6. Differentiable FK gives the Jacobian via autograd, so no analytic Jacobian is needed.
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


def _solve_foot_dls(
    kin: RobotKinematics,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    dof_pos: torch.Tensor,
    foot_body_index: int,
    leg_dof_indices: torch.Tensor,
    target: torch.Tensor,
    iters: int,
    step_scale: float,
    damping: float,
) -> torch.Tensor:
    """Solve one frame with a task-space damped least-squares FK update."""
    q = dof_pos.detach().clone()
    lo, hi = kin.dof_limits()
    lo = lo.to(device=q.device, dtype=q.dtype)
    hi = hi.to(device=q.device, dtype=q.dtype)
    eye = torch.eye(3, device=q.device, dtype=q.dtype)

    for _ in range(iters):
        q = q.detach().requires_grad_(True)
        body_pos, _ = kin.forward_kinematics(root_pos, root_quat, q)
        foot_pos = body_pos[foot_body_index]
        residual = foot_pos - target
        current_error = residual.square().sum().detach()
        if current_error <= 1e-10:
            break

        jacobian = torch.stack(
            [
                torch.autograd.grad(
                    foot_pos[axis],
                    q,
                    retain_graph=axis < 2,
                )[0][leg_dof_indices]
                for axis in range(3)
            ]
        )
        normal = jacobian @ jacobian.transpose(0, 1) + damping * eye
        delta = -jacobian.transpose(0, 1) @ torch.linalg.solve(normal, residual.detach())
        delta_norm = delta.norm()
        if delta_norm > 0.25:
            delta = delta * (0.25 / delta_norm)

        base = q.detach()
        base_leg = base[leg_dof_indices]
        alpha = min(max(step_scale, 0.0), 1.0)
        accepted = base
        for _ in range(6):
            candidate_leg = (
                base_leg + alpha * delta
            ).clamp(lo[leg_dof_indices], hi[leg_dof_indices])
            candidate = base.index_copy(0, leg_dof_indices, candidate_leg)
            with torch.no_grad():
                candidate_pos, _ = kin.forward_kinematics(root_pos, root_quat, candidate)
                candidate_error = (
                    candidate_pos[foot_body_index] - target
                ).square().sum()
            if candidate_error < current_error:
                accepted = candidate
                break
            alpha *= 0.5
        q = accepted

    return q.detach()


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
                dof[t] = _solve_foot_dls(
                    kin,
                    root_pos[t],
                    root_quat[t],
                    dof[t],
                    bidx,
                    leg_dofs_t,
                    target,
                    iters,
                    lr,
                    damping,
                )
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


def dilate_contact_mask(mask: torch.Tensor, merge_gap: int = 6, extend: int = 2) -> torch.Tensor:
    """Close gaps <= ``merge_gap`` frames between contact intervals and extend each by ``extend``.

    Any stance frame the mask misses keeps its full raw skate, so a deployable mask (e.g. the
    human source's contact flags, ~80% agreement with the teacher stance) must over- rather than
    under-cover. Over-locking swing frames is comparatively harmless: the blend ramp in
    :func:`foot_lock_masked` keeps their targets near the unedited prediction."""
    out = torch.zeros_like(mask)
    T = mask.shape[0]
    for fi in range(mask.shape[1]):
        merged: list[tuple[int, int]] = []
        for lo, hi in _intervals(mask[:, fi]):
            if merged and lo - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], hi)
            else:
                merged.append((lo, hi))
        for lo, hi in merged:
            out[max(lo - extend, 0):min(hi + extend, T), fi] = True
    return out


def smooth_correction(dof_raw: torch.Tensor, dof_locked: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian low-pass the IK *correction* (locked − raw) over time, then re-apply it.

    Frame-independent IK solves can leave frame-to-frame noise in the edit. Smoothing the
    correction rather than the full motion preserves the raw trajectory outside edited regions.
    Callers must re-project stance targets afterward because filtering relaxes the foot lock."""
    if sigma <= 0:
        return dof_locked

    delta = dof_locked - dof_raw
    radius = max(int(3 * sigma), 1)
    x = torch.arange(-radius, radius + 1, dtype=dof_raw.dtype, device=dof_raw.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    # depthwise conv over time, per dof channel
    d = delta.t().unsqueeze(1)                                   # (D, 1, T)
    pad_mode = "reflect" if radius < dof_raw.shape[0] else "replicate"
    d = torch.nn.functional.pad(d, (radius, radius), mode=pad_mode)
    sm = torch.nn.functional.conv1d(d, kernel.view(1, 1, -1))    # (D, 1, T)
    return dof_raw + sm.squeeze(1).t()


def foot_lock_masked(
    kin: RobotKinematics,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    dof_pos: torch.Tensor,
    foot_body_names: list[str],
    contact_mask: torch.Tensor,
    iters: int = 12,
    lr: float = 0.5,
    damping: float = 1e-2,
    blend: int = 2,
    smooth_sigma: float = 0.0,
    merge_gap: int = 6,
    extend: int = 2,
) -> torch.Tensor:
    """Foot-lock driven by a CALLER-SUPPLIED contact mask (T, F) — the E26 production variant.

    E24's :func:`foot_lock` failed (0.160→0.153 m/s) because it detected contact on the decoded
    motion itself, whose stance-phase xy oscillation exceeds the 0.3 m/s speed test — contact
    under-fires (frac 0.03 vs teacher 0.29) and most stance frames are never locked. The
    literature-standard fix is to take contact labels from the CLEAN signal; at inference the
    clean signal we always have is the *human source* motion, whose contact flags agree with the
    teacher stance at ~0.80 (vs 0.74 for decoded detection). Pipeline:
      1. dilate the mask (:func:`dilate_contact_mask`) so it over-covers stance;
      2. per interval, pin the foot to (median xy, min z) with a `blend`-frame linear ramp at the
         boundaries (Kovar-style ease-in/out — avoids entry/exit pops);
      3. damped least-squares leg IK per frame (root untouched, only that leg's dofs move);
      4. Gaussian-smooth the correction and re-project the stance targets.

    The original normalized-gradient E26 pilot reported 0.49 to 0.06 m/s on three windows.
    Re-evaluate this DLS implementation with ``scripts/eval_footlock.py`` before making claims."""
    dof = dof_pos.clone()
    foot_idx = [kin.body_index(n) for n in foot_body_names]
    expected_shape = (dof_pos.shape[0], len(foot_body_names))
    if contact_mask.shape != expected_shape:
        raise ValueError(
            f"contact mask shape {tuple(contact_mask.shape)} != expected {expected_shape}"
        )
    mask = dilate_contact_mask(
        contact_mask.bool(),
        merge_gap=merge_gap,
        extend=extend,
    )
    targets = torch.full(
        (dof_pos.shape[0], len(foot_body_names), 3),
        float("nan"),
        device=dof.device,
        dtype=dof.dtype,
    )

    with torch.no_grad():
        body_pos, _ = kin.forward_kinematics(root_pos, root_quat, dof)
        feet0 = body_pos[:, foot_idx, :]                 # (T, F, 3)

    for fi, foot in enumerate(foot_body_names):
        leg_dofs_t = torch.tensor(_leg_dof_indices(kin, foot), dtype=torch.long, device=dof.device)
        bidx = foot_idx[fi]
        for lo, hi in _intervals(mask[:, fi]):
            if hi - lo < 2:
                continue
            with torch.no_grad():
                seg = feet0[lo:hi, fi, :]
                target = torch.empty(3, device=dof.device, dtype=dof.dtype)
                target[:2] = seg[:, :2].median(dim=0).values
                target[2] = seg[:, 2].min()
            for t in range(lo, hi):
                w = min(1.0, (t - lo + 1) / max(blend, 1), (hi - t) / max(blend, 1))
                tgt_t = w * target + (1 - w) * feet0[t, fi, :]
                targets[t, fi] = tgt_t
                dof[t] = _solve_foot_dls(
                    kin,
                    root_pos[t],
                    root_quat[t],
                    dof[t],
                    bidx,
                    leg_dofs_t,
                    tgt_t,
                    iters,
                    lr,
                    damping,
                )

    if smooth_sigma > 0:
        dof = smooth_correction(dof_pos, dof, sigma=smooth_sigma)
        # Smoothing can relax the stance constraint because the IK correction must cancel
        # frame-varying decoder error. Re-project after filtering instead of reporting a
        # visually smoother joint trajectory whose foot is no longer locked.
        reprojection_iters = max(2, iters // 3)
        for fi, foot in enumerate(foot_body_names):
            leg_dofs_t = torch.tensor(
                _leg_dof_indices(kin, foot),
                dtype=torch.long,
                device=dof.device,
            )
            bidx = foot_idx[fi]
            valid_targets = torch.isfinite(targets[:, fi]).all(dim=-1)
            for t in torch.nonzero(valid_targets, as_tuple=False).flatten().tolist():
                dof[t] = _solve_foot_dls(
                    kin,
                    root_pos[t],
                    root_quat[t],
                    dof[t],
                    bidx,
                    leg_dofs_t,
                    targets[t, fi],
                    reprojection_iters,
                    lr,
                    damping,
                )
    lo, hi = kin.dof_limits()
    return dof.clamp(
        lo.to(device=dof.device, dtype=dof.dtype),
        hi.to(device=dof.device, dtype=dof.dtype),
    )
