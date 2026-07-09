"""Rotation and transform utilities for SNMR.

Conventions (fixed for the whole package, matching MuJoCo / GMR intermediate format):
  * Quaternions are **scalar-first (wxyz)** — same as MuJoCo ``qpos[3:7]`` and the GMR
    intermediate human dict. (Note: GMR's *saved pkl* stores ``root_rot`` as xyzw and its
    ``torch_utils`` uses xyzw; we deliberately standardise on wxyz internally and convert only
    at the numpy I/O boundary. See ``snmr.data`` for the conversion points.)
  * 6D rotation representation follows Zhou et al. 2019 (CVPR): the first two columns of the
    rotation matrix, flattened column-major to a 6-vector ``[m00,m10,m20, m01,m11,m21]``.
  * All functions are batched over arbitrary leading dims and differentiable.

These are re-implemented here (rather than imported from GMR) so the package is self-contained
and so the quaternion order is unambiguous; correctness is checked against scipy in the tests.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------------------
# quaternion ops (wxyz)
# --------------------------------------------------------------------------------------
def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two wxyz quaternions, broadcasting over leading dims."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) ``v`` (..., 3) by wxyz quaternion(s) ``q`` (..., 4)."""
    w = q[..., 0:1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """axis (..., 3) assumed unit; angle (...,) -> wxyz quat (..., 4)."""
    half = 0.5 * angle
    w = torch.cos(half)
    xyz = axis * torch.sin(half).unsqueeze(-1)
    return torch.cat((w.unsqueeze(-1), xyz), dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """wxyz quat (..., 4) -> rotation matrix (..., 3, 3)."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    m = torch.stack(
        (
            1.0 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1.0 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1.0 - (txx + tyy),
        ),
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """rotation matrix (..., 3, 3) -> wxyz quat (..., 4). Branch-robust (Shepperd's method)."""
    batch = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    # Four candidate reconstructions; pick the numerically largest denominator.
    q_candidates = torch.stack(
        (
            torch.stack((1.0 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, 1.0 + m00 - m11 - m22, m01 + m10, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m01 + m10, 1.0 - m00 + m11 - m22, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, 1.0 - m00 - m11 + m22), dim=-1),
        ),
        dim=1,
    )  # (N, 4, 4)
    diag = torch.stack(
        (1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22),
        dim=-1,
    )  # (N, 4)
    best = diag.argmax(dim=-1)  # (N,)
    idx = torch.arange(m.shape[0], device=m.device)  # keep on the same device as the input (CUDA-safe)
    q = q_candidates[idx, best]  # (N, 4)
    q = q / (2.0 * torch.sqrt(diag[idx, best].clamp_min(1e-8))).unsqueeze(-1)
    q = quat_normalize(q)
    # canonical sign (w >= 0)
    q = torch.where(q[..., :1] < 0, -q, q)
    return q.reshape(batch + (4,))


# --------------------------------------------------------------------------------------
# 6D rotation representation (Zhou et al. 2019)
# --------------------------------------------------------------------------------------
def matrix_to_rot6d(m: torch.Tensor) -> torch.Tensor:
    """rotation matrix (..., 3, 3) -> 6D rep (..., 6): first two columns, column-major.

    Layout is ``[m00, m10, m20, m01, m11, m21]`` so that ``r6[...,0:3]`` is the first column and
    ``r6[...,3:6]`` is the second column, matching :func:`rot6d_to_matrix`.
    """
    return torch.cat((m[..., :, 0], m[..., :, 1]), dim=-1)


def rot6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """6D rep (..., 6) -> rotation matrix (..., 3, 3) via Gram-Schmidt.

    Columns are recovered as [b1 | b2 | b1 x b2].
    """
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    a2_proj = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_proj, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)  # columns


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    return matrix_to_rot6d(quat_to_matrix(q))


def rot6d_to_quat(r6: torch.Tensor) -> torch.Tensor:
    return matrix_to_quat(rot6d_to_matrix(r6))


# --------------------------------------------------------------------------------------
# geodesic distance
# --------------------------------------------------------------------------------------
def quat_geodesic_angle(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Shortest geodesic angle (radians) between two wxyz quaternions, batched (...,).

    Uses an ``atan2`` formulation that is accurate near zero angle (no ``acos``-clamp floor) and
    finite everywhere. Intended for **metrics / eval**; for a differentiable rotation *loss* prefer
    :func:`matrix_to_rot6d` L2 or a Frobenius matrix loss, which are smooth at zero angle.
    """
    dot = (quat_normalize(q0) * quat_normalize(q1)).sum(-1)
    # angle = 2*atan2(|sin|, |cos|); |cos| = |dot| via the double cover.
    abs_dot = dot.abs().clamp(max=1.0)
    sin_half = torch.sqrt((1.0 - abs_dot * abs_dot).clamp_min(0.0))
    return 2.0 * torch.atan2(sin_half, abs_dot)


def matrix_geodesic_angle(m0: torch.Tensor, m1: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) between two rotation matrices, batched (...,). Metric use."""
    rel = m0.transpose(-1, -2) @ m1
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    # sin from off-diagonal skew part, for atan2 stability near 0 and pi.
    sin = 0.5 * torch.sqrt(
        (
            (rel[..., 2, 1] - rel[..., 1, 2]) ** 2
            + (rel[..., 0, 2] - rel[..., 2, 0]) ** 2
            + (rel[..., 1, 0] - rel[..., 0, 1]) ** 2
        ).clamp_min(0.0)
    )
    return torch.atan2(sin, cos)


# --------------------------------------------------------------------------------------
# numpy <-> convention helpers (I/O boundary only)
# --------------------------------------------------------------------------------------
def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:], q[..., 0:1]), dim=-1)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 3:4], q[..., 0:3]), dim=-1)
