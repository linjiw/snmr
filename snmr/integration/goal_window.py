"""Explicit future-goal window: the window-matched control for the SNMR latent window.

The E70 explicit arm reads the current-frame goal ``g_t = [q_ref, qdot_ref, R_rel]`` (58 + 6 =
64-d).  The SNMR arm reads a two-sample latent window ``[z_t, z_{t+0.1 s}]``.  Any A/C
contrast therefore confounds representation with a 100 ms horizon (paper Limitations).  This
module builds ``[g_t, g_{t+k}]`` from the HoloSoma motion library so an explicit arm can be
given exactly the same horizon through exactly the same projection path (E78 arm ``mGf``; the
paper's ``C-future`` arm).

Frame semantics match HoloSoma's own observation terms (``managers/observation/terms/wbt.py``):

* ``joint_pos`` / ``joint_vel`` are the motion library rows at the (offset, clip-clamped) frame;
* ``motion_ref_ori_b`` is the reference body's orientation at that frame expressed in the robot's
  *current* reference-body frame — the future goal is "where the reference will be relative to
  where I am now", which is the only causal reading at deployment.

Frame indices are clamped at the clip end (no boundary crossing), exactly like
:func:`snmr.integration.wbt_latent._gather_at_offsets`.

Normalization: the raw 64-d goal is standardised with the teacher actor normalizer's statistics
for the goal slice, so ``g_t`` produced here is bit-for-bit the goal slice the E70 trainer
already feeds (offset 0), and ``g_{t+k}`` is on the same scale.
"""

from __future__ import annotations

import torch

from snmr.integration.wbt_latent import _gather_at_offsets

GOAL_DIM = 64  # motion_command (58) + motion_ref_ori_b (6)


def rotation_6d_from_quaternion(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, flattened — HoloSoma's ``motion_ref_ori_b``.

    Local re-implementation (identical math to ``quaternion_to_matrix(..., w_last=True)[..., :2]``)
    so this module stays importable without HoloSoma for unit tests.
    """
    x, y, z, w = quat_xyzw.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m00 = 1 - 2 * (yy + zz); m01 = 2 * (xy - wz)
    m10 = 2 * (xy + wz);     m11 = 1 - 2 * (xx + zz)
    m20 = 2 * (xz - wy);     m21 = 2 * (yz + wx)
    # column-major first two columns, matching mat[..., :2].reshape(N, -1) (row-major over rows)
    mat = torch.stack(
        (torch.stack((m00, m01), -1), torch.stack((m10, m11), -1), torch.stack((m20, m21), -1)),
        dim=-2,
    )  # (N, 3, 2)
    return mat.reshape(mat.shape[0], -1)


def quat_inverse_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)


def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def relative_orientation_6d(robot_quat_w: torch.Tensor, ref_quat_w: torch.Tensor) -> torch.Tensor:
    """6-d representation of ``R_robot^T R_ref`` (HoloSoma ``subtract_frame_transforms`` rotation)."""
    rel = quat_mul_xyzw(quat_inverse_xyzw(robot_quat_w), ref_quat_w)
    return rotation_6d_from_quaternion(rel)


def raw_goal_at_offsets(motion_command, offsets: tuple[int, ...]) -> list[torch.Tensor]:
    """Un-normalised 64-d goals ``[q_ref, qdot_ref, R_rel]`` at each frame offset.

    ``R_rel`` is taken relative to the robot's *current* reference-body pose.
    """
    motion = motion_command.motion
    ref = motion_command.ref_body_index
    q = _gather_at_offsets(motion_command, motion.joint_pos, offsets)
    qd = _gather_at_offsets(motion_command, motion.joint_vel, offsets)
    ref_quat = _gather_at_offsets(motion_command, motion.body_quat_w[:, ref], offsets)
    robot_quat = motion_command.robot_ref_quat_w
    goals = []
    for qi, qdi, rq in zip(q, qd, ref_quat):
        goals.append(torch.cat((qi, qdi, relative_orientation_6d(robot_quat, rq)), dim=-1))
    return goals


class GoalWindow:
    """Callable producing the standardised explicit window ``[g_t, g_{t+k}, ...]``.

    ``goal_mean`` / ``goal_std`` are the teacher actor normalizer's goal-slice statistics
    (``EmpiricalNormalization._mean[0, 90:154]``, ``_std[0, 90:154]``, plus its ``eps``).
    """

    def __init__(self, goal_mean: torch.Tensor, goal_std: torch.Tensor, eps: float, offsets: tuple[int, ...]):
        if goal_mean.shape != (GOAL_DIM,) or goal_std.shape != (GOAL_DIM,):
            raise ValueError("goal statistics must have shape (64,)")
        if not offsets or offsets[0] != 0:
            raise ValueError("offsets must start with 0 so g_t is the ordinary explicit goal")
        self.mean, self.std, self.eps, self.offsets = goal_mean, goal_std, float(eps), tuple(offsets)

    @property
    def dim(self) -> int:
        return GOAL_DIM * len(self.offsets)

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self.mean) / (self.std + self.eps)

    def __call__(self, motion_command) -> torch.Tensor:
        raws = raw_goal_at_offsets(motion_command, self.offsets)
        return torch.cat([self.normalize(r) for r in raws], dim=-1)


def goal_stats_from_normalizer(normalizer, goal_slice: slice) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Pull the goal-slice statistics out of a HoloSoma ``EmpiricalNormalization``."""
    mean = normalizer._mean[0, goal_slice].detach().clone()
    std = normalizer._std[0, goal_slice].detach().clone()
    return mean, std, float(normalizer.eps)
