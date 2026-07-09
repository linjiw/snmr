"""Human-side skeleton + pose features (LAFAN1 BVH path).

This is the encoder's human input path (design doc §3.1 left side). The LAFAN1 skeleton is the fixed
24-body tree emitted by GMR's ``load_bvh_file`` (22 BVH joints + the two synthetic ``*FootMod``
bodies GMR adds at the feet). Pose features mirror ``snmr.data.robot_pose_features`` exactly —
positions/6D orientations in the root heading frame plus finite-difference velocities — so the
skeleton-agnostic encoder consumes human and robot motion through the same interface.

Contact flags use the standard height+velocity heuristic (SAME / holosoma convention): a foot body is
"in contact" when it is low AND slow. They are returned separately from the pose features so the
caller can (a) append them as an extra node feature channel or (b) use them for the foot-skate loss
on the decoded robot motion.
"""

from __future__ import annotations

import numpy as np
import torch

from . import rotation as rot
from .data import heading_quat_inverse
from .skeleton import SkeletonGraph

# GMR's LAFAN1 loader body order (see snmr/scripts/make_pairs_lafan1.py — stored in each pair NPZ).
LAFAN1_BODY_NAMES: list[str] = [
    "Hips",                                             # 0  root
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",      # 1-4
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",  # 5-8
    "Spine", "Spine1", "Spine2", "Neck", "Head",        # 9-13
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",   # 14-17
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",  # 18-21
    "LeftFootMod", "RightFootMod",                      # 22-23 (synthetic: foot pos + toe ori)
]
LAFAN1_PARENTS: list[int] = [
    -1,                # Hips
    0, 1, 2, 3,        # left leg chain
    0, 5, 6, 7,        # right leg chain
    0, 9, 10, 11, 12,  # spine chain -> head
    11, 14, 15, 16,    # left arm chain (from Spine2)
    11, 18, 19, 20,    # right arm chain (from Spine2)
    3, 7,              # FootMod bodies hang off the feet
]

# Bodies used for contact detection (toes are the most reliable ground indicators in LAFAN1).
LAFAN1_CONTACT_BODIES: list[str] = ["LeftToe", "RightToe"]


def lafan1_skeleton(device: str | torch.device = "cpu") -> SkeletonGraph:
    """The LAFAN1 24-body skeleton as a :class:`SkeletonGraph` (topology only; rest offsets zero)."""
    device = torch.device(device)
    n = len(LAFAN1_BODY_NAMES)
    parent = torch.tensor(LAFAN1_PARENTS, dtype=torch.long, device=device)
    is_parent = torch.zeros(n, dtype=torch.bool, device=device)
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            is_parent[p] = True
    return SkeletonGraph(
        names=list(LAFAN1_BODY_NAMES),
        parent_index=parent,
        rest_offset=torch.zeros(n, 3, device=device),
        is_end_effector=~is_parent,
    )


def human_pose_features(
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    root_index: int = 0,
    include_velocity: bool = True,
) -> torch.Tensor:
    """Per-body features for a human motion in the root heading frame — mirror of
    ``robot_pose_features`` but computed from observed world kinematics instead of FK.

    Args:
        body_pos:  (T, J, 3) world positions.
        body_quat: (T, J, 4) wxyz world orientations.
        root_index: index of the root body (Hips for LAFAN1).

    Returns:
        (T, J, F) with F = 3 + 6 [+ 3 velocity].
    """
    T, J, _ = body_pos.shape
    root_quat = body_quat[:, root_index, :]
    root_pos = body_pos[:, root_index, :]

    inv_head = heading_quat_inverse(root_quat)             # (T, 4)
    root_xy = root_pos.clone()
    root_xy[..., 2] = 0.0

    inv_head_b = inv_head[:, None, :].expand(T, J, 4)
    local_pos = rot.quat_rotate(inv_head_b, body_pos - root_xy[:, None, :])
    local_quat = rot.quat_mul(inv_head_b, body_quat)
    local_rot6d = rot.quat_to_rot6d(local_quat)

    feats = [local_pos, local_rot6d]
    if include_velocity:
        vel = torch.zeros_like(local_pos)
        if T > 1:
            vel[1:] = local_pos[1:] - local_pos[:-1]
        feats.append(vel)
    return torch.cat(feats, dim=-1)


def human_static_features(
    skel: SkeletonGraph, body_pos_sample: torch.Tensor | None = None, static_dim: int = 8
) -> torch.Tensor:
    """Static per-node features for a human skeleton, matching the robot static feature layout
    (offset(3) + axis(3) + has_dof(1) + ee(1) = 8) so the shared encoder sees one schema.

    Humans have no hinge axes; the axis slot is zeroed and has_dof set to 1 for all non-root nodes
    (every human joint is actuated). Rest offsets come from the skeleton (or, better, from a sample
    frame's mean bone vectors if ``body_pos_sample`` (T, J, 3) is given).
    """
    n = skel.num_nodes
    offsets = skel.rest_offset.clone()
    if body_pos_sample is not None:
        # mean bone vector in the first frames = a data-derived rest offset
        pos = body_pos_sample[: min(10, body_pos_sample.shape[0])].mean(dim=0)  # (J, 3)
        for i in range(n):
            p = int(skel.parent_index[i])
            offsets[i] = pos[i] - pos[p] if p >= 0 else 0.0
    axis = torch.zeros(n, 3, dtype=offsets.dtype, device=offsets.device)
    has_dof = torch.ones(n, 1, dtype=offsets.dtype, device=offsets.device)
    has_dof[0] = 0.0  # root is driven by the free joint, as on the robot side
    ee = skel.is_end_effector.to(offsets.dtype).unsqueeze(-1)
    out = torch.cat([offsets, axis, has_dof, ee], dim=-1)
    assert out.shape[-1] == static_dim
    return out


def foot_contact_flags(
    body_pos: torch.Tensor,
    body_names: list[str],
    contact_bodies: list[str] | None = None,
    height_threshold: float = 0.08,
    velocity_threshold: float = 0.008,
) -> torch.Tensor:
    """Height+velocity contact heuristic.

    A body is in contact at frame t when its height is below ``height_threshold`` (relative to the
    clip's per-body minimum, robust to ground offset) AND its per-frame XY displacement is below
    ``velocity_threshold`` (meters/frame; ~0.24 m/s at 30 fps).

    Returns (T, C) float flags in {0, 1} for the ``contact_bodies``.
    """
    contact_bodies = contact_bodies or LAFAN1_CONTACT_BODIES
    idx = [body_names.index(b) for b in contact_bodies]
    feet = body_pos[:, idx, :]  # (T, C, 3)
    T = feet.shape[0]

    height = feet[..., 2] - feet[..., 2].min(dim=0, keepdim=True).values  # per-body ground-relative
    low = height < height_threshold

    disp = torch.zeros_like(height)
    if T > 1:
        disp[1:] = (feet[1:, :, :2] - feet[:-1, :, :2]).norm(dim=-1)
        disp[0] = disp[1]
    slow = disp < velocity_threshold

    return (low & slow).to(body_pos.dtype)


def load_pair_npz(path: str, dtype: torch.dtype = torch.float32) -> dict:
    """Load one training pair produced by ``scripts/make_pairs_lafan1.py``.

    Returns dict with: human_pos (T,J,3), human_quat (T,J,4) wxyz, human_names, qpos (T,7+D),
    fps, robot, human_height — human arrays as torch, metadata as python types.
    """
    data = np.load(path, allow_pickle=True)
    return {
        "human_pos": torch.tensor(np.asarray(data["human_pos"]), dtype=dtype),
        "human_quat": torch.tensor(np.asarray(data["human_quat"]), dtype=dtype),
        "human_names": [str(x) for x in data["human_names"]],
        "qpos": torch.tensor(np.asarray(data["qpos"]), dtype=dtype),
        "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
        "robot": str(np.asarray(data["robot"]).reshape(-1)[0]),
        "human_height": float(np.asarray(data["human_height"]).reshape(-1)[0]),
    }
