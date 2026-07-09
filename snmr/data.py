"""Data structures, loaders, and feature extraction.

Two responsibilities:

  1. **Canonical motion representation.** A motion is a window of frames, each frame being a robot
     configuration ``qpos = [root_pos(3), root_quat wxyz(4), dof_pos(D)]`` plus (optionally) the human
     source keypoints. We standardise on wxyz quaternions internally and convert at the numpy boundary.

  2. **Graph pose features.** Both the encoder (which must accept any skeleton) and the latent-
     consistency loss need a *per-node* feature tensor computed from a motion. :func:`robot_pose_features`
     turns a robot qpos sequence into ``(T, B, F)`` node features via differentiable FK: for each body,
     its position and 6D orientation expressed in the **root (facing) frame**, plus linear velocity.
     Root-frame normalisation makes the features invariant to global translation/heading, matching
     SAME's "reset transformation" idea and what a retargeting encoder should be blind to.

There is no SMPL-X / AMASS data in this environment, so the training-format loader targets the real
holosoma whole-body-tracking NPZ (``joint_pos`` = root+dof at 50 fps). That NPZ is the ground-truth
consumer contract and doubles as an overfit-a-batch fixture (see ``scripts/overfit_batch.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import rotation as rot


@dataclass
class RobotMotion:
    """A robot motion in canonical form (all torch, wxyz quats)."""

    root_pos: torch.Tensor   # (T, 3)
    root_quat: torch.Tensor  # (T, 4) wxyz
    dof_pos: torch.Tensor    # (T, D)
    fps: float

    @property
    def num_frames(self) -> int:
        return self.root_pos.shape[0]

    def qpos(self) -> torch.Tensor:
        """(T, 7 + D) in MuJoCo layout: [root_pos, root_quat wxyz, dof]."""
        return torch.cat((self.root_pos, self.root_quat, self.dof_pos), dim=-1)

    def to(self, device) -> "RobotMotion":
        return RobotMotion(
            self.root_pos.to(device), self.root_quat.to(device), self.dof_pos.to(device), self.fps
        )


def load_holosoma_wbt_npz(path: str, dtype: torch.dtype = torch.float32) -> RobotMotion:
    """Load a holosoma whole-body-tracking NPZ into a :class:`RobotMotion`.

    The NPZ ``joint_pos`` is ``(T, 7 + D)`` = ``[root_pos(3), root_quat wxyz(4), dof(D)]`` (root quat
    is wxyz on disk, per the holosoma converter). ``fps`` is stored as a length-1 array.
    """
    data = np.load(path, allow_pickle=True)
    jp = np.asarray(data["joint_pos"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    root_pos = torch.tensor(jp[:, 0:3], dtype=dtype)
    root_quat = torch.tensor(jp[:, 3:7], dtype=dtype)  # already wxyz
    dof_pos = torch.tensor(jp[:, 7:], dtype=dtype)
    return RobotMotion(root_pos, root_quat, dof_pos, fps)


def heading_quat_inverse(root_quat: torch.Tensor) -> torch.Tensor:
    """Inverse of the yaw-only (heading) rotation of ``root_quat`` (wxyz), batched (..., 4).

    Used to express body kinematics in a gravity-aligned, heading-normalised frame.
    """
    # heading = atan2 of the rotated +x axis onto the xy plane
    xaxis = torch.zeros(root_quat.shape[:-1] + (3,), dtype=root_quat.dtype, device=root_quat.device)
    xaxis[..., 0] = 1.0
    fwd = rot.quat_rotate(root_quat, xaxis)
    heading = torch.atan2(fwd[..., 1], fwd[..., 0])
    zaxis = torch.zeros_like(xaxis)
    zaxis[..., 2] = 1.0
    return rot.axis_angle_to_quat(zaxis, -heading)


def robot_pose_features(
    robot_kin,
    motion: RobotMotion,
    include_velocity: bool = True,
) -> torch.Tensor:
    """Per-body graph features for a robot motion, expressed in the root heading frame.

    Returns ``(T, B, F)`` with F = 3 (pos) + 6 (rot6d) [+ 3 (lin vel) if ``include_velocity``].
    All quantities are in a frame centred at the root xy and de-rotated by root heading (yaw), so the
    features are invariant to global translation and facing direction.
    """
    body_pos, body_quat = robot_kin.forward_kinematics(motion.root_pos, motion.root_quat, motion.dof_pos)
    T, B, _ = body_pos.shape

    inv_head = heading_quat_inverse(motion.root_quat)  # (T, 4)
    root_xy = motion.root_pos.clone()
    root_xy[..., 2] = 0.0  # keep height (z) absolute, only remove planar translation

    inv_head_b = inv_head[:, None, :].expand(T, B, 4)
    rel_pos = body_pos - root_xy[:, None, :]
    local_pos = rot.quat_rotate(inv_head_b, rel_pos)                       # (T, B, 3)
    local_quat = rot.quat_mul(inv_head_b, body_quat)                       # (T, B, 4)
    local_rot6d = rot.quat_to_rot6d(local_quat)                            # (T, B, 6)

    feats = [local_pos, local_rot6d]
    if include_velocity:
        vel = torch.zeros_like(local_pos)
        if T > 1:
            vel[1:] = local_pos[1:] - local_pos[:-1]
        feats.append(vel)
    return torch.cat(feats, dim=-1)


def world_root_to_local(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a root pose in the per-frame heading frame of an anchor (e.g. the human root).

    Predicting the robot root in *world* coordinates is ill-posed for SNMR: the encoder features are
    deliberately invariant to global translation and heading, so the network cannot know where in the
    world the motion happens. The well-posed target is the robot root pose *relative to the source
    human root* — local xy offset, absolute z, and orientation de-rotated by the human heading. At
    inference the world pose is recovered by composing with the (known) human root trajectory via
    :func:`local_root_to_world`.

    Args:
        anchor_pos: (T, 3) world position of the anchor (human root).
        anchor_quat: (T, 4) wxyz world orientation of the anchor.
        root_pos / root_quat: (T, 3)/(T, 4) world pose to localise (robot root).

    Returns:
        local_pos (T, 3) — xy relative to anchor xy, de-rotated by anchor heading; z absolute.
        local_quat (T, 4) — anchor-heading-inverse ∘ root_quat.
    """
    inv_head = heading_quat_inverse(anchor_quat)  # (T, 4)
    anchor_xy = anchor_pos.clone()
    anchor_xy[..., 2] = 0.0
    local_pos = rot.quat_rotate(inv_head, root_pos - anchor_xy)
    local_quat = rot.quat_mul(inv_head, root_quat)
    return local_pos, local_quat


def local_root_to_world(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    local_pos: torch.Tensor,
    local_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`world_root_to_local`: recover the world root pose from the local one."""
    inv_head = heading_quat_inverse(anchor_quat)
    head = rot.quat_conjugate(inv_head)
    anchor_xy = anchor_pos.clone()
    anchor_xy[..., 2] = 0.0
    world_pos = rot.quat_rotate(head, local_pos) + anchor_xy
    world_quat = rot.quat_mul(head, local_quat)
    return world_pos, world_quat


def robot_node_static_features(robot_graph) -> torch.Tensor:
    """Static per-node features that define the target embodiment for the decoder.

    Returns ``(B, F_s)`` with F_s = 3 (rest offset) + 3 (joint axis) + 1 (has-dof) + 1 (end-effector).
    """
    B = robot_graph.num_bodies
    has_dof = (robot_graph.joint_dof_index >= 0).to(robot_graph.local_translation.dtype).unsqueeze(-1)
    # recompute end-effector flag
    is_parent = torch.zeros(B, dtype=torch.bool, device=robot_graph.device)
    for i in range(B):
        p = int(robot_graph.parent_index[i])
        if p >= 0:
            is_parent[p] = True
    ee = (~is_parent).to(has_dof.dtype).unsqueeze(-1)
    return torch.cat(
        [robot_graph.local_translation, robot_graph.joint_axis, has_dof, ee], dim=-1
    )
