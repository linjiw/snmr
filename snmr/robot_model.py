"""Differentiable robot kinematics + robot embodiment graph, extracted from an MJCF.

Design decisions (and why):

  * **Static params come from MuJoCo, FK is re-implemented in torch.** We load the MJCF with
    ``mujoco.MjModel`` once to get the resolved kinematic tree (body parents, local pos/quat, joint
    axes, ranges, qpos addressing). This is robust to MJCF defaults/includes/meshes that a hand-rolled
    XML parser (like GMR's ``KinematicsModel``) would miss. We then run forward kinematics in pure
    torch so it is differentiable and batched on GPU. ``tests/test_fk.py`` asserts the torch FK matches
    ``mujoco.mj_forward`` to <1e-4 m / <1e-4 rad on random configurations, so any indexing/ordering bug
    surfaces immediately.

  * **Assumes hinge joints sit at the body origin (``jnt_pos == 0``).** True for the humanoid robots in
    scope (verified for Unitree G1). If a model violates this we raise, rather than silently emitting
    wrong FK.

  * **wxyz quaternions throughout**, matching MuJoCo ``qpos`` and ``snmr.rotation``.

The class also exposes the *embodiment graph* (nodes = bodies, edges = parent links) plus per-node
features (local offset, joint axis, joint range) that condition the neural decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import rotation as rot


@dataclass
class RobotGraph:
    """Static, embodiment-defining description of a robot (numpy/torch tensors on one device)."""

    name: str
    body_names: list[str]
    parent_index: torch.Tensor          # (B,) long, parent body index; root -> -1
    local_translation: torch.Tensor     # (B, 3) body pos relative to parent
    local_rotation: torch.Tensor        # (B, 4) wxyz body quat relative to parent
    # Per-body single-hinge description (0 dof bodies get axis=0, dof_index=-1):
    joint_axis: torch.Tensor            # (B, 3) hinge axis in body frame (zeros if no dof)
    joint_dof_index: torch.Tensor       # (B,) long index into dof vector, -1 if no dof
    dof_lower: torch.Tensor             # (D,) lower joint limits (rad)
    dof_upper: torch.Tensor             # (D,) upper joint limits (rad)
    dof_body_index: torch.Tensor        # (D,) long, which body each dof drives
    num_dof: int

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def device(self) -> torch.device:
        return self.parent_index.device

    def edge_index(self) -> torch.Tensor:
        """Undirected graph edges (2, 2*E) as (child<->parent) pairs incl. both directions.

        Root (parent == -1) contributes no edge. Suitable for message passing.
        """
        children = [i for i in range(self.num_bodies) if int(self.parent_index[i]) >= 0]
        parents = [int(self.parent_index[i]) for i in children]
        src = children + parents
        dst = parents + children
        return torch.tensor([src, dst], dtype=torch.long, device=self.device)


class RobotKinematics:
    """Loads an MJCF, exposes a :class:`RobotGraph`, and does differentiable forward kinematics."""

    def __init__(self, mjcf_path: str, device: str | torch.device = "cpu"):
        import mujoco  # local import so the package imports without mujoco for pure-model use

        self.mjcf_path = str(mjcf_path)
        self.device = torch.device(device)
        model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self._mujoco = mujoco

        # --- identify the free (root) joint -------------------------------------------------
        free_joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free_joints) != 1:
            raise ValueError(
                f"{self.mjcf_path}: expected exactly one free root joint, found {len(free_joints)}"
            )
        root_body = int(model.jnt_bodyid[free_joints[0]])

        # --- collect the subtree rooted at the floating base (skip the MuJoCo 'world' body) --
        body_names: list[str] = []
        old_to_new: dict[int, int] = {}
        order: list[int] = []

        def _walk(bid: int) -> None:
            order.append(bid)
            for child in range(model.nbody):
                if int(model.body_parentid[child]) == bid and child != bid:
                    _walk(child)

        _walk(root_body)
        for new_idx, old_idx in enumerate(order):
            old_to_new[old_idx] = new_idx
            body_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, old_idx))

        B = len(order)
        parent_index = np.full(B, -1, dtype=np.int64)
        local_translation = np.zeros((B, 3), dtype=np.float64)
        local_rotation = np.zeros((B, 4), dtype=np.float64)
        joint_axis = np.zeros((B, 3), dtype=np.float64)
        joint_dof_index = np.full(B, -1, dtype=np.int64)

        dof_lower: list[float] = []
        dof_upper: list[float] = []
        dof_body_index: list[int] = []

        for new_idx, old_idx in enumerate(order):
            local_translation[new_idx] = model.body_pos[old_idx]
            local_rotation[new_idx] = model.body_quat[old_idx]  # MJCF quats are wxyz
            p_old = int(model.body_parentid[old_idx])
            if old_idx == root_body:
                parent_index[new_idx] = -1
            else:
                parent_index[new_idx] = old_to_new[p_old]

            # hinge joints attached to this body
            jids = [
                j
                for j in range(model.njnt)
                if int(model.jnt_bodyid[j]) == old_idx and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            ]
            if old_idx == root_body:
                continue  # root motion handled by the free joint, not a hinge

            # Fail loud on any non-hinge actuated joint (slide/ball). Silently dropping these would
            # (a) contradict this class's fail-loud contract and (b) mis-align the dof vector: every
            # hinge after a dropped slide/ball would receive the wrong element of ``dof_pos``.
            non_hinge = [
                j
                for j in range(model.njnt)
                if int(model.jnt_bodyid[j]) == old_idx
                and model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_FREE)
            ]
            if non_hinge:
                types = [int(model.jnt_type[j]) for j in non_hinge]
                raise NotImplementedError(
                    f"{self.mjcf_path}: body '{body_names[new_idx]}' has non-hinge joint(s) of "
                    f"MuJoCo type(s) {types} (slide/ball); only single-hinge actuated bodies are "
                    "supported."
                )
            if len(jids) == 0:
                continue
            if len(jids) > 1:
                raise NotImplementedError(
                    f"{self.mjcf_path}: body '{body_names[new_idx]}' has {len(jids)} hinge joints; "
                    "only single-hinge bodies are supported."
                )
            j = jids[0]
            if np.linalg.norm(model.jnt_pos[j]) > 1e-6:
                raise NotImplementedError(
                    f"{self.mjcf_path}: joint on body '{body_names[new_idx]}' has non-zero jnt_pos "
                    f"{model.jnt_pos[j]}; only origin-anchored hinges are supported."
                )
            dof = len(dof_lower)
            joint_axis[new_idx] = model.jnt_axis[j]
            joint_dof_index[new_idx] = dof
            lo, hi = model.jnt_range[j]
            # A hinge with limited==0 has range [0,0]; treat as unlimited.
            if bool(model.jnt_limited[j]) and not (lo == 0.0 and hi == 0.0):
                dof_lower.append(float(lo))
                dof_upper.append(float(hi))
            else:
                dof_lower.append(-np.pi)
                dof_upper.append(np.pi)
            dof_body_index.append(new_idx)

        t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=self.device)  # noqa: E731
        self.graph = RobotGraph(
            name=body_names[0],
            body_names=body_names,
            parent_index=t(parent_index, torch.long),
            local_translation=t(local_translation),
            local_rotation=t(local_rotation),
            joint_axis=t(joint_axis),
            joint_dof_index=t(joint_dof_index, torch.long),
            dof_lower=t(dof_lower),
            dof_upper=t(dof_upper),
            dof_body_index=t(dof_body_index, torch.long),
            num_dof=len(dof_lower),
        )
        # Traversal order guarantees parent index < child index, so a single forward sweep is valid.
        self._check_topo_order()
        # Precompute tree-depth levels for the vectorized FK: bodies at the same depth are
        # independent given their parents, so one op per depth (~tree height) replaces the per-body
        # Python loop (~num_bodies). Cuts kernel-launch overhead ~6x on the CPU-bound contact path.
        self._build_depth_levels()

    def _build_depth_levels(self) -> None:
        parent = self.graph.parent_index.tolist()
        depth = [0] * self.graph.num_bodies
        for i in range(self.graph.num_bodies):
            p = parent[i]
            depth[i] = 0 if p < 0 else depth[p] + 1
        levels: dict[int, list[int]] = {}
        for i, d in enumerate(depth):
            levels.setdefault(d, []).append(i)
        # per level (excluding root level 0): child body indices + their parent indices
        self._fk_levels = []
        dev = self.graph.parent_index.device
        for d in sorted(levels):
            if d == 0:
                continue
            children = levels[d]
            parents = [parent[c] for c in children]
            self._fk_levels.append((
                torch.tensor(children, dtype=torch.long, device=dev),
                torch.tensor(parents, dtype=torch.long, device=dev),
            ))

    def _check_topo_order(self) -> None:
        p = self.graph.parent_index
        for i in range(self.graph.num_bodies):
            assert int(p[i]) < i, "bodies must be in parent-before-child order"

    # ------------------------------------------------------------------------------------
    @property
    def num_dof(self) -> int:
        return self.graph.num_dof

    @property
    def num_bodies(self) -> int:
        return self.graph.num_bodies

    @property
    def body_names(self) -> list[str]:
        return self.graph.body_names

    def body_index(self, name: str) -> int:
        return self.graph.body_names.index(name)

    def dof_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.graph.dof_lower, self.graph.dof_upper

    # ------------------------------------------------------------------------------------
    def forward_kinematics(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        dof_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched differentiable FK.

        Args:
            root_pos:  (..., 3) world position of the floating base.
            root_quat: (..., 4) wxyz world orientation of the floating base.
            dof_pos:   (..., D) hinge joint angles in MuJoCo dof order.

        Returns:
            body_pos:  (..., B, 3) world positions of every body.
            body_quat: (..., B, 4) wxyz world orientations of every body.
        """
        g = self.graph
        lead = root_pos.shape[:-1]
        B = g.num_bodies
        dtype = root_pos.dtype
        local_translation = g.local_translation.to(dtype)
        local_rotation = g.local_rotation.to(dtype)

        # Per-body local hinge rotation (identity for dof-less bodies).
        local_joint_quat = self._dof_to_body_quat(dof_pos, lead)  # (..., B, 4)

        # Body-local composed rotation R_local[j] = body_quat[j] * joint_quat[j] (jnt_pos == 0),
        # precomputed for all bodies at once (this and quat_rotate are the batched primitives).
        local_rot_full = local_rotation.expand(lead + (B, 4))
        local_rot_composed = rot.quat_mul(local_rot_full, local_joint_quat)  # (..., B, 4)
        local_trans_full = local_translation.expand(lead + (B, 3))

        # World buffers, filled level by level (root at depth 0 is the input). We accumulate with
        # index_add on fresh zero tensors each level (out-of-place -> autograd-safe), building the
        # full (...,B,·) arrays across levels via a running scatter that never writes in place.
        world_pos = root_pos.new_zeros(lead + (B, 3))
        world_quat = root_quat.new_zeros(lead + (B, 4))
        root_oh = torch.zeros(B, 1, dtype=dtype, device=root_pos.device)
        root_oh[0, 0] = 1.0
        world_pos = world_pos + root_oh * root_pos.unsqueeze(-2)
        world_quat = world_quat + root_oh * root_quat.unsqueeze(-2)

        for children, parents in self._fk_levels:
            pq = world_quat.index_select(-2, parents)               # (..., L, 4) parent world rot
            pp = world_pos.index_select(-2, parents)                # (..., L, 3) parent world pos
            lt = local_trans_full.index_select(-2, children)        # (..., L, 3)
            lr = local_rot_composed.index_select(-2, children)      # (..., L, 4)
            cur_pos = pp + rot.quat_rotate(pq, lt)                   # (..., L, 3)
            cur_quat = rot.quat_mul(pq, lr)                          # (..., L, 4)
            # scatter this level's results into fresh buffers (out-of-place add of a masked term)
            world_pos = world_pos.index_add(-2, children, cur_pos)
            world_quat = world_quat.index_add(-2, children, cur_quat)

        return world_pos, world_quat

    def _dof_to_body_quat(self, dof_pos: torch.Tensor, lead: tuple[int, ...]) -> torch.Tensor:
        g = self.graph
        B = g.num_bodies
        out = torch.zeros(lead + (B, 4), dtype=dof_pos.dtype, device=dof_pos.device)
        out[..., 0] = 1.0  # identity quats everywhere by default
        has_dof = g.joint_dof_index >= 0
        if has_dof.any():
            body_ids = torch.nonzero(has_dof, as_tuple=False).squeeze(-1)  # (D,)
            dof_ids = g.joint_dof_index[body_ids]                          # (D,)
            axes = g.joint_axis[body_ids].to(dof_pos.dtype)                # (D, 3)
            angles = dof_pos[..., dof_ids]                                 # (..., D)
            axes_b = axes.expand(lead + axes.shape)                        # (..., D, 3)
            quats = rot.axis_angle_to_quat(axes_b, angles)                 # (..., D, 4)
            out[..., body_ids, :] = quats
        return out
