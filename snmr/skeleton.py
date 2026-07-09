"""Generic skeleton graph shared by the human and robot sides.

The shared-latent idea (SAME) requires that *both* the human source skeleton and every robot target
embodiment are described by the **same** structure: a graph of joints/bodies with a parent tree and
per-node static features. The neural encoder/decoder then operate on this common structure, which is
what lets a single latent be embodiment-agnostic.

``SkeletonGraph`` is that common structure. :func:`smplx_body_skeleton` builds the canonical SMPL-X
body skeleton (first 22 joints), whose topology is fixed and provider-independent (it does not need
the SMPL-X ``.pkl`` body model — only the joint tree). Robot skeletons are produced by
``snmr.robot_model.RobotKinematics`` and adapted via :meth:`SkeletonGraph.from_robot_graph`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Canonical SMPL / SMPL-X kinematic tree for the 22 body joints (index -> name, parent index).
# This ordering matches ``smplx.joint_names.JOINT_NAMES[:22]`` used by GMR's SMPL-X loader.
SMPLX_BODY_JOINT_NAMES: list[str] = [
    "pelvis",        # 0  (root)
    "left_hip",      # 1
    "right_hip",     # 2
    "spine1",        # 3
    "left_knee",     # 4
    "right_knee",    # 5
    "spine2",        # 6
    "left_ankle",    # 7
    "right_ankle",   # 8
    "spine3",        # 9
    "left_foot",     # 10
    "right_foot",    # 11
    "neck",          # 12
    "left_collar",   # 13
    "right_collar",  # 14
    "head",          # 15
    "left_shoulder", # 16
    "right_shoulder",# 17
    "left_elbow",    # 18
    "right_elbow",   # 19
    "left_wrist",    # 20
    "right_wrist",   # 21
]
SMPLX_BODY_PARENTS: list[int] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]


@dataclass
class SkeletonGraph:
    """A joint/body graph with a parent tree and static per-node features.

    Attributes:
        names: node names (root first, parent-before-child order).
        parent_index: (N,) long, parent node index; root -> -1.
        rest_offset: (N, 3) local translation of each node relative to its parent in the rest pose
            (used as a static geometric feature; may be zeros if unknown at construction time).
        is_end_effector: (N,) bool, True for leaf nodes (feet/hands/head), used by contact/limb logic.
    """

    names: list[str]
    parent_index: torch.Tensor
    rest_offset: torch.Tensor
    is_end_effector: torch.Tensor

    def __post_init__(self) -> None:
        n = len(self.names)
        assert self.parent_index.shape == (n,)
        assert self.rest_offset.shape == (n, 3)
        assert self.is_end_effector.shape == (n,)
        for i in range(n):
            assert int(self.parent_index[i]) < i, "nodes must be in parent-before-child order"

    @property
    def num_nodes(self) -> int:
        return len(self.names)

    @property
    def device(self) -> torch.device:
        return self.parent_index.device

    def edge_index(self) -> torch.Tensor:
        """Undirected edges (2, 2E) including both directions; root contributes none."""
        children = [i for i in range(self.num_nodes) if int(self.parent_index[i]) >= 0]
        parents = [int(self.parent_index[i]) for i in children]
        src = children + parents
        dst = parents + children
        return torch.tensor([src, dst], dtype=torch.long, device=self.device)

    def to(self, device: str | torch.device) -> "SkeletonGraph":
        return SkeletonGraph(
            names=self.names,
            parent_index=self.parent_index.to(device),
            rest_offset=self.rest_offset.to(device),
            is_end_effector=self.is_end_effector.to(device),
        )

    @staticmethod
    def from_robot_graph(robot_graph) -> "SkeletonGraph":
        """Adapt a :class:`snmr.robot_model.RobotGraph` into the shared skeleton structure."""
        parent = robot_graph.parent_index
        n = robot_graph.num_bodies
        # A body is an end effector if it is nobody's parent.
        is_parent = torch.zeros(n, dtype=torch.bool, device=parent.device)
        for i in range(n):
            p = int(parent[i])
            if p >= 0:
                is_parent[p] = True
        return SkeletonGraph(
            names=list(robot_graph.body_names),
            parent_index=parent.clone(),
            rest_offset=robot_graph.local_translation.clone(),
            is_end_effector=~is_parent,
        )


def smplx_body_skeleton(
    rest_offset: torch.Tensor | None = None, device: str | torch.device = "cpu"
) -> SkeletonGraph:
    """Canonical SMPL-X body skeleton (22 joints).

    Args:
        rest_offset: optional (22, 3) rest-pose local offsets. If ``None``, zeros are used (topology
            is still correct; supply mean bone vectors from data for geometric features in training).
    """
    device = torch.device(device)
    names = list(SMPLX_BODY_JOINT_NAMES)
    parent = torch.tensor(SMPLX_BODY_PARENTS, dtype=torch.long, device=device)
    n = len(names)
    if rest_offset is None:
        rest_offset = torch.zeros(n, 3, device=device)
    else:
        rest_offset = rest_offset.to(device)
    is_parent = torch.zeros(n, dtype=torch.bool, device=device)
    for i in range(n):
        p = int(parent[i])
        if p >= 0:
            is_parent[p] = True
    return SkeletonGraph(
        names=names,
        parent_index=parent,
        rest_offset=rest_offset,
        is_end_effector=~is_parent,
    )
