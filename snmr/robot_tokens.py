"""Serialization-equivariant tensorization of :mod:`snmr.robot_spec` graphs.

The legacy SNMR decoder receives nodes in one fixed body order.  MorphoRetarget must
instead treat link order as serialization: permuting a ``RobotSpec`` may permute its
output rows, but must not change their physical meaning.  This module therefore:

* exposes no names, hashes, paths, robot IDs, or raw parent indices as model features;
* pads heterogeneous link counts with an explicit node mask;
* represents topology through all-pairs tree distance and a structural parent tensor;
* keeps dynamics availability separate from numeric values; and
* emits joint limits per node for a shared, limit-by-construction output head.

The first registered MorphoRetarget experiment should use ``feature_set="kinematic"``.
The full 37-D RobotSpec features are available for the later dynamics experiment, but
this tensor contract does not itself establish that a learned model uses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn

from .robot_spec import RobotFeatureBundle, RobotSpec


FeatureSet = Literal["kinematic", "full"]

# Fields whose values do not change under a dynamics-only RobotSpec intervention.  The
# structural scalars appended below are also kinematic.  ``parent_index`` is deliberately
# absent: its integer value depends on serialization and belongs in the topology tensor.
KINEMATIC_FEATURE_NAMES: tuple[str, ...] = (
    "parent_to_joint_x_over_height",
    "parent_to_joint_y_over_height",
    "parent_to_joint_z_over_height",
    "rest_rot6d_0",
    "rest_rot6d_1",
    "rest_rot6d_2",
    "rest_rot6d_3",
    "rest_rot6d_4",
    "rest_rot6d_5",
    "collision_radius_over_height",
    "joint_axis_x",
    "joint_axis_y",
    "joint_axis_z",
    "joint_lower_over_pi",
    "joint_upper_over_pi",
    "has_dof",
    "is_root",
    "is_torso",
    "is_head",
    "is_left_hand",
    "is_right_hand",
    "is_left_foot",
    "is_right_foot",
    "is_allowed_contact",
)

TOPOLOGY_FEATURE_NAMES: tuple[str, ...] = (
    "link_length_over_height",
    "depth_over_max_depth",
    "child_count_over_max_child_count",
    "is_left_symmetry_member",
    "is_right_symmetry_member",
)


@dataclass(frozen=True)
class RobotTokenBatch:
    """Padded graph tokens for a batch of variable-link-count robots.

    Names are retained only as non-model metadata so callers can invert a serialization
    permutation and audit outputs.  They must never be embedded or passed to a model.
    """

    node_features: torch.Tensor  # (B, N, F), exactly 37 fields for feature_set="full"
    topology_features: torch.Tensor  # (B, N, 5), derived and kept separately auditable
    # (B, N, 4), separate from numeric values; deliberately zero for kinematic-only batches.
    dynamics_available: torch.Tensor
    node_mask: torch.Tensor  # (B, N), True for real nodes
    parent_index: torch.Tensor  # (B, N), -1 root, -2 padding; structural metadata
    tree_distance: torch.Tensor  # (B, N, N), -1 wherever either node is padding
    joint_mask: torch.Tensor  # (B, N), one revolute DoF attached to this link
    joint_lower: torch.Tensor  # (B, N), radians
    joint_upper: torch.Tensor  # (B, N), radians
    root_mask: torch.Tensor  # (B, N), exactly one root per robot
    node_names: tuple[tuple[str, ...], ...]  # audit metadata only
    feature_names: tuple[str, ...]
    topology_feature_names: tuple[str, ...]
    availability_names: tuple[str, ...]

    def to(self, device: torch.device | str) -> "RobotTokenBatch":
        """Move tensor fields while preserving immutable audit metadata."""

        return RobotTokenBatch(
            node_features=self.node_features.to(device),
            topology_features=self.topology_features.to(device),
            dynamics_available=self.dynamics_available.to(device),
            node_mask=self.node_mask.to(device),
            parent_index=self.parent_index.to(device),
            tree_distance=self.tree_distance.to(device),
            joint_mask=self.joint_mask.to(device),
            joint_lower=self.joint_lower.to(device),
            joint_upper=self.joint_upper.to(device),
            root_mask=self.root_mask.to(device),
            node_names=self.node_names,
            feature_names=self.feature_names,
            topology_feature_names=self.topology_feature_names,
            availability_names=self.availability_names,
        )

    def attention_bias(self, gamma: float) -> torch.Tensor:
        """Return ``-gamma * tree_distance`` with padding masked to ``-inf``."""

        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError(f"gamma must be finite and nonnegative, got {gamma}")
        valid = self.tree_distance >= 0
        bias = -float(gamma) * self.tree_distance.clamp_min(0).to(self.node_features.dtype)
        return bias.masked_fill(~valid, float("-inf"))


class RobotGraphTokenizer(nn.Module):
    """Convert one or more ``RobotSpec`` objects into padded graph tensors.

    This module is intentionally parameter-free.  A learned graph/topology embedder can
    project ``node_features`` after this contract has been tested independently.
    """

    def __init__(self, feature_set: FeatureSet = "kinematic") -> None:
        super().__init__()
        if feature_set not in ("kinematic", "full"):
            raise ValueError(f"unknown feature_set {feature_set!r}")
        self.feature_set: FeatureSet = feature_set

    def forward(self, specs: Sequence[RobotSpec]) -> RobotTokenBatch:
        if not specs:
            raise ValueError("at least one RobotSpec is required")
        for spec in specs:
            spec.validate()

        bundles = [spec.model_features() for spec in specs]
        feature_names = self._feature_names(bundles[0])
        max_nodes = max(len(spec.links) for spec in specs)
        batch_size = len(specs)
        feature_dim = len(feature_names)
        availability_dim = len(bundles[0].dynamics_available_names)

        node_features = torch.zeros(batch_size, max_nodes, feature_dim, dtype=torch.float32)
        topology_features = torch.zeros(
            batch_size, max_nodes, len(TOPOLOGY_FEATURE_NAMES), dtype=torch.float32
        )
        dynamics_available = torch.zeros(
            batch_size, max_nodes, availability_dim, dtype=torch.float32
        )
        node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        parent_index = torch.full((batch_size, max_nodes), -2, dtype=torch.long)
        tree_distance = torch.full((batch_size, max_nodes, max_nodes), -1, dtype=torch.long)
        joint_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
        joint_lower = torch.zeros(batch_size, max_nodes, dtype=torch.float32)
        joint_upper = torch.zeros(batch_size, max_nodes, dtype=torch.float32)
        root_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)

        for batch_index, (spec, bundle) in enumerate(zip(specs, bundles)):
            if bundle.dynamics_available_names != bundles[0].dynamics_available_names:
                raise ValueError("RobotSpecs disagree on dynamics availability fields")
            values = self._features_for_spec(spec, bundle)
            if values.shape[1] != feature_dim:
                raise AssertionError((values.shape, feature_dim))
            count = values.shape[0]
            distances = _tree_distances(bundle.parent_index)

            node_features[batch_index, :count] = torch.from_numpy(values)
            topology_features[batch_index, :count] = torch.from_numpy(
                _topology_features(spec, bundle.parent_index)
            )
            dynamics_available[batch_index, :count] = torch.from_numpy(
                np.asarray(bundle.dynamics_available, dtype=np.float32)
                if self.feature_set == "full"
                else np.zeros_like(bundle.dynamics_available, dtype=np.float32)
            )
            node_mask[batch_index, :count] = True
            parent_index[batch_index, :count] = torch.from_numpy(bundle.parent_index)
            tree_distance[batch_index, :count, :count] = torch.from_numpy(distances)

            link_index = {link.name: i for i, link in enumerate(spec.links)}
            root_mask[batch_index, link_index[spec.semantics.root_link]] = True
            for joint in spec.joints:
                if joint.joint_type != "revolute":
                    raise ValueError(
                        f"joint {joint.name!r} has unsupported type {joint.joint_type!r}"
                    )
                node_index = link_index[joint.child_link]
                if joint_mask[batch_index, node_index]:
                    raise ValueError(
                        f"link {joint.child_link!r} has more than one scalar joint; "
                        "multi-DoF links need separate joint tokens"
                    )
                joint_mask[batch_index, node_index] = True
                joint_lower[batch_index, node_index] = joint.lower_limit
                joint_upper[batch_index, node_index] = joint.upper_limit

        return RobotTokenBatch(
            node_features=node_features,
            topology_features=topology_features,
            dynamics_available=dynamics_available,
            node_mask=node_mask,
            parent_index=parent_index,
            tree_distance=tree_distance,
            joint_mask=joint_mask,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            root_mask=root_mask,
            node_names=tuple(tuple(bundle.node_names) for bundle in bundles),
            feature_names=feature_names,
            topology_feature_names=TOPOLOGY_FEATURE_NAMES,
            availability_names=bundles[0].dynamics_available_names,
        )

    def _feature_names(self, bundle: RobotFeatureBundle) -> tuple[str, ...]:
        if self.feature_set == "full":
            return bundle.node_feature_names
        missing = set(KINEMATIC_FEATURE_NAMES) - set(bundle.node_feature_names)
        if missing:
            raise ValueError(f"RobotSpec is missing kinematic model fields: {sorted(missing)}")
        return KINEMATIC_FEATURE_NAMES

    def _features_for_spec(
        self,
        spec: RobotSpec,
        bundle: RobotFeatureBundle,
    ) -> np.ndarray:
        field_index = {name: i for i, name in enumerate(bundle.node_feature_names)}
        if self.feature_set == "full":
            base = np.asarray(bundle.node_features, dtype=np.float32)
        else:
            base = np.asarray(
                bundle.node_features[:, [field_index[name] for name in KINEMATIC_FEATURE_NAMES]],
                dtype=np.float32,
            )
        return base.astype(np.float32, copy=False)


def bounded_joint_positions(
    logits: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    joint_mask: torch.Tensor,
) -> torch.Tensor:
    """Map shared per-node logits into joint ranges without serialization assumptions.

    Non-joint and padded nodes are returned as zero; callers should use ``joint_mask``
    when gathering the variable number of actual DoFs.
    """

    if logits.shape != joint_lower.shape or logits.shape != joint_upper.shape:
        raise ValueError("logits and joint limits must have identical shapes")
    if joint_mask.shape != logits.shape:
        raise ValueError("joint_mask must match logits")
    if torch.any(joint_lower[joint_mask] >= joint_upper[joint_mask]):
        raise ValueError("every active joint must have lower < upper")
    unit = torch.tanh(logits)
    values = joint_lower + 0.5 * (joint_upper - joint_lower) * (unit + 1.0)
    return torch.where(joint_mask, values, torch.zeros_like(values))


def _tree_distances(parent_index: np.ndarray) -> np.ndarray:
    """All-pairs distances for a valid tree in arbitrary node order."""

    parents = np.asarray(parent_index, dtype=np.int64)
    if parents.ndim != 1 or parents.size == 0:
        raise ValueError("parent_index must be a nonempty vector")
    node_count = int(parents.size)
    roots = np.flatnonzero(parents == -1)
    if roots.size != 1:
        raise ValueError(f"expected one root, found {roots.size}")
    if np.any((parents < -1) | (parents >= node_count)):
        raise ValueError("parent_index contains an out-of-range node")
    if np.any(parents == np.arange(node_count)):
        raise ValueError("a node cannot be its own parent")

    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for child, parent in enumerate(parents.tolist()):
        if parent >= 0:
            adjacency[child].append(parent)
            adjacency[parent].append(child)

    distances = np.full((node_count, node_count), -1, dtype=np.int64)
    for source in range(node_count):
        distances[source, source] = 0
        frontier = [source]
        while frontier:
            node = frontier.pop(0)
            for neighbor in adjacency[node]:
                if distances[source, neighbor] >= 0:
                    continue
                distances[source, neighbor] = distances[source, node] + 1
                frontier.append(neighbor)
    if np.any(distances < 0):
        raise ValueError("parent_index does not define a connected tree")
    if sum(len(neighbors) for neighbors in adjacency) != 2 * (node_count - 1):
        raise ValueError("parent_index does not define an acyclic tree")
    return distances


def _topology_features(spec: RobotSpec, parent_index: np.ndarray) -> np.ndarray:
    parents = np.asarray(parent_index, dtype=np.int64)
    distances = _tree_distances(parents)
    root = int(np.flatnonzero(parents == -1)[0])
    depth = distances[root].astype(np.float32)
    max_depth = max(float(depth.max()), 1.0)
    child_count = np.bincount(parents[parents >= 0], minlength=len(parents)).astype(np.float32)
    max_children = max(float(child_count.max()), 1.0)
    link_length = np.asarray(
        [np.linalg.norm(link.local_position) / spec.standing_height for link in spec.links],
        dtype=np.float32,
    )
    link_index = {link.name: i for i, link in enumerate(spec.links)}
    left = np.zeros(len(spec.links), dtype=np.float32)
    right = np.zeros(len(spec.links), dtype=np.float32)
    for left_name, right_name in spec.semantics.symmetry_pairs:
        left[link_index[left_name]] = 1.0
        right[link_index[right_name]] = 1.0
    return np.stack(
        (link_length, depth / max_depth, child_count / max_children, left, right),
        axis=1,
    )
