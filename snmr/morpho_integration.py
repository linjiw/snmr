"""Fixed-target integration of the SNMR human encoder and MorphoRetarget decoder.

This module is the deliberately small bridge needed for the first fixed-G1
amortisation check.  It reuses :class:`snmr.model.MotionEncoder`,
``human_pose_features``/``human_static_features``, and
:class:`snmr.morpho_model.KinematicMorphoRetargeter` without introducing a second
human feature path.

One kinematic-only :class:`snmr.robot_tokens.RobotTokenBatch` is bound into the
module at construction time.  Its tensors and audit metadata are checkpoint state,
while names and hashes remain audit-only and never enter the learned forward pass.
The public tensor path requires explicit time and node masks.  The canonical
``HumanMotionSpec`` path preserves the motion hashes in the returned audit record.

Scalar joint-node trajectories and a learned global root head are implemented here.
The root is expressed in the same source-heading-local convention as
``snmr.data.world_root_to_local``: local XY, absolute Z, and heading-relative ``wxyz``
orientation.  It is predicted from human tokens and permutation-invariant pooled
kinematic robot embeddings; it is never teacher-forced or copied from the legacy
decoder.  Contacts remain outside the model and must be derived from robot FK under a
registered sole/contact protocol.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import json
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from . import rotation as rot
from .human import human_pose_features, human_static_features
from .model import MotionEncoder, SNMRConfig, _adjacency
from .morpho_model import (
    KinematicMorphoRetargeter,
    MorphoRetargetConfig,
    MorphoRetargetOutput,
)
from .motion_spec import HumanMotionSpec
from .robot_tokens import (
    KINEMATIC_FEATURE_NAMES,
    ROBOT_TOKEN_SCHEMA_VERSION,
    TOPOLOGY_FEATURE_NAMES,
    RobotTokenBatch,
)
from .skeleton import SkeletonGraph


INTEGRATION_SCHEMA_VERSION = "snmr.morpho-fixed-target.v0.1"


@dataclass(frozen=True)
class MorphoIntegrationOutput:
    """Joint-only prediction plus the exact input-contract bindings.

    ``joint_output`` remains node-aligned.  Gather physical scalar DoFs with its
    ``valid_joint_mask``; do not assume a fixed joint index order.  Contact fields are
    intentionally absent rather than silently sourced from a teacher.
    """

    joint_output: MorphoRetargetOutput
    root_position_local: torch.Tensor  # (B, T, 3), source-heading-local XY + absolute Z
    root_orientation_local_wxyz: torch.Tensor  # (B, T, 4), unit quaternion
    human_tokens: torch.Tensor  # (B, T, latent_dim), zero at invalid times
    human_time_mask: torch.Tensor  # (B, T), explicit valid-prefix mask
    robot_token_audit_manifest: Mapping[str, Any]
    human_motion_spec_sha256: str | None = None
    human_motion_buffer_sha256: str | None = None
    schema_version: str = INTEGRATION_SCHEMA_VERSION


class _KinematicRootHead(nn.Module):
    """Identity-free global root head over motion and pooled robot kinematics."""

    def __init__(self, human_dim: int, robot_dim: int) -> None:
        super().__init__()
        hidden = max(human_dim, robot_dim)
        self.network = nn.Sequential(
            nn.Linear(human_dim + robot_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 9),  # local position (3) + continuous rotation 6D
        )

    def forward(
        self,
        human_tokens: torch.Tensor,
        robot_embeddings: torch.Tensor,
        robot_node_mask: torch.Tensor,
        human_time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = robot_node_mask.to(robot_embeddings.dtype).unsqueeze(-1)
        pooled_robot = (robot_embeddings * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        pooled_by_time = pooled_robot[:, None, :].expand(
            -1, human_tokens.shape[1], -1
        )
        raw = self.network(torch.cat((human_tokens, pooled_by_time), dim=-1))
        valid = human_time_mask.unsqueeze(-1)
        local_position = raw[..., :3] * valid
        local_orientation = rot.rot6d_to_quat(raw[..., 3:9])
        identity = torch.zeros_like(local_orientation)
        identity[..., 0] = 1.0
        local_orientation = torch.where(valid, local_orientation, identity)
        return local_position, local_orientation


class _BoundRobotTokens(nn.Module):
    """Persistent singleton RobotTokenBatch with checkpointed audit metadata."""

    _TENSOR_FIELDS = (
        "node_features",
        "topology_features",
        "dynamics_available",
        "node_mask",
        "parent_index",
        "tree_distance",
        "joint_mask",
        "joint_lower",
        "joint_upper",
        "root_mask",
    )

    def __init__(self, tokens: RobotTokenBatch) -> None:
        super().__init__()
        if not isinstance(tokens, RobotTokenBatch):
            raise TypeError("target_robot_tokens must be a RobotTokenBatch")
        if tokens.node_features.shape[0] != 1:
            raise ValueError("fixed-target integration binds exactly one robot")
        if tokens.schema_version != ROBOT_TOKEN_SCHEMA_VERSION:
            raise ValueError("unsupported RobotTokenBatch schema")
        if tuple(tokens.feature_names) != KINEMATIC_FEATURE_NAMES:
            raise ValueError(
                "fixed-target integration requires RobotGraphTokenizer(feature_set='kinematic')"
            )
        if tuple(tokens.topology_feature_names) != TOPOLOGY_FEATURE_NAMES:
            raise ValueError("unexpected topology feature contract")
        if torch.count_nonzero(tokens.dynamics_available).item() != 0:
            raise ValueError("dynamics availability must be zero for the kinematic-only gate")

        for name in self._TENSOR_FIELDS:
            self.register_buffer(name, getattr(tokens, name).detach().clone(), persistent=True)
        self._set_metadata(
            node_names=tokens.node_names,
            feature_names=tokens.feature_names,
            topology_feature_names=tokens.topology_feature_names,
            availability_names=tokens.availability_names,
            source_spec_hashes=tokens.source_spec_hashes,
            source_kinematic_hashes=tokens.source_kinematic_hashes,
            schema_version=tokens.schema_version,
            expected_manifest=tokens.audit_manifest(),
        )
        self._verify_binding()

    def _set_metadata(
        self,
        *,
        node_names: Any,
        feature_names: Any,
        topology_feature_names: Any,
        availability_names: Any,
        source_spec_hashes: Any,
        source_kinematic_hashes: Any,
        schema_version: Any,
        expected_manifest: Any,
    ) -> None:
        self._node_names = tuple(tuple(str(name) for name in row) for row in node_names)
        self._feature_names = tuple(str(name) for name in feature_names)
        self._topology_feature_names = tuple(
            str(name) for name in topology_feature_names
        )
        self._availability_names = tuple(str(name) for name in availability_names)
        self._source_spec_hashes = tuple(str(value) for value in source_spec_hashes)
        self._source_kinematic_hashes = tuple(
            str(value) for value in source_kinematic_hashes
        )
        self._schema_version = str(schema_version)
        # A JSON round-trip removes caller-owned mutable containers and rejects NaN.
        encoded = json.dumps(
            expected_manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._expected_manifest = json.loads(encoded)

    def get_extra_state(self) -> dict[str, Any]:
        """Make audit-only metadata part of ``state_dict`` checkpoints."""

        return {
            "binding_schema": "snmr.bound-robot-tokens.v0.1",
            "node_names": [list(row) for row in self._node_names],
            "feature_names": list(self._feature_names),
            "topology_feature_names": list(self._topology_feature_names),
            "availability_names": list(self._availability_names),
            "source_spec_hashes": list(self._source_spec_hashes),
            "source_kinematic_hashes": list(self._source_kinematic_hashes),
            "robot_token_schema": self._schema_version,
            "expected_manifest": copy.deepcopy(self._expected_manifest),
        }

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, dict) or state.get("binding_schema") != (
            "snmr.bound-robot-tokens.v0.1"
        ):
            raise ValueError("unsupported bound RobotToken checkpoint metadata")
        required = {
            "binding_schema",
            "node_names",
            "feature_names",
            "topology_feature_names",
            "availability_names",
            "source_spec_hashes",
            "source_kinematic_hashes",
            "robot_token_schema",
            "expected_manifest",
        }
        if set(state) != required:
            raise ValueError("bound RobotToken checkpoint metadata fields do not match")
        self._set_metadata(
            node_names=state["node_names"],
            feature_names=state["feature_names"],
            topology_feature_names=state["topology_feature_names"],
            availability_names=state["availability_names"],
            source_spec_hashes=state["source_spec_hashes"],
            source_kinematic_hashes=state["source_kinematic_hashes"],
            schema_version=state["robot_token_schema"],
            expected_manifest=state["expected_manifest"],
        )

    def _singleton(self) -> RobotTokenBatch:
        return RobotTokenBatch(
            **{name: getattr(self, name) for name in self._TENSOR_FIELDS},
            node_names=self._node_names,
            feature_names=self._feature_names,
            topology_feature_names=self._topology_feature_names,
            availability_names=self._availability_names,
            source_spec_hashes=self._source_spec_hashes,
            source_kinematic_hashes=self._source_kinematic_hashes,
            schema_version=self._schema_version,
        )

    def _verify_binding(self) -> dict[str, Any]:
        current = self._singleton().audit_manifest()
        if current != self._expected_manifest:
            raise ValueError(
                "bound RobotToken tensors or audit metadata changed after construction/load"
            )
        return current

    def audit_manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._verify_binding())

    def for_batch(self, batch_size: int) -> RobotTokenBatch:
        self._verify_binding()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        def repeat(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.expand(batch_size, *tensor.shape[1:])

        return RobotTokenBatch(
            **{name: repeat(getattr(self, name)) for name in self._TENSOR_FIELDS},
            node_names=self._node_names * batch_size,
            feature_names=self._feature_names,
            topology_feature_names=self._topology_feature_names,
            availability_names=self._availability_names,
            source_spec_hashes=self._source_spec_hashes * batch_size,
            source_kinematic_hashes=self._source_kinematic_hashes * batch_size,
            schema_version=self._schema_version,
        )


class FixedTargetMorphoRetargeter(nn.Module):
    """Existing human temporal encoder followed by the kinematic joint-token decoder.

    The target binding is fixed (G1 for the first gate) but identity-free: changing
    names or source hashes changes the returned audit manifest, never learned inputs.
    """

    def __init__(
        self,
        target_robot_tokens: RobotTokenBatch,
        *,
        human_encoder_config: SNMRConfig | None = None,
        morpho_config: MorphoRetargetConfig | None = None,
    ) -> None:
        super().__init__()
        encoder_config = copy.deepcopy(human_encoder_config or SNMRConfig())
        if encoder_config.node_feat_dim != 12:
            raise ValueError("current human_pose_features requires node_feat_dim=12")
        if encoder_config.static_feat_dim != 8:
            raise ValueError("current human_static_features requires static_feat_dim=8")
        decoder_config = morpho_config or MorphoRetargetConfig(
            human_token_dim=encoder_config.latent_dim
        )
        if decoder_config.human_token_dim != encoder_config.latent_dim:
            raise ValueError(
                "MorphoRetarget human_token_dim must equal MotionEncoder latent_dim"
            )

        self.human_encoder_config = encoder_config
        self.morpho_config = decoder_config
        self.target_binding = _BoundRobotTokens(target_robot_tokens)
        self.human_encoder = MotionEncoder(encoder_config)
        self.joint_decoder = KinematicMorphoRetargeter(decoder_config)
        self.root_head = _KinematicRootHead(
            encoder_config.latent_dim, decoder_config.hidden_dim
        )
        self._checkpoint_contract = {
            "integration_schema": INTEGRATION_SCHEMA_VERSION,
            "human_encoder_config": asdict(encoder_config),
            "morpho_config": asdict(decoder_config),
            "root_output_convention": (
                "source_heading_local_xy_absolute_z_heading_relative_wxyz"
            ),
            "contact_output": "derived_from_fk_not_learned",
        }

    def get_extra_state(self) -> dict[str, Any]:
        """Bind behavior-affecting, parameter-free configuration in checkpoints."""

        return copy.deepcopy(self._checkpoint_contract)

    def set_extra_state(self, state: Any) -> None:
        if state != self._checkpoint_contract:
            raise ValueError(
                "checkpoint integration config/output convention does not match model"
            )

    @property
    def target_robot_audit_manifest(self) -> dict[str, Any]:
        return self.target_binding.audit_manifest()

    def forward(
        self,
        human_positions_w: torch.Tensor,
        human_orientations_wxyz: torch.Tensor,
        *,
        human_skeleton: SkeletonGraph,
        human_time_mask: torch.Tensor,
        human_node_mask: torch.Tensor,
        human_static: torch.Tensor | None = None,
        require_contacts: bool = False,
    ) -> MorphoIntegrationOutput:
        """Run a masked current-tensor human→joint-node forward pass.

        Inputs are batched ``(B,T,J,3/4)`` tensors.  Unbatched ``(T,J,3/4)``
        tensors are accepted only when their explicit masks are also unbatched.  Time
        masks must be valid prefixes because the reused temporal encoder has no elapsed-
        time representation for gaps.  Inactive/padded tensor values may be nonfinite;
        active values must be finite and quaternions must be unit ``wxyz``.
        """

        if require_contacts:
            raise NotImplementedError(
                "contacts are not a learned/teacher-forced output; derive them from "
                "predicted robot FK under the registered sole/contact protocol"
            )
        positions, orientations, time_mask, node_mask, static = self._validate_human_inputs(
            human_positions_w,
            human_orientations_wxyz,
            human_skeleton,
            human_time_mask,
            human_node_mask,
            human_static,
        )
        batch, padded_time, _, _ = positions.shape
        skeleton = human_skeleton.to(positions.device)
        full_adjacency = _adjacency(skeleton)
        root_indices = torch.nonzero(
            skeleton.parent_index == -1, as_tuple=False
        ).flatten()
        if root_indices.numel() != 1:
            raise ValueError("human_skeleton must contain exactly one root")
        root_index = int(root_indices.item())

        padded_tokens: list[torch.Tensor] = []
        for batch_index in range(batch):
            length = int(time_mask[batch_index].sum().item())
            active_nodes = torch.nonzero(
                node_mask[batch_index], as_tuple=False
            ).flatten()
            active_root = torch.nonzero(
                active_nodes == root_index, as_tuple=False
            ).flatten()
            if active_root.numel() != 1:
                raise ValueError("human_node_mask must retain the skeleton root")
            local_root_index = int(active_root.item())
            pos = positions[batch_index, :length, active_nodes]
            quat = orientations[batch_index, :length, active_nodes]
            features = human_pose_features(pos, quat, root_index=local_root_index)
            encoded = self.human_encoder(
                features,
                static[batch_index, active_nodes],
                full_adjacency[active_nodes][:, active_nodes],
            )
            # ``pad`` preserves autograd, unlike assigning encoded rows into a fresh tensor.
            padded_tokens.append(
                torch.nn.functional.pad(encoded, (0, 0, 0, padded_time - length))
            )
        human_tokens = torch.stack(padded_tokens, dim=0)
        human_tokens = human_tokens * time_mask.unsqueeze(-1)
        robot_tokens = self.target_binding.for_batch(batch)
        joint_output = self.joint_decoder(human_tokens, robot_tokens, time_mask)
        root_position, root_orientation = self.root_head(
            human_tokens,
            joint_output.robot_embeddings,
            robot_tokens.node_mask,
            time_mask,
        )
        return MorphoIntegrationOutput(
            joint_output=joint_output,
            root_position_local=root_position,
            root_orientation_local_wxyz=root_orientation,
            human_tokens=human_tokens,
            human_time_mask=time_mask,
            robot_token_audit_manifest=self.target_binding.audit_manifest(),
        )

    def forward_motion_spec(
        self,
        motion: HumanMotionSpec,
        *,
        human_skeleton: SkeletonGraph,
        require_contacts: bool = False,
    ) -> MorphoIntegrationOutput:
        """Run one canonical HumanMotionSpec and retain its exact hashes in output."""

        if not isinstance(motion, HumanMotionSpec):
            raise TypeError("motion must be a HumanMotionSpec")
        motion.validate()
        if tuple(human_skeleton.names) != tuple(motion.body_names):
            raise ValueError("HumanMotionSpec body_names must exactly match human_skeleton.names")
        roots = torch.nonzero(
            human_skeleton.parent_index == -1, as_tuple=False
        ).flatten()
        if roots.numel() != 1:
            raise ValueError("human_skeleton must contain exactly one root")
        root_index = int(roots.item())
        if not np.allclose(
            motion.root_position,
            motion.body_positions[:, root_index],
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError("HumanMotionSpec root_position disagrees with its root body")
        root_dot = np.abs(
            np.sum(
                motion.root_orientation_wxyz
                * motion.body_orientations_wxyz[:, root_index],
                axis=-1,
            )
        )
        if not np.allclose(root_dot, 1.0, rtol=0.0, atol=1.0e-7):
            raise ValueError("HumanMotionSpec root orientation disagrees with its root body")

        target = self.target_binding.node_features
        device = target.device
        dtype = target.dtype
        positions = torch.from_numpy(np.array(motion.body_positions, copy=True)).to(
            device=device, dtype=dtype
        )
        orientations = torch.from_numpy(
            np.array(motion.body_orientations_wxyz, copy=True)
        ).to(device=device, dtype=dtype)
        validity = torch.from_numpy(np.array(motion.validity_mask, copy=True)).to(
            device=device
        )
        output = self(
            positions,
            orientations,
            human_skeleton=human_skeleton,
            human_time_mask=validity.all(dim=1),
            human_node_mask=validity.all(dim=0),
            require_contacts=require_contacts,
        )
        return replace(
            output,
            human_motion_spec_sha256=motion.spec_sha256,
            human_motion_buffer_sha256=motion.buffer_sha256,
        )

    def _validate_human_inputs(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        skeleton: SkeletonGraph,
        time_mask: torch.Tensor,
        node_mask: torch.Tensor,
        static: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(positions, torch.Tensor) or not isinstance(
            orientations, torch.Tensor
        ):
            raise TypeError("human positions and orientations must be torch tensors")
        unbatched = positions.ndim == 3
        if unbatched:
            positions = positions.unsqueeze(0)
            orientations = orientations.unsqueeze(0)
            if time_mask.ndim != 1 or node_mask.ndim != 1:
                raise ValueError("unbatched human tensors require one-dimensional masks")
            time_mask = time_mask.unsqueeze(0)
            node_mask = node_mask.unsqueeze(0)
            if static is not None and static.ndim == 2:
                static = static.unsqueeze(0)
        if positions.ndim != 4 or positions.shape[-1] != 3:
            raise ValueError("human_positions_w must have shape (B,T,J,3) or (T,J,3)")
        if orientations.shape != positions.shape[:-1] + (4,):
            raise ValueError("human_orientations_wxyz must match positions with width four")
        if not positions.is_floating_point() or not orientations.is_floating_point():
            raise TypeError("human positions and orientations must be floating point")
        if positions.dtype != orientations.dtype or positions.device != orientations.device:
            raise ValueError("human positions and orientations must share dtype and device")

        batch, time, nodes, _ = positions.shape
        if time == 0 or nodes == 0:
            raise ValueError("human tensors must contain at least one time and node")
        if not isinstance(skeleton, SkeletonGraph) or skeleton.num_nodes != nodes:
            raise ValueError("human_skeleton node count must match human tensors")
        if time_mask.shape != (batch, time) or time_mask.dtype != torch.bool:
            raise ValueError("human_time_mask must be boolean with shape (B,T)")
        if node_mask.shape != (batch, nodes) or node_mask.dtype != torch.bool:
            raise ValueError("human_node_mask must be boolean with shape (B,J)")
        if time_mask.device != positions.device or node_mask.device != positions.device:
            raise ValueError("human masks must share the tensor device")
        if torch.any(~time_mask.any(dim=1)) or torch.any(~node_mask.any(dim=1)):
            raise ValueError("each human batch item needs a valid time and node")

        expected_prefix = torch.arange(time, device=positions.device)[None, :] < (
            time_mask.sum(dim=1, keepdim=True)
        )
        if not torch.equal(time_mask, expected_prefix):
            raise ValueError("human_time_mask must be a contiguous valid prefix")

        parent = skeleton.parent_index.to(node_mask.device)
        roots = torch.nonzero(parent == -1, as_tuple=False).flatten()
        if roots.numel() != 1:
            raise ValueError("human_skeleton must contain exactly one root")
        root_index = int(roots.item())
        for batch_index in range(batch):
            if not bool(node_mask[batch_index, root_index]):
                raise ValueError("human_node_mask must retain the skeleton root")
            for child in torch.nonzero(
                node_mask[batch_index], as_tuple=False
            ).flatten().tolist():
                parent_index = int(parent[child])
                if parent_index >= 0 and not bool(node_mask[batch_index, parent_index]):
                    raise ValueError(
                        "human_node_mask must be ancestor-closed for the graph encoder"
                    )

        active = time_mask[:, :, None] & node_mask[:, None, :]
        if not torch.isfinite(positions[active]).all() or not torch.isfinite(
            orientations[active]
        ).all():
            raise ValueError("active human position/orientation values must be finite")
        quaternion_norm = torch.linalg.vector_norm(orientations[active], dim=-1)
        if not torch.allclose(
            quaternion_norm,
            torch.ones_like(quaternion_norm),
            rtol=0.0,
            atol=1.0e-4,
        ):
            raise ValueError("active human orientations must be unit wxyz quaternions")

        target = self.target_binding.node_features
        if positions.device != target.device or positions.dtype != target.dtype:
            raise ValueError(
                "human tensors must share dtype/device with the bound RobotToken buffers"
            )
        if static is None:
            moved_skeleton = skeleton.to(positions.device)
            static_rows = []
            for batch_index in range(batch):
                length = int(time_mask[batch_index].sum().item())
                static_rows.append(
                    human_static_features(
                        moved_skeleton,
                        body_pos_sample=positions[batch_index, :length],
                        static_dim=self.human_encoder_config.static_feat_dim,
                    )
                )
            parsed_static = torch.stack(static_rows, dim=0)
        else:
            if static.ndim == 2 and static.shape == (
                nodes,
                self.human_encoder_config.static_feat_dim,
            ):
                static = static.unsqueeze(0).expand(batch, -1, -1)
            if static.shape != (
                batch,
                nodes,
                self.human_encoder_config.static_feat_dim,
            ):
                raise ValueError("human_static has unexpected shape")
            if static.dtype != positions.dtype or static.device != positions.device:
                raise ValueError("human_static must share human tensor dtype/device")
            parsed_static = static
        if not torch.isfinite(parsed_static[node_mask]).all():
            raise ValueError("active human static features must be finite")
        return positions, orientations, time_mask, node_mask, parsed_static


__all__ = [
    "FixedTargetMorphoRetargeter",
    "INTEGRATION_SCHEMA_VERSION",
    "MorphoIntegrationOutput",
]
