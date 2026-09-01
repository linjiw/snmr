"""Learned kinematic MorphoRetarget skeleton with serialization-equivariant decoding.

This module deliberately starts after the human temporal encoder.  Callers provide
already-encoded human tokens ``(batch, time, human_dim)`` and a kinematic
:class:`~snmr.robot_tokens.RobotTokenBatch`.  The model then:

1. embeds identity-free kinematic and topology fields with shared projections;
2. contextualizes robot nodes using tree-distance-biased self-attention;
3. forms time-by-node queries and cross-attends them to the human token sequence; and
4. applies one shared scalar head followed by joint-limit parameterization.

No node positional embeddings, serialization indices, names, robot IDs, prompts, or
dynamics fields enter the learned computation.  Outputs remain node-aligned and padded;
``valid_joint_mask`` identifies the variable number of physical scalar DoFs.

The implementation is a bounded P3 skeleton, not the registered experiment model: it
does not predict root motion or contacts, and its dense time-by-node cross-attention is
quadratic in sequence length.  It preserves permutation equivariance algebraically;
floating-point reduction order can still introduce normal device-level roundoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .robot_tokens import (
    KINEMATIC_FEATURE_NAMES,
    ROBOT_TOKEN_SCHEMA_VERSION,
    TOPOLOGY_FEATURE_NAMES,
    RobotTokenBatch,
    _tree_distances,
    bounded_joint_positions,
)


@dataclass(frozen=True)
class MorphoRetargetConfig:
    """Dimensions for the kinematic-only learned skeleton."""

    human_token_dim: int
    hidden_dim: int = 128
    num_heads: int = 4
    graph_layers: int = 3
    feedforward_multiplier: int = 4
    tree_bias_gamma: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "human_token_dim",
            "hidden_dim",
            "num_heads",
            "graph_layers",
            "feedforward_multiplier",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not math.isfinite(self.tree_bias_gamma) or self.tree_bias_gamma < 0.0:
            raise ValueError("tree_bias_gamma must be finite and nonnegative")


@dataclass(frozen=True)
class MorphoRetargetOutput:
    """Node-aligned joint predictions and the mask defining real scalar DoFs."""

    joint_positions: torch.Tensor  # (B, T, N), zero outside valid_joint_mask
    joint_logits: torch.Tensor  # (B, T, N), zero for padding/invalid output times
    valid_joint_mask: torch.Tensor  # (B, T, N)
    robot_embeddings: torch.Tensor  # (B, N, H), zero for padding


class _TreeBiasedSelfAttention(nn.Module):
    """Dense multi-head node attention with shortest-path bias and safe padding."""

    def __init__(self, hidden_dim: int, num_heads: int, gamma: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.gamma = float(gamma)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        values: torch.Tensor,
        node_mask: torch.Tensor,
        tree_distance: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, hidden = values.shape
        qkv = self.qkv(values).reshape(
            batch, nodes, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        scores = torch.einsum("bhid,bhjd->bhij", query, key)
        scores = scores / math.sqrt(self.head_dim)
        distances = tree_distance.clamp_min(0).to(dtype=scores.dtype)
        scores = scores - self.gamma * distances.unsqueeze(1)

        valid_pairs = (
            node_mask[:, None, :, None]
            & node_mask[:, None, None, :]
            & (tree_distance[:, None, :, :] >= 0)
        )
        scores = scores.masked_fill(~valid_pairs, float("-inf"))
        # A padding query has no valid keys.  Zero its row before softmax, then erase
        # its weights afterwards, avoiding NaNs without introducing a padding signal.
        valid_query = node_mask[:, None, :, None]
        scores = torch.where(valid_query, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1).masked_fill(~valid_pairs, 0.0)
        attended = torch.einsum("bhij,bhjd->bhid", weights, value)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, nodes, hidden)
        return self.output(attended) * node_mask.unsqueeze(-1)


class _GraphEncoderLayer(nn.Module):
    def __init__(self, config: MorphoRetargetConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = _TreeBiasedSelfAttention(
            hidden,
            config.num_heads,
            config.tree_bias_gamma,
        )
        self.feedforward_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, config.feedforward_multiplier * hidden),
            nn.GELU(),
            nn.Linear(config.feedforward_multiplier * hidden, hidden),
        )

    def forward(
        self,
        values: torch.Tensor,
        node_mask: torch.Tensor,
        tree_distance: torch.Tensor,
    ) -> torch.Tensor:
        mask = node_mask.unsqueeze(-1)
        values = values + self.attention(
            self.attention_norm(values), node_mask, tree_distance
        )
        values = values * mask
        values = values + self.feedforward(self.feedforward_norm(values))
        return values * mask


class _HumanRobotCrossAttention(nn.Module):
    """Cross-attend every time/node query to the human temporal sequence."""

    def __init__(self, config: MorphoRetargetConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = hidden // config.num_heads
        self.node_query = nn.Linear(hidden, hidden)
        self.time_query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.feedforward_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, config.feedforward_multiplier * hidden),
            nn.GELU(),
            nn.Linear(config.feedforward_multiplier * hidden, hidden),
        )

    def forward(
        self,
        human: torch.Tensor,
        robot: torch.Tensor,
        human_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, hidden = human.shape
        nodes = robot.shape[1]
        query = self.time_query(human)[:, :, None, :] + self.node_query(robot)[:, None, :, :]
        query = query.reshape(batch, time, nodes, self.num_heads, self.head_dim)
        query = query.permute(0, 3, 1, 2, 4)
        key = self.key(human).reshape(batch, time, self.num_heads, self.head_dim)
        value = self.value(human).reshape(batch, time, self.num_heads, self.head_dim)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        scores = torch.einsum("bhtnd,bhsd->bhtns", query, key)
        scores = scores / math.sqrt(self.head_dim)
        key_mask = human_mask[:, None, None, None, :]
        scores = scores.masked_fill(~key_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).masked_fill(~key_mask, 0.0)
        attended = torch.einsum("bhtns,bhsd->bhtnd", weights, value)
        attended = attended.permute(0, 2, 3, 1, 4).reshape(
            batch, time, nodes, hidden
        )

        base = human[:, :, None, :] + robot[:, None, :, :]
        output = self.output_norm(base + self.output(attended))
        output = output + self.feedforward(self.feedforward_norm(output))
        valid = human_mask[:, :, None, None] & node_mask[:, None, :, None]
        return output * valid


class KinematicMorphoRetargeter(nn.Module):
    """Kinematic RobotSpec-conditioned, shared-head joint trajectory decoder."""

    def __init__(self, config: MorphoRetargetConfig) -> None:
        super().__init__()
        self.config = config
        robot_input_dim = len(KINEMATIC_FEATURE_NAMES) + len(TOPOLOGY_FEATURE_NAMES)
        self.robot_input = nn.Sequential(
            nn.Linear(robot_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.graph_layers = nn.ModuleList(
            _GraphEncoderLayer(config) for _ in range(config.graph_layers)
        )
        self.human_input = nn.Sequential(
            nn.Linear(config.human_token_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.cross_attention = _HumanRobotCrossAttention(config)
        self.joint_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        human_tokens: torch.Tensor,
        robot_tokens: RobotTokenBatch,
        human_mask: torch.Tensor | None = None,
    ) -> MorphoRetargetOutput:
        """Decode padded per-node joint positions for an encoded human sequence."""

        parsed_human_mask = self._validate_inputs(human_tokens, robot_tokens, human_mask)
        node_mask = robot_tokens.node_mask
        robot_input = torch.cat(
            (robot_tokens.node_features, robot_tokens.topology_features), dim=-1
        )
        robot = self.robot_input(robot_input) * node_mask.unsqueeze(-1)
        for layer in self.graph_layers:
            robot = layer(robot, node_mask, robot_tokens.tree_distance)

        human = self.human_input(human_tokens)
        fused = self.cross_attention(
            human,
            robot,
            parsed_human_mask,
            node_mask,
        )
        output_mask = (
            parsed_human_mask[:, :, None]
            & robot_tokens.joint_mask[:, None, :]
            & node_mask[:, None, :]
        )
        logits = self.joint_head(fused).squeeze(-1)
        logits = logits * (
            parsed_human_mask[:, :, None] & node_mask[:, None, :]
        )
        lower = robot_tokens.joint_lower[:, None, :].expand_as(logits)
        upper = robot_tokens.joint_upper[:, None, :].expand_as(logits)
        positions = bounded_joint_positions(logits, lower, upper, output_mask)
        return MorphoRetargetOutput(
            joint_positions=positions,
            joint_logits=logits,
            valid_joint_mask=output_mask,
            robot_embeddings=robot,
        )

    def _validate_inputs(
        self,
        human_tokens: torch.Tensor,
        robot_tokens: RobotTokenBatch,
        human_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not isinstance(robot_tokens, RobotTokenBatch):
            raise TypeError("robot_tokens must be a RobotTokenBatch")
        if robot_tokens.schema_version != ROBOT_TOKEN_SCHEMA_VERSION:
            raise ValueError("unsupported RobotTokenBatch schema")
        if tuple(robot_tokens.feature_names) != KINEMATIC_FEATURE_NAMES:
            raise ValueError(
                "KinematicMorphoRetargeter requires RobotGraphTokenizer(feature_set='kinematic')"
            )
        if tuple(robot_tokens.topology_feature_names) != TOPOLOGY_FEATURE_NAMES:
            raise ValueError("unexpected topology feature contract")
        if human_tokens.ndim != 3:
            raise ValueError("human_tokens must have shape (batch, time, human_token_dim)")
        batch, time, feature_dim = human_tokens.shape
        if time == 0 or feature_dim != self.config.human_token_dim:
            raise ValueError(
                f"human_tokens must have nonzero time and final dimension "
                f"{self.config.human_token_dim}"
            )
        if not human_tokens.is_floating_point() or not torch.isfinite(human_tokens).all():
            raise ValueError("human_tokens must be finite floating-point values")

        node_features = robot_tokens.node_features
        topology = robot_tokens.topology_features
        node_mask = robot_tokens.node_mask
        if node_features.ndim != 3 or node_features.shape[0] != batch:
            raise ValueError("robot node features must match the human batch")
        nodes = node_features.shape[1]
        if node_features.shape[2] != len(KINEMATIC_FEATURE_NAMES):
            raise ValueError("unexpected kinematic node feature width")
        if topology.shape != (batch, nodes, len(TOPOLOGY_FEATURE_NAMES)):
            raise ValueError("unexpected topology feature shape")
        if node_mask.shape != (batch, nodes) or node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be boolean with shape (batch, nodes)")
        if robot_tokens.tree_distance.shape != (batch, nodes, nodes):
            raise ValueError("tree_distance must have shape (batch, nodes, nodes)")
        if robot_tokens.tree_distance.dtype != torch.long:
            raise TypeError("tree_distance must use torch.long integer distances")
        if robot_tokens.parent_index.shape != (batch, nodes):
            raise ValueError("parent_index must have shape (batch, nodes)")
        if robot_tokens.parent_index.dtype != torch.long:
            raise TypeError("parent_index must use torch.long indices")
        if robot_tokens.joint_mask.shape != (batch, nodes):
            raise ValueError("joint_mask must have shape (batch, nodes)")
        if robot_tokens.joint_mask.dtype != torch.bool:
            raise ValueError("joint_mask must be boolean")
        if robot_tokens.joint_lower.shape != (batch, nodes) or robot_tokens.joint_upper.shape != (
            batch,
            nodes,
        ):
            raise ValueError("joint limits must have shape (batch, nodes)")
        if torch.any(robot_tokens.joint_mask & ~node_mask):
            raise ValueError("joint_mask cannot select padding")
        if robot_tokens.root_mask.shape != (batch, nodes):
            raise ValueError("root_mask must have shape (batch, nodes)")
        if robot_tokens.root_mask.dtype != torch.bool:
            raise TypeError("root_mask must be boolean")
        if torch.any(robot_tokens.root_mask & ~node_mask):
            raise ValueError("root_mask cannot select padding")
        if torch.any(node_mask.sum(dim=1) == 0):
            raise ValueError("each robot must contain at least one real node")
        if torch.any(robot_tokens.joint_mask.sum(dim=1) == 0):
            raise ValueError("each robot must contain at least one scalar joint")
        if torch.any(robot_tokens.root_mask.sum(dim=1) != 1):
            raise ValueError("each robot must contain exactly one root node")
        if not torch.isfinite(robot_tokens.joint_lower).all() or not torch.isfinite(
            robot_tokens.joint_upper
        ).all():
            raise ValueError("joint limits must contain only finite values")
        if torch.any(
            robot_tokens.joint_lower[robot_tokens.joint_mask]
            >= robot_tokens.joint_upper[robot_tokens.joint_mask]
        ):
            raise ValueError("every active joint must have lower < upper")

        # Recompute the tree metric from the structural parent contract.  This is a
        # deliberately strict research-time guard: RobotTokenBatch tensors are mutable,
        # and accepting a tampered distance matrix could inject serialization/index cues.
        for batch_index in range(batch):
            count = int(node_mask[batch_index].sum().item())
            if not torch.all(node_mask[batch_index, :count]) or torch.any(
                node_mask[batch_index, count:]
            ):
                raise ValueError("real robot nodes must precede padding in RobotTokenBatch")
            parents = robot_tokens.parent_index[batch_index].detach().cpu().numpy()
            if not np.all(parents[count:] == -2):
                raise ValueError("parent_index padding entries must equal -2")
            try:
                expected_distance = _tree_distances(parents[:count])
            except ValueError as exc:
                raise ValueError(f"invalid parent tree for robot {batch_index}: {exc}") from exc
            actual_distance = (
                robot_tokens.tree_distance[batch_index].detach().cpu().numpy()
            )
            if not np.array_equal(
                actual_distance[:count, :count], expected_distance
            ) or not (
                np.all(actual_distance[count:, :] == -1)
                and np.all(actual_distance[:, count:] == -1)
            ):
                raise ValueError("tree_distance must exactly match parent_index with -1 padding")
            expected_root = parents[:count] == -1
            actual_root = (
                robot_tokens.root_mask[batch_index, :count].detach().cpu().numpy()
            )
            if not np.array_equal(actual_root, expected_root):
                raise ValueError("root_mask must exactly identify the parent_index root")

        root_feature = KINEMATIC_FEATURE_NAMES.index("is_root")
        dof_feature = KINEMATIC_FEATURE_NAMES.index("has_dof")
        if not torch.equal(
            node_features[..., root_feature] > 0.5,
            robot_tokens.root_mask,
        ):
            raise ValueError("is_root model feature must match root_mask")
        if not torch.equal(
            node_features[..., dof_feature] > 0.5,
            robot_tokens.joint_mask,
        ):
            raise ValueError("has_dof model feature must match joint_mask")

        tensor_fields = (
            node_features,
            topology,
            node_mask,
            robot_tokens.parent_index,
            robot_tokens.tree_distance,
            robot_tokens.joint_mask,
            robot_tokens.joint_lower,
            robot_tokens.joint_upper,
            robot_tokens.root_mask,
        )
        if any(tensor.device != human_tokens.device for tensor in tensor_fields):
            raise ValueError("human and robot tensors must be on the same device")
        if node_features.dtype != human_tokens.dtype or topology.dtype != human_tokens.dtype:
            raise ValueError("human and robot floating-point tensors must share a dtype")
        if not torch.isfinite(node_features).all() or not torch.isfinite(topology).all():
            raise ValueError("robot model features must be finite")

        if human_mask is None:
            parsed = torch.ones((batch, time), dtype=torch.bool, device=human_tokens.device)
        else:
            if human_mask.shape != (batch, time) or human_mask.dtype != torch.bool:
                raise ValueError("human_mask must be boolean with shape (batch, time)")
            if human_mask.device != human_tokens.device:
                raise ValueError("human_mask must be on the human token device")
            parsed = human_mask
        if torch.any(~parsed.any(dim=1)):
            raise ValueError("each batch item must contain at least one valid human token")
        return parsed


__all__ = [
    "KinematicMorphoRetargeter",
    "MorphoRetargetConfig",
    "MorphoRetargetOutput",
]
