"""SNMR neural retargeter: skeleton-agnostic graph encoder → shared latent → embodiment decoder.

Architecture (following SAME, adapted for robot targets — see NEURAL_RETARGETING_DESIGN.md §3.1):

  Encoder   Enc(S_src, D_src) -> z_{1:T}
      * graph attention over the source skeleton (node = joint/body, edges = bones), with
        node-shared weights so any topology/joint-count is accepted;
      * global max-pool over nodes -> a single per-frame latent z_t (skeleton-agnostic);
      * a temporal transformer over z_{1:T} for velocity-consistent, non-jittery latents.

  Decoder   Dec(S_tgt, embodiment_code, z) -> robot qpos
      * broadcasts z to every target-skeleton node, concatenates the node's static features at
        *every* layer (SAME's re-injection), and conditions each layer on the target embodiment
        code via AdaLN (AdaMorph recipe);
      * output heads: per-node scalar -> hinge angle squashed by tanh into the joint's limits
        (limit satisfaction by construction); the root node additionally emits root position + a
        6D rotation.

Everything is dependency-light (torch only). Dense masked attention is used because the skeletons are
small (≤ ~51 nodes); it also makes variable node counts across embodiments trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rotation as rot
from .data import robot_node_static_features, robot_pose_features
from .robot_model import RobotKinematics
from .skeleton import SkeletonGraph


# --------------------------------------------------------------------------------------
# graph attention (dense, masked)
# --------------------------------------------------------------------------------------
class GraphAttention(nn.Module):
    """Multi-head GAT layer with node-shared weights, over a dense adjacency mask.

    Operates on ``h`` of shape ``(*, N, in_dim)`` and a boolean ``adj`` of shape ``(N, N)`` where
    ``adj[i, j]`` is True if node ``j`` is a neighbour of ``i`` (self-loops included). Returns
    ``(*, N, out_dim)``.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, negative_slope: float = 0.2):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        lead = h.shape[:-2]
        N = h.shape[-2]
        wh = self.lin(h).reshape(lead + (N, self.heads, self.head_dim))  # (*, N, H, Dh)

        # attention logits e[i, j] = LeakyReLU(a_dst . wh_i + a_src . wh_j)
        alpha_dst = (wh * self.att_dst).sum(-1)  # (*, N, H)  contribution of the query node i
        alpha_src = (wh * self.att_src).sum(-1)  # (*, N, H)  contribution of the key node j
        # e[..., i, j, h] = alpha_dst[..., i, h] + alpha_src[..., j, h]
        e = alpha_dst.unsqueeze(-2) + alpha_src.unsqueeze(-3)  # (*, N, N, H)
        e = F.leaky_relu(e, self.negative_slope)

        mask = adj.to(torch.bool)  # (N, N)
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        att = torch.softmax(e, dim=-2)  # softmax over neighbours j (dim = -2 is the j axis)
        # A fully-masked row cannot occur because self-loops are always present.

        # out_i = sum_j att[i, j] * wh_j
        out = torch.einsum("...ijh,...jhd->...ihd", att, wh)  # (*, N, H, Dh)
        out = out.reshape(lead + (N, self.heads * self.head_dim)) + self.bias
        return out


class GraphAttentionStack(nn.Module):
    """A stack of GraphAttention layers with residual connections + LayerNorm + ELU."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, heads: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projs = nn.ModuleList()
        d = in_dim
        for _ in range(num_layers):
            self.layers.append(GraphAttention(d, hidden_dim, heads=heads))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.projs.append(nn.Linear(d, hidden_dim) if d != hidden_dim else nn.Identity())
            d = hidden_dim
        self.out_dim = hidden_dim

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer, norm, proj in zip(self.layers, self.norms, self.projs):
            residual = proj(h)
            h = norm(residual + F.elu(layer(h, adj)))
        return h


# --------------------------------------------------------------------------------------
# AdaLN conditioning for the decoder
# --------------------------------------------------------------------------------------
class AdaLN(nn.Module):
    """LayerNorm whose scale/shift are produced from a conditioning vector (embodiment code)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # h: (*, N, dim); cond: (*, cond_dim) broadcast over N.
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-2)
        shift = shift.unsqueeze(-2)
        return self.norm(h) * (1.0 + scale) + shift


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
@dataclass
class SNMRConfig:
    node_feat_dim: int = 12          # per-node dynamic pose feature dim (3 pos + 6 rot6d + 3 vel)
    static_feat_dim: int = 8         # per-node static skeleton feature dim
    latent_dim: int = 64
    enc_hidden: int = 128
    enc_layers: int = 4
    dec_hidden: int = 128
    dec_layers: int = 4
    heads: int = 4
    embodiment_dim: int = 32
    temporal_layers: int = 2
    temporal_heads: int = 4
    use_temporal: bool = True


# --------------------------------------------------------------------------------------
# encoder / decoder
# --------------------------------------------------------------------------------------
class MotionEncoder(nn.Module):
    """Enc(skeleton, node_features) -> per-frame latent z_{1:T}."""

    def __init__(self, cfg: SNMRConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.node_feat_dim + cfg.static_feat_dim, cfg.enc_hidden)
        self.gat = GraphAttentionStack(cfg.enc_hidden, cfg.enc_hidden, cfg.enc_layers, cfg.heads)
        self.to_latent = nn.Linear(cfg.enc_hidden, cfg.latent_dim)
        if cfg.use_temporal:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.latent_dim,
                nhead=cfg.temporal_heads,
                dim_feedforward=4 * cfg.latent_dim,
                batch_first=True,
                activation="gelu",
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=cfg.temporal_layers)
        else:
            self.temporal = None

    def forward(
        self,
        node_features: torch.Tensor,   # (T, N, node_feat_dim)
        static_features: torch.Tensor,  # (N, static_feat_dim)
        adj: torch.Tensor,             # (N, N)
    ) -> torch.Tensor:
        T, N, _ = node_features.shape
        stat = static_features.unsqueeze(0).expand(T, N, -1)
        h = torch.cat([node_features, stat], dim=-1)
        h = F.elu(self.input_proj(h))
        h = self.gat(h, adj)                     # (T, N, enc_hidden)
        h = h.max(dim=-2).values                 # global max-pool over nodes -> (T, enc_hidden)
        z = self.to_latent(h)                    # (T, latent_dim)
        if self.temporal is not None:
            z = self.temporal(z.unsqueeze(0)).squeeze(0)  # temporal mixing over frames
        return z


class EmbodimentEncoder(nn.Module):
    """Pools a target robot's static graph features into an embodiment code (zero-shot from MJCF)."""

    def __init__(self, cfg: SNMRConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.static_feat_dim, cfg.embodiment_dim),
            nn.ELU(),
            nn.Linear(cfg.embodiment_dim, cfg.embodiment_dim),
        )

    def forward(self, static_features: torch.Tensor) -> torch.Tensor:
        # static_features: (N, static_feat_dim) -> (embodiment_dim,)
        h = self.net(static_features)
        return h.max(dim=-2).values


class MotionDecoder(nn.Module):
    """Dec(skeleton, embodiment_code, z) -> robot qpos.

    Per target node we compute a hidden vector conditioned on ``z`` + node static features + the
    embodiment code (AdaLN). Node heads emit hinge angles (tanh-scaled to limits); the root node head
    emits root position + 6D root rotation.
    """

    def __init__(self, cfg: SNMRConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.latent_dim + cfg.static_feat_dim
        self.input_proj = nn.Linear(in_dim, cfg.dec_hidden)
        self.layers = nn.ModuleList()
        self.adalns = nn.ModuleList()
        self.reinject = nn.ModuleList()
        for _ in range(cfg.dec_layers):
            self.layers.append(GraphAttention(cfg.dec_hidden, cfg.dec_hidden, heads=cfg.heads))
            self.adalns.append(AdaLN(cfg.dec_hidden, cfg.embodiment_dim))
            # re-inject latent + static features at every layer (SAME)
            self.reinject.append(nn.Linear(cfg.dec_hidden + in_dim, cfg.dec_hidden))
        self.angle_head = nn.Sequential(nn.Linear(cfg.dec_hidden, cfg.dec_hidden // 2), nn.ELU(),
                                        nn.Linear(cfg.dec_hidden // 2, 1))
        self.root_head = nn.Sequential(nn.Linear(cfg.dec_hidden, cfg.dec_hidden // 2), nn.ELU(),
                                       nn.Linear(cfg.dec_hidden // 2, 9))  # 3 pos + 6 rot6d

    def forward(
        self,
        z: torch.Tensor,               # (T, latent_dim)
        static_features: torch.Tensor,  # (N, static_feat_dim)
        adj: torch.Tensor,             # (N, N)
        embodiment_code: torch.Tensor,  # (embodiment_dim,)
        robot_graph,
    ) -> dict[str, torch.Tensor]:
        T = z.shape[0]
        N = static_features.shape[0]
        stat = static_features.unsqueeze(0).expand(T, N, -1)
        z_nodes = z.unsqueeze(1).expand(T, N, self.cfg.latent_dim)
        base = torch.cat([z_nodes, stat], dim=-1)  # (T, N, in_dim) re-injected each layer

        cond = embodiment_code.unsqueeze(0).expand(T, -1)  # (T, embodiment_dim)
        h = F.elu(self.input_proj(base))
        for layer, adaln, reinj in zip(self.layers, self.adalns, self.reinject):
            h_att = F.elu(layer(h, adj))
            h = adaln(h + h_att, cond)
            h = F.elu(reinj(torch.cat([h, base], dim=-1)))

        # angles for every node, then gather dof-bearing bodies in dof order
        node_angle = torch.tanh(self.angle_head(h).squeeze(-1))  # (T, N) in [-1, 1]
        dof_body = robot_graph.dof_body_index                    # (D,)
        gathered = node_angle[:, dof_body]                       # (T, D)
        lo = robot_graph.dof_lower.to(z.dtype)
        hi = robot_graph.dof_upper.to(z.dtype)
        dof_pos = lo + (hi - lo) * 0.5 * (gathered + 1.0)        # scaled to limits

        root_out = self.root_head(h[:, 0, :])                    # (T, 9)
        root_pos = root_out[:, :3]
        root_quat = rot.rot6d_to_quat(root_out[:, 3:9])
        return {"root_pos": root_pos, "root_quat": root_quat, "dof_pos": dof_pos}


class SNMR(nn.Module):
    """Full model tying encoder, embodiment encoder, and decoder together."""

    def __init__(self, cfg: SNMRConfig | None = None):
        super().__init__()
        self.cfg = cfg or SNMRConfig()
        self.encoder = MotionEncoder(self.cfg)
        self.embodiment_encoder = EmbodimentEncoder(self.cfg)
        self.decoder = MotionDecoder(self.cfg)

    def encode(self, node_features, static_features, adj) -> torch.Tensor:
        return self.encoder(node_features, static_features, adj)

    def decode(self, z, target_kin: RobotKinematics) -> dict[str, torch.Tensor]:
        static = robot_node_static_features(target_kin.graph)
        adj = _adjacency(SkeletonGraph.from_robot_graph(target_kin.graph))
        code = self.embodiment_encoder(static)
        return self.decoder(z, static, adj, code, target_kin.graph)

    def retarget_human_to_robot(
        self,
        human_pos: torch.Tensor,
        human_quat: torch.Tensor,
        human_skel: SkeletonGraph,
        human_static: torch.Tensor,
        target_kin: RobotKinematics,
    ) -> dict[str, torch.Tensor]:
        """The Phase-1 production path: human world kinematics -> shared latent -> robot qpos.

        Args:
            human_pos:  (T, J, 3) world body positions (e.g. LAFAN1 via ``snmr.human``).
            human_quat: (T, J, 4) wxyz world orientations.
            human_skel: the human :class:`SkeletonGraph`.
            human_static: (J, static_feat_dim) from ``snmr.human.human_static_features``.
        """
        from .human import human_pose_features  # local import to avoid a module cycle

        feats = human_pose_features(human_pos, human_quat)
        adj = _adjacency(human_skel)
        z = self.encode(feats, human_static, adj)
        return self.decode(z, target_kin)

    def retarget_robot_to_robot(
        self, source_kin: RobotKinematics, motion, target_kin: RobotKinematics
    ) -> dict[str, torch.Tensor]:
        """Encode a robot motion and decode it onto ``target_kin`` (autoencoding if same robot).

        This is the fully-testable path in this environment (no SMPL-X data needed) and exercises the
        whole encode→latent→decode pipeline end-to-end.
        """
        node_features = robot_pose_features(source_kin, motion)  # (T, N, node_feat_dim)
        static = robot_node_static_features(source_kin.graph)
        adj = _adjacency(SkeletonGraph.from_robot_graph(source_kin.graph))
        z = self.encode(node_features, static, adj)
        return self.decode(z, target_kin)


def _adjacency(skel: SkeletonGraph) -> torch.Tensor:
    """Dense (N, N) boolean adjacency with self-loops from a skeleton's edge list."""
    N = skel.num_nodes
    adj = torch.eye(N, dtype=torch.bool, device=skel.device)
    ei = skel.edge_index()
    adj[ei[0], ei[1]] = True
    return adj
