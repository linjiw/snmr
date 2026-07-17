"""Latent rectified-flow retargeting head (side project — see docs/FLOW_RETARGETING_SIDE_PROJECT.md).

A small conditional flow-matching model over the FROZEN SNMR latent space. It learns the
transport `p(z_target | z_cond)` where both sequences come from the frozen encoder: the
condition is the human encoding `z_h` and the target is (per gate F0) the teacher-robot
encoding `z_r` or `z_h` itself. Sampling integrates the learned ODE with Euler steps and can
steer the trajectory with the gradient of a differentiable physics cost evaluated through the
frozen decoder + FK (SafeFlow-style guidance, applied at the one-step endpoint prediction
because the frozen decoder has only ever seen clean latents).

Everything here is torch-only and SNMR-frozen: no module in this file owns or updates SNMR
weights. Conventions follow the repo: CondOT path `z_u = u·z1 + (1−u)·z0`, u ∈ [0, 1] with
u=0 noise and u=1 data; velocity target `z1 − z0`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
@dataclass
class LatentFlowConfig:
    latent_dim: int = 128          # must equal the frozen SNMR checkpoint's latent_dim
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    time_embed_dim: int = 128      # Fourier feature dim for the flow time u


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------
def flow_time_embedding(u: torch.Tensor, dim: int) -> torch.Tensor:
    """Fourier features of the flow time u ∈ [0, 1] — (B,) → (B, dim)."""
    if dim % 2 != 0:
        raise ValueError("time embedding dim must be even")
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=u.device, dtype=u.dtype)
        * (-torch.log(torch.tensor(10000.0, device=u.device, dtype=u.dtype)) / max(half - 1, 1))
    )
    angles = 2.0 * torch.pi * u[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _frame_positions(T: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard fixed sinusoidal positional encoding over frames, (T, dim)."""
    pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
    half = torch.arange(0, dim, 2, device=device, dtype=dtype)
    freq = torch.exp(-half * (torch.log(torch.tensor(10000.0, dtype=dtype)) / dim))
    pe = torch.zeros(T, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq[: dim // 2])
    return pe


class FlowAdaLN(nn.Module):
    """AdaLN-zero: LayerNorm modulated by the flow-time embedding, plus a zero-init gate."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_mod = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.to_mod.weight)
        nn.init.zeros_(self.to_mod.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, dim); cond: (B, cond_dim) broadcast over T.
        scale, shift, gate = self.to_mod(cond).unsqueeze(1).chunk(3, dim=-1)
        return self.norm(h) * (1.0 + scale) + shift, gate


class FlowBlock(nn.Module):
    """Pre-norm temporal self-attention + MLP, both AdaLN-zero-conditioned on flow time."""

    def __init__(self, cfg: LatentFlowConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.attn_norm = FlowAdaLN(d, cfg.time_embed_dim)
        self.attn = nn.MultiheadAttention(d, cfg.num_heads, batch_first=True)
        self.mlp_norm = FlowAdaLN(d, cfg.time_embed_dim)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x, gate = self.attn_norm(h, t_emb)
        att, _ = self.attn(x, x, x, need_weights=False)
        h = h + gate * att
        x, gate = self.mlp_norm(h, t_emb)
        h = h + gate * self.mlp(x)
        return h


class LatentFlowNet(nn.Module):
    """v_θ(z_u, u | c): velocity field over latent sequences.

    Inputs are batched sequences: noisy latent ``z_u`` (B, T, latent_dim), per-frame condition
    ``cond`` (B, T, latent_dim) from the frozen encoder, and flow time ``u`` (B,). The condition
    is concatenated at the input (per-frame, so the transport stays frame-aligned) and the flow
    time modulates every block via AdaLN-zero. Output head is zero-initialized, so the untrained
    model predicts v=0 — sampling from an untrained model returns its noise unchanged.
    """

    def __init__(self, cfg: LatentFlowConfig | None = None):
        super().__init__()
        self.cfg = cfg or LatentFlowConfig()
        cfg = self.cfg
        self.input_proj = nn.Linear(2 * cfg.latent_dim, cfg.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.time_embed_dim, cfg.time_embed_dim),
        )
        self.blocks = nn.ModuleList(FlowBlock(cfg) for _ in range(cfg.num_layers))
        self.out_norm = FlowAdaLN(cfg.hidden_dim, cfg.time_embed_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_u: torch.Tensor, u: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if z_u.ndim != 3 or cond.shape != z_u.shape:
            raise ValueError(
                f"expected z_u and cond shaped (B, T, latent); got {tuple(z_u.shape)} "
                f"and {tuple(cond.shape)}"
            )
        B, T, _ = z_u.shape
        t_emb = self.time_mlp(flow_time_embedding(u.reshape(B), self.cfg.time_embed_dim))
        h = self.input_proj(torch.cat([z_u, cond], dim=-1))
        h = h + _frame_positions(T, h.shape[-1], h.device, h.dtype)
        for block in self.blocks:
            h = block(h, t_emb)
        h, _ = self.out_norm(h, t_emb)
        return self.out_proj(h)


# --------------------------------------------------------------------------------------
# training objective
# --------------------------------------------------------------------------------------
def rectified_flow_loss(
    net: LatentFlowNet,
    z1: torch.Tensor,
    cond: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Conditional flow-matching MSE on the CondOT path.

    z1/cond: (B, T, latent_dim) target and condition latents (both detached frozen-encoder
    outputs). Samples z0 ∼ N(0, I) and u ∼ Unif[0, 1] per batch element.
    """
    if z1.ndim != 3 or cond.shape != z1.shape:
        raise ValueError("z1 and cond must both be (B, T, latent)")
    B = z1.shape[0]
    z0 = torch.randn(z1.shape, device=z1.device, dtype=z1.dtype, generator=generator)
    u = torch.rand(B, device=z1.device, dtype=z1.dtype, generator=generator)
    z_u = u[:, None, None] * z1 + (1.0 - u[:, None, None]) * z0
    v = net(z_u, u, cond)
    return torch.mean((v - (z1 - z0)) ** 2)


# --------------------------------------------------------------------------------------
# sampling (optionally physics-guided)
# --------------------------------------------------------------------------------------
@dataclass
class GuidanceConfig:
    """Endpoint-cost guidance during Euler integration (SafeFlow Eq. 5, endpoint variant)."""

    cost_fn: Callable[[torch.Tensor], torch.Tensor]  # ẑ1 (B, T, latent) → scalar cost
    alpha_start: float = 0.0        # guidance scale at u=0
    alpha_end: float = 3.0          # guidance scale at u=1 (linear schedule in between)
    grad_clamp: float = 0.2         # per-element clamp on ∇_{ẑ1} C, as in SafeFlow
    skip_final: float = 0.9         # no guidance for u ≥ this (endpoint prediction ≈ z_u there,
    # and un-guided final steps let the learned field settle back onto the data manifold)


@torch.no_grad()
def sample_flow(
    net: LatentFlowNet,
    cond: torch.Tensor,
    num_steps: int = 25,
    z0: torch.Tensor | None = None,
    guidance: GuidanceConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler-integrate dz/du = v_θ(z, u | cond) from u=0 to u=1; returns ẑ1 (B, T, latent).

    With ``guidance``, each step forms the endpoint prediction ẑ1 = z + (1−u)·v, differentiates
    the cost at ẑ1 (re-enabling grad locally), clamps the gradient per element, and subtracts
    α(u)·∇C from the velocity.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if cond.ndim != 3:
        raise ValueError(f"cond must be (B, T, latent); got {tuple(cond.shape)}")
    z = (
        torch.randn(cond.shape, device=cond.device, dtype=cond.dtype, generator=generator)
        if z0 is None
        else z0.clone()
    )
    h = 1.0 / num_steps
    for i in range(num_steps):
        u_val = i * h
        u = torch.full((cond.shape[0],), u_val, device=cond.device, dtype=cond.dtype)
        v = net(z, u, cond)
        if guidance is not None and u_val < guidance.skip_final:
            z1_hat = (z + (1.0 - u_val) * v).detach().requires_grad_(True)
            with torch.enable_grad():
                cost = guidance.cost_fn(z1_hat)
                (grad,) = torch.autograd.grad(cost, z1_hat)
            grad = grad.clamp(-guidance.grad_clamp, guidance.grad_clamp)
            alpha = guidance.alpha_start + (guidance.alpha_end - guidance.alpha_start) * u_val
            v = v - alpha * grad
        z = z + h * v
    return z


# --------------------------------------------------------------------------------------
# physics cost through the frozen decoder (+ FK)
# --------------------------------------------------------------------------------------
def soft_stance_weights(
    foot_heights: torch.Tensor,
    height_threshold: float = 0.03,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Soft contact weight per (frame, foot) from decoded foot heights — no binary threshold.

    Heights should be ground-relative (per-foot clip minimum removed, matching
    ``snmr.metrics.detect_contact`` conventions). Returns σ((thr − h)/τ) ∈ (0, 1); E24's hard
    decoded-height mask under-fired (prevalence 0.03 vs 0.29), so the deployable guidance signal
    deliberately stays soft.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid((height_threshold - foot_heights) / temperature)


@dataclass
class PhysicsCostConfig:
    skate_weight: float = 1.0
    penetration_weight: float = 1.0
    smooth_weight: float = 0.1
    height_threshold: float = 0.03
    temperature: float = 0.01
    fps: float = 30.0


def make_decoded_physics_cost(
    decode_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    cfg: PhysicsCostConfig,
    stance_weights: torch.Tensor | None = None,
    stance_gate: torch.Tensor | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build C(z): latent (B, T, latent) → scalar, differentiable through the frozen decoder.

    ``decode_fn`` maps a latent batch to ``(foot_pos_w, dof_pos)`` with shapes (B, T, F, 3) and
    (B, T, D) — world-frame foot positions (after ``local_root_to_world`` + FK) and joint
    angles. It must keep the graph (no ``torch.no_grad``) and treat SNMR weights as constants.

    Stance signal per (B, T, F): ``stance_weights`` fixes it (oracle/teacher mode); otherwise it
    is recomputed each call from the *decoded* heights (soft, self-consistent mode), optionally
    multiplied by a binary/float ``stance_gate`` (e.g. the human-source contact mask).
    """
    def cost(z: torch.Tensor) -> torch.Tensor:
        foot_pos, dof_pos = decode_fn(z)
        if foot_pos.ndim != 4 or foot_pos.shape[-1] != 3:
            raise ValueError(f"decode_fn foot positions must be (B,T,F,3); got {tuple(foot_pos.shape)}")
        total = foot_pos.new_zeros(())

        heights = foot_pos[..., 2]
        if cfg.skate_weight > 0 and foot_pos.shape[1] > 1:
            if stance_weights is not None:
                w = stance_weights.to(foot_pos.dtype)
            else:
                # ground-relative heights per foot over the window; detached so the guidance
                # gradient pushes the feet to stop, not to levitate out of the stance weight
                rel = heights - heights.min(dim=1, keepdim=True).values
                w = soft_stance_weights(rel, cfg.height_threshold, cfg.temperature).detach()
                if stance_gate is not None:
                    w = w * stance_gate.to(w.dtype)
            vel_xy = (foot_pos[:, 1:, :, :2] - foot_pos[:, :-1, :, :2]) * cfg.fps
            w_v = w[:, 1:, :]
            total = total + cfg.skate_weight * (
                (vel_xy.square().sum(-1) * w_v).sum() / w_v.sum().clamp_min(1.0)
            )
        if cfg.penetration_weight > 0:
            total = total + cfg.penetration_weight * torch.relu(-heights).square().mean()
        if cfg.smooth_weight > 0 and dof_pos.shape[1] >= 3:
            acc = dof_pos[:, 2:] - 2 * dof_pos[:, 1:-1] + dof_pos[:, :-2]
            total = total + cfg.smooth_weight * acc.square().mean()
            if dof_pos.shape[1] >= 4:
                jerk = dof_pos[:, 3:] - 3 * dof_pos[:, 2:-1] + 3 * dof_pos[:, 1:-2] - dof_pos[:, :-3]
                total = total + 0.1 * cfg.smooth_weight * jerk.square().mean()
        return total

    return cost
