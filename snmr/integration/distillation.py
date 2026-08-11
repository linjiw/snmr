"""Utilities for stable retarget-to-track online distillation.

This module deliberately has no Holosoma dependency.  Keeping the stability and
information-isolation primitives here makes the load-bearing E67 contracts unit-testable
without constructing a simulator.
"""

from __future__ import annotations

import collections
import dataclasses
import math
from collections.abc import Mapping

import torch


def mlp(sizes: list[int], out_dim: int) -> torch.nn.Sequential:
    """Build the ELU MLP used by the command student."""
    layers: list[torch.nn.Module] = []
    for in_dim, hidden_dim in zip(sizes[:-1], sizes[1:]):
        layers.extend((torch.nn.Linear(in_dim, hidden_dim), torch.nn.ELU()))
    layers.append(torch.nn.Linear(sizes[-1], out_dim))
    return torch.nn.Sequential(*layers)


class CommandStudent(torch.nn.Module):
    """Exclusive command student.

    ``prior_goal`` controls what reaches the code encoder.  The action decoder is
    structurally limited to ``[proprio, z_cmd]`` and therefore cannot receive the explicit
    goal, retargeting latent, clock, or motion ID through another argument.
    """

    VALID_GOALS = frozenset({"none", "snmr", "explicit", "explicit+snmr"})

    def __init__(
        self,
        proprio_dim: int,
        priv_dim: int,
        num_act: int,
        prior_goal: str,
        cmd_dim: int,
        *,
        z_window_dim: int = 256,
        z_cmd_dim: int = 64,
    ) -> None:
        super().__init__()
        if prior_goal not in self.VALID_GOALS:
            raise ValueError(f"unknown prior goal {prior_goal!r}")
        self.prior_goal = prior_goal
        self.z_cmd_dim = z_cmd_dim
        self.z_proj = mlp([z_window_dim, 256], z_cmd_dim)
        goal_dim = {
            "none": 0,
            "snmr": z_cmd_dim,
            "explicit": cmd_dim,
            "explicit+snmr": cmd_dim + z_cmd_dim,
        }[prior_goal]
        prior_in = proprio_dim + goal_dim
        self.prior = mlp([prior_in, 512, 256], z_cmd_dim)
        self.posterior = mlp([prior_in + priv_dim, 512, 256], z_cmd_dim)
        self.decoder = mlp([proprio_dim + z_cmd_dim, 512, 256, 128], num_act)

    def prior_input(
        self,
        proprio: torch.Tensor,
        z_snmr_window: torch.Tensor,
        cmd: torch.Tensor,
    ) -> torch.Tensor:
        parts = [proprio]
        if "explicit" in self.prior_goal:
            parts.append(cmd)
        if "snmr" in self.prior_goal:
            parts.append(self.z_proj(z_snmr_window))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else proprio

    def mu_prior(
        self,
        proprio: torch.Tensor,
        z_snmr_window: torch.Tensor,
        cmd: torch.Tensor,
    ) -> torch.Tensor:
        return self.prior(self.prior_input(proprio, z_snmr_window, cmd))

    def mu_residual(
        self,
        proprio: torch.Tensor,
        z_snmr_window: torch.Tensor,
        cmd: torch.Tensor,
        priv: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat((self.prior_input(proprio, z_snmr_window, cmd), priv), dim=-1)
        return self.posterior(inputs)

    def act(self, proprio: torch.Tensor, z_cmd: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((proprio, z_cmd), dim=-1))


def paired_temporal_smoothness(
    current: torch.Tensor,
    previous: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean squared code change over valid paired transitions.

    Both ``current`` and ``previous`` must be recomputed with the *current* parameters.
    Averaging first over the latent dimension and then over valid transitions keeps the
    scale independent of code width and reset frequency.
    """
    if current.shape != previous.shape or current.ndim != 2:
        raise ValueError(
            f"paired codes must have identical (N,D) shapes, got "
            f"{tuple(current.shape)} and {tuple(previous.shape)}"
        )
    if valid.shape not in {(current.shape[0],), (current.shape[0], 1)}:
        raise ValueError(f"valid mask has incompatible shape {tuple(valid.shape)}")
    weights = valid.reshape(-1).to(dtype=current.dtype, device=current.device)
    per_transition = (current - previous).square().mean(dim=-1)
    return (per_transition * weights).sum() / weights.sum().clamp_min(1.0)


def teacher_mix_probability(
    round_index: int,
    *,
    anneal_rounds: int = 200,
    floor: float = 0.1,
) -> float:
    """Linear teacher mixing schedule with a nonzero safety floor."""
    if round_index < 0 or anneal_rounds <= 0:
        raise ValueError("round_index must be nonnegative and anneal_rounds positive")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("teacher floor must lie in [0, 1]")
    return max(floor, 1.0 - round_index / float(anneal_rounds))


class RoundReplayBuffer:
    """Bounded round-level replay with deterministic FIFO eviction."""

    def __init__(self, max_rounds: int) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.max_rounds = max_rounds
        self._rounds: collections.deque[dict[str, torch.Tensor]] = collections.deque(
            maxlen=max_rounds
        )

    def append(self, data: Mapping[str, torch.Tensor]) -> None:
        if not data:
            raise ValueError("cannot append an empty round")
        lengths = {tensor.shape[0] for tensor in data.values()}
        if len(lengths) != 1:
            raise ValueError(f"round tensors have different lengths: {sorted(lengths)}")
        self._rounds.append({key: value.detach() for key, value in data.items()})

    def merged(self) -> dict[str, torch.Tensor]:
        if not self._rounds:
            raise ValueError("replay buffer is empty")
        keys = set(self._rounds[0])
        if any(set(round_data) != keys for round_data in self._rounds):
            raise ValueError("replay rounds do not have identical fields")
        return {
            key: torch.cat([round_data[key] for round_data in self._rounds], dim=0)
            for key in self._rounds[0]
        }

    def __len__(self) -> int:
        return len(self._rounds)


@dataclasses.dataclass
class DivergenceGate:
    """Pre-specified abort gate for an online-distillation run."""

    smooth_warmup_rounds: int = 200
    smooth_multiplier: float = 100.0
    max_kl: float = 1.0e4
    max_latent_norm: float = 100.0
    _smooth_history: list[float] = dataclasses.field(default_factory=list, init=False)
    _smooth_baseline: float | None = dataclasses.field(default=None, init=False)

    def check(
        self,
        round_index: int,
        *,
        action: float,
        kl: float,
        smooth: float,
        latent_norm: float,
    ) -> str | None:
        metrics = {
            "action": action,
            "kl": kl,
            "smooth": smooth,
            "latent_norm": latent_norm,
        }
        for name, value in metrics.items():
            if not math.isfinite(value):
                return f"nonfinite_{name}"
        if kl > self.max_kl:
            return f"kl_above_{self.max_kl:g}"
        if latent_norm > self.max_latent_norm:
            return f"latent_norm_above_{self.max_latent_norm:g}"

        if round_index <= self.smooth_warmup_rounds:
            self._smooth_history.append(smooth)
        if round_index == self.smooth_warmup_rounds:
            self._smooth_baseline = max(
                float(torch.tensor(self._smooth_history).median()), 1.0e-8
            )
        if (
            self._smooth_baseline is not None
            and round_index > self.smooth_warmup_rounds
            and smooth > self.smooth_multiplier * self._smooth_baseline
        ):
            return f"smooth_above_{self.smooth_multiplier:g}x_warmup_median"
        return None


def same_phase_shuffled_latents(
    latents: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """Replace every clip's latent with the next clip at matched normalized time.

    The cyclic, deterministic mapping is a derangement for every pool with more than one
    clip.  Values are nearest-neighbor sampled in normalized time and written back into the
    destination clip's original frame range, so the normal offset gather remains valid.
    """
    if latents.ndim != 2:
        raise ValueError(f"latents must be (T,D), got {tuple(latents.shape)}")
    starts = starts.to(device=latents.device, dtype=torch.long)
    ends = ends.to(device=latents.device, dtype=torch.long)
    if starts.shape != ends.shape or starts.ndim != 1 or len(starts) < 2:
        raise ValueError("same-phase shuffling requires at least two motion boundaries")
    if int(starts[0]) != 0 or int(ends[-1]) != latents.shape[0]:
        raise ValueError("motion boundaries do not cover the latent tensor")

    shuffled = torch.empty_like(latents)
    num_motions = len(starts)
    for destination in range(num_motions):
        source = (destination + 1) % num_motions
        dst_start, dst_end = int(starts[destination]), int(ends[destination])
        src_start, src_end = int(starts[source]), int(ends[source])
        dst_length, src_length = dst_end - dst_start, src_end - src_start
        phase = torch.linspace(0.0, 1.0, dst_length, device=latents.device)
        source_index = src_start + torch.round(phase * (src_length - 1)).long()
        shuffled[dst_start:dst_end] = latents[source_index]
    return shuffled


def shared_time_index_latents(
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    output_dim: int,
    seed: int = 63,
    num_frequencies: int = 16,
) -> torch.Tensor:
    """Build a time code that resets identically at every motion boundary.

    All clips share one frequency bank and fixed projection.  Consequently, equal local
    frame indices receive bit-identical codes regardless of motion ID; concatenated global
    frame offsets cannot leak clip identity.
    """
    if starts.shape != ends.shape or starts.ndim != 1 or len(starts) < 1:
        raise ValueError("invalid motion boundaries")
    if int(starts[0]) != 0 or torch.any(ends <= starts):
        raise ValueError("motion boundaries must start at zero and have positive lengths")
    lengths = ends - starts
    total_frames = int(ends[-1])
    max_length = int(lengths.max())
    device = starts.device
    dtype = torch.float32
    periods = torch.logspace(
        1,
        math.log10(max(max_length, 20)),
        num_frequencies,
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(
        2 * num_frequencies, output_dim, generator=generator, dtype=dtype
    ).to(device)
    result = torch.empty(total_frames, output_dim, device=device, dtype=dtype)
    for start, end in zip(starts.tolist(), ends.tolist()):
        local_index = torch.arange(end - start, device=device, dtype=dtype)
        angles = 2 * math.pi * local_index[:, None] / periods[None, :]
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        result[start:end] = features @ projection
    return result


def route_teacher_actions(
    actions_by_teacher: torch.Tensor,
    motion_ids: torch.Tensor,
) -> torch.Tensor:
    """Select labels by motion ID without exposing that ID to the student."""
    if actions_by_teacher.ndim != 3:
        raise ValueError("actions_by_teacher must have shape (teachers, envs, actions)")
    num_teachers, num_envs, _ = actions_by_teacher.shape
    if motion_ids.shape != (num_envs,):
        raise ValueError("motion_ids must have one entry per environment")
    if torch.any((motion_ids < 0) | (motion_ids >= num_teachers)):
        raise ValueError("motion_ids index an unavailable teacher")
    env_ids = torch.arange(num_envs, device=motion_ids.device)
    return actions_by_teacher[motion_ids.long(), env_ids]


def destroy_command_code(code: torch.Tensor, mode: str) -> torch.Tensor:
    """Causal deployment interventions for the exclusive command channel."""
    if code.ndim != 2:
        raise ValueError("command code must have shape (envs, dim)")
    if mode == "none":
        return code
    if mode == "zero":
        return torch.zeros_like(code)
    if mode == "shuffle":
        return code.roll(shifts=1, dims=0)
    if mode == "marginal_random":
        mean = code.mean(dim=0, keepdim=True)
        std = code.std(dim=0, keepdim=True).clamp_min(1.0e-6)
        return mean + std * torch.randn_like(code)
    raise ValueError(f"unknown command destruction mode {mode!r}")
