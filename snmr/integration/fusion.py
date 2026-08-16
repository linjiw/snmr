"""E78 — reference-channel dropout masking and structured explicit/latent fusion.

This module extends the frozen E70 distillation primitives without modifying them
(``snmr/integration/distillation.py`` and ``scripts/train_e52_dagger.py`` are bound in
the E70 confirmation hash manifest).  Everything here is unit-testable on CPU.

Two ideas, both from the 2026-08-15 program plan (``docs/LATENT_BENEFIT_PROGRAM_2026-08-15.md``):

1. **Reference dropout masking.**  Under action-MSE distillation from a deterministic
   teacher, ``I(a*; z | x, g) = 0`` and the latent is redundant beside the explicit
   reference.  Masking the reference channel during *training* manufactures, on purpose,
   the ambiguity that E70 manufactured only at evaluation time, which forces gradient
   into the latent pathway.  The same masker is used at evaluation to sweep dropout
   severity in physical units (ticks), with a seeded generator so every arm sees the
   identical dropout schedule (paired design).

2. **Structured fusion.**  A FiLM-conditioned or gated-residual code encoder in which
   the latent modulates the explicit trunk instead of competing with it as a concatenated
   input feature.  The action decoder still sees exactly ``[proprio, z_cmd]`` — the
   E70 exclusivity contract is unchanged.
"""

from __future__ import annotations

import dataclasses

import torch

from snmr.integration.distillation import CommandStudent, mlp

FLAG_DIM = 2  # [is_masked, staleness / max_segment]

VALID_SCOPES = frozenset({"all", "explicit", "snmr"})
VALID_MODES = frozenset({"hold", "zero", "extrapolate", "cycle"})


def dropout_hazard(target_fraction: float, mean_segment_ticks: float) -> float:
    """Per-tick segment-start hazard giving an expected masked-tick fraction.

    A renewal process alternating clean runs (geometric, mean ``1/h``) with dropout
    segments (mean ``L``) spends fraction ``hL / (1 + hL)`` of ticks masked, so
    ``h = f / (L (1 - f))``.
    """
    if not 0.0 <= target_fraction < 1.0:
        raise ValueError("target fraction must lie in [0, 1)")
    if mean_segment_ticks <= 0:
        raise ValueError("mean segment length must be positive")
    if target_fraction == 0.0:
        return 0.0
    return target_fraction / (mean_segment_ticks * (1.0 - target_fraction))


def ramp(round_index: int, ramp_rounds: int) -> float:
    """Linear 0 -> 1 schedule used to phase masking into training."""
    if round_index < 0:
        raise ValueError("round index must be nonnegative")
    if ramp_rounds <= 0:
        return 1.0
    return min(1.0, round_index / float(ramp_rounds))


@dataclasses.dataclass
class ReferenceDropoutMasker:
    """Per-environment Bernoulli-segment dropout of a reference-derived signal.

    State is one segment countdown per environment plus the last valid value of each
    masked tensor.  Fill modes:

    * ``hold``        — replay the last valid value (a mocap/teleop stall; the naive default);
    * ``zero``        — blank the channel (an explicit "no command" signal);
    * ``cycle``       — replay the channel's own most recent matching cycle
      (:class:`CycleContinuationExtrapolator`); the only fill that stays on the motion
      manifold at long outages;
    * ``extrapolate`` — advance the last valid value with a per-signal extrapolator
      ``f(held, staleness_ticks) -> value``, registered via :meth:`set_extrapolator`.
      Signals without one fall back to ``hold``.

    The extrapolating mode exists because E78-F showed the harm under dropout comes from a
    *stale* command conflicting with live proprioception, not from a missing one
    (``docs/E78F_FROZEN_DROPOUT_BASELINE_2026-08-16.md``).  Extrapolation is causal: it uses
    only what the channel delivered before it failed.

    The flag tensor returned by :meth:`step` is ``[is_masked, staleness_ticks / max_segment]``.
    """

    num_envs: int
    hazard: float
    min_segment: int
    max_segment: int
    device: torch.device | str = "cpu"
    mode: str = "hold"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.hazard < 0.0 or self.hazard > 1.0:
            raise ValueError("hazard must lie in [0, 1]")
        if self.min_segment < 1 or self.max_segment < self.min_segment:
            raise ValueError("segment bounds must satisfy 1 <= min <= max")
        if self.mode not in VALID_MODES:
            raise ValueError(f"unknown mask mode {self.mode!r}")
        self.device = torch.device(self.device)
        self.generator = torch.Generator(device="cpu")
        if self.seed is not None:
            self.generator.manual_seed(int(self.seed))
        self.remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.staleness = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._held: dict[str, torch.Tensor] = {}
        # Envs whose next tick must be clean (fresh episode: nothing valid to hold yet).
        self._force_clean = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_masked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._extrapolators: dict[str, object] = {}
        self.masked_ticks = 0
        self.total_ticks = 0

    # -- state ------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.remaining.zero_()
            self.staleness.zero_()
            self._held.clear()
            self._force_clean[:] = True
            return
        self.remaining[env_ids] = 0
        self.staleness[env_ids] = 0
        self._force_clean[env_ids] = True  # held value refreshes on that clean tick
        for extrapolator in self._extrapolators.values():
            if hasattr(extrapolator, "reset"):
                extrapolator.reset(env_ids)

    def set_extrapolator(self, name: str, fn) -> None:
        """Register ``f(held_value, staleness_ticks) -> value`` for one signal (mode='extrapolate')."""
        self._extrapolators[name] = fn

    def set_hazard(self, hazard: float) -> None:
        if hazard < 0.0 or hazard > 1.0:
            raise ValueError("hazard must lie in [0, 1]")
        self.hazard = float(hazard)

    def _advance(self) -> torch.Tensor:
        """Draw new segments, decrement running ones; return the per-env masked flag."""
        n = self.num_envs
        idle = (self.remaining <= 0) & ~self._force_clean
        self._force_clean[:] = False
        if self.hazard > 0.0:
            u = torch.rand(n, generator=self.generator).to(self.device)
            start = idle & (u < self.hazard)
            if bool(start.any()):
                lengths = torch.randint(
                    self.min_segment, self.max_segment + 1, (n,), generator=self.generator
                ).to(self.device)
                self.remaining = torch.where(start, lengths, self.remaining)
        masked = self.remaining > 0
        self.remaining = torch.where(masked, self.remaining - 1, self.remaining)
        self.staleness = torch.where(masked, self.staleness + 1, torch.zeros_like(self.staleness))
        self.masked_ticks += int(masked.sum())
        self.total_ticks += n
        self.last_masked = masked
        return masked

    def step(self, **signals: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Mask the named signals for this tick.

        Returns ``(masked_signals, flags)`` where ``flags`` has shape ``(N, FLAG_DIM)``.
        Every named signal shares one dropout schedule (a dropout is an event on the
        reference stream, not on one tensor).
        """
        masked = self._advance()
        out: dict[str, torch.Tensor] = {}
        m = masked.unsqueeze(-1)
        for name, value in signals.items():
            if value.shape[0] != self.num_envs:
                raise ValueError(f"signal {name!r} has {value.shape[0]} rows, expected {self.num_envs}")
            held = self._held.get(name)
            if held is None or held.shape != value.shape:
                held = value.detach().clone()
            if self.mode == "zero":
                shown = torch.where(m, torch.zeros_like(value), value)
            elif self.mode in ("extrapolate", "cycle") and name in self._extrapolators:
                extrapolator = self._extrapolators[name]
                if hasattr(extrapolator, "observe"):
                    extrapolator.observe(value, masked)
                filled = extrapolator(held, self.staleness)
                shown = torch.where(m, filled, value)
            else:  # hold (and extrapolate fallback for signals without an extrapolator)
                shown = torch.where(m, held, value)
            # Refresh the held copy on clean ticks only.
            self._held[name] = torch.where(m, held, value.detach())
            out[name] = shown
        flags = torch.stack(
            (
                masked.to(dtype=torch.float32),
                self.staleness.to(dtype=torch.float32) / float(self.max_segment),
            ),
            dim=-1,
        )
        return out, flags

    @property
    def masked_fraction(self) -> float:
        return self.masked_ticks / max(self.total_ticks, 1)


class FiLM(torch.nn.Module):
    """Feature-wise affine modulation ``h * (1 + gamma) + beta`` from a conditioner."""

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = torch.nn.Linear(cond_dim, 2 * feature_dim)
        torch.nn.init.zeros_(self.to_scale_shift.weight)
        torch.nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class FusionCommandStudent(CommandStudent):
    """Command student with reference-dropout flags and structured explicit/latent fusion.

    ``fusion`` applies only when ``prior_goal == 'explicit+snmr'``:

    * ``concat`` — the E52 arm-D encoder, plus flag bits (baseline fusion);
    * ``film``   — trunk over ``[proprio, cmd, flags]``; the projected latent emits
      per-layer FiLM scale/shift (initialised to identity, so training starts at the
      explicit solution and must *earn* latent use);
    * ``gated``  — ``mu = prior_base([proprio, cmd, flags]) + sigmoid(w) * prior_res([proprio, z', flags])``.

    ``flag_dim=0`` with ``fusion='concat'`` is shape-identical to the frozen
    :class:`CommandStudent`, so frozen E70 checkpoints load into it for evaluation under
    the same dropout masker (the unmasked-trained baseline).

    The decoder is inherited unchanged: ``act(proprio, z_cmd)`` and nothing else.
    """

    VALID_FUSION = frozenset({"concat", "film", "gated"})

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
        fusion: str = "concat",
        flag_dim: int = FLAG_DIM,
    ) -> None:
        super().__init__(
            proprio_dim, priv_dim, num_act, prior_goal, cmd_dim,
            z_window_dim=z_window_dim, z_cmd_dim=z_cmd_dim,
        )
        if fusion not in self.VALID_FUSION:
            raise ValueError(f"unknown fusion {fusion!r}")
        if fusion != "concat" and prior_goal != "explicit+snmr":
            raise ValueError("film/gated fusion requires prior_goal='explicit+snmr'")
        if flag_dim < 0:
            raise ValueError("flag_dim must be nonnegative")
        self.fusion = fusion
        self.flag_dim = flag_dim
        self.cmd_dim = cmd_dim
        self.proprio_dim = proprio_dim
        goal_dim = {
            "none": 0, "snmr": z_cmd_dim, "explicit": cmd_dim,
            "explicit+snmr": cmd_dim + z_cmd_dim,
        }[prior_goal]
        prior_in = proprio_dim + goal_dim + flag_dim
        # Rebuild the flag-aware encoders; the inherited ``decoder`` is kept verbatim.
        self.prior = mlp([prior_in, 512, 256], z_cmd_dim)
        self.posterior = mlp([prior_in + priv_dim, 512, 256], z_cmd_dim)
        if fusion == "film":
            trunk_in = proprio_dim + cmd_dim + flag_dim
            self.trunk_1 = torch.nn.Linear(trunk_in, 512)
            self.trunk_2 = torch.nn.Linear(512, 256)
            self.trunk_out = torch.nn.Linear(256, z_cmd_dim)
            cond_dim = z_cmd_dim + flag_dim
            self.film_1 = FiLM(cond_dim, 512)
            self.film_2 = FiLM(cond_dim, 256)
        elif fusion == "gated":
            self.prior_base = mlp([proprio_dim + cmd_dim + flag_dim, 512, 256], z_cmd_dim)
            self.prior_res = mlp([proprio_dim + z_cmd_dim + flag_dim, 512, 256], z_cmd_dim)
            self.gate_logit = torch.nn.Parameter(torch.zeros(()))

    # -- encoder ---------------------------------------------------------------------
    def prior_input(  # type: ignore[override]
        self,
        proprio: torch.Tensor,
        z_snmr_window: torch.Tensor,
        cmd: torch.Tensor,
        flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flags is None:
            flags = torch.zeros(proprio.shape[0], self.flag_dim, device=proprio.device, dtype=proprio.dtype)
        parts = [proprio]
        if "explicit" in self.prior_goal:
            parts.append(cmd)
        if "snmr" in self.prior_goal:
            parts.append(self.z_proj(z_snmr_window))
        parts.append(flags)
        return torch.cat(parts, dim=-1)

    def mu_prior(  # type: ignore[override]
        self,
        proprio: torch.Tensor,
        z_snmr_window: torch.Tensor,
        cmd: torch.Tensor,
        flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flags is None:
            flags = torch.zeros(proprio.shape[0], self.flag_dim, device=proprio.device, dtype=proprio.dtype)
        if self.fusion == "concat":
            return self.prior(self.prior_input(proprio, z_snmr_window, cmd, flags))
        if self.fusion == "film":
            cond = torch.cat((self.z_proj(z_snmr_window), flags), dim=-1)
            h = torch.nn.functional.elu(self.film_1(self.trunk_1(torch.cat((proprio, cmd, flags), -1)), cond))
            h = torch.nn.functional.elu(self.film_2(self.trunk_2(h), cond))
            return self.trunk_out(h)
        # gated
        base = self.prior_base(torch.cat((proprio, cmd, flags), dim=-1))
        res = self.prior_res(torch.cat((proprio, self.z_proj(z_snmr_window), flags), dim=-1))
        return base + torch.sigmoid(self.gate_logit) * res

    def mu_residual(  # type: ignore[override]
        self,
        proprio: torch.Tensor,
        z_snmr_window: torch.Tensor,
        cmd: torch.Tensor,
        priv: torch.Tensor,
        flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs = torch.cat((self.prior_input(proprio, z_snmr_window, cmd, flags), priv), dim=-1)
        return self.posterior(inputs)

    @property
    def gate(self) -> float | None:
        """Sigmoid gate value for the gated fusion (diagnostic); None otherwise."""
        if self.fusion != "gated":
            return None
        return float(torch.sigmoid(self.gate_logit.detach()))


def constant_velocity_goal_extrapolator(
    mean: torch.Tensor, std: torch.Tensor, eps: float, dt: float,
    pos_slice: slice = slice(0, 29), vel_slice: slice = slice(29, 58),
):
    """Dead-reckon a held explicit goal with the joint velocities it already carries.

    The 64-d WBT goal is ``[q_ref (29), qdot_ref (29), R_rel (6)]``: a stale sample already
    contains the reference's own velocity field, so ``q_ref + qdot_ref * staleness*dt`` is a
    strictly causal first-order prediction of where the reference went — free, no model.
    Velocities and the relative orientation block are held (they cannot be integrated from
    the goal alone).  Operates on the normalised tensor the student consumes.
    """
    def extrapolate(held_norm: torch.Tensor, staleness: torch.Tensor) -> torch.Tensor:
        raw = held_norm * (std + eps) + mean
        out = raw.clone()
        out[:, pos_slice] = raw[:, pos_slice] + raw[:, vel_slice] * (
            staleness.to(raw.dtype).unsqueeze(-1) * dt
        )
        return (out - mean) / (std + eps)

    return extrapolate


def window_linear_extrapolator(sample_dim: int, offset_ticks: int):
    """Linearly continue a two-sample window ``[u_t0, u_{t0+k}]`` past its own horizon.

    The held window already states where the signal was heading; at staleness ``s`` the
    first-order continuation is ``u_t0 + (u_{t0+k} - u_t0) * s / k`` for the current sample,
    and one step further for the lookahead sample.  Causal: uses only the last valid window.
    """
    def extrapolate(held: torch.Tensor, staleness: torch.Tensor) -> torch.Tensor:
        u0, u1 = held[:, :sample_dim], held[:, sample_dim:2 * sample_dim]
        slope = (u1 - u0) / float(offset_ticks)
        s = staleness.to(held.dtype).unsqueeze(-1)
        cur = u0 + slope * s
        nxt = u0 + slope * (s + offset_ticks)
        out = held.clone()
        out[:, :sample_dim] = cur
        out[:, sample_dim:2 * sample_dim] = nxt
        return out

    return extrapolate


class CycleContinuationExtrapolator:
    """Continue a signal by replaying its own most recent matching cycle.

    Strictly causal and model-free: keeps a per-environment ring buffer of the values the
    channel actually delivered, and at the start of an outage picks the lag ``L`` in
    ``[min_lag, max_lag]`` whose past ``match_ticks`` window best matches the most recent
    ``match_ticks`` window. During the outage it emits ``value[t - L]``.

    Motivation (`docs/COMMAND_INTERFACE_SYNTHESIS_2026-08-16.md`): on the E70 walks the
    reference-prediction error of a held sample drifts to 0.22–0.31 rad and of a
    constant-velocity extrapolation to 0.93–1.39 rad at a 1 s outage, while cycle
    continuation stays **flat at 0.155–0.250 rad from 0.1 s to 1.5 s**. A fill that stays on
    the motion manifold turns an unbounded-horizon problem into a bounded one.

    Falls back to holding whenever an environment has not yet accumulated enough valid
    history (fresh episode, or a long outage that consumed the buffer).
    """

    def __init__(
        self,
        num_envs: int,
        dim: int,
        *,
        device: torch.device | str = "cpu",
        min_lag: int = 25,
        max_lag: int = 80,
        match_ticks: int = 20,
    ) -> None:
        if not 1 <= min_lag <= max_lag or match_ticks < 1:
            raise ValueError("require 1 <= min_lag <= max_lag and match_ticks >= 1")
        self.num_envs, self.dim = num_envs, dim
        self.device = torch.device(device)
        self.min_lag, self.max_lag, self.match_ticks = min_lag, max_lag, match_ticks
        self.capacity = max_lag + match_ticks + 1
        self.buffer = torch.zeros(num_envs, self.capacity, dim, device=self.device)
        self.head = torch.zeros(num_envs, dtype=torch.long, device=self.device)   # next write slot
        self.valid = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # consecutive valid ticks
        self.lag = torch.full((num_envs,), max_lag, dtype=torch.long, device=self.device)
        self.fallbacks = 0
        self.uses = 0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.valid.zero_()
        else:
            self.valid[env_ids] = 0

    def _gather(self, offsets_back: torch.Tensor) -> torch.Tensor:
        """Values ``offsets_back`` ticks before the newest write, per environment."""
        index = (self.head - 1 - offsets_back) % self.capacity
        return self.buffer[torch.arange(self.num_envs, device=self.device), index]

    def observe(self, value: torch.Tensor, masked: torch.Tensor, emitted: torch.Tensor | None = None) -> None:
        """Advance the timeline by one tick for every environment.

        The buffer must stay **contiguous in time** or a lag in samples is not a lag in
        ticks: with 30 % of ticks masked, a clean-only buffer time-compresses by 1/0.7 and
        the matched cycle is wrong. Clean ticks record the delivered value; masked ticks
        record what was emitted in their place (self-consistent recursion), so index
        arithmetic stays in control ticks throughout.
        """
        write = value.detach() if emitted is None else torch.where(masked.unsqueeze(-1), emitted.detach(), value.detach())
        index = self.head % self.capacity
        rows = torch.arange(self.num_envs, device=self.device)
        self.buffer[rows, index] = write
        self.head += 1
        self.valid = torch.clamp(self.valid + 1, max=self.capacity)
        onset = masked & (self.valid >= self.max_lag + self.match_ticks)
        if bool(onset.any()):
            recent = torch.stack([self._gather(torch.full_like(self.head, k))
                                  for k in range(self.match_ticks)], dim=1)      # (N, M, D)
            best = torch.full((self.num_envs,), float("inf"), device=self.device)
            for lag in range(self.min_lag, self.max_lag + 1):
                past = torch.stack([self._gather(torch.full_like(self.head, k + lag))
                                    for k in range(self.match_ticks)], dim=1)
                err = (recent - past).square().mean(dim=(1, 2))
                better = onset & (err < best)
                best = torch.where(better, err, best)
                self.lag = torch.where(better, torch.full_like(self.lag, lag), self.lag)

    def __call__(self, held: torch.Tensor, staleness: torch.Tensor) -> torch.Tensor:
        """Emit ``value[t - lag]`` where history allows, else the held value.

        The emitted value is written into the buffer slot for this tick, keeping the
        timeline contiguous (see :meth:`observe`).
        """
        usable = self.valid >= self.max_lag + self.match_ticks
        offsets = torch.clamp(self.lag, min=1)   # lag is measured from the current tick
        candidate = self._gather(offsets)
        self.uses += int(usable.sum())
        self.fallbacks += int((~usable).sum())
        out = torch.where(usable.unsqueeze(-1), candidate, held)
        rows = torch.arange(self.num_envs, device=self.device)
        self.buffer[rows, (self.head - 1) % self.capacity] = out.detach()
        return out


def paired_dropout_summary(
    completed_a: torch.Tensor,
    completed_b: torch.Tensor,
    clean_a: torch.Tensor,
    clean_b: torch.Tensor,
) -> dict[str, float]:
    """Paired matched-subset contrast (E77 addendum lesson).

    Restricts to rollouts both arms complete under *clean* conditions and reports the
    completion difference under the degraded condition on that subset, plus McNemar
    discordant counts.  Marginal retention ratios launder a clean-condition gap; this
    does not.
    """
    for t in (completed_a, completed_b, clean_a, clean_b):
        if t.shape != completed_a.shape or t.ndim != 1:
            raise ValueError("all completion vectors must share one 1-D shape")
    both_clean = clean_a.bool() & clean_b.bool()
    n = int(both_clean.sum())
    if n == 0:
        return {"matched_n": 0, "a": float("nan"), "b": float("nan"),
                "diff_a_minus_b": float("nan"), "a_only": 0, "b_only": 0}
    a = completed_a.bool()[both_clean]
    b = completed_b.bool()[both_clean]
    return {
        "matched_n": n,
        "a": float(a.float().mean()),
        "b": float(b.float().mean()),
        "diff_a_minus_b": float(a.float().mean() - b.float().mean()),
        "a_only": int((a & ~b).sum()),
        "b_only": int((b & ~a).sum()),
    }
