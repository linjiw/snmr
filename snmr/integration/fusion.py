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
VALID_MODES = frozenset({"hold", "zero"})


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
    masked tensor.  ``mode='hold'`` replays the last valid value (mocap/teleop stall);
    ``mode='zero'`` blanks it.  The flag tensor returned by :meth:`step` is
    ``[is_masked, staleness_ticks / max_segment]``.
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
            if self.mode == "hold":
                shown = torch.where(m, held, value)
            else:
                shown = torch.where(m, torch.zeros_like(value), value)
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
