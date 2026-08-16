"""CPU tests for the E78 masking + fusion primitives (snmr/integration/fusion.py)."""

import math

import pytest
import torch

from snmr.integration.fusion import (
    FLAG_DIM,
    FusionCommandStudent,
    ReferenceDropoutMasker,
    dropout_hazard,
    paired_dropout_summary,
    ramp,
)


def test_dropout_hazard_reproduces_target_fraction_in_expectation():
    torch.manual_seed(0)
    frac, seg = 0.3, (5, 25)
    mean_len = (seg[0] + seg[1]) / 2
    masker = ReferenceDropoutMasker(512, dropout_hazard(frac, mean_len), *seg, seed=1)
    x = torch.zeros(512, 3)
    for _ in range(4000):
        masker.step(x=x)
    assert abs(masker.masked_fraction - frac) < 0.03
    assert dropout_hazard(0.0, 10) == 0.0
    with pytest.raises(ValueError):
        dropout_hazard(1.0, 10)


def test_hold_mode_replays_last_clean_value_and_flags_staleness():
    masker = ReferenceDropoutMasker(1, hazard=0.0, min_segment=3, max_segment=3, seed=0)
    # Force a segment by hand.
    (out, flags) = masker.step(g=torch.tensor([[1.0]]))
    assert out["g"].item() == 1.0 and flags[0, 0] == 0.0
    masker.remaining[:] = 3
    seen = []
    for t in range(2, 6):
        out, flags = masker.step(g=torch.tensor([[float(t)]]))
        seen.append((out["g"].item(), flags[0, 0].item(), flags[0, 1].item()))
    # three masked ticks replay 1.0 with rising staleness, then the live value returns
    for got, want in zip(seen[:3], [(1.0, 1.0, 1 / 3), (1.0, 1.0, 2 / 3), (1.0, 1.0, 1.0)]):
        assert got == pytest.approx(want)
    assert seen[3] == (5.0, 0.0, 0.0)


def test_zero_mode_blanks_and_shared_schedule_masks_all_signals_together():
    masker = ReferenceDropoutMasker(4, hazard=0.0, min_segment=2, max_segment=2, mode="zero")
    masker.step(g=torch.ones(4, 2), z=torch.ones(4, 5))
    masker.remaining[:] = torch.tensor([2, 0, 2, 0])
    out, flags = masker.step(g=torch.ones(4, 2), z=torch.ones(4, 5))
    assert torch.equal(flags[:, 0], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.equal(out["g"][:, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.equal(out["z"][:, 0], torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_seeded_maskers_produce_identical_schedules_across_arms():
    a = ReferenceDropoutMasker(64, 0.05, 2, 10, seed=404)
    b = ReferenceDropoutMasker(64, 0.05, 2, 10, seed=404)
    x = torch.zeros(64, 1)
    for _ in range(200):
        _, fa = a.step(x=x)
        _, fb = b.step(x=x)
        assert torch.equal(fa, fb)


def test_reset_clears_segments_for_named_envs_only():
    masker = ReferenceDropoutMasker(3, 0.0, 4, 4)
    masker.step(x=torch.zeros(3, 1))
    masker.remaining[:] = 4
    masker.reset(torch.tensor([1]))
    _, flags = masker.step(x=torch.zeros(3, 1))
    assert flags[:, 0].tolist() == [1.0, 0.0, 1.0]


def test_first_tick_after_reset_is_always_clean():
    masker = ReferenceDropoutMasker(8, hazard=1.0, min_segment=3, max_segment=3, seed=0)
    _, flags = masker.step(x=torch.zeros(8, 1))
    assert flags[:, 0].sum() == 0.0          # fresh episodes: nothing to hold yet
    _, flags = masker.step(x=torch.zeros(8, 1))
    assert flags[:, 0].sum() == 8.0          # hazard 1 starts a segment on tick 2
    masker.reset(torch.tensor([0, 1]))
    _, flags = masker.step(x=torch.zeros(8, 1))
    assert flags[:2, 0].tolist() == [0.0, 0.0] and flags[2:, 0].sum() == 6.0


def test_ramp_schedule():
    assert ramp(0, 100) == 0.0 and ramp(50, 100) == 0.5 and ramp(500, 100) == 1.0
    assert ramp(3, 0) == 1.0


@pytest.mark.parametrize("fusion", ["concat", "film", "gated"])
def test_fusion_students_keep_decoder_contract_and_accept_flags(fusion):
    torch.manual_seed(0)
    s = FusionCommandStudent(90, 7, 29, "explicit+snmr", 64, fusion=fusion)
    p, z, c = torch.randn(5, 90), torch.randn(5, 256), torch.randn(5, 64)
    flags = torch.zeros(5, FLAG_DIM)
    mu = s.mu_prior(p, z, c, flags)
    assert mu.shape == (5, 64)
    assert s.mu_prior(p, z, c).shape == (5, 64)  # flags default to zeros
    assert s.act(p, mu).shape == (5, 29)
    # decoder input is exactly proprio + z_cmd
    assert s.decoder[0].in_features == 90 + 64
    if fusion == "gated":
        assert abs(s.gate - 0.5) < 1e-6


def test_film_initialises_to_identity_so_latent_must_earn_influence():
    torch.manual_seed(1)
    s = FusionCommandStudent(90, 7, 29, "explicit+snmr", 64, fusion="film")
    p, c = torch.randn(4, 90), torch.randn(4, 64)
    flags = torch.zeros(4, FLAG_DIM)
    mu1 = s.mu_prior(p, torch.randn(4, 256), c, flags)
    mu2 = s.mu_prior(p, torch.randn(4, 256), c, flags)
    assert torch.allclose(mu1, mu2)  # zero-init FiLM: latent has no effect at init


def test_flags_change_the_code_only_when_flag_dim_is_wired():
    torch.manual_seed(2)
    s = FusionCommandStudent(90, 7, 29, "explicit", 64)
    p, z, c = torch.randn(4, 90), torch.randn(4, 256), torch.randn(4, 64)
    on = torch.tensor([[1.0, 0.5]]).expand(4, 2)
    assert not torch.allclose(s.mu_prior(p, z, c, on), s.mu_prior(p, z, c))


def test_film_and_gated_require_explicit_plus_snmr():
    with pytest.raises(ValueError):
        FusionCommandStudent(3, 4, 2, "explicit", 5, z_window_dim=6, z_cmd_dim=4, fusion="film")


def test_paired_dropout_summary_uses_matched_subset():
    clean_a = torch.tensor([1, 1, 1, 0, 1])
    clean_b = torch.tensor([1, 1, 0, 1, 1])
    deg_a = torch.tensor([1, 0, 1, 0, 1])
    deg_b = torch.tensor([1, 1, 0, 1, 0])
    out = paired_dropout_summary(deg_a, deg_b, clean_a, clean_b)
    assert out["matched_n"] == 3  # rollouts 0, 1, 4
    assert out["a"] == pytest.approx(2 / 3) and out["b"] == pytest.approx(2 / 3)
    assert out["a_only"] == 1 and out["b_only"] == 1


def test_e78_trainer_is_derived_from_the_frozen_harness():
    """The E78 trainer must be exactly what derive_e78_trainer.py produces from the frozen file."""
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("derive_e78", root / "scripts" / "derive_e78_trainer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (root / "scripts" / "train_e78_masked_fusion.py").read_text() == mod.derive()


def test_flag_dim_zero_is_shape_compatible_with_frozen_command_student():
    from snmr.integration.distillation import CommandStudent

    torch.manual_seed(3)
    frozen = CommandStudent(90, 7, 29, "explicit", 64, z_window_dim=256, z_cmd_dim=64)
    compat = FusionCommandStudent(90, 7, 29, "explicit", 64, z_window_dim=256, z_cmd_dim=64, flag_dim=0)
    compat.load_state_dict(frozen.state_dict())  # strict: identical keys and shapes
    p, z, c = torch.randn(4, 90), torch.randn(4, 256), torch.randn(4, 64)
    assert torch.allclose(compat.mu_prior(p, z, c, torch.zeros(4, 0)), frozen.mu_prior(p, z, c))
    assert torch.allclose(compat.mu_prior(p, z, c), frozen.mu_prior(p, z, c))


def test_constant_velocity_extrapolator_dead_reckons_positions_only():
    from snmr.integration.fusion import constant_velocity_goal_extrapolator

    mean = torch.zeros(64); std = torch.ones(64)
    f = constant_velocity_goal_extrapolator(mean, std, eps=0.0, dt=0.02,
                                            pos_slice=slice(0, 29), vel_slice=slice(29, 58))
    held = torch.zeros(2, 64)
    held[:, 0] = 1.0        # q_ref[0] = 1.0
    held[:, 29] = 0.5       # qdot_ref[0] = 0.5 rad/s
    held[:, 58] = 7.0       # R_rel block
    out = f(held, torch.tensor([0, 10]))
    assert out[0, 0] == pytest.approx(1.0)                 # zero staleness -> unchanged
    assert out[1, 0] == pytest.approx(1.0 + 0.5 * 10 * 0.02)   # 0.2 s of dead reckoning
    assert out[1, 29] == pytest.approx(0.5) and out[1, 58] == pytest.approx(7.0)  # held


def test_constant_velocity_extrapolator_respects_normalisation():
    from snmr.integration.fusion import constant_velocity_goal_extrapolator

    mean = torch.full((64,), 3.0); std = torch.full((64,), 2.0)
    f = constant_velocity_goal_extrapolator(mean, std, eps=0.0, dt=0.02)
    held_norm = torch.zeros(1, 64)          # normalised zero == raw 3.0 everywhere
    out = f(held_norm, torch.tensor([5]))
    raw_q = out[0, 0] * 2.0 + 3.0
    assert raw_q == pytest.approx(3.0 + 3.0 * 5 * 0.02)     # q + qdot*dt with raw values


def test_window_linear_extrapolator_continues_the_window_slope():
    from snmr.integration.fusion import window_linear_extrapolator

    f = window_linear_extrapolator(sample_dim=2, offset_ticks=5)
    held = torch.tensor([[0.0, 10.0, 5.0, 20.0]])   # u_t0 = (0,10), u_t0+5 = (5,20)
    out = f(held, torch.tensor([5]))
    assert out[0, 0] == pytest.approx(5.0) and out[0, 1] == pytest.approx(20.0)   # current -> old lookahead
    assert out[0, 2] == pytest.approx(10.0) and out[0, 3] == pytest.approx(30.0)  # lookahead one step on
    assert torch.allclose(f(held, torch.tensor([0])), held)


def test_extrapolate_mode_uses_registered_fill_and_falls_back_to_hold():
    masker = ReferenceDropoutMasker(2, hazard=0.0, min_segment=3, max_segment=3, mode="extrapolate")
    masker.set_extrapolator("g", lambda held, stale: held + stale.unsqueeze(-1).float())
    masker.step(g=torch.ones(2, 1), z=torch.ones(2, 1))
    masker.remaining[:] = 3
    out, _ = masker.step(g=torch.full((2, 1), 9.0), z=torch.full((2, 1), 9.0))
    assert out["g"].flatten().tolist() == [2.0, 2.0]   # held 1.0 + staleness 1
    assert out["z"].flatten().tolist() == [1.0, 1.0]   # no extrapolator -> hold


def test_cycle_continuation_replays_the_matching_cycle_on_a_periodic_signal():
    from snmr.integration.fusion import CycleContinuationExtrapolator

    period, n, dim = 30, 2, 3
    ext = CycleContinuationExtrapolator(n, dim, min_lag=25, max_lag=40, match_ticks=10)
    signal = lambda t: torch.tensor(  # noqa: E731
        [[math.sin(2 * math.pi * t / period), math.cos(2 * math.pi * t / period), 0.5]] * n
    )
    clean = torch.zeros(n, dtype=torch.bool)
    for t in range(200):                       # fill the buffer with clean history
        ext.observe(signal(t), clean)
    masked = torch.ones(n, dtype=torch.bool)
    held = signal(199)
    errs = []
    for s in range(1, 26):                     # a 0.5 s outage at 50 Hz; t0 = 199
        ext.observe(signal(199 + s), masked)   # timeline advances; the emitted value is stored
        pred = ext(held, torch.full((n,), s))
        errs.append(float((pred - signal(199 + s)).abs().max()))
    assert max(errs) < 0.05                    # cycle continuation tracks the true signal
    assert float((held - signal(224)).abs().max()) > 0.5   # holding would not have
    assert ext.lag.tolist() == [period, period]            # it found the true period


def test_cycle_buffer_stays_contiguous_in_time_under_intermittent_dropout():
    """A lag in samples must equal a lag in ticks even when 30 % of ticks are masked."""
    from snmr.integration.fusion import CycleContinuationExtrapolator

    period = 30
    ext = CycleContinuationExtrapolator(1, 1, min_lag=25, max_lag=40, match_ticks=10)
    sig = lambda t: torch.tensor([[math.sin(2 * math.pi * t / period)]])  # noqa: E731
    rng = torch.Generator().manual_seed(0)
    t = 0
    for _ in range(400):                       # burn-in with intermittent masking
        masked = torch.rand(1, generator=rng) < 0.3
        ext.observe(sig(t), masked)
        if bool(masked):
            ext(sig(t), torch.tensor([1]))
        t += 1
    errs = []
    for s in range(1, 21):
        ext.observe(sig(t), torch.ones(1, dtype=torch.bool))
        errs.append(float((ext(sig(t - s), torch.tensor([s])) - sig(t)).abs().max()))
        t += 1
    assert max(errs) < 0.2                     # phase stays locked despite the punched holes


def test_cycle_continuation_falls_back_to_hold_without_enough_history():
    from snmr.integration.fusion import CycleContinuationExtrapolator

    ext = CycleContinuationExtrapolator(2, 3, min_lag=25, max_lag=40, match_ticks=10)
    held = torch.full((2, 3), 7.0)
    out = ext(held, torch.tensor([1, 1]))
    assert torch.equal(out, held) and ext.fallbacks == 2


def test_cycle_continuation_reset_clears_history_for_named_envs():
    from snmr.integration.fusion import CycleContinuationExtrapolator

    ext = CycleContinuationExtrapolator(3, 2, min_lag=5, max_lag=8, match_ticks=2)
    clean = torch.zeros(3, dtype=torch.bool)
    for t in range(60):
        ext.observe(torch.full((3, 2), float(t % 7)), clean)  # noqa: E501
    assert (ext.valid >= ext.max_lag + ext.match_ticks).all()
    ext.reset(torch.tensor([1]))
    assert ext.valid[1] == 0 and ext.valid[0] > 0
