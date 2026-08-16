"""CPU tests for the E78 masking + fusion primitives (snmr/integration/fusion.py)."""

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
