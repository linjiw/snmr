import pytest
import torch

from snmr.integration.distillation import (
    CommandStudent,
    DivergenceGate,
    RoundReplayBuffer,
    destroy_command_code,
    paired_temporal_smoothness,
    route_teacher_actions,
    same_phase_shuffled_latents,
    shared_time_index_latents,
    teacher_mix_probability,
)


def test_paired_smoothness_recomputes_a_dimension_normalized_valid_mean():
    current = torch.tensor([[1.0, 3.0], [20.0, 20.0], [4.0, 8.0]], requires_grad=True)
    previous = torch.tensor([[0.0, 1.0], [0.0, 0.0], [2.0, 4.0]], requires_grad=True)
    valid = torch.tensor([1.0, 0.0, 1.0])
    loss = paired_temporal_smoothness(current, previous, valid)
    assert loss.item() == pytest.approx(((1 + 4) / 2 + (4 + 16) / 2) / 2)
    loss.backward()
    assert current.grad is not None and previous.grad is not None
    assert torch.equal(current.grad[1], torch.zeros(2))


def test_teacher_mix_keeps_floor_after_annealing():
    assert teacher_mix_probability(0, anneal_rounds=200, floor=0.1) == 1.0
    assert teacher_mix_probability(100, anneal_rounds=200, floor=0.1) == 0.5
    assert teacher_mix_probability(200, anneal_rounds=200, floor=0.1) == 0.1
    assert teacher_mix_probability(10_000, anneal_rounds=200, floor=0.1) == 0.1


def test_round_replay_evicts_oldest_round():
    replay = RoundReplayBuffer(max_rounds=2)
    replay.append({"x": torch.tensor([[1.0]])})
    replay.append({"x": torch.tensor([[2.0], [3.0]])})
    replay.append({"x": torch.tensor([[4.0]])})
    assert len(replay) == 2
    assert torch.equal(replay.merged()["x"], torch.tensor([[2.0], [3.0], [4.0]]))


def test_divergence_gate_detects_smoothness_feedback_after_warmup():
    gate = DivergenceGate(smooth_warmup_rounds=2, smooth_multiplier=10.0)
    for round_index, smooth in enumerate((1.0, 2.0, 1.5)):
        assert gate.check(round_index, action=0.1, kl=0.2, smooth=smooth, latent_norm=1.0) is None
    assert gate.check(3, action=0.1, kl=0.2, smooth=20.0, latent_norm=1.0).startswith("smooth_")
    assert gate.check(4, action=float("nan"), kl=0.0, smooth=0.0, latent_norm=0.0) == "nonfinite_action"


def test_same_phase_shuffle_is_cross_clip_and_length_safe():
    first = torch.arange(4.0).unsqueeze(1)
    second = (10.0 + torch.arange(6.0)).unsqueeze(1)
    latents = torch.cat((first, second))
    shuffled = same_phase_shuffled_latents(
        latents, torch.tensor([0, 4]), torch.tensor([4, 10])
    )
    assert shuffled[:4, 0].tolist() == [10.0, 12.0, 13.0, 15.0]
    assert shuffled[4:, 0].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0]


def test_shared_time_index_resets_without_motion_identity_leak():
    code = shared_time_index_latents(
        torch.tensor([0, 4]), torch.tensor([4, 10]), output_dim=8
    )
    assert torch.equal(code[0], code[4])
    assert torch.equal(code[1], code[5])
    assert not torch.equal(code[0], code[1])


def test_motion_id_routes_labels_but_is_not_a_student_input():
    student = CommandStudent(3, 4, 2, "explicit", 5, z_window_dim=6, z_cmd_dim=4)
    assert student.prior[0].in_features == 8  # proprio + explicit goal only
    assert student.decoder[0].in_features == 7  # proprio + command code only

    actions = torch.tensor([[[1.0], [2.0]], [[10.0], [20.0]]])
    routed = route_teacher_actions(actions, torch.tensor([1, 0]))
    assert routed[:, 0].tolist() == [10.0, 2.0]


def test_command_destruction_controls_are_channel_local():
    code = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    assert torch.equal(destroy_command_code(code, "none"), code)
    assert torch.equal(destroy_command_code(code, "zero"), torch.zeros_like(code))
    assert torch.equal(
        destroy_command_code(code, "shuffle"),
        torch.tensor([[7.0, 11.0], [1.0, 2.0], [3.0, 5.0]]),
    )
    randomized = destroy_command_code(code, "marginal_random")
    assert randomized.shape == code.shape and torch.isfinite(randomized).all()


# ---------------------------------------------------------------------------
# Identity-channel probes for the same-phase shuffle control.
#
# `same_phase_shuffled_latents` maps every clip to `(destination + 1) % n`.
# That is a derangement, as its docstring says, but it is also a DETERMINISTIC
# BIJECTION, so the donor's identity determines the destination's identity
# exactly: destination = (donor - 1) mod n.  The arm is therefore a
# phase-matched, misaligned-reference control -- NOT an identity-erasing or
# content-free null.  These tests pin that property so the distinction cannot
# be lost again.  See docs/E70_SHUFFLE_CONTROL_AUDIT_2026-09-10.md.
# ---------------------------------------------------------------------------


def _identity_coded_pool(num_clips: int, length: int = 8):
    """One clip per identity, channel 0 = clip ID, channel 1 = normalized phase."""
    blocks, starts, ends, cursor = [], [], [], 0
    for clip in range(num_clips):
        phase = torch.linspace(0.0, 1.0, length).unsqueeze(1)
        ident = torch.full((length, 1), float(clip))
        blocks.append(torch.cat((ident, phase), dim=1))
        starts.append(cursor)
        ends.append(cursor + length)
        cursor += length
    return torch.cat(blocks), torch.tensor(starts), torch.tensor(ends), length


@pytest.mark.parametrize("num_clips", [2, 3, 5])
def test_same_phase_shuffle_leaks_recoverable_clip_identity(num_clips):
    """The donor ID is a lossless code for the destination ID at every pool size.

    A student that reads the identity channel can invert the fixed map, so a
    drop in this arm's score cannot be attributed to the removal of clip
    identity.  This is not special to the two-clip E70 pool -- adding clips does
    not fix it, because the map stays deterministic.
    """
    latents, starts, ends, length = _identity_coded_pool(num_clips)
    shuffled = same_phase_shuffled_latents(latents, starts, ends)

    recovered = []
    for destination in range(num_clips):
        donor_id = int(round(float(shuffled[int(starts[destination]), 0])))
        recovered.append((donor_id - 1) % num_clips)

    assert recovered == list(range(num_clips)), (
        "destination clip identity is fully recoverable from the donor channel"
    )

    # The phase channel is preserved, which is the property the arm is meant to
    # have; it is the identity channel that also survives.
    for destination in range(num_clips):
        block = shuffled[int(starts[destination]) : int(ends[destination]), 1]
        assert block[0].item() == pytest.approx(0.0)
        assert block[-1].item() == pytest.approx(1.0)


def test_same_phase_shuffle_is_an_involution_for_the_two_clip_e70_pool():
    """With the E70 pool (walk1_subject1, walk1_subject5) the map is a pure swap."""
    latents, starts, ends, _ = _identity_coded_pool(2)
    once = same_phase_shuffled_latents(latents, starts, ends)
    twice = same_phase_shuffled_latents(once, starts, ends)
    assert torch.allclose(twice, latents), "n=2 cyclic shift is self-inverse"
    # And the swap is total: neither clip keeps any of its own frames.
    assert not torch.equal(once[: len(latents) // 2, 0], latents[: len(latents) // 2, 0])


def test_an_identity_independent_donor_must_break_the_donor_to_target_map():
    """Specification for the replacement control.

    A donor draw that is independent of target identity cannot admit a single
    lookup table from donor ID to destination ID.  This test states the property
    the current control fails, so a future implementation has a target to meet.
    """
    num_clips = 4
    latents, starts, ends, _ = _identity_coded_pool(num_clips)
    fixed = same_phase_shuffled_latents(latents, starts, ends)

    # Current control: exactly one donor ever appears per destination.
    donors_per_destination = {
        destination: {int(round(float(fixed[int(starts[destination]), 0])))}
        for destination in range(num_clips)
    }
    assert all(len(d) == 1 for d in donors_per_destination.values())
    # ...and the donor sets are disjoint, i.e. donor ID identifies destination.
    seen = [next(iter(d)) for d in donors_per_destination.values()]
    assert len(set(seen)) == num_clips, "donor ID is a bijective code for destination"
