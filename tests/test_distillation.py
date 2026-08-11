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
