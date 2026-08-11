import pytest
import torch

from snmr.deployment import (
    OfflineExplicitMotionPolicy,
    OfflineSnmrMotionPolicy,
    action_bounds_from_joint_limits,
)
from snmr.integration.distillation import CommandStudent


def _wrapper(frames=12):
    torch.manual_seed(4)
    student = CommandStudent(
        90, 7, 29, "snmr", 64, z_window_dim=256, z_cmd_dim=64
    ).eval()
    return OfflineSnmrMotionPolicy(
        student,
        observation_mean=torch.zeros(1, 154),
        observation_std=torch.ones(1, 154),
        latent_mean=torch.zeros(1, 128),
        latent_std=torch.ones(1, 128),
        latent_z=torch.arange(frames * 128).reshape(frames, 128).float() / 1000,
        joint_pos=torch.arange(frames * 29).reshape(frames, 29).float(),
        joint_vel=-torch.arange(frames * 29).reshape(frames, 29).float(),
        ref_quat_xyzw=torch.arange(frames * 4).reshape(frames, 4).float(),
    )


def test_offline_policy_outputs_runtime_contract_and_clamps_preview():
    policy = _wrapper()
    obs = torch.zeros(2, 154)
    actions, joint_pos, joint_vel, quat = policy(
        obs, torch.tensor([[2.0], [999.0]])
    )
    assert actions.shape == (2, 29)
    assert torch.equal(joint_pos[:, 0], torch.tensor([58.0, 319.0]))
    assert torch.equal(joint_vel, -joint_pos)
    assert torch.equal(quat[:, 0], torch.tensor([8.0, 44.0]))
    assert torch.isfinite(actions).all()


def test_offline_policy_matches_manual_normalization_and_student_path():
    policy = _wrapper()
    obs = torch.randn(1, 154)
    index = torch.tensor([[3.0]])
    actual = policy(obs, index)[0]
    proprio = obs[:, :90] / 1.01
    zwin = torch.cat((policy.latent_z[3:4], policy.latent_z[8:9]), dim=-1)
    expected = policy.student.act(
        proprio, policy.student.mu_prior(proprio, zwin, obs[:, :0])
    )
    assert torch.allclose(actual, expected)


def test_offline_policy_rejects_non_snmr_or_misaligned_motion():
    student = CommandStudent(90, 7, 29, "none", 64)
    with pytest.raises(ValueError, match="snmr-only"):
        OfflineSnmrMotionPolicy(
            student,
            observation_mean=torch.zeros(1, 154),
            observation_std=torch.ones(1, 154),
            latent_mean=torch.zeros(1, 128),
            latent_std=torch.ones(1, 128),
            latent_z=torch.zeros(3, 128),
            joint_pos=torch.zeros(3, 29),
            joint_vel=torch.zeros(3, 29),
            ref_quat_xyzw=torch.zeros(3, 4),
        )


def test_explicit_policy_uses_normalized_goal_and_runtime_outputs():
    torch.manual_seed(5)
    student = CommandStudent(90, 7, 29, "explicit", 64).eval()
    policy = OfflineExplicitMotionPolicy(
        student,
        observation_mean=torch.zeros(1, 154),
        observation_std=torch.ones(1, 154),
        joint_pos=torch.zeros(4, 29),
        joint_vel=torch.ones(4, 29),
        ref_quat_xyzw=torch.zeros(4, 4),
    )
    observation = torch.randn(1, 154)
    actions, _, velocities, _ = policy(observation, torch.tensor([[2.0]]))
    normalized = observation / 1.01
    proprio, goal = normalized[:, :90], normalized[:, 90:]
    expected = student.act(
        proprio, student.mu_prior(proprio, observation[:, :0], goal)
    )
    assert torch.allclose(actions, expected)
    assert torch.equal(velocities, torch.ones(1, 29))


def test_action_bounds_map_centered_joint_envelope_to_policy_space():
    lower, upper = action_bounds_from_joint_limits(
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.5, 2.0]),
        torch.tensor([-2.0, -3.0]),
        torch.tensor([2.0, 5.0]),
        limit_fraction=0.5,
    )
    assert torch.equal(lower, torch.tensor([[-2.0, -1.0]]))
    assert torch.equal(upper, torch.tensor([[2.0, 1.0]]))


def test_explicit_policy_clamps_actions_to_deployment_envelope():
    torch.manual_seed(5)
    student = CommandStudent(90, 7, 29, "explicit", 64).eval()
    for parameter in student.parameters():
        parameter.data.fill_(2.0)
    lower = torch.full((1, 29), -0.25)
    upper = torch.full((1, 29), 0.5)
    policy = OfflineExplicitMotionPolicy(
        student,
        observation_mean=torch.zeros(1, 154),
        observation_std=torch.ones(1, 154),
        joint_pos=torch.zeros(4, 29),
        joint_vel=torch.ones(4, 29),
        ref_quat_xyzw=torch.zeros(4, 4),
        action_lower=lower,
        action_upper=upper,
    )
    actions = policy(torch.ones(1, 154), torch.zeros(1, 1))[0]
    assert torch.all(actions >= lower)
    assert torch.all(actions <= upper)
