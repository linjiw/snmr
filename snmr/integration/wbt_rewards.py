"""E51: joint-space tracking reward terms for holosoma WBT, injected without clone edits.

Diagnosis (E50-A error decomposition, 2026-07-24): holosoma's g1 WBT reward tracks only 14
body poses + velocities — there is NO joint-space term (`config_values/wbt/g1/reward.py`).
Joint error is flat from iteration ~1k while episode length keeps climbing, 48% of the joint
MSE is a constant bias toward the default pose (hip pitch −0.37 rad compensated by waist
pitch −0.36 rad so torso tracking stays good), and gait amplitude is undershot ~1/3 — the
signature of a reward that cannot see joint detail, not of undertraining.

These terms follow the existing exp-kernel convention in
``holosoma.managers.reward.terms.wbt`` (squared error → ``exp(-err/sigma**2)``) and are
referenced from the reward config by func string, resolved through the same
``_resolve_function`` path as observation terms (reward manager.py:66). Injection of the new
dict keys happens in ``scripts/train_agent_joint_reward.py`` (tyro CLI cannot add dict keys).

Import only inside the WBT env (.venv-wbt).
"""

from __future__ import annotations

import torch

from holosoma.managers.reward.terms.wbt import _get_motion_command_and_assert_type


def motion_joint_pos_error_exp(env, sigma: float) -> torch.Tensor:
    """DeepMimic-style joint position tracking: exp(-mean_sq_err / sigma^2)."""
    command = _get_motion_command_and_assert_type(env)
    error = torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_joint_vel_error_exp(env, sigma: float) -> torch.Tensor:
    """Joint velocity tracking: exp(-mean_sq_err / sigma^2)."""
    command = _get_motion_command_and_assert_type(env)
    error = torch.mean(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
    return torch.exp(-error / sigma**2)
