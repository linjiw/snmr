"""Deployment wrappers for exclusive SNMR command students.

These modules contain no simulator or ONNX dependency.  The export entry point in
``scripts/export_e70_policy_onnx.py`` supplies frozen normalizers and motion data.
"""

from __future__ import annotations

import torch

from snmr.integration.distillation import CommandStudent


def action_bounds_from_joint_limits(
    default_joint_pos: torch.Tensor,
    action_scale: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    *,
    limit_fraction: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert centered joint limits into bounds on residual policy actions."""
    tensors = (default_joint_pos, action_scale, joint_lower, joint_upper)
    if any(value.ndim != 1 for value in tensors) or len({value.shape for value in tensors}) != 1:
        raise ValueError("deployment limit vectors must have one matching dimension")
    if not 0.0 < limit_fraction <= 1.0:
        raise ValueError("deployment limit fraction must be in (0, 1]")
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("deployment limit vectors must be finite")
    if torch.any(action_scale <= 0) or torch.any(joint_lower >= joint_upper):
        raise ValueError("deployment scales and joint limits are invalid")

    center = 0.5 * (joint_lower + joint_upper)
    half_range = 0.5 * (joint_upper - joint_lower) * limit_fraction
    safe_lower = center - half_range
    safe_upper = center + half_range
    if torch.any(default_joint_pos < safe_lower) or torch.any(default_joint_pos > safe_upper):
        raise ValueError("default joint pose lies outside the deployment safety envelope")
    return (
        ((safe_lower - default_joint_pos) / action_scale).reshape(1, -1),
        ((safe_upper - default_joint_pos) / action_scale).reshape(1, -1),
    )


def _validated_action_bounds(
    num_actions: int,
    action_lower: torch.Tensor | None,
    action_upper: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action_lower is None and action_upper is None:
        return torch.full((1, num_actions), -torch.inf), torch.full(
            (1, num_actions), torch.inf
        )
    if action_lower is None or action_upper is None:
        raise ValueError("both deployment action bounds are required")
    if action_lower.shape != (1, num_actions) or action_upper.shape != (1, num_actions):
        raise ValueError(f"deployment action bounds must have shape (1,{num_actions})")
    if not torch.isfinite(action_lower).all() or not torch.isfinite(action_upper).all():
        raise ValueError("deployment action bounds must be finite")
    if torch.any(action_lower >= action_upper):
        raise ValueError("deployment action lower bounds must precede upper bounds")
    return action_lower.float(), action_upper.float()


class OfflineSnmrMotionPolicy(torch.nn.Module):
    """Make an SNMR command student match Holosoma's offline WBT ONNX contract.

    The real-time input is the raw 154-d WBT observation and a motion time step.
    The wrapper embeds the precomputed SNMR trajectory and reference outputs used by
    Holosoma inference.  It is suitable for a fixed, preplanned motion clip.  A live
    teleoperation system needs a separately validated online latent producer.
    """

    def __init__(
        self,
        student: CommandStudent,
        *,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        latent_mean: torch.Tensor,
        latent_std: torch.Tensor,
        latent_z: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        ref_quat_xyzw: torch.Tensor,
        proprio_dim: int = 90,
        future_offset: int = 5,
        observation_eps: float = 1.0e-2,
        action_lower: torch.Tensor | None = None,
        action_upper: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if student.prior_goal != "snmr":
            raise ValueError("offline SNMR wrapper requires a snmr-only prior")
        if observation_mean.shape != observation_std.shape:
            raise ValueError("observation mean/std shapes differ")
        if observation_mean.ndim != 2 or observation_mean.shape[0] != 1:
            raise ValueError("observation statistics must have shape (1,D)")
        if observation_mean.shape[1] < proprio_dim:
            raise ValueError("observation statistics do not cover proprioception")
        if latent_mean.shape != latent_std.shape or latent_mean.ndim != 2:
            raise ValueError("latent statistics must have matching shape (1,D)")
        if latent_z.ndim != 2 or latent_z.shape[1] != latent_mean.shape[1]:
            raise ValueError("latent trajectory is incompatible with latent statistics")
        lengths = {latent_z.shape[0], joint_pos.shape[0], joint_vel.shape[0], ref_quat_xyzw.shape[0]}
        if len(lengths) != 1:
            raise ValueError(f"motion channels have different lengths: {sorted(lengths)}")
        if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
            raise ValueError("joint position/velocity must have matching (T,J) shapes")
        if ref_quat_xyzw.shape != (latent_z.shape[0], 4):
            raise ValueError("reference quaternion must have shape (T,4)")
        if future_offset < 0:
            raise ValueError("future_offset must be nonnegative")

        self.student = student
        self.proprio_dim = proprio_dim
        self.future_offset = future_offset
        self.observation_eps = observation_eps
        lower, upper = _validated_action_bounds(joint_pos.shape[1], action_lower, action_upper)
        self.register_buffer("observation_mean", observation_mean.float())
        self.register_buffer("observation_std", observation_std.float())
        self.register_buffer("latent_mean", latent_mean.float())
        self.register_buffer("latent_std", latent_std.float())
        self.register_buffer("latent_z", latent_z.float())
        self.register_buffer("joint_pos", joint_pos.float())
        self.register_buffer("joint_vel", joint_vel.float())
        self.register_buffer("ref_quat_xyzw", ref_quat_xyzw.float())
        self.register_buffer("action_lower", lower)
        self.register_buffer("action_upper", upper)

    def forward(
        self, observation: torch.Tensor, time_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing() and (
            observation.ndim != 2
            or observation.shape[1] != self.observation_mean.shape[1]
        ):
            raise ValueError(
                f"expected raw observation shape (N,{self.observation_mean.shape[1]})"
            )
        index = time_step.long().reshape(-1).clamp(0, self.latent_z.shape[0] - 1)
        future_index = (index + self.future_offset).clamp(max=self.latent_z.shape[0] - 1)
        proprio = (
            observation[:, : self.proprio_dim]
            - self.observation_mean[:, : self.proprio_dim]
        ) / (self.observation_std[:, : self.proprio_dim] + self.observation_eps)
        latent_now = (self.latent_z[index] - self.latent_mean) / self.latent_std
        latent_future = (self.latent_z[future_index] - self.latent_mean) / self.latent_std
        latent_window = torch.cat((latent_now, latent_future), dim=-1)
        # The snmr-only prior ignores the explicit-goal argument by construction.
        unused_goal = observation[:, :0]
        z_cmd = self.student.mu_prior(proprio, latent_window, unused_goal)
        actions = self.student.act(proprio, z_cmd)
        actions = torch.maximum(torch.minimum(actions, self.action_upper), self.action_lower)
        return (
            actions,
            self.joint_pos[index],
            self.joint_vel[index],
            self.ref_quat_xyzw[index],
        )


class OfflineExplicitMotionPolicy(torch.nn.Module):
    """Export the high-performing explicit-goal control to the WBT runtime."""

    def __init__(
        self,
        student: CommandStudent,
        *,
        observation_mean: torch.Tensor,
        observation_std: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        ref_quat_xyzw: torch.Tensor,
        proprio_dim: int = 90,
        observation_eps: float = 1.0e-2,
        action_lower: torch.Tensor | None = None,
        action_upper: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if student.prior_goal != "explicit":
            raise ValueError("offline explicit wrapper requires an explicit-only prior")
        if observation_mean.shape != observation_std.shape:
            raise ValueError("observation mean/std shapes differ")
        if observation_mean.ndim != 2 or observation_mean.shape[0] != 1:
            raise ValueError("observation statistics must have shape (1,D)")
        lengths = {joint_pos.shape[0], joint_vel.shape[0], ref_quat_xyzw.shape[0]}
        if len(lengths) != 1:
            raise ValueError(f"motion channels have different lengths: {sorted(lengths)}")
        if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
            raise ValueError("joint position/velocity must have matching (T,J) shapes")
        if ref_quat_xyzw.shape != (joint_pos.shape[0], 4):
            raise ValueError("reference quaternion must have shape (T,4)")
        self.student = student
        self.proprio_dim = proprio_dim
        self.observation_eps = observation_eps
        lower, upper = _validated_action_bounds(joint_pos.shape[1], action_lower, action_upper)
        self.register_buffer("observation_mean", observation_mean.float())
        self.register_buffer("observation_std", observation_std.float())
        self.register_buffer("joint_pos", joint_pos.float())
        self.register_buffer("joint_vel", joint_vel.float())
        self.register_buffer("ref_quat_xyzw", ref_quat_xyzw.float())
        self.register_buffer("action_lower", lower)
        self.register_buffer("action_upper", upper)

    def forward(
        self, observation: torch.Tensor, time_step: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing() and (
            observation.ndim != 2
            or observation.shape[1] != self.observation_mean.shape[1]
        ):
            raise ValueError(
                f"expected raw observation shape (N,{self.observation_mean.shape[1]})"
            )
        index = time_step.long().reshape(-1).clamp(0, self.joint_pos.shape[0] - 1)
        normalized = (observation - self.observation_mean) / (
            self.observation_std + self.observation_eps
        )
        proprio = normalized[:, : self.proprio_dim]
        explicit_goal = normalized[:, self.proprio_dim :]
        unused_latent = observation[:, :0]
        z_cmd = self.student.mu_prior(proprio, unused_latent, explicit_goal)
        actions = self.student.act(proprio, z_cmd)
        actions = torch.maximum(torch.minimum(actions, self.action_upper), self.action_lower)
        return (
            actions,
            self.joint_pos[index],
            self.joint_vel[index],
            self.ref_quat_xyzw[index],
        )
