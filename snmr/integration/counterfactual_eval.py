"""Counterfactual evaluation utilities for independent state and command starts.

The frozen E70 evaluator couples the physical reset state, commanded reference, teacher
route, and latent cursor through ``MotionCommand.time_steps``.  A target-specific command
intervention needs two cursors instead:

* ``state_start_steps`` initializes the simulated robot; and
* ``command_start_steps`` drives the reference, termination tube, teacher, and latent.

This module is additive.  It does not modify the frozen :mod:`wbt_bodyfix` source or any
recorded E70 artifact.  Call :func:`patch` after ``wbt_bodyfix.patch()`` and before environment
construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


# The complete command-independent state contract for the pinned MJWarp/G1 runtime.
# Keeping this list in the simulator-independent utility module lets the freeze-manifest
# generator bind it before a GPU environment is constructed.  The evaluator fails closed
# if the pinned backend exposes a different contract during smoke testing.
E71_FULL_STATE_TENSOR_NAMES = (
    "action",
    "action_term.joint_control.actions_after_delay",
    "action_term.joint_control.kd_scale",
    "action_term.joint_control.kp_scale",
    "action_term.joint_control.prev_dof_vel",
    "action_term.joint_control.processed_actions",
    "action_term.joint_control.raw_actions",
    "action_term.joint_control.rfi_lim_scale",
    "action_term.joint_control.torques",
    "contact_forces_history",
    "default_dof_pos",
    "dof_pos",
    "dof_vel",
    "integration_act",
    "integration_act_dot",
    "integration_cfrc",
    "integration_ctrl",
    "integration_eq_active",
    "integration_history",
    "integration_mocap_pos",
    "integration_mocap_quat",
    "integration_overflow",
    "integration_qacc",
    "integration_qacc_warmstart",
    "integration_qfrc_applied",
    "integration_qpos_local",
    "integration_qvel",
    "integration_time",
    "integration_tree_asleep",
    "integration_tree_awake",
    "integration_userdata",
    "integration_xfrc_applied",
    "previous_action",
    "push_robot_vel_buf",
    "record_push_robot_vel_buf",
    "rigid_body_ang_vel",
    "rigid_body_pos_local",
    "rigid_body_rot",
    "rigid_body_vel",
    "root_state_local",
)

E71_RUNTIME_CONTRACT = {
    "environment_class": "holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    "simulator_target": "holosoma.simulator.mujoco.mujoco.MuJoCo",
    "simulator_backend": "warp",
    "simulator_backend_class": (
        "holosoma.simulator.mujoco.backends.warp_backend.WarpBackend"
    ),
    "device": "cuda:0",
    "num_dof": 29,
    "num_actions": 29,
    "physics_fps": 200,
    "control_decimation": 4,
    "policy_dt_s": 0.02,
    "max_episode_length_s": 20.0,
    "terrain_mesh_type": "plane",
    "control_type": "P",
    "clip_actions": True,
    "action_clip_value": 100.0,
    "clip_torques": True,
    "action_scale": 0.25,
    "action_scales_by_effort_limit_over_p_gain": True,
    "action_manager_contract": {
        "joint_control": {
            "func": (
                "holosoma.managers.action.terms.joint_control:"
                "JointPositionActionTerm"
            ),
            "scale": 1.0,
            "clip": None,
            "params": {},
        }
    },
    "actor_observation_contract": {
        "concatenate": True,
        "history_length": 1,
        "clip_observations": 100.0,
        "terms": {
            "actions": {
                "func": "holosoma.managers.observation.terms.wbt:actions",
                "scale": 1.0,
                "noise": 0.0,
                "clip": None,
                "params": {},
            },
            "base_ang_vel": {
                "func": "holosoma.managers.observation.terms.wbt:base_ang_vel",
                "scale": 1.0,
                "noise": 0.2,
                "clip": None,
                "params": {},
            },
            "dof_pos": {
                "func": "holosoma.managers.observation.terms.wbt:dof_pos",
                "scale": 1.0,
                "noise": 0.01,
                "clip": None,
                "params": {},
            },
            "dof_vel": {
                "func": "holosoma.managers.observation.terms.wbt:dof_vel",
                "scale": 1.0,
                "noise": 0.5,
                "clip": None,
                "params": {},
            },
            "motion_command": {
                "func": "holosoma.managers.observation.terms.wbt:motion_command",
                "scale": 1.0,
                "noise": 0.0,
                "clip": None,
                "params": {},
            },
            "motion_ref_ori_b": {
                "func": "holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
                "scale": 1.0,
                "noise": 0.05,
                "clip": None,
                "params": {},
            },
        },
    },
    "termination_terms": {
        "bad_tracking": {
            "func": "holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            "is_timeout": False,
            "params": {
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                "body_names_to_track": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        },
        "timeout": {
            "func": "holosoma.managers.termination.terms.common:timeout_exceeded",
            "is_timeout": True,
            "params": {},
        },
    },
}

E71_EVALUATION_CONDITIONS = {
    "actor_observation_noise": False,
    "all_observation_noise": False,
    "startup_physics_randomization": False,
    "reset_state_randomization": False,
    "external_pushes": False,
    "action_delay": False,
    "pd_gain_randomization": False,
    "torque_rfi": False,
    "initial_pose_noise_scale": 0.0,
    "tracking_termination_suppressed_steps": 50,
    "future_only_branch_samples": True,
    "reference_termination_after_primary_horizon": True,
    "nonfinite_state_guard": True,
    "mujoco_warp_overflow_guard": True,
    "warmup_callback_order_override": True,
    "rollout_task_update_before_termination": False,
    "torch_deterministic": True,
    "adaptive_timestep_sampler": False,
    "terrain_spawn_randomization": False,
    "expanded_mujoco_model_fields": [],
    "default_dof_pose_equal": True,
    "runtime_contract": E71_RUNTIME_CONTRACT,
    "passed": True,
}


@dataclass(frozen=True)
class CounterfactualGrid:
    """The four ``state side x command side`` cells for each ambiguity pair."""

    pair_ids: torch.Tensor
    state_sides: torch.Tensor
    command_sides: torch.Tensor
    state_start_steps: torch.Tensor
    command_start_steps: torch.Tensor

    @property
    def num_cells(self) -> int:
        return int(self.pair_ids.numel())


def branch_start_steps_for_grid(
    grid: CounterfactualGrid,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two reference-future starts aligned to every four-cell row."""

    starts_by_pair: dict[tuple[int, int], int] = {}
    for pair_id in torch.unique(grid.pair_ids, sorted=True).tolist():
        for side in (0, 1):
            mask = (grid.pair_ids == pair_id) & (grid.state_sides == side)
            values = torch.unique(grid.state_start_steps[mask])
            if values.numel() != 1:
                raise ValueError("each pair and state side must map to one reference start")
            starts_by_pair[(int(pair_id), side)] = int(values[0])
    first = torch.empty_like(grid.state_start_steps)
    second = torch.empty_like(grid.state_start_steps)
    for row, pair_id in enumerate(grid.pair_ids.tolist()):
        first[row] = starts_by_pair[(int(pair_id), 0)]
        second[row] = starts_by_pair[(int(pair_id), 1)]
    return first, second


def pooled_goal_scale(joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    """Per-dimension pooled scale for the registered joint-position/velocity goal."""

    if joint_pos.ndim != 2 or joint_vel.ndim != 2:
        raise ValueError("joint position and velocity references must be matrices")
    if joint_pos.shape != joint_vel.shape:
        raise ValueError("joint position and velocity references must have identical shape")
    if not torch.isfinite(joint_pos).all() or not torch.isfinite(joint_vel).all():
        raise ValueError("reference goals contain nonfinite values")
    goal = torch.cat((joint_pos, joint_vel), dim=-1)
    scale = goal.std(dim=0, unbiased=False)
    return torch.where(scale < 1.0e-6, torch.ones_like(scale), scale)


def standardized_branch_squared_error(
    observed_goal: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-rollout mean squared standardized errors to two reference branches."""

    if observed_goal.ndim != 2:
        raise ValueError("observed goals must have shape (rollouts, dimensions)")
    if reference_a.shape != observed_goal.shape or reference_b.shape != observed_goal.shape:
        raise ValueError("observed and reference goals must have identical shape")
    if scale.shape != (observed_goal.shape[1],):
        raise ValueError("goal scale must have one entry per dimension")
    if torch.any(scale <= 0) or not all(
        torch.isfinite(value).all()
        for value in (observed_goal, reference_a, reference_b, scale)
    ):
        raise ValueError("branch distance inputs must be finite with positive scale")
    error_a = ((observed_goal - reference_a) / scale).square().mean(dim=-1)
    error_b = ((observed_goal - reference_b) / scale).square().mean(dim=-1)
    return error_a, error_b


def normalized_branch_coordinate(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    q_ab: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Project rollout error onto the standardized A--B branch axis.

    ``q_a`` and ``q_b`` are mean squared standardized rollout errors to the two
    references, and ``q_ab`` is their mean squared standardized separation over the
    same samples.  A perfect A reference has coordinate -1 and a perfect B reference
    has coordinate +1.  The squared-distance difference cancels error orthogonal to
    the branch axis, while division by ``q_ab`` prevents more-divergent pairs from
    receiving a mechanically larger ideal effect.
    """

    if q_a.shape != q_b.shape or q_a.shape != q_ab.shape:
        raise ValueError("branch-error tensors must have identical shapes")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not all(torch.isfinite(value).all() for value in (q_a, q_b, q_ab)):
        raise ValueError("branch-error tensors must be finite")
    if torch.any(q_a < 0) or torch.any(q_b < 0) or torch.any(q_ab < 0):
        raise ValueError("squared branch errors and separation must be nonnegative")
    return (q_a - q_b) / (q_ab + epsilon)


def _as_long_tensor(values: Any, *, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=torch.long)


def motion_ids_for_steps(steps: torch.Tensor, motion_ends: torch.Tensor) -> torch.Tensor:
    """Map concatenated global frame indices to motion IDs."""

    steps = _as_long_tensor(steps, device=motion_ends.device)
    motion_ends = _as_long_tensor(motion_ends, device=steps.device)
    if motion_ends.ndim != 1 or not len(motion_ends):
        raise ValueError("motion_ends must be a nonempty vector")
    motion_ids = torch.searchsorted(motion_ends, steps, right=True)
    if torch.any(motion_ids >= len(motion_ends)):
        raise ValueError("a frame maps beyond the final motion")
    return motion_ids


def validate_start_steps(
    steps: torch.Tensor,
    motion_starts: torch.Tensor,
    motion_ends: torch.Tensor,
    *,
    horizon_steps: int,
    label: str,
) -> torch.Tensor:
    """Validate interior global frames and return their motion IDs.

    The lower-bound requirement matches ``wbt_bodyfix``: reset initializes the preceding
    frame and the environment's warm-up step advances to the requested frame.
    """

    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    steps = _as_long_tensor(steps, device=motion_ends.device)
    motion_starts = _as_long_tensor(motion_starts, device=steps.device)
    motion_ends = _as_long_tensor(motion_ends, device=steps.device)
    if steps.ndim != 1:
        raise ValueError(f"{label} starts must be a vector")
    if motion_starts.shape != motion_ends.shape or motion_starts.ndim != 1:
        raise ValueError("motion boundaries must be aligned vectors")
    if int(motion_starts[0]) != 0 or torch.any(motion_ends <= motion_starts):
        raise ValueError("motion boundaries must begin at zero and have positive lengths")
    if len(motion_starts) > 1 and not torch.equal(motion_starts[1:], motion_ends[:-1]):
        raise ValueError("motion boundaries must describe contiguous concatenated clips")
    motion_ids = motion_ids_for_steps(steps, motion_ends)
    lower = motion_starts[motion_ids]
    upper = motion_ends[motion_ids]
    if torch.any(steps <= lower):
        raise ValueError(f"{label} starts must have a preceding in-clip frame")
    if torch.any(steps + horizon_steps >= upper):
        raise ValueError(f"{label} starts do not support the registered horizon")
    return motion_ids


def build_four_cell_grid(
    pair_report: dict[str, Any],
    *,
    clip_names: Sequence[str],
    motion_starts: torch.Tensor,
    motion_ends: torch.Tensor,
    fps: float,
    horizon_steps: int,
    device: torch.device | str | None = None,
) -> CounterfactualGrid:
    """Build one deterministic four-cell evaluation grid per frozen ambiguity pair."""

    if list(pair_report.get("clips", [])) != list(clip_names):
        raise ValueError("ambiguity pair clips do not match the loaded motion order")
    windows = pair_report.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("ambiguity pair report contains no windows")
    if fps <= 0:
        raise ValueError("fps must be positive")
    motion_starts = _as_long_tensor(motion_starts, device=device)
    motion_ends = _as_long_tensor(motion_ends, device=device)
    if len(motion_starts) != 2:
        raise ValueError("the command-swap assay requires exactly two motions")

    pair_ids: list[int] = []
    state_sides: list[int] = []
    command_sides: list[int] = []
    state_steps: list[int] = []
    command_steps: list[int] = []
    for pair_id, window in enumerate(windows):
        local = [
            int(round(float(window["time_seconds_first"]) * fps)),
            int(round(float(window["time_seconds_second"]) * fps)),
        ]
        global_steps = [int(motion_starts[side]) + local[side] for side in (0, 1)]
        for state_side in (0, 1):
            for command_side in (0, 1):
                pair_ids.append(pair_id)
                state_sides.append(state_side)
                command_sides.append(command_side)
                state_steps.append(global_steps[state_side])
                command_steps.append(global_steps[command_side])

    grid = CounterfactualGrid(
        pair_ids=_as_long_tensor(pair_ids, device=device),
        state_sides=_as_long_tensor(state_sides, device=device),
        command_sides=_as_long_tensor(command_sides, device=device),
        state_start_steps=_as_long_tensor(state_steps, device=device),
        command_start_steps=_as_long_tensor(command_steps, device=device),
    )
    state_motion_ids = validate_start_steps(
        grid.state_start_steps,
        motion_starts,
        motion_ends,
        horizon_steps=horizon_steps,
        label="state",
    )
    command_motion_ids = validate_start_steps(
        grid.command_start_steps,
        motion_starts,
        motion_ends,
        horizon_steps=horizon_steps,
        label="command",
    )
    if not torch.equal(state_motion_ids, grid.state_sides):
        raise ValueError("state frames do not map to their declared sides")
    if not torch.equal(command_motion_ids, grid.command_sides):
        raise ValueError("command frames do not map to their declared sides")
    return grid


def audit_same_state_proprio(
    proprio: torch.Tensor,
    grid: CounterfactualGrid,
    *,
    tolerance: float = 1.0e-6,
) -> dict[str, float | int | bool]:
    """Fail-closed check that changing command side leaves initial proprioception fixed."""

    if proprio.ndim != 2 or proprio.shape[0] != grid.num_cells:
        raise ValueError("proprioception must align with the four-cell grid")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    differences: list[torch.Tensor] = []
    for pair_id in torch.unique(grid.pair_ids, sorted=True).tolist():
        for state_side in (0, 1):
            rows = []
            for command_side in (0, 1):
                mask = (
                    (grid.pair_ids == pair_id)
                    & (grid.state_sides == state_side)
                    & (grid.command_sides == command_side)
                )
                indexes = torch.nonzero(mask, as_tuple=False).flatten()
                if indexes.numel() != 1:
                    raise ValueError("each pair/state must contain exactly one cell per command")
                rows.append(proprio[indexes[0]])
            differences.append((rows[0] - rows[1]).abs().max())
    stacked = torch.stack(differences)
    max_abs = float(stacked.max())
    return {
        "num_state_comparisons": len(differences),
        "max_abs_difference": max_abs,
        "tolerance": float(tolerance),
        "passed": bool(max_abs <= tolerance),
    }


def audit_same_state_tensors(
    tensors: Mapping[str, torch.Tensor],
    grid: CounterfactualGrid,
    *,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare complete command-independent state tensors across each command swap.

    Every tensor must have the four-cell environment dimension first.  Empty trailing
    dimensions are allowed and recorded as exact by construction.  Callers are responsible
    for expressing positions in a common local frame before this audit.
    """

    if not tensors:
        raise ValueError("at least one same-state tensor is required")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    per_tensor: dict[str, dict[str, float | int | bool]] = {}
    overall = 0.0
    num_comparisons = 0
    for name, tensor in tensors.items():
        if not name or tensor.ndim < 1 or tensor.shape[0] != grid.num_cells:
            raise ValueError(f"state tensor {name!r} does not align with the four-cell grid")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"state tensor {name!r} contains nonfinite values")
        differences: list[torch.Tensor] = []
        for pair_id in torch.unique(grid.pair_ids, sorted=True).tolist():
            for state_side in (0, 1):
                rows = []
                for command_side in (0, 1):
                    mask = (
                        (grid.pair_ids == pair_id)
                        & (grid.state_sides == state_side)
                        & (grid.command_sides == command_side)
                    )
                    indexes = torch.nonzero(mask, as_tuple=False).flatten()
                    if indexes.numel() != 1:
                        raise ValueError(
                            "each pair/state must contain exactly one cell per command"
                        )
                    rows.append(tensor[indexes[0]])
                if rows[0].numel():
                    if rows[0].is_floating_point() or rows[0].is_complex():
                        differences.append((rows[0] - rows[1]).abs().max())
                    else:
                        differences.append(
                            torch.any(rows[0] != rows[1]).to(dtype=torch.float32)
                        )
                else:
                    differences.append(torch.zeros((), device=tensor.device))
        maximum = float(torch.stack(differences).max())
        overall = max(overall, maximum)
        num_comparisons += len(differences)
        per_tensor[name] = {
            "dtype": str(tensor.dtype),
            "num_state_comparisons": len(differences),
            "max_abs_difference": maximum,
            "passed": bool(maximum <= tolerance),
        }
    return {
        "num_tensors": len(tensors),
        "num_tensor_state_comparisons": num_comparisons,
        "max_abs_difference": overall,
        "tolerance": float(tolerance),
        "per_tensor": per_tensor,
        "passed": bool(overall <= tolerance),
    }


# Body slot of ``motion.body_*_w`` that the frozen E70 reset writes into
# ``simulator.robot_root_states``.
#
# Provenance, re-verified 2026-08-14 against the pinned Holosoma working tree
# (``/home/robotixx/holosoma``, clean at ``20699ff``, 2026-04-17 -- i.e. the exact source
# every frozen E70 student was trained and evaluated under):
#
#   ``wbt_bodyfix.reset`` (lines 119-122, the frozen reset) writes ``self.root_pos_w`` /
#   ``root_quat_w`` / ``root_lin_vel_w`` / ``root_ang_vel_w``.  Those four Holosoma
#   ``MotionCommand`` properties live at ``managers/command/terms/wbt.py``:876-890 and each
#   reads ``self.motion.body_*_w[self.time_steps, 0]`` -- a literal ``0``.
#
# ``ref_body_index`` (``torso_link`` for G1, ``config_values/wbt/g1/command.py``:35) belongs to
# the *``ref_*``* family one block earlier, ``wbt.py``:860-874.  That family drives the tracking
# reference, the termination tube, and the ``motion_ref_ori_b`` observation -- it is never read
# by any reset path.  Confusing ``wbt.py``:862 (``ref_pos_w``) for ``wbt.py``:877
# (``root_pos_w``) is the trap this constant exists to close.
#
# Slot 0 is also the only physically admissible choice: ``MotionLoader.body_pos_w`` re-orders
# the raw npz bodies into simulator order via ``_body_indexes``
# (``wbt.py``:46/167), so slot 0 is ``simulator._body_list[0]`` -- ``pelvis``, the free-joint
# body that *is* ``robot_root_states``.  (The raw npz for these clips carries 51 bodies with
# ``world`` at 0, ``pelvis`` at 1 and ``torso_link`` at 29; those raw indices never reach this
# code.)  Writing the torso pose into the pelvis free joint would both teleport the robot and
# diverge from every frozen E70 rollout.
#
# The value is named rather than inlined, and ``assert_frozen_root_convention`` re-derives the
# equality against the live ``root_*_w`` properties on every evaluation reset, so an upstream
# change to that family fails closed instead of silently re-initializing E71 off-convention.
FROZEN_ROOT_BODY_INDEX = 0

_ROOT_STATE_FIELDS = ("pos", "quat", "lin_vel", "ang_vel")


def frozen_root_state(
    motion: Any, steps: torch.Tensor, origins: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(pos, quat, lin_vel, ang_vel)`` exactly as the frozen E70 reset computes them.

    ``origins`` is added to the position only; the frozen ``root_*_w`` properties leave
    orientation and both velocities in the raw motion frame (``wbt.py``:877-890).
    """

    index = FROZEN_ROOT_BODY_INDEX
    return (
        motion.body_pos_w[steps, index] + origins,
        motion.body_quat_w[steps, index],
        motion.body_lin_vel_w[steps, index],
        motion.body_ang_vel_w[steps, index],
    )


def assert_frozen_root_convention(command: Any, env_ids: torch.Tensor) -> None:
    """Fail closed unless :func:`frozen_root_state` reproduces the frozen reset bit for bit.

    Evaluated at the *command* cursor -- the only cursor for which the Holosoma ``root_*_w``
    properties are defined -- this is a pure identity: it compares two ways of spelling the
    same frozen expression and is independent of the E71 state/command split.  It therefore
    holds on every reset, diagonal or crossed, and detects any upstream redefinition of the
    ``root_*_w`` family (for example a switch to ``ref_body_index``) before a single rollout
    is recorded.
    """

    steps = command.time_steps[env_ids]
    origins = command._env.simulator.scene.env_origins[env_ids]
    expected = frozen_root_state(command.motion, steps, origins)
    observed = (
        command.root_pos_w[env_ids],
        command.root_quat_w[env_ids],
        command.root_lin_vel_w[env_ids],
        command.root_ang_vel_w[env_ids],
    )
    for field, want, got in zip(_ROOT_STATE_FIELDS, expected, observed):
        if want.shape != got.shape:
            raise RuntimeError(
                f"frozen root convention broken: root_{field}_w has shape "
                f"{tuple(got.shape)}, expected {tuple(want.shape)}"
            )
        if not torch.equal(want, got):
            raise RuntimeError(
                f"frozen root convention broken: motion body slot {FROZEN_ROOT_BODY_INDEX} no "
                f"longer reproduces root_{field}_w (max abs difference "
                f"{float((want - got).abs().max())}); the frozen E70 reset and the E71 "
                "counterfactual reset would initialize different bodies"
            )


def _write_reference_state(command: Any, env_ids: torch.Tensor, state_steps: torch.Tensor) -> None:
    """Overwrite simulator state from reference frames without changing command cursors."""

    motion = command.motion
    sim = command._env.simulator
    origins = sim.scene.env_origins[env_ids]
    sim.dof_pos[env_ids] = motion.joint_pos[state_steps]
    sim.dof_vel[env_ids] = motion.joint_vel[state_steps]
    root_pos, root_quat, root_lin_vel, root_ang_vel = frozen_root_state(
        motion, state_steps, origins
    )
    sim.robot_root_states[env_ids, :3] = root_pos
    sim.robot_root_states[env_ids, 3:7] = root_quat
    sim.robot_root_states[env_ids, 7:10] = root_lin_vel
    sim.robot_root_states[env_ids, 10:13] = root_ang_vel

    if motion.has_object:
        object_vel = motion.object_lin_vel_w[state_steps]
        object_states = torch.cat(
            (
                motion.object_pos_w[state_steps] + origins,
                motion.object_quat_w[state_steps],
                object_vel,
                torch.zeros_like(object_vel),
            ),
            dim=-1,
        )
        sim.set_actor_states([command.object_name], env_ids, object_states)


def patch() -> None:
    """Add independent evaluation state/command starts to Holosoma ``MotionCommand``."""

    from holosoma.managers.command.terms import wbt as cmd_wbt

    MotionCommand = cmd_wbt.MotionCommand
    if getattr(MotionCommand, "_snmr_counterfactual_patched", False):
        return
    if not getattr(MotionCommand, "_snmr_bodyfix_patched", False):
        raise RuntimeError("counterfactual_eval.patch() requires wbt_bodyfix.patch() first")

    base_reset = MotionCommand.reset

    def set_counterfactual_evaluation_start_steps(
        self: Any,
        state_starts: torch.Tensor,
        command_starts: torch.Tensor,
        *,
        horizon_steps: int,
    ) -> None:
        state_starts = _as_long_tensor(state_starts, device=self.device)
        command_starts = _as_long_tensor(command_starts, device=self.device)
        expected = (self.num_envs,)
        if state_starts.shape != expected or command_starts.shape != expected:
            raise ValueError(
                f"counterfactual starts must both have shape {expected}, got "
                f"{tuple(state_starts.shape)} and {tuple(command_starts.shape)}"
            )
        validate_start_steps(
            state_starts,
            self.motion.motion_start_idx,
            self.motion.motion_end_idx,
            horizon_steps=horizon_steps,
            label="state",
        )
        validate_start_steps(
            command_starts,
            self.motion.motion_start_idx,
            self.motion.motion_end_idx,
            horizon_steps=horizon_steps,
            label="command",
        )
        self._snmr_counterfactual_state_steps = state_starts.clone()
        self._snmr_counterfactual_horizon_steps = int(horizon_steps)
        self.set_evaluation_start_steps(command_starts)

    def reset(self: Any, env_ids: Any) -> None:
        base_reset(self, env_ids)
        requested = getattr(self, "_snmr_counterfactual_state_steps", None)
        if requested is None or not self._env.is_evaluating:
            return
        env_ids = self._ensure_index_tensor(env_ids)
        # ``wbt_bodyfix`` and BaseTask use a warm-up transition.  Initialize the physical
        # state one frame before the requested observation, just as the command cursor does.
        state_steps = requested[env_ids] - 1
        # Prove, on the live runtime and before any state is written, that the body slot this
        # module writes into ``robot_root_states`` is the one the frozen E70 reset used.
        assert_frozen_root_convention(self, env_ids)
        _write_reference_state(self, env_ids, state_steps)

    MotionCommand.set_counterfactual_evaluation_start_steps = (
        set_counterfactual_evaluation_start_steps
    )
    MotionCommand.reset = reset
    MotionCommand._snmr_counterfactual_patched = True
