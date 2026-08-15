#!/usr/bin/env python
"""Evaluate frozen E70 students under E71 same-state valid-command swaps.

This evaluator is intentionally separate from the frozen E70 trainer.  For every selected
ambiguity pair it constructs the four physical-state x valid-command cells, holds student
weights and initial proprioception fixed within each state comparison, and measures whether
the rollout moves toward the commanded reference branch.

Required environment variables:

``E52_ARM``
    ``c_prior_explicit`` or ``a_prior_snmr``.
``E52_OUT``
    Frozen E70 student directory containing ``<arm>_student.pt``.
``E52_TEACHER_MANIFEST``
    Frozen two-specialist E70 manifest.
``E71_STARTS_JSON``
    Frozen E70 ambiguity precheck.
``E71_REPORT_PATH``
    New write-once output JSON.
``E71_TRAINING_SEED``
    Frozen student seed (0, 1, or 2).
``E71_FREEZE_MANIFEST``
    Machine-readable, preregistered E71 freeze manifest binding code and inputs.

``E71_SMOKE_PAIR_ID`` may select one pair and four simulator environments.  Smoke reports use
a distinct protocol and are never accepted by the confirmatory analyzer or bundle auditor.
"""

from __future__ import annotations

import glob
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import pathlib
import platform
import subprocess
import sys
import uuid
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import tyro

from snmr.integration import counterfactual_eval, distillation, wbt_bodyfix, wbt_latent
from snmr.integration.counterfactual_eval import (
    CounterfactualGrid,
    E71_EVALUATION_CONDITIONS,
    E71_FULL_STATE_TENSOR_NAMES,
    audit_same_state_proprio,
    audit_same_state_tensors,
    branch_start_steps_for_grid,
    build_four_cell_grid,
    normalized_branch_coordinate,
    pooled_goal_scale,
    standardized_branch_squared_error,
)
from snmr.integration.distillation import CommandStudent, route_teacher_actions

wbt_bodyfix.patch()
counterfactual_eval.patch()

from holosoma.agents.modules.module_utils import setup_ppo_actor_module  # noqa: E402
from holosoma.agents.ppo.ppo import EmpiricalNormalization  # noqa: E402
from holosoma.train_agent import AnnotatedExperimentConfig  # noqa: E402
from holosoma.utils.sim_utils import (  # noqa: E402
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


PROTOCOL = "E71 same-state valid-command swap v1"
SMOKE_PROTOCOL = "E71 same-state valid-command swap smoke v1"
PROPRIO_SLICE = slice(0, 90)
RAW_DOF_POS_SLICE = slice(32, 61)
RAW_DOF_VEL_SLICE = slice(61, 90)
GOAL_SLICE = slice(90, 154)
MOTION_CMD_DIM = 64
Z_SNMR_DIM = 128
Z_OFFSETS = (0, 5)
Z_CMD_DIM = 64
HORIZON_STEPS = 500
HALF_SECOND_STEPS = 25
ONE_SECOND_STEPS = 50
EVALUATION_SEED = 404
EXPECTED_CLIPS = ("walk1_subject1", "walk1_subject5")
EXPECTED_PRECHECK_SHA256 = "3c03b89e2bda939e9a8c5a6dd58caeb771944bb9e5755b14eb37ab75ed9c502e"
EXPECTED_MANIFEST_SHA256 = "2d0005a7a9504056ce5944a71388e3992d0e4ee30bae441da26a460a5b163504"
EXPECTED_MOTION_SHA256 = (
    "b78f294395a5c74f37edc1c09dd6de0909d966a9b7fd948291c8dd803f7106aa",
    "d8de93425c14e90dce2930450d722d3eb2b6fcbb09e9c4ff3d59725025424f51",
)
EXPECTED_STUDENT_SHA256 = {
    ("c_prior_explicit", 0): "8fb6f2e1645d8070e978c34f642f82b5be8b43a67c3f11e0adf87ea1fa272916",
    ("a_prior_snmr", 0): "f88984971c3435e3c377f038ed2ef5abef788aa1a0f68a80ad7011b23bb9b93a",
    ("c_prior_explicit", 1): "a6a53f09e6bfc1f886aa331330c7456fa7bdd2cf59a2450eecda3d4788cb80d5",
    ("a_prior_snmr", 1): "185ac3991cb6bcdd451d719c72d5d72273d4351a45bd93e7f532ebeaec730d38",
    ("c_prior_explicit", 2): "c8f262f1284eb883fdd2a07c1713241e6dd00c6c7449c7d811fe892b82f69779",
    ("a_prior_snmr", 2): "6d23363133df4ba30f6ddb12887aa22b932f6255a9e27031bdf57daf683e42c1",
}
ARM_GOALS = {
    "c_prior_explicit": "explicit",
    "a_prior_snmr": "snmr",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: pathlib.Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash changed: expected {expected}, got {actual}")
    return actual


def _resolve_manifest_record(
    manifest: dict[str, Any],
    manifest_path: pathlib.Path,
    *,
    section: str,
    key: str,
) -> tuple[pathlib.Path, str]:
    records = manifest.get(section)
    if not isinstance(records, (dict, list)):
        raise ValueError(f"freeze manifest {section} is missing")
    if isinstance(records, dict):
        record = records.get(key)
    else:
        arm, seed_raw = key.rsplit(":", 1)
        record = next(
            (
                item
                for item in records
                if item.get("arm") == arm
                and item.get("training_seed") == int(seed_raw)
            ),
            None,
        )
    if not isinstance(record, dict):
        raise ValueError(f"freeze manifest record {section}.{key} is missing")
    path = pathlib.Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = manifest_path.parent / path
    digest = str(record.get("sha256", ""))
    if len(digest) != 64:
        raise ValueError(f"freeze manifest record {section}.{key} has no SHA-256")
    return path.resolve(), digest


def _require_manifest_binding(
    manifest: dict[str, Any],
    manifest_path: pathlib.Path,
    *,
    section: str,
    key: str,
    actual_path: pathlib.Path,
    actual_hash: str,
) -> None:
    frozen_path, frozen_hash = _resolve_manifest_record(
        manifest, manifest_path, section=section, key=key
    )
    if frozen_path != actual_path.resolve() or frozen_hash != actual_hash:
        raise ValueError(f"{section}.{key} differs from the E71 freeze manifest")


def _select_smoke_pair(grid: CounterfactualGrid, pair_id: int) -> CounterfactualGrid:
    mask = grid.pair_ids == pair_id
    if int(mask.sum()) != 4:
        raise ValueError(f"smoke pair {pair_id} does not identify exactly four cells")
    return CounterfactualGrid(
        pair_ids=grid.pair_ids[mask],
        state_sides=grid.state_sides[mask],
        command_sides=grid.command_sides[mask],
        state_start_steps=grid.state_start_steps[mask],
        command_start_steps=grid.command_start_steps[mask],
    )


def _terminal_joint_goal(
    extras: dict[str, Any],
    failed: torch.Tensor,
    default_dof_pos: torch.Tensor,
) -> torch.Tensor:
    """Recover the pre-autoreset terminal joint state from Holosoma final observations."""

    final = extras.get("final_observations", {}).get("actor_obs")
    if final is None:
        raise RuntimeError("Holosoma did not expose terminal actor observations")
    if final.ndim != 2 or final.shape[1] != 154:
        raise RuntimeError(f"terminal actor observation layout changed: {tuple(final.shape)}")
    joint_pos = final[failed, RAW_DOF_POS_SLICE] + default_dof_pos[failed]
    joint_vel = final[failed, RAW_DOF_VEL_SLICE]
    return torch.cat((joint_pos, joint_vel), dim=-1)


def _json_vector(values: torch.Tensor) -> list[Any]:
    return values.detach().cpu().tolist()


def _write_json_once(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Atomically create a report and refuse a pre-existing destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite E71 report {path}")
    data = (json.dumps(payload, indent=2) + "\n").encode()
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite E71 report {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _git_state(root: pathlib.Path) -> tuple[str, dict[str, Any]]:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, {
        "root": str(root.resolve()),
        "tracked_changes": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _gpu_driver() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unavailable"
    value = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    return value or "unavailable"


def _gpu_uuid() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unavailable"
    value = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    return value or "unavailable"


def _runtime_environment() -> dict[str, str]:
    import mujoco
    import mujoco_warp._src.types as mjw_types
    import numpy
    import warp

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "mujoco": mujoco.__version__,
        "mujoco_warp": importlib.metadata.version("mujoco-warp"),
        "warp": warp.__version__,
        "numpy": numpy.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_logical_device": (
            str(torch.cuda.current_device()) if torch.cuda.is_available() else "unavailable"
        ),
        "gpu_total_memory_mb": (
            str(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
            if torch.cuda.is_available()
            else "unavailable"
        ),
        "gpu_uuid": _gpu_uuid(),
        "mujoco_warp_types": str(
            pathlib.Path(inspect.getsourcefile(mjw_types)).resolve()
        ),
        "nvidia_driver": _gpu_driver(),
    }


def _nominal_evaluation_config(config: Any) -> Any:
    """Remove observation and dynamics randomness before environment construction.

    The command-swap estimand holds the physical state and dynamics fixed while only
    replacing a source-valid command.  Merely seeding Holosoma is insufficient because
    its WBT defaults independently randomize observations, model parameters, pushes,
    and reset state per environment.
    """

    if config.observation is None or config.randomization is None or config.command is None:
        raise ValueError("E71 requires observation, randomization, and command managers")
    if "actor_obs" not in config.observation.groups:
        raise ValueError("E71 requires the registered actor_obs group")
    observation_groups = {
        name: dataclasses.replace(group, enable_noise=False)
        for name, group in config.observation.groups.items()
    }
    observation = dataclasses.replace(config.observation, groups=observation_groups)

    delay_term = config.randomization.setup_terms.get("setup_action_delay_buffers")
    if delay_term is None:
        raise ValueError("WBT action-delay setup term is missing")
    delay_term = dataclasses.replace(
        delay_term,
        params={"ctrl_delay_step_range": [0, 0], "enabled": False},
    )
    randomization = dataclasses.replace(
        config.randomization,
        setup_terms={"setup_action_delay_buffers": delay_term},
        reset_terms={},
        step_terms={},
        ignore_unsupported=False,
    )

    motion_term = config.command.setup_terms.get("motion_command")
    if motion_term is None:
        raise ValueError("E71 requires the WBT motion command")
    motion_params = dict(motion_term.params)
    motion_config = motion_params.get("motion_config")
    if motion_config is None:
        raise ValueError("motion command is missing motion_config")
    if isinstance(motion_config, dict):
        motion_config = dict(motion_config)
        initial_pose = motion_config.get("noise_to_initial_pose", {})
        if dataclasses.is_dataclass(initial_pose):
            initial_pose = dataclasses.replace(initial_pose, overall_noise_scale=0.0)
        else:
            initial_pose = {**dict(initial_pose), "overall_noise_scale": 0.0}
        motion_config["noise_to_initial_pose"] = initial_pose
    else:
        initial_pose = dataclasses.replace(
            motion_config.noise_to_initial_pose, overall_noise_scale=0.0
        )
        motion_config = dataclasses.replace(
            motion_config,
            noise_to_initial_pose=initial_pose,
            use_adaptive_timesteps_sampler=False,
        )
    if isinstance(motion_config, dict):
        motion_config["use_adaptive_timesteps_sampler"] = False
    motion_params["motion_config"] = motion_config
    motion_term = dataclasses.replace(motion_term, params=motion_params)
    command_terms = dict(config.command.setup_terms)
    command_terms["motion_command"] = motion_term
    command = dataclasses.replace(config.command, setup_terms=command_terms)

    spawn = dataclasses.replace(
        config.terrain.terrain_term.spawn,
        randomize_tiles=False,
        xy_offset_range=0.0,
    )
    terrain_term = dataclasses.replace(config.terrain.terrain_term, spawn=spawn)
    terrain = dataclasses.replace(config.terrain, terrain_term=terrain_term)
    sim_engine = dataclasses.replace(
        config.simulator.config.sim, max_episode_length_s=20.0
    )
    simulator_init = dataclasses.replace(config.simulator.config, sim=sim_engine)
    simulator = dataclasses.replace(config.simulator, config=simulator_init)
    return dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, torch_deterministic=True),
        observation=observation,
        randomization=randomization,
        command=command,
        terrain=terrain,
        simulator=simulator,
    )


def _initial_pose_noise_scale(config: Any) -> float:
    motion = config.command.setup_terms["motion_command"].params["motion_config"]
    if isinstance(motion, dict):
        noise = motion["noise_to_initial_pose"]
        if isinstance(noise, dict):
            return float(noise["overall_noise_scale"])
        return float(noise.overall_noise_scale)
    return float(motion.noise_to_initial_pose.overall_noise_scale)


def _nominal_condition_audit(config: Any, env: Any) -> dict[str, Any]:
    """Fail closed if the constructed environment violates nominal E71 conditions."""

    setup_names = set(config.randomization.setup_terms)
    delay = config.randomization.setup_terms["setup_action_delay_buffers"].params
    expanded = sorted(
        getattr(getattr(env.simulator.backend, "mjw_model", object()), "_expanded_fields", set())
    )
    action_terms = list(env.action_manager.iter_terms())
    action_delay_off = bool(
        getattr(env, "_randomize_ctrl_delay", None) is False
        and delay == {"ctrl_delay_step_range": [0, 0], "enabled": False}
        and all(getattr(term, "action_queue", None) is None for _, term in action_terms)
    )
    gain_terms = [
        term
        for _, term in action_terms
        if all(
            hasattr(term, name)
            for name in ("_kp_scale", "_kd_scale", "_rfi_lim_scale")
        )
    ]
    pd_gain_randomization = bool(
        len(gain_terms) != len(action_terms)
        or any(
            not torch.equal(term._kp_scale, torch.ones_like(term._kp_scale))
            or not torch.equal(term._kd_scale, torch.ones_like(term._kd_scale))
            or not torch.equal(
                term._rfi_lim_scale, torch.ones_like(term._rfi_lim_scale)
            )
            for term in gain_terms
        )
    )
    torque_rfi = any(
        bool(getattr(term, "_randomize_torque_rfi", False)) for _, term in action_terms
    )
    default_pose_equal = torch.equal(
        env.default_dof_pos,
        env.default_dof_pos_base.expand_as(env.default_dof_pos),
    )
    motion_config = config.command.setup_terms["motion_command"].params["motion_config"]
    adaptive_sampler = (
        bool(motion_config.get("use_adaptive_timesteps_sampler", False))
        if isinstance(motion_config, dict)
        else bool(motion_config.use_adaptive_timesteps_sampler)
    )
    termination_contract = {
        name: {
            "func": term.func,
            "is_timeout": bool(term.is_timeout),
            "params": term.params,
        }
        for name, term in sorted(config.termination.terms.items())
    }
    backend_class = (
        f"{type(env.simulator.backend).__module__}."
        f"{type(env.simulator.backend).__name__}"
    )
    backend_value = getattr(
        config.simulator.config.mujoco_backend,
        "value",
        config.simulator.config.mujoco_backend,
    )
    actor_group = config.observation.groups["actor_obs"]
    actor_observation_contract = {
        "concatenate": bool(actor_group.concatenate),
        "history_length": int(actor_group.history_length),
        "clip_observations": float(config.observation.clip_observations),
        "terms": {
            name: {
                "func": term.func,
                "scale": term.scale,
                "noise": float(term.noise),
                "clip": term.clip,
                "params": term.params,
            }
            for name, term in sorted(actor_group.terms.items())
        },
    }
    action_manager_contract = {
        name: {
            "func": term.func,
            "scale": term.scale,
            "clip": term.clip,
            "params": term.params,
        }
        for name, term in sorted(config.action.terms.items())
    }
    runtime_contract = {
        "environment_class": config.env_class,
        "simulator_target": config.simulator._target_,
        "simulator_backend": str(backend_value),
        "simulator_backend_class": backend_class,
        "device": str(env.device),
        "num_dof": int(env.num_dof),
        "num_actions": int(env.robot_config.actions_dim),
        "physics_fps": int(config.simulator.config.sim.fps),
        "control_decimation": int(config.simulator.config.sim.control_decimation),
        "policy_dt_s": float(env.dt),
        "max_episode_length_s": float(config.simulator.config.sim.max_episode_length_s),
        "terrain_mesh_type": str(config.terrain.terrain_term.mesh_type),
        "control_type": str(env.robot_config.control.control_type),
        "clip_actions": bool(env.robot_config.control.clip_actions),
        "action_clip_value": float(env.robot_config.control.action_clip_value),
        "clip_torques": bool(env.robot_config.control.clip_torques),
        "action_scale": float(env.robot_config.control.action_scale),
        "action_scales_by_effort_limit_over_p_gain": bool(
            env.robot_config.control.action_scales_by_effort_limit_over_p_gain
        ),
        "action_manager_contract": action_manager_contract,
        "actor_observation_contract": actor_observation_contract,
        "termination_terms": termination_contract,
    }
    runtime_contract_passed = bool(
        runtime_contract["environment_class"]
        == "holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager"
        and runtime_contract["simulator_target"]
        == "holosoma.simulator.mujoco.mujoco.MuJoCo"
        and runtime_contract["simulator_backend"] == "warp"
        and runtime_contract["simulator_backend_class"].endswith(".WarpBackend")
        and runtime_contract["device"].startswith("cuda")
        and runtime_contract["num_dof"] == 29
        and runtime_contract["num_actions"] == 29
        and runtime_contract["physics_fps"] == 200
        and runtime_contract["control_decimation"] == 4
        and runtime_contract["policy_dt_s"] == 0.02
        and runtime_contract["max_episode_length_s"] == 20.0
        and runtime_contract["terrain_mesh_type"] == "plane"
        and runtime_contract["control_type"] == "P"
        and runtime_contract["clip_actions"] is True
        and runtime_contract["action_scale"] == 0.25
        and runtime_contract["action_scales_by_effort_limit_over_p_gain"] is True
        and set(termination_contract) == {"timeout", "bad_tracking"}
        and termination_contract["timeout"]["is_timeout"] is True
        and termination_contract["bad_tracking"]["is_timeout"] is False
    )
    conditions = {
        "actor_observation_noise": bool(
            config.observation.groups["actor_obs"].enable_noise
        ),
        "all_observation_noise": bool(
            any(group.enable_noise for group in config.observation.groups.values())
        ),
        "startup_physics_randomization": bool(
            setup_names != {"setup_action_delay_buffers"} or expanded
        ),
        "reset_state_randomization": bool(config.randomization.reset_terms),
        "external_pushes": bool(
            config.randomization.step_terms
            or getattr(env, "_randomize_push_robots", False)
            or torch.any(getattr(env, "_max_push_vel", torch.ones(1, device=env.device)) != 0)
        ),
        "action_delay": not action_delay_off,
        "pd_gain_randomization": pd_gain_randomization,
        "torque_rfi": torque_rfi,
        "initial_pose_noise_scale": _initial_pose_noise_scale(config),
        "tracking_termination_suppressed_steps": ONE_SECOND_STEPS,
        "future_only_branch_samples": True,
        "reference_termination_after_primary_horizon": True,
        "nonfinite_state_guard": True,
        "mujoco_warp_overflow_guard": True,
        "warmup_callback_order_override": True,
        "rollout_task_update_before_termination": bool(
            env._update_tasks_before_termination
        ),
        "torch_deterministic": bool(config.training.torch_deterministic),
        "adaptive_timestep_sampler": adaptive_sampler,
        "terrain_spawn_randomization": bool(
            config.terrain.terrain_term.spawn.randomize_tiles
            or config.terrain.terrain_term.spawn.xy_offset_range != 0.0
        ),
        "expanded_mujoco_model_fields": expanded,
        "default_dof_pose_equal": bool(default_pose_equal),
        "runtime_contract": runtime_contract,
    }
    conditions["passed"] = bool(
        conditions["actor_observation_noise"] is False
        and conditions["all_observation_noise"] is False
        and conditions["startup_physics_randomization"] is False
        and conditions["reset_state_randomization"] is False
        and conditions["external_pushes"] is False
        and conditions["action_delay"] is False
        and conditions["pd_gain_randomization"] is False
        and conditions["torque_rfi"] is False
        and conditions["initial_pose_noise_scale"] == 0.0
        and conditions["terrain_spawn_randomization"] is False
        and conditions["torch_deterministic"] is True
        and conditions["adaptive_timestep_sampler"] is False
        and conditions["default_dof_pose_equal"] is True
        and conditions["rollout_task_update_before_termination"] is False
        and runtime_contract_passed
    )
    expected_without_verdict = {
        key: value for key, value in E71_EVALUATION_CONDITIONS.items() if key != "passed"
    }
    observed_without_verdict = {
        key: value for key, value in conditions.items() if key != "passed"
    }
    conditions["passed"] = bool(
        conditions["passed"]
        and observed_without_verdict == expected_without_verdict
    )
    return conditions


def _full_same_state_tensors(env: Any) -> dict[str, torch.Tensor]:
    """Collect command-independent integration, physics, and controller state."""

    if torch.cuda.is_available():
        torch.cuda.synchronize(device=env.device)
    sim = env.simulator
    origins = sim.scene.env_origins
    tensors: dict[str, torch.Tensor] = {
        "dof_pos": sim.dof_pos.clone(),
        "dof_vel": sim.dof_vel.clone(),
        "root_state_local": sim.robot_root_states.clone(),
        "action": env.action_manager.action.clone(),
        "previous_action": env.action_manager.prev_action.clone(),
        "default_dof_pos": env.default_dof_pos.clone(),
    }
    tensors["root_state_local"][:, :3] -= origins
    for name in (
        "contact_forces_history",
        "push_robot_vel_buf",
        "record_push_robot_vel_buf",
    ):
        value = getattr(sim if name == "contact_forces_history" else env, name, None)
        if isinstance(value, torch.Tensor) and value.shape[0] == env.num_envs:
            tensors[name] = value.clone()
    rigid_positions = getattr(sim, "_rigid_body_pos", None)
    if isinstance(rigid_positions, torch.Tensor):
        tensors["rigid_body_pos_local"] = rigid_positions.clone() - origins[:, None, :]
    for source, label in (
        ("_rigid_body_rot", "rigid_body_rot"),
        ("_rigid_body_vel", "rigid_body_vel"),
        ("_rigid_body_ang_vel", "rigid_body_ang_vel"),
    ):
        value = getattr(sim, source, None)
        if isinstance(value, torch.Tensor) and value.shape[0] == env.num_envs:
            tensors[label] = value.clone()

    backend = sim.backend
    qpos = getattr(backend, "qpos_t", None)
    if isinstance(qpos, torch.Tensor):
        localized = qpos.clone()
        root_start = int(sim.robot_qpos_addr)
        localized[:, root_start : root_start + 3] -= origins
        tensors["integration_qpos_local"] = localized
    for name in (
        "qvel_t",
        "qacc_t",
        "ctrl_t",
        "cfrc_t",
        "xfrc_applied_t",
    ):
        value = getattr(backend, name, None)
        if isinstance(value, torch.Tensor) and value.shape[0] == env.num_envs:
            tensors[f"integration_{name.removesuffix('_t')}"] = value.clone()

    mjw_data = getattr(backend, "mjw_data", None)
    if mjw_data is not None:
        try:
            import warp as wp
        except ImportError as exc:  # pragma: no cover - simulator runtime dependency
            raise RuntimeError("MJWarp state audit requires the warp package") from exc
        wp.synchronize()
        for name in (
            "time",
            "act",
            "act_dot",
            "qacc_warmstart",
            "qfrc_applied",
            "eq_active",
            "mocap_pos",
            "mocap_quat",
            "history",
            "userdata",
            "tree_awake",
            "tree_asleep",
            "overflow",
        ):
            value = getattr(mjw_data, name, None)
            if value is None:
                continue
            tensor = wp.to_torch(value)
            if tensor.shape[0] == env.num_envs:
                tensors[f"integration_{name}"] = tensor.clone()

    for term_name, term in env.action_manager.iter_terms():
        for attr in (
            "_raw_actions",
            "_processed_actions",
            "_actions_after_delay",
            "torques",
            "_prev_dof_vel",
            "_kp_scale",
            "_kd_scale",
            "_rfi_lim_scale",
        ):
            value = getattr(term, attr, None)
            if isinstance(value, torch.Tensor) and value.shape[0] == env.num_envs:
                tensors[f"action_term.{term_name}.{attr.lstrip('_')}"] = value.clone()
        queue = getattr(term, "action_queue", None)
        if isinstance(queue, torch.Tensor) and queue.shape[0] == env.num_envs:
            tensors[f"action_term.{term_name}.action_queue"] = queue.clone()
    observed_names = set(tensors)
    expected_names = set(E71_FULL_STATE_TENSOR_NAMES)
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise RuntimeError(
            "pinned MJWarp full-state contract changed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return tensors


def _require_zero_mujoco_warp_overflow(env: Any, *, label: str) -> int:
    """Fail if MJWarp reports any accumulated capacity overflow bit."""

    try:
        import warp as wp
    except ImportError as exc:  # pragma: no cover - simulator runtime dependency
        raise RuntimeError("MJWarp overflow audit requires the warp package") from exc
    data = getattr(env.simulator.backend, "mjw_data", None)
    overflow = getattr(data, "overflow", None)
    if overflow is None:
        raise RuntimeError("pinned MJWarp runtime exposes no overflow state")
    wp.synchronize()
    tensor = wp.to_torch(overflow)
    if tensor.shape[0] != env.num_envs:
        raise RuntimeError(f"MJWarp overflow state has changed shape: {tuple(tensor.shape)}")
    nonzero = int(torch.count_nonzero(tensor).item())
    if nonzero:
        raise RuntimeError(f"MJWarp capacity overflow at {label}: {nonzero} nonzero entries")
    return nonzero


def _observation_layout_audit(
    env: Any,
    motion_command: Any,
    obs_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Verify the semantic 154-D layout instead of trusting width alone."""

    from holosoma.managers.observation.terms import wbt as observation_terms

    expected_terms = (
        "actions",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
        "motion_command",
        "motion_ref_ori_b",
    )
    configured = tuple(sorted(env.observation_manager.cfg.groups["actor_obs"].terms))
    group = env.observation_manager.cfg.groups["actor_obs"]
    configuration = {
        "concatenate": bool(group.concatenate),
        "history_length": int(group.history_length),
        "clip_observations": float(env.observation_manager.cfg.clip_observations),
        "terms": {
            name: {
                "func": term.func,
                "scale": term.scale,
                "noise": float(term.noise),
                "clip": term.clip,
                "params": term.params,
            }
            for name, term in sorted(group.terms.items())
        },
    }
    expected = torch.cat(
        (
            env.action_manager.action,
            observation_terms.base_ang_vel(env),
            env.simulator.dof_pos - env.default_dof_pos,
            env.simulator.dof_vel,
            motion_command.command,
            observation_terms.motion_ref_ori_b(env),
        ),
        dim=-1,
    )
    observed = obs_dict["actor_obs"]
    if expected.shape != observed.shape:
        max_abs = float("inf")
    else:
        max_abs = float((observed - expected).abs().max())
    report = {
        "configured_terms_sorted": list(configured),
        "expected_terms_sorted": list(expected_terms),
        "observed_width": int(observed.shape[1]),
        "expected_width": 154,
        "max_abs_difference": max_abs,
        "tolerance": 1.0e-6,
        "configuration": configuration,
    }
    report["passed"] = bool(
        configured == expected_terms
        and observed.shape == expected.shape
        and observed.shape[1] == 154
        and max_abs <= report["tolerance"]
        and configuration
        == E71_EVALUATION_CONDITIONS["runtime_contract"][
            "actor_observation_contract"
        ]
    )
    return report


def main() -> None:
    arm = os.environ["E52_ARM"]
    if arm not in ARM_GOALS:
        raise ValueError(f"E71 only supports the frozen explicit and SNMR arms, got {arm!r}")
    training_seed = int(os.environ["E71_TRAINING_SEED"])
    if (arm, training_seed) not in EXPECTED_STUDENT_SHA256:
        raise ValueError("E71 training seed must be one of the frozen seeds 0, 1, or 2")
    out = pathlib.Path(os.environ["E52_OUT"])
    student_path = out / f"{arm}_student.pt"
    student_hash = _require_hash(
        student_path,
        EXPECTED_STUDENT_SHA256[(arm, training_seed)],
        label="student checkpoint",
    )
    manifest_path = pathlib.Path(os.environ["E52_TEACHER_MANIFEST"])
    manifest_hash = _require_hash(
        manifest_path, EXPECTED_MANIFEST_SHA256, label="teacher manifest"
    )
    precheck_path = pathlib.Path(os.environ["E71_STARTS_JSON"])
    precheck_hash = _require_hash(
        precheck_path, EXPECTED_PRECHECK_SHA256, label="ambiguity precheck"
    )
    smoke_pair_raw = os.environ.get("E71_SMOKE_PAIR_ID", "")
    smoke_pair_id = int(smoke_pair_raw) if smoke_pair_raw else None
    freeze_manifest_path = pathlib.Path(os.environ["E71_FREEZE_MANIFEST"])
    if not freeze_manifest_path.is_file():
        raise FileNotFoundError(f"missing E71 freeze manifest: {freeze_manifest_path}")
    freeze_manifest_hash = sha256_file(freeze_manifest_path)
    freeze_manifest = json.loads(freeze_manifest_path.read_text())
    from scripts.audit_e71_bundle import validate_manifest as validate_freeze_manifest

    freeze_state = validate_freeze_manifest(
        freeze_manifest,
        base_dir=freeze_manifest_path.parent,
        require_preregistered=smoke_pair_id is None,
    )
    artifact_paths = freeze_state["artifact_paths"]
    expected_manifest_path = artifact_paths[
        "draft_manifest" if smoke_pair_id is not None else "preregistered_manifest"
    ]
    if str(freeze_manifest_path.resolve()) != expected_manifest_path:
        raise ValueError("freeze manifest path differs from its artifact contract")
    report_path = pathlib.Path(os.environ["E71_REPORT_PATH"])
    expected_report_path = (
        artifact_paths["smoke_report"]
        if smoke_pair_id is not None
        else artifact_paths[f"{ARM_GOALS[arm]}_reports"][training_seed]
    )
    if str(report_path.resolve()) != expected_report_path:
        raise ValueError("E71 report path differs from the freeze artifact contract")
    if (
        freeze_manifest.get("protocol") != "E71 command-swap freeze manifest v1"
        or freeze_manifest.get("report_protocol") != PROTOCOL
    ):
        raise ValueError("E71 requires a valid command-swap freeze manifest")
    allowed_manifest_status = (
        {"DRAFT", "PREREGISTERED"} if smoke_pair_id is not None else {"PREREGISTERED"}
    )
    if freeze_manifest.get("status") not in allowed_manifest_status:
        raise ValueError("confirmatory E71 requires a PREREGISTERED freeze manifest")
    if freeze_manifest.get("runtime_ready") is not True:
        raise ValueError("E71 simulator execution requires a runtime-ready freeze manifest")
    if (
        freeze_manifest.get("evaluation_seed") != EVALUATION_SEED
        or freeze_manifest.get("training_seeds") != [0, 1, 2]
        or freeze_manifest.get("num_pairs") != 69
        or freeze_manifest.get("cells_per_report") != 276
    ):
        raise ValueError("freeze manifest changed the registered E71 sample grid")
    invoked_argv = [str(pathlib.Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    if smoke_pair_id is None:
        argv_key = f"{ARM_GOALS[arm]}_seed{training_seed}"
        expected_argv = freeze_manifest.get("confirmatory_argv_by_arm_seed", {}).get(
            argv_key
        )
        argv_label = f"confirmatory manifest entry {argv_key}"
    else:
        if smoke_pair_id != freeze_manifest.get("smoke_pair_id"):
            raise ValueError("smoke pair differs from the DRAFT freeze manifest")
        if arm != "c_prior_explicit" or training_seed != 0:
            raise ValueError("E71 smoke is frozen to the explicit seed-0 controller")
        expected_argv = freeze_manifest.get("smoke_argv")
        argv_label = "DRAFT smoke manifest entry"
    if invoked_argv != expected_argv:
        raise ValueError(f"evaluator argv differs from {argv_label}")
    snmr_commit, snmr_worktree = _git_state(ROOT)
    holosoma_root = ROOT.parent / "holosoma"
    holosoma_commit, holosoma_worktree = _git_state(holosoma_root)
    observed_commits = {"snmr": snmr_commit, "holosoma": holosoma_commit}
    observed_worktrees = {"snmr": snmr_worktree, "holosoma": holosoma_worktree}
    if observed_commits != freeze_manifest.get("commits"):
        raise ValueError("repository revisions differ from the E71 freeze manifest")
    if observed_worktrees != freeze_manifest.get("working_trees"):
        raise ValueError("tracked working-tree state differs from the E71 freeze manifest")
    observed_environment = _runtime_environment()
    if observed_environment != freeze_manifest.get("environment"):
        raise ValueError("runtime package/GPU environment differs from the E71 freeze manifest")
    runtime_paths = {
        "evaluator": pathlib.Path(__file__).resolve(),
        "reset_layer": pathlib.Path(counterfactual_eval.__file__).resolve(),
        "bodyfix_layer": pathlib.Path(wbt_bodyfix.__file__).resolve(),
        "latent_runtime": pathlib.Path(wbt_latent.__file__).resolve(),
        "distillation_runtime": pathlib.Path(distillation.__file__).resolve(),
    }
    runtime_hashes_start = {
        name: sha256_file(path) for name, path in runtime_paths.items()
    }
    for name, path in runtime_paths.items():
        _require_manifest_binding(
            freeze_manifest,
            freeze_manifest_path,
            section="frozen_files",
            key=name,
            actual_path=path,
            actual_hash=runtime_hashes_start[name],
        )
    for key, path, digest in (
        ("ambiguity_precheck", precheck_path, precheck_hash),
        ("teacher_manifest", manifest_path, manifest_hash),
    ):
        _require_manifest_binding(
            freeze_manifest,
            freeze_manifest_path,
            section="frozen_files",
            key=key,
            actual_path=path,
            actual_hash=digest,
        )
    _require_manifest_binding(
        freeze_manifest,
        freeze_manifest_path,
        section="checkpoints",
        key=f"{ARM_GOALS[arm]}:{training_seed}",
        actual_path=student_path,
        actual_hash=student_hash,
    )
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite E71 report {report_path}")

    manifest = json.loads(manifest_path.read_text())
    teacher_entries = manifest.get("motions", [])
    if [entry.get("clip") for entry in teacher_entries] != list(EXPECTED_CLIPS):
        raise ValueError("teacher manifest no longer has the frozen E71 clip order")
    teacher_paths = [pathlib.Path(str(entry["checkpoint"])) for entry in teacher_entries]
    teacher_hashes = [
        _require_hash(path, str(entry["checkpoint_sha256"]), label="teacher checkpoint")
        for path, entry in zip(teacher_paths, teacher_entries, strict=True)
    ]
    for index, (path, digest) in enumerate(
        zip(teacher_paths, teacher_hashes, strict=True)
    ):
        _require_manifest_binding(
            freeze_manifest,
            freeze_manifest_path,
            section="frozen_files",
            key=f"teacher_checkpoint_{index}",
            actual_path=path,
            actual_hash=digest,
        )
    precheck = json.loads(precheck_path.read_text())
    preferred_pair = precheck.get("preferred_pair")
    pair_report = precheck.get("pairs", {}).get(preferred_pair)
    if preferred_pair != ",".join(EXPECTED_CLIPS) or not isinstance(pair_report, dict):
        raise ValueError("ambiguity precheck no longer selects the frozen E71 pair")
    if len(pair_report.get("windows", [])) != 69:
        raise ValueError("E71 requires the frozen set of exactly 69 ambiguity windows")

    tyro_cfg = _nominal_evaluation_config(
        tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    )
    if int(tyro_cfg.training.seed) != EVALUATION_SEED:
        raise ValueError(f"E71 must run at evaluation seed {EVALUATION_SEED}")
    env, device, sim_app = setup_simulation_environment(tyro_cfg)
    try:
        expected_envs = 4 if smoke_pair_id is not None else 69 * 4
        if env.num_envs != expected_envs:
            raise ValueError(f"E71 requires {expected_envs} environments, got {env.num_envs}")
        if abs(float(env.dt) - 0.02) > 1.0e-9:
            raise ValueError(f"E71 requires the frozen 50 Hz policy rate, got dt={env.dt}")
        obs_dims = env.observation_manager.get_obs_dims()
        actor_dim = int(obs_dims["actor_obs"])
        critic_dim = int(obs_dims["critic_obs"])
        if actor_dim != 154:
            raise ValueError(f"WBT actor observation layout changed: {actor_dim}")
        num_actions = int(env.robot_config.actions_dim)
        if num_actions != 29:
            raise ValueError(f"frozen G1 action width changed: {num_actions}")
        history_length = {
            key: env.observation_manager.cfg.groups[key].history_length
            for key in ("actor_obs", "critic_obs")
        }

        teachers = []
        teacher_norms = []
        for teacher_path in teacher_paths:
            checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)
            teacher = setup_ppo_actor_module(
                obs_dim_dict=obs_dims,
                module_config=tyro_cfg.algo.config.module_dict.actor,
                num_actions=num_actions,
                init_noise_std=tyro_cfg.algo.config.init_noise_std,
                device=device,
                history_length=history_length,
            )
            teacher.load_state_dict(checkpoint["actor_model_state_dict"])
            teacher.eval()
            normalizer = EmpiricalNormalization(shape=actor_dim, device=device)
            normalizer.load_state_dict(checkpoint["actor_obs_normalizer_state_dict"])
            normalizer.eval()
            teachers.append(teacher)
            teacher_norms.append(normalizer)
        actor_norm = teacher_norms[0]
        for module in (*teachers, *teacher_norms):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        motion_command = env.command_manager.get_state("motion_command")
        from holosoma.managers.observation.terms import wbt as observation_terms_runtime
        import mujoco_warp._src.forward as mujoco_warp_forward
        import mujoco_warp._src.io as mujoco_warp_io
        import mujoco_warp._src.sleep as mujoco_warp_sleep

        imported_runtime_paths = {
            **{name: str(path) for name, path in runtime_paths.items()},
            "holosoma_base_task": str(
                pathlib.Path(inspect.getsourcefile(type(env).__mro__[1])).resolve()
            ),
            "holosoma_action_manager": str(
                pathlib.Path(inspect.getsourcefile(type(env.action_manager))).resolve()
            ),
            "holosoma_joint_action_term": str(
                pathlib.Path(
                    inspect.getsourcefile(
                        type(dict(env.action_manager.iter_terms())["joint_control"])
                    )
                ).resolve()
            ),
            "holosoma_mujoco_simulator": str(
                pathlib.Path(inspect.getsourcefile(type(env.simulator))).resolve()
            ),
            "holosoma_wbt_manager": str(
                pathlib.Path(inspect.getsourcefile(type(env))).resolve()
            ),
            "holosoma_warp_backend": str(
                pathlib.Path(inspect.getsourcefile(type(env.simulator.backend))).resolve()
            ),
            "holosoma_command_runtime": str(
                pathlib.Path(inspect.getsourcefile(type(motion_command))).resolve()
            ),
            "holosoma_observation_manager": str(
                pathlib.Path(inspect.getsourcefile(type(env.observation_manager))).resolve()
            ),
            "holosoma_observation_terms": str(
                pathlib.Path(observation_terms_runtime.__file__).resolve()
            ),
            "holosoma_termination_manager": str(
                pathlib.Path(inspect.getsourcefile(type(env.termination_manager))).resolve()
            ),
            "holosoma_termination_terms": str(
                pathlib.Path(
                    inspect.getsourcefile(
                        type(env.termination_manager._term_instances["bad_tracking"])
                    )
                ).resolve()
            ),
            "mujoco_warp_forward": str(
                pathlib.Path(inspect.getsourcefile(mujoco_warp_forward)).resolve()
            ),
            "mujoco_warp_io": str(
                pathlib.Path(inspect.getsourcefile(mujoco_warp_io)).resolve()
            ),
            "mujoco_warp_sleep": str(
                pathlib.Path(inspect.getsourcefile(mujoco_warp_sleep)).resolve()
            ),
            "mujoco_warp_types": observed_environment["mujoco_warp_types"],
        }
        imported_runtime_hashes: dict[str, str] = {}
        for name, raw_path in imported_runtime_paths.items():
            path = pathlib.Path(raw_path)
            digest = sha256_file(path)
            _require_manifest_binding(
                freeze_manifest,
                freeze_manifest_path,
                section="frozen_files",
                key=name,
                actual_path=path,
                actual_hash=digest,
            )
            imported_runtime_hashes[name] = digest
        motion_files: list[str] = []
        for directory in str(motion_command.motion_cfg.motion_dir).split(","):
            motion_files.extend(
                sorted(glob.glob(str(pathlib.Path(directory.strip()).expanduser() / "*.npz")))
            )
        motion_paths = [pathlib.Path(path).resolve() for path in motion_files]
        if len(motion_paths) != 2:
            raise ValueError("E71 requires exactly the two frozen motion files")
        if not all(
            path.stem.startswith(clip)
            for path, clip in zip(motion_paths, EXPECTED_CLIPS, strict=True)
        ):
            raise ValueError("loaded motion order differs from the frozen E71 clip order")
        motion_hashes = [sha256_file(path) for path in motion_paths]
        if tuple(motion_hashes) != EXPECTED_MOTION_SHA256:
            raise ValueError("a loaded motion differs from the frozen E71 input")
        for index, (path, digest) in enumerate(
            zip(motion_paths, motion_hashes, strict=True)
        ):
            _require_manifest_binding(
                freeze_manifest,
                freeze_manifest_path,
                section="frozen_files",
                key=f"motion_{index}",
                actual_path=path,
                actual_hash=digest,
            )
        if abs(float(motion_command.motion.fps) - 50.0) > 1.0e-9:
            raise ValueError("loaded E71 motions must be sampled at 50 Hz")

        latents = wbt_latent._ensure_latent_loaded(motion_command)
        saved = torch.load(student_path, map_location=device, weights_only=False)
        config = saved.get("config", {})
        if saved.get("arm") != arm or config.get("deterministic") is not True:
            raise ValueError("E71 requires the matching deterministic E70 student")
        if bool(config.get("time_index")) or bool(config.get("shuffle_latent")):
            raise ValueError("E71 requires an intact explicit or SNMR student")
        if tuple(config.get("offsets", ())) != Z_OFFSETS or config.get("z_cmd_dim") != Z_CMD_DIM:
            raise ValueError("student latent-window contract differs from frozen E71")
        student = CommandStudent(
            PROPRIO_SLICE.stop,
            critic_dim,
            num_actions,
            ARM_GOALS[arm],
            MOTION_CMD_DIM,
            z_window_dim=Z_SNMR_DIM * len(Z_OFFSETS),
            z_cmd_dim=Z_CMD_DIM,
        ).to(device)
        student.load_state_dict(saved["student"])
        student.eval()
        z_mean = saved["z_mean"].to(device)
        z_std = saved["z_std"].to(device)
        if z_mean.shape != (1, Z_SNMR_DIM) or z_std.shape != (1, Z_SNMR_DIM):
            raise ValueError("student latent standardizer shape changed")
        if not torch.isfinite(z_mean).all() or not torch.isfinite(z_std).all() or torch.any(z_std <= 0):
            raise ValueError("student latent standardizer is invalid")

        all_frozen_paths = {
            "freeze_manifest": freeze_manifest_path.resolve(),
            "student_checkpoint": student_path.resolve(),
            "teacher_manifest": manifest_path.resolve(),
            "teacher_checkpoint_0": teacher_paths[0].resolve(),
            "teacher_checkpoint_1": teacher_paths[1].resolve(),
            "motion_0": motion_paths[0].resolve(),
            "motion_1": motion_paths[1].resolve(),
            "ambiguity_precheck": precheck_path.resolve(),
            **{
                f"runtime.{name}": pathlib.Path(path).resolve()
                for name, path in imported_runtime_paths.items()
            },
        }
        all_frozen_hashes_start = {
            name: sha256_file(path) for name, path in all_frozen_paths.items()
        }
        if all_frozen_hashes_start["freeze_manifest"] != freeze_manifest_hash:
            raise RuntimeError("freeze manifest changed after startup")

        grid = build_four_cell_grid(
            pair_report,
            clip_names=EXPECTED_CLIPS,
            motion_starts=motion_command.motion.motion_start_idx,
            motion_ends=motion_command.motion.motion_end_idx,
            fps=float(motion_command.motion.fps),
            horizon_steps=HORIZON_STEPS,
            device=device,
        )
        if smoke_pair_id is not None:
            grid = _select_smoke_pair(grid, smoke_pair_id)
        if grid.num_cells != expected_envs:
            raise RuntimeError("four-cell grid does not match the simulator environment count")
        branch_a_starts, branch_b_starts = branch_start_steps_for_grid(grid)

        def split_obs(obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            full = actor_norm(obs_dict["actor_obs"], update=False)
            return full[:, PROPRIO_SLICE], full[:, GOAL_SLICE]

        def z_window() -> torch.Tensor:
            values = wbt_latent._latent_at_offsets(motion_command, Z_OFFSETS)
            return torch.cat([(value - z_mean) / z_std for value in values], dim=-1)

        motion_teacher_ids = torch.arange(2, device=device, dtype=torch.long)

        def teacher_action(raw_actor_obs: torch.Tensor) -> torch.Tensor:
            actions = [
                teacher.act_inference({"actor_obs": normalizer(raw_actor_obs, update=False)})
                for teacher, normalizer in zip(teachers, teacher_norms, strict=True)
            ]
            routed_ids = motion_teacher_ids[motion_command.motion_ids]
            return route_teacher_actions(torch.stack(actions, dim=0), routed_ids)

        env.set_is_evaluating()
        motion_command.set_counterfactual_evaluation_start_steps(
            grid.state_start_steps,
            grid.command_start_steps,
            horizon_steps=HORIZON_STEPS,
        )
        if env.termination_manager is None:
            raise RuntimeError("E71 requires the registered termination manager")
        termination_control = {"suppress_reference": True}
        original_termination_check = env.termination_manager.check

        def registered_termination_check() -> tuple[torch.Tensor, torch.Tensor]:
            reset_flags, timeout_flags = original_termination_check()
            if termination_control["suppress_reference"]:
                reset_flags = torch.zeros_like(reset_flags)
                env.termination_manager.terminated = reset_flags.clone()
            return reset_flags, timeout_flags

        env.termination_manager.check = registered_termination_check
        previous_callback_order = env._update_tasks_before_termination
        if previous_callback_order is not False:
            raise RuntimeError(
                "MJWarp rollout callback order changed before the E71 warm-up override"
            )
        callback_order_during_warmup = False
        try:
            # The first WBT target buffers are populated by MotionCommand.step().  On
            # MJWarp that callback normally follows termination, so use the alternate
            # order for this single zero-action warm-up transition only.
            env._update_tasks_before_termination = True
            callback_order_during_warmup = (
                env._update_tasks_before_termination is True
            )
            if not callback_order_during_warmup:
                raise RuntimeError("failed to realize the registered warm-up callback order")
            obs_dict = env.reset_all()
        finally:
            env._update_tasks_before_termination = previous_callback_order
        callback_order_after_warmup = env._update_tasks_before_termination
        if callback_order_after_warmup is not False:
            raise RuntimeError("failed to restore the registered rollout callback order")
        warmup_reset_mask = env.reset_buf.bool().clone()
        episode_lengths = env.episode_length_buf.clone()
        warmup_audit = {
            "reset_count": int(warmup_reset_mask.sum()),
            "command_dependent_reset": False,
            "episode_length_min": int(episode_lengths.min()),
            "episode_length_max": int(episode_lengths.max()),
            "callback_order_before_warmup": previous_callback_order,
            "callback_order_during_warmup": callback_order_during_warmup,
            "callback_order_after_warmup": callback_order_after_warmup,
            "passed": bool(
                not torch.any(warmup_reset_mask)
                and torch.all(episode_lengths == 1)
                and previous_callback_order is False
                and callback_order_during_warmup is True
                and callback_order_after_warmup is False
            ),
        }
        if not warmup_audit["passed"]:
            raise RuntimeError(f"counterfactual reset warm-up failed audit: {warmup_audit}")
        overflow_checks = 1
        _require_zero_mujoco_warp_overflow(env, label="post-warmup")
        if not torch.equal(motion_command.time_steps, grid.command_start_steps):
            raise RuntimeError("reset did not realize the requested command cursor")
        if not torch.equal(motion_command.motion_ids, grid.command_sides):
            raise RuntimeError("reset did not realize command-side motion routing")
        observation_layout_audit = _observation_layout_audit(
            env, motion_command, obs_dict
        )
        if not observation_layout_audit["passed"]:
            raise RuntimeError(
                f"registered actor observation layout changed: {observation_layout_audit}"
            )
        normalized_proprio, _ = split_obs(obs_dict)
        proprio_audit = audit_same_state_proprio(normalized_proprio, grid)
        raw_proprio_audit = audit_same_state_proprio(
            obs_dict["actor_obs"][:, PROPRIO_SLICE], grid
        )
        if not proprio_audit["passed"] or not raw_proprio_audit["passed"]:
            raise RuntimeError(
                "changing command side changed initial proprioception: "
                f"normalized={proprio_audit}, raw={raw_proprio_audit}"
            )
        evaluation_conditions = _nominal_condition_audit(tyro_cfg, env)
        if not evaluation_conditions["passed"]:
            raise RuntimeError(
                f"E71 nominal evaluation conditions were not realized: {evaluation_conditions}"
            )
        if not torch.allclose(
            env.simulator.scene.env_origins,
            env.simulator.scene.env_origins[:1].expand_as(env.simulator.scene.env_origins),
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError("deterministic terrain spawn produced unequal environment origins")
        same_state_tensors = _full_same_state_tensors(env)
        full_state_audit = audit_same_state_tensors(same_state_tensors, grid)
        if not full_state_audit["passed"]:
            raise RuntimeError(
                f"command replacement changed hidden integration state: {full_state_audit}"
            )
        full_state_contract = {
            "tensor_names": sorted(same_state_tensors),
            "expanded_mujoco_model_fields": evaluation_conditions[
                "expanded_mujoco_model_fields"
            ],
            "tolerance": full_state_audit["tolerance"],
        }
        if evaluation_conditions != freeze_manifest.get("evaluation_conditions"):
            raise RuntimeError(
                "realized evaluation conditions differ from the freeze manifest"
            )
        if full_state_contract != freeze_manifest.get("full_state_contract"):
            raise RuntimeError(
                "realized full-state contract differs from the freeze manifest"
            )

        raw_window = wbt_latent._latent_at_offsets(motion_command, Z_OFFSETS)
        direct_window = []
        for offset in Z_OFFSETS:
            indexes = torch.minimum(
                grid.command_start_steps + offset,
                motion_command.motion.motion_end_idx[grid.command_sides] - 1,
            )
            direct_window.append(latents[indexes])
        latent_direct_max = max(
            float((actual - expected).abs().max())
            for actual, expected in zip(raw_window, direct_window, strict=True)
        )
        latent_state_max = 0.0
        for pair_id in torch.unique(grid.pair_ids, sorted=True).tolist():
            for command_side in (0, 1):
                rows = []
                for state_side in (0, 1):
                    mask = (
                        (grid.pair_ids == pair_id)
                        & (grid.command_sides == command_side)
                        & (grid.state_sides == state_side)
                    )
                    indexes = torch.nonzero(mask, as_tuple=False).flatten()
                    if indexes.numel() != 1:
                        raise RuntimeError("four-cell latent routing grid is incomplete")
                    rows.append(torch.cat([value[indexes[0]] for value in raw_window]))
                latent_state_max = max(
                    latent_state_max, float((rows[0] - rows[1]).abs().max())
                )
        latent_route_audit = {
            "direct_lookup_max_abs_difference": latent_direct_max,
            "same_command_across_state_max_abs_difference": latent_state_max,
            "tolerance": 1.0e-6,
            "passed": bool(latent_direct_max <= 1.0e-6 and latent_state_max <= 1.0e-6),
        }
        if not latent_route_audit["passed"]:
            raise RuntimeError(f"latent command routing failed audit: {latent_route_audit}")

        reference_goal = torch.cat(
            (motion_command.motion.joint_pos, motion_command.motion.joint_vel), dim=-1
        )
        goal_scale = pooled_goal_scale(
            motion_command.motion.joint_pos, motion_command.motion.joint_vel
        )
        goal_dim = int(reference_goal.shape[1])
        if goal_dim != 58:
            raise RuntimeError(f"registered branch goal width changed: {goal_dim}")

        active = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        survival = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        sum_error_a = torch.zeros(env.num_envs, device=device)
        sum_error_b = torch.zeros(env.num_envs, device=device)
        sum_separation_ab = torch.zeros(env.num_envs, device=device)
        branch_samples = 0
        snapshots: dict[int, dict[str, torch.Tensor]] = {}

        def physical_goal() -> torch.Tensor:
            return torch.cat(
                (motion_command.robot_joint_pos, motion_command.robot_joint_vel), dim=-1
            )

        def accumulate_branch(goal: torch.Tensor, offset: int) -> None:
            nonlocal branch_samples, sum_error_a, sum_error_b, sum_separation_ab
            if offset <= 0:
                raise ValueError("confirmatory branch samples must be strictly future-only")
            reference_a = reference_goal[branch_a_starts + offset]
            reference_b = reference_goal[branch_b_starts + offset]
            error_a, error_b = standardized_branch_squared_error(
                goal, reference_a, reference_b, goal_scale
            )
            separation_ab = ((reference_a - reference_b) / goal_scale).square().mean(
                dim=-1
            )
            sum_error_a = sum_error_a + error_a
            sum_error_b = sum_error_b + error_b
            sum_separation_ab = sum_separation_ab + separation_ab
            branch_samples += 1
            if offset in (HALF_SECOND_STEPS, ONE_SECOND_STEPS):
                q_a = sum_error_a / branch_samples
                q_b = sum_error_b / branch_samples
                q_ab = sum_separation_ab / branch_samples
                snapshots[offset] = {
                    "q_a": q_a.clone(),
                    "q_b": q_b.clone(),
                    "q_ab": q_ab.clone(),
                    "branch_coordinate": normalized_branch_coordinate(q_a, q_b, q_ab),
                }

        first_student_action: torch.Tensor | None = None
        first_teacher_action: torch.Tensor | None = None
        first_z_cmd: torch.Tensor | None = None
        primary_horizon_done_count = 0
        with torch.no_grad():
            for step in range(HORIZON_STEPS):
                expected_cursor = grid.command_start_steps + step
                if not torch.equal(
                    motion_command.time_steps[active], expected_cursor[active]
                ):
                    raise RuntimeError(f"active command cursor drifted at rollout step {step}")
                if not torch.equal(
                    motion_command.motion_ids[active], grid.command_sides[active]
                ):
                    raise RuntimeError(f"active motion route drifted at rollout step {step}")
                proprio, command = split_obs(obs_dict)
                latent_window = z_window()
                z_cmd = student.mu_prior(proprio, latent_window, command)
                actions = student.act(proprio, z_cmd)
                if not torch.isfinite(z_cmd).all() or not torch.isfinite(actions).all():
                    raise RuntimeError(f"nonfinite student output at rollout step {step}")
                if step == 0:
                    first_z_cmd = z_cmd.clone()
                    first_student_action = actions.clone()
                    first_teacher_action = teacher_action(obs_dict["actor_obs"]).clone()
                    if not torch.isfinite(first_teacher_action).all():
                        raise RuntimeError("routed teacher produced a nonfinite first action")
                actions = torch.where(active[:, None], actions, torch.zeros_like(actions))
                obs_dict, _, dones, extras = env.step({"actions": actions})
                _require_zero_mujoco_warp_overflow(
                    env, label=f"rollout step {step + 1}"
                )
                overflow_checks += 1
                dones = dones.bool()
                survival[active] += 1
                failed_now = active & dones
                survived_now = active & ~dones
                current_goal = physical_goal()
                current_root = env.simulator.robot_root_states
                if torch.any(survived_now) and (
                    not torch.isfinite(current_goal[survived_now]).all()
                    or not torch.isfinite(current_root[survived_now]).all()
                ):
                    raise RuntimeError(
                        f"nonfinite surviving physical state at rollout step {step + 1}"
                    )
                if torch.any(failed_now):
                    final_actor = extras.get("final_observations", {}).get("actor_obs")
                    if (
                        final_actor is None
                        or final_actor.shape != (env.num_envs, 154)
                        or not torch.isfinite(final_actor[failed_now]).all()
                    ):
                        raise RuntimeError(
                            f"missing or nonfinite terminal state at rollout step {step + 1}"
                        )
                offset = step + 1
                if offset <= ONE_SECOND_STEPS:
                    primary_horizon_done_count += int(dones.sum())
                    if torch.any(dones):
                        raise RuntimeError(
                            "a rollout reset inside the termination-suppressed primary horizon"
                        )
                    accumulate_branch(current_goal, offset)
                    if offset == ONE_SECOND_STEPS:
                        termination_control["suppress_reference"] = False
                active &= ~dones

        if set(snapshots) != {HALF_SECOND_STEPS, ONE_SECOND_STEPS}:
            raise RuntimeError("branch-distance snapshots were not completed")
        if first_z_cmd is None or first_student_action is None or first_teacher_action is None:
            raise RuntimeError("first-action audit was not recorded")
        completed = active.clone()
        half = snapshots[HALF_SECOND_STEPS]
        one = snapshots[ONE_SECOND_STEPS]
        env.termination_manager.check = original_termination_check
        runtime_hashes_end = {
            name: sha256_file(path) for name, path in runtime_paths.items()
        }
        if runtime_hashes_end != runtime_hashes_start:
            raise RuntimeError("a load-bearing E71 runtime changed during evaluation")
        all_frozen_hashes_end = {
            name: sha256_file(path) for name, path in all_frozen_paths.items()
        }
        if all_frozen_hashes_end != all_frozen_hashes_start:
            raise RuntimeError("a frozen E71 input or runtime changed during evaluation")
        report = {
            "protocol": SMOKE_PROTOCOL if smoke_pair_id is not None else PROTOCOL,
            "status": "complete",
            "arm": ARM_GOALS[arm],
            "student_arm": arm,
            "training_seed": training_seed,
            "evaluation_seed": int(tyro_cfg.training.seed),
            "smoke_pair_id": smoke_pair_id,
            "num_rollouts": env.num_envs,
            "pair_ids": _json_vector(grid.pair_ids),
            "state_sides": _json_vector(grid.state_sides),
            "command_sides": _json_vector(grid.command_sides),
            "state_start_steps": _json_vector(grid.state_start_steps),
            "command_start_steps": _json_vector(grid.command_start_steps),
            "state_motion_ids": _json_vector(grid.state_sides),
            "command_motion_ids": _json_vector(grid.command_sides),
            "q_a_0p5_s": _json_vector(half["q_a"]),
            "q_b_0p5_s": _json_vector(half["q_b"]),
            "q_ab_0p5_s": _json_vector(half["q_ab"]),
            "d_a_0p5_s": _json_vector(torch.sqrt(half["q_a"])),
            "d_b_0p5_s": _json_vector(torch.sqrt(half["q_b"])),
            "branch_coordinate_0p5_s": _json_vector(half["branch_coordinate"]),
            "q_a_1p0_s": _json_vector(one["q_a"]),
            "q_b_1p0_s": _json_vector(one["q_b"]),
            "q_ab_1p0_s": _json_vector(one["q_ab"]),
            "d_a_1p0_s": _json_vector(torch.sqrt(one["q_a"])),
            "d_b_1p0_s": _json_vector(torch.sqrt(one["q_b"])),
            "branch_coordinate_1p0_s": _json_vector(one["branch_coordinate"]),
            "branch_samples_0p5_s": HALF_SECOND_STEPS,
            "branch_samples_1p0_s": ONE_SECOND_STEPS,
            "goal_definition": (
                "mapped joint_pos + joint_vel; pooled per-dimension scale; future-only "
                "mean squared distances; C=(Q_A-Q_B)/(Q_AB+1e-8)"
            ),
            "goal_scale": _json_vector(goal_scale),
            "completed": _json_vector(completed),
            "completion_rate": float(completed.float().mean()),
            "survival_s": _json_vector(survival * float(env.dt)),
            "mean_survival_s": float(survival.float().mean() * env.dt),
            "survival_definition": (
                "reference termination suppressed through 1.0 s, then restored through 10.0 s"
            ),
            "first_z_cmd": _json_vector(first_z_cmd),
            "first_student_action": _json_vector(first_student_action),
            "first_teacher_action": _json_vector(first_teacher_action),
            "proprio_audit": proprio_audit,
            "raw_proprio_audit": raw_proprio_audit,
            "observation_layout_audit": observation_layout_audit,
            "latent_route_audit": latent_route_audit,
            "full_state_contract": full_state_contract,
            "full_state_audit": full_state_audit,
            "overflow_audit": {
                "checked_transitions": overflow_checks,
                "nonzero_entries": 0,
                "passed": True,
            },
            "evaluation_conditions": evaluation_conditions,
            "termination_audit": {
                "primary_horizon_done_count": primary_horizon_done_count,
                "suppressed_steps": ONE_SECOND_STEPS,
                "reference_termination_restored_after_primary_horizon": True,
                "passed": bool(primary_horizon_done_count == 0),
            },
            "warmup_audit": warmup_audit,
            "student_checkpoint": str(student_path.resolve()),
            "student_checkpoint_sha256": student_hash,
            "teacher_manifest": str(manifest_path.resolve()),
            "teacher_manifest_sha256": manifest_hash,
            "teacher_ckpts": [str(path.resolve()) for path in teacher_paths],
            "teacher_checkpoint_sha256": teacher_hashes,
            "motion_files": [str(path) for path in motion_paths],
            "motion_sha256": motion_hashes,
            "ambiguity_precheck": str(precheck_path.resolve()),
            "ambiguity_precheck_sha256": precheck_hash,
            "freeze_manifest": str(freeze_manifest_path.resolve()),
            "freeze_manifest_sha256": freeze_manifest_hash,
            "runtime": str(pathlib.Path(__file__).resolve()),
            "runtime_sha256": runtime_hashes_start["evaluator"],
            "reset_runtime": str(pathlib.Path(counterfactual_eval.__file__).resolve()),
            "reset_runtime_sha256": runtime_hashes_start["reset_layer"],
            "runtime_toctou_audit": {
                "startup_sha256": runtime_hashes_start,
                "completion_sha256": runtime_hashes_end,
                "all_frozen_startup_sha256": all_frozen_hashes_start,
                "all_frozen_completion_sha256": all_frozen_hashes_end,
                "passed": True,
            },
            "invoked_argv": invoked_argv,
            "imported_runtime_paths": imported_runtime_paths,
            "imported_runtime_sha256": imported_runtime_hashes,
            "repository_commits": observed_commits,
            "working_trees": observed_worktrees,
            "runtime_environment": observed_environment,
            "simulator_num_envs": env.num_envs,
            "policy_dt_s": float(env.dt),
            "motion_fps": float(motion_command.motion.fps),
        }
        _write_json_once(report_path, report)
        print(json.dumps(report), flush=True)
    finally:
        close_simulation_app(sim_app)


if __name__ == "__main__":
    main()
