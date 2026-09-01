#!/usr/bin/env python3
"""Generate a hash-bound, fail-closed G0 controller/standing-state report.

The report compares one concrete saved Isaac Lab tracker recipe with one concrete
saved MuJoCo/MJWarp recipe and the exact G1 URDF, MJCF, and currently registered USD
bundle.  It performs no simulation and never upgrades source inspection into live
runtime evidence.  Missing live nominal/controller/reset observations therefore keep
the overall G0 gate false.

Run this script in an environment containing PyYAML (Holosoma's ``hssim`` environment
already does).  It does not launch Isaac Sim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from snmr.controller_parity import (  # noqa: E402
    CONTROLLER_PARITY_SCHEMA_VERSION,
    ControllerRecipe,
    compute_action_scales,
    evaluate_controller_parity,
    expand_joint_pattern_values,
    parse_mjcf_joint_model,
    parse_urdf_joint_model,
)
from snmr.fk_parity import capture_directory_bundle, capture_urdf_bundle  # noqa: E402
from snmr.provenance import ArtifactSnapshot, git_revision  # noqa: E402


DEFAULT_HOLOSOMA = Path("/home/robotixx/holosoma")
DEFAULT_ROBOT_ROOT = DEFAULT_HOLOSOMA / "src/holosoma/holosoma/data/robots"
DEFAULT_MJCF_ROOT = DEFAULT_ROBOT_ROOT / "g1"
DEFAULT_MJCF_ENTRYPOINT = "scenes/scene_g1_29dof_wbt_plane.xml"
DEFAULT_MJCF_ROBOT_MEMBER = "g1_29dof.xml"
DEFAULT_URDF = DEFAULT_ROBOT_ROOT / "g1/g1_29dof.urdf"
DEFAULT_USD_ROOT = DEFAULT_ROBOT_ROOT / "converted_rank0"
DEFAULT_USD_ENTRYPOINT = "g1_29dof.usd"
DEFAULT_PHYSX_RUN = (
    DEFAULT_HOLOSOMA
    / "logs/WholeBodyTracking/20260121_223142-g1_29dof_wbt_manager-locomotion"
)
DEFAULT_PHYSX_CONFIG = DEFAULT_PHYSX_RUN / "holosoma_config.yaml"
DEFAULT_PHYSX_CHECKPOINT = DEFAULT_PHYSX_RUN / "model_39999.pt"
DEFAULT_PHYSX_TRAIN_LOG = DEFAULT_PHYSX_RUN / "train.log"
DEFAULT_MUJOCO_RUN = Path(
    "/data/robotixx/snmr-research/e67/teacher_holosoma_logs/WholeBodyTracking/"
    "20260808_084940-e67_walk1_subject5_teacher_seed0-locomotion"
)
DEFAULT_MUJOCO_CONFIG = DEFAULT_MUJOCO_RUN / "holosoma_config.yaml"
DEFAULT_MUJOCO_CHECKPOINT = DEFAULT_MUJOCO_RUN / "model_08000.pt"
DEFAULT_ISAAC_LAB = Path("/home/robotixx/.holosoma_deps/IsaacLab")
DEFAULT_NEWTON = Path("/home/robotixx/newton")

SOURCE_PATHS: Mapping[str, str] = {
    "experiment_config": "src/holosoma/holosoma/config_values/wbt/g1/experiment.py",
    "robot_config": "src/holosoma/holosoma/config_values/robot.py",
    "randomization_config": "src/holosoma/holosoma/config_values/wbt/g1/randomization.py",
    "motion_command_config": "src/holosoma/holosoma/config_values/wbt/g1/command.py",
    "simulator_config": "src/holosoma/holosoma/config_values/simulator.py",
    "joint_position_action": "src/holosoma/holosoma/managers/action/terms/joint_control.py",
    "base_task": "src/holosoma/holosoma/envs/base_task/base_task.py",
    "wbt_manager": "src/holosoma/holosoma/envs/wbt/wbt_manager.py",
    "wbt_motion_command": "src/holosoma/holosoma/managers/command/terms/wbt.py",
    "isaac_adapter": "src/holosoma/holosoma/simulator/isaacsim/isaacsim.py",
    "mujoco_adapter": "src/holosoma/holosoma/simulator/mujoco/mujoco.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physx-config", type=Path, default=DEFAULT_PHYSX_CONFIG)
    parser.add_argument("--physx-checkpoint", type=Path, default=DEFAULT_PHYSX_CHECKPOINT)
    parser.add_argument("--physx-train-log", type=Path, default=DEFAULT_PHYSX_TRAIN_LOG)
    parser.add_argument("--mujoco-config", type=Path, default=DEFAULT_MUJOCO_CONFIG)
    parser.add_argument("--mujoco-checkpoint", type=Path, default=DEFAULT_MUJOCO_CHECKPOINT)
    parser.add_argument(
        "--physx-fk-report",
        type=Path,
        help="Optional prior live PhysX FK report used only for joint-mapping evidence.",
    )
    parser.add_argument("--mjcf-root", type=Path, default=DEFAULT_MJCF_ROOT)
    parser.add_argument("--mjcf-entrypoint", default=DEFAULT_MJCF_ENTRYPOINT)
    parser.add_argument("--mjcf-robot-member", default=DEFAULT_MJCF_ROBOT_MEMBER)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--usd-root", type=Path, default=DEFAULT_USD_ROOT)
    parser.add_argument("--usd-entrypoint", default=DEFAULT_USD_ENTRYPOINT)
    parser.add_argument("--holosoma-root", type=Path, default=DEFAULT_HOLOSOMA)
    parser.add_argument("--snmr-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--isaac-lab-root", type=Path, default=DEFAULT_ISAAC_LAB)
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    """Atomically publish a new JSON artifact without overwriting any existing path."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # Hard-link publication is atomic and fails if the destination already exists.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_yaml_bytes(data: bytes, *, label: str) -> dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read saved Holosoma recipes; run with hssim Python"
        ) from exc
    parsed = yaml.safe_load(data)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to a mapping")
    return parsed


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _recipe_from_saved_config(document: Mapping[str, object], *, source: str) -> ControllerRecipe:
    robot = _mapping(document.get("robot"), label=f"{source}.robot")
    control = _mapping(robot.get("control"), label=f"{source}.robot.control")
    init_state = _mapping(robot.get("init_state"), label=f"{source}.robot.init_state")
    simulator = _mapping(document.get("simulator"), label=f"{source}.simulator")
    simulator_cfg = _mapping(simulator.get("config"), label=f"{source}.simulator.config")
    sim = _mapping(simulator_cfg.get("sim"), label=f"{source}.simulator.config.sim")

    joint_names = tuple(str(value) for value in robot.get("dof_names", []))
    stiffness = _mapping(control.get("stiffness"), label=f"{source}.stiffness")
    damping = _mapping(control.get("damping"), label=f"{source}.damping")
    kp = expand_joint_pattern_values(joint_names, stiffness, label=f"{source}.stiffness")
    kd = expand_joint_pattern_values(joint_names, damping, label=f"{source}.damping")
    effort = np.asarray(robot.get("dof_effort_limit_list"), dtype=np.float64)
    velocity = np.asarray(robot.get("dof_vel_limit_list"), dtype=np.float64)
    lower = np.asarray(robot.get("dof_pos_lower_limit_list"), dtype=np.float64)
    upper = np.asarray(robot.get("dof_pos_upper_limit_list"), dtype=np.float64)
    joint_friction = np.asarray(robot.get("dof_joint_friction_list"), dtype=np.float64)
    base_scale = float(control.get("action_scale"))
    by_effort = bool(control.get("action_scales_by_effort_limit_over_p_gain"))
    action_scales = compute_action_scales(
        base_scale=base_scale,
        effort_limits=effort,
        kp=kp,
        by_effort_over_kp=by_effort,
    )

    randomization = _mapping(document.get("randomization"), label=f"{source}.randomization")
    setup_terms = _mapping(randomization.get("setup_terms"), label=f"{source}.setup_terms")
    delay_term = _mapping(
        setup_terms.get("setup_action_delay_buffers"), label=f"{source}.action_delay_term"
    )
    delay_params = _mapping(delay_term.get("params"), label=f"{source}.action_delay_params")
    reset_terms = _mapping(randomization.get("reset_terms"), label=f"{source}.reset_terms")
    dof_reset = _mapping(reset_terms.get("randomize_dof_state"), label=f"{source}.dof_reset")
    dof_reset_params = _mapping(dof_reset.get("params"), label=f"{source}.dof_reset.params")

    command = _mapping(document.get("command"), label=f"{source}.command")
    command_setup = _mapping(command.get("setup_terms"), label=f"{source}.command.setup_terms")
    motion_command = _mapping(command_setup.get("motion_command"), label=f"{source}.motion_command")
    motion_params = _mapping(motion_command.get("params"), label=f"{source}.motion_command.params")
    motion_config = _mapping(motion_params.get("motion_config"), label=f"{source}.motion_config")
    motion_noise = _mapping(
        motion_config.get("noise_to_initial_pose"), label=f"{source}.motion_reset_noise"
    )
    reset_noise = {
        "motion_state_source": "reference_motion_at_sampled_time_step",
        "motion_noise": motion_noise,
        "post_motion_dof_randomization": dof_reset_params,
        "startup_randomization": setup_terms,
        "phase_sampling": {
            key: motion_config.get(key)
            for key in (
                "use_adaptive_timesteps_sampler",
                "start_at_timestep_zero_prob",
                "reassign_on_reset",
                "sampling_mode",
            )
        },
    }

    fps = int(sim.get("fps"))
    decimation = int(sim.get("control_decimation"))
    nominal = _mapping(init_state.get("default_joint_angles"), label=f"{source}.nominal_q")
    return ControllerRecipe.from_sequences(
        source=source,
        backend=str(simulator.get("_target_")),
        joint_names=joint_names,
        nominal_q=[float(nominal[name]) for name in joint_names],
        root_position=init_state.get("pos", []),
        root_quaternion_xyzw=init_state.get("rot", []),
        root_linear_velocity=init_state.get("lin_vel", []),
        root_angular_velocity=init_state.get("ang_vel", []),
        kp=kp,
        kd=kd,
        lower_limits=lower,
        upper_limits=upper,
        effort_limits=effort,
        velocity_limits=velocity,
        joint_friction=joint_friction,
        action_scales=action_scales,
        action_scale_base=base_scale,
        action_scale_by_effort_over_kp=by_effort,
        control_type=str(control.get("control_type")),
        clip_actions=bool(control.get("clip_actions")),
        action_clip_value=float(control.get("action_clip_value")),
        clip_torques=bool(control.get("clip_torques")),
        simulation_dt=1.0 / fps,
        control_dt=decimation / fps,
        decimation=decimation,
        latency_enabled=bool(delay_params.get("enabled")),
        latency_step_range=delay_params.get("ctrl_delay_step_range", []),
        reset_semantics="motion_reference_state_plus_registered_noise_then_shared_dof_reset",
        reset_noise=reset_noise,
    )


def _contains_all(snapshot: ArtifactSnapshot, snippets: tuple[bytes, ...]) -> bool:
    return all(snippet in snapshot.data for snippet in snippets)


def _mujoco_nominal_source_binding(data: bytes) -> bool:
    """Return true only if a nominal joint write is present after the last data reset."""

    initial_joint_call = data.rfind(b"self._set_initial_joint_angles()")
    reset_data_call = data.rfind(b"mujoco.mj_resetData(self.root_model, self.root_data)")
    return bool(
        initial_joint_call >= 0
        and reset_data_call >= 0
        and initial_joint_call > reset_data_call
    )


def _source_evidence(sources: Mapping[str, ArtifactSnapshot]) -> dict[str, bool | None]:
    action = sources["joint_position_action"]
    isaac = sources["isaac_adapter"]
    mujoco = sources["mujoco_adapter"]
    base_task = sources["base_task"]
    motion = sources["wbt_motion_command"]
    mujoco_nominal_survives_prepare = _mujoco_nominal_source_binding(mujoco.data)
    return {
        "shared_position_target_formula": _contains_all(
            action,
            (
                b"actions_scaled = actions * self.action_scales",
                b"actions_scaled + self.env.default_dof_pos - self.env.simulator.dof_pos",
                b"torch.clip(torques, -self.env.torque_limits, self.env.torque_limits)",
            ),
        ),
        "isaac_effort_target_adapter": b"set_joint_effort_target(torques, joint_ids=self.dof_ids)"
        in isaac.data,
        "mujoco_effort_target_adapter": _contains_all(
            mujoco,
            (
                b"ctrl_tensor[:] = torques",
                b"self.root_data.ctrl[actuator_id] = torques_np[i]",
            ),
        ),
        "isaac_simulation_dt_binding": b"dt=1.0 / self.simulator_config.sim.fps" in isaac.data,
        "mujoco_simulation_dt_binding": _contains_all(
            mujoco,
            (
                b"self.sim_dt = 1.0 / self.simulator_config.sim.fps",
                b"self.root_model.opt.timestep = self.sim_dt",
            ),
        ),
        "control_decimation_binding": _contains_all(
            base_task,
            (
                b"self.dt = simulator_config.config.sim.control_decimation * self.sim_dt",
                b"for _ in range(self.simulator.simulator_config.sim.control_decimation)",
            ),
        ),
        "isaac_standing_state_binding": _contains_all(
            isaac,
            (
                b"ArticulationCfg.InitialStateCfg(",
                b"joint_pos={joint_name: joint_angle",
                b"joint_vel={\".*\": 0.0}",
            ),
        ),
        # Current source sets nominal joint q before prepare_sim calls mj_resetData,
        # and _set_robot_initial_state restores only the free root.  This is a hard
        # source-level failure until a post-reset nominal write or live proof exists.
        "mujoco_standing_state_binding": mujoco_nominal_survives_prepare,
        "shared_motion_reset_binding": _contains_all(
            motion,
            (
                b"root_pos = self.root_pos_w[env_ids].clone()",
                b"dof_pos = self.joint_pos[env_ids].clone()",
                b"self._env.simulator.dof_pos[env_ids] = target_dof_pos",
                b"self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos",
            ),
        ),
    }


def _physx_fk_evidence(
    snapshot: ArtifactSnapshot | None,
    *,
    canonical_joint_names: tuple[str, ...],
    usd_sha256: str,
    urdf_sha256: str,
) -> tuple[bool | None, dict[str, object] | None]:
    if snapshot is None:
        return None, None
    try:
        report = json.loads(snapshot.data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid PhysX FK report JSON: {exc}") from exc
    runtime = _mapping(report.get("runtime"), label="physx_fk_report.runtime")
    inputs = _mapping(report.get("inputs"), label="physx_fk_report.inputs")
    usd = _mapping(inputs.get("usd_bundle"), label="physx_fk_report.inputs.usd_bundle")
    urdf = _mapping(inputs.get("urdf_bundle"), label="physx_fk_report.inputs.urdf_bundle")
    runtime_names = tuple(str(value) for value in runtime.get("joint_names", []))
    mapping_pass = bool(
        report.get("status") == "pass"
        and report.get("g0_pass") is True
        and runtime.get("pose_source") == "isaaclab.assets.Articulation.data.body_link_{pos,quat}_w"
        and len(runtime_names) == len(canonical_joint_names)
        and set(runtime_names) == set(canonical_joint_names)
        and usd.get("sha256") == usd_sha256
        and urdf.get("sha256") == urdf_sha256
    )
    return mapping_pass, {
        "artifact": snapshot.manifest(),
        "status": report.get("status"),
        "g0_pass": report.get("g0_pass"),
        "pose_source": runtime.get("pose_source"),
        "runtime_joint_names": list(runtime_names),
        "usd_bundle_sha256": usd.get("sha256"),
        "urdf_bundle_sha256": urdf.get("sha256"),
    }


def _revision(path: Path) -> dict[str, object]:
    return asdict(git_revision(path))


def main() -> int:
    args = parse_args()
    out_json = args.out_json.expanduser().resolve()
    if out_json.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {out_json}")
    base_report: dict[str, object] = {
        "schema_version": CONTROLLER_PARITY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "status": "backend_error",
        "g0_controller_contract_pass": False,
        "g0_pass": False,
    }
    try:
        worker_source = ArtifactSnapshot.capture(Path(__file__))
        contract_source = ArtifactSnapshot.capture(REPO_ROOT / "snmr/controller_parity.py")
        physx_config = ArtifactSnapshot.capture(args.physx_config)
        physx_checkpoint = ArtifactSnapshot.capture(args.physx_checkpoint)
        physx_train_log = ArtifactSnapshot.capture(args.physx_train_log)
        mujoco_config = ArtifactSnapshot.capture(args.mujoco_config)
        mujoco_checkpoint = ArtifactSnapshot.capture(args.mujoco_checkpoint)
        physx_fk = (
            ArtifactSnapshot.capture(args.physx_fk_report)
            if args.physx_fk_report is not None
            else None
        )

        urdf_manifest, urdf_bytes = capture_urdf_bundle(args.urdf)
        mjcf_manifest, mjcf_members = capture_directory_bundle(
            args.mjcf_root, entrypoint=args.mjcf_entrypoint
        )
        member_map = dict(mjcf_members)
        if args.mjcf_robot_member not in member_map:
            raise FileNotFoundError(args.mjcf_root / args.mjcf_robot_member)
        usd_manifest, _usd_members = capture_directory_bundle(
            args.usd_root, entrypoint=args.usd_entrypoint
        )

        sources = {
            name: ArtifactSnapshot.capture(args.holosoma_root / relative)
            for name, relative in SOURCE_PATHS.items()
        }
        physx_document = _load_yaml_bytes(physx_config.data, label="physx_config")
        mujoco_document = _load_yaml_bytes(mujoco_config.data, label="mujoco_config")
        physx_recipe = _recipe_from_saved_config(
            physx_document, source="saved_physx_tracker_recipe"
        )
        mujoco_recipe = _recipe_from_saved_config(
            mujoco_document, source="saved_e67_mjwarp_recipe"
        )
        urdf = parse_urdf_joint_model(urdf_bytes, source="urdf")
        mjcf = parse_mjcf_joint_model(member_map[args.mjcf_robot_member], source="mjcf")

        mapping_evidence, fk_manifest = _physx_fk_evidence(
            physx_fk,
            canonical_joint_names=physx_recipe.joint_names,
            usd_sha256=str(usd_manifest["sha256"]),
            urdf_sha256=str(urdf_manifest["sha256"]),
        )
        live_evidence: dict[str, bool | None] = {
            "physx_runtime_joint_mapping": mapping_evidence,
            "physx_live_nominal_state": None,
            "mujoco_live_nominal_state": None,
            "physx_live_controller_parameters": None,
            "mujoco_live_controller_parameters": None,
            "paired_reset_same_motion_seed": None,
            # The saved January run did not preserve a USD hash or Holosoma Git SHA.
            "registered_usd_matches_tracker_training_asset": None,
            "tracker_training_holosoma_revision": None,
            # These are two independently trained, backend-specific checkpoints.
            "same_frozen_checkpoint_both_backends": False,
        }
        evaluation = evaluate_controller_parity(
            physx_recipe=physx_recipe,
            mujoco_recipe=mujoco_recipe,
            urdf=urdf,
            mjcf=mjcf,
            source_evidence=_source_evidence(sources),
            live_evidence=live_evidence,
        )

        base_report.update(evaluation)
        base_report["status"] = (
            "pass"
            if evaluation["g0_controller_contract_pass"]
            else ("incomplete" if not evaluation["failed_checks"] else "fail")
        )
        base_report["gate_reason"] = (
            "All source and required live controller evidence passed."
            if evaluation["g0_controller_contract_pass"]
            else "Controller/standing-state parity fails closed on hard mismatches and missing live evidence."
        )
        base_report["inputs"] = {
            "worker_source": worker_source.manifest(),
            "contract_source": contract_source.manifest(),
            "physx_tracker_config": physx_config.manifest(),
            "physx_tracker_checkpoint": physx_checkpoint.manifest(),
            "physx_training_log": physx_train_log.manifest(),
            "mujoco_recipe_config": mujoco_config.manifest(),
            "mujoco_checkpoint": mujoco_checkpoint.manifest(),
            "physx_fk_report": fk_manifest,
            "urdf_bundle": urdf_manifest,
            "mjcf_bundle": mjcf_manifest,
            "usd_bundle": usd_manifest,
            "holosoma_sources": {
                name: snapshot.manifest() for name, snapshot in sources.items()
            },
        }
        base_report["revisions"] = {
            "snmr": _revision(args.snmr_root),
            "holosoma_current_checkout": _revision(args.holosoma_root),
            "isaac_lab": _revision(args.isaac_lab_root),
            "newton": _revision(args.newton_root),
            "tracker_training_holosoma_revision": None,
        }
        base_report["provenance_notes"] = [
            "The current Holosoma checkout is source evidence only; the saved January Isaac run records no Git SHA.",
            "The current converted_rank0 USD is hash-bound here and in the supplied live FK "
            "report, but its identity at tracker training time is not recorded.",
            "The saved PhysX and E67 MJWarp checkpoints are distinct policies; this report "
            "does not call them a paired frozen tracker.",
            "The standalone FK scene declares timestep 0.002 s, while Holosoma overrides "
            "the tracker runtime to the saved recipe's 0.005 s simulation_dt.",
        ]
        for snapshot in (
            worker_source,
            contract_source,
            physx_config,
            physx_checkpoint,
            physx_train_log,
            mujoco_config,
            mujoco_checkpoint,
            *sources.values(),
        ):
            snapshot.assert_source_unchanged()
        base_report["completed_at"] = _utc_now()
        _write_json_exclusive(out_json, base_report)
        print(
            json.dumps(
                {
                    "report": str(out_json),
                    "status": base_report["status"],
                    "g0_controller_contract_pass": base_report["g0_controller_contract_pass"],
                    "g0_pass": base_report["g0_pass"],
                },
                sort_keys=True,
            )
        )
        return 0 if bool(base_report["g0_controller_contract_pass"]) else 1
    except BaseException as exc:
        base_report["status"] = "backend_error"
        base_report["gate_reason"] = "Contract extraction did not complete; G0 fails closed."
        base_report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        base_report["completed_at"] = _utc_now()
        _write_json_exclusive(out_json, base_report)
        print(
            json.dumps(
                {"report": str(out_json), "status": "backend_error", "g0_pass": False},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
