#!/usr/bin/env python
"""Qualify an E70 ONNX through Holosoma's loopback-only CPU MuJoCo path.

Run this script with Holosoma's ``hsmujoco`` Python environment.  It starts the
production inference process in its separate pinned environment, binds both
ends to ``lo``, performs a gantry-assisted first-pose transition, starts motion
while supported, and releases only after an upright preflight.  It then runs
500 registered motion frames without gantry support, requests the production
stiff-hold stop, and writes a machine-readable report.  It cannot select a
physical interface.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import platform
import pty
import signal
import subprocess
import time
from typing import Any

import numpy as np


HOLOSOMA_ROOT = pathlib.Path("/home/robotixx/holosoma")
INFERENCE_PYTHON = pathlib.Path(
    "/home/robotixx/.holosoma_deps/miniconda3/envs/hsinference/bin/python"
)
POLICY_ENTRY = (
    HOLOSOMA_ROOT
    / "src/holosoma_inference/holosoma_inference/run_policy.py"
)
MOTION_END_TIMESTEP = 700
SIMULATION_HZ = 2000
SAMPLE_HZ = 200

FIRST_POSE_S = 3.0
POLICY_START_S = 12.0
MOTION_START_S = 13.0
GANTRY_LOWER_START_S = 14.0
GANTRY_RELEASE_S = 17.0
SAFE_HOLD_S = 28.0
END_S = 31.0
SAFETY_HANDOFF_S = 27.0


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_stats() -> dict[str, Any]:
    return {
        "samples": 0,
        "minimum_base_height_m": float("inf"),
        "minimum_up_axis_z": float("inf"),
        "maximum_abs_joint_velocity_rad_s": 0.0,
        "maximum_abs_torque_nm": 0.0,
        "minimum_measured_joint_limit_margin_rad": float("inf"),
        "minimum_commanded_joint_limit_margin_rad": float("inf"),
        "nonfinite_samples": 0,
        "measured_joint_limit_violation_samples": 0,
        "commanded_joint_limit_violation_samples": 0,
        "velocity_limit_violation_samples": 0,
        "torque_limit_violation_samples": 0,
        "torque_saturation_samples": 0,
        "first_base_height_violation_time_s": None,
        "first_up_axis_violation_time_s": None,
        "first_measured_joint_limit_violation_time_s": None,
        "first_commanded_joint_limit_violation_time_s": None,
        "first_velocity_limit_violation_time_s": None,
        "first_torque_limit_violation_time_s": None,
        "minimum_measured_joint_limit_index": None,
        "minimum_commanded_joint_limit_index": None,
    }


def _update_phase(
    stats: dict[str, Any],
    simulator: Any,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    velocity_limit: np.ndarray,
    torque_limit: np.ndarray,
    *,
    elapsed_s: float,
    minimum_base_height_m: float,
    minimum_up_axis_z: float,
) -> None:
    root = simulator.robot_root_states[0].detach().cpu().numpy()
    joint_pos = simulator.dof_pos[0].detach().cpu().numpy()
    joint_vel = simulator.dof_vel[0].detach().cpu().numpy()
    torque = simulator.get_dof_forces().detach().cpu().numpy()
    quat_xyzw = root[3:7]
    up_axis_z = 1.0 - 2.0 * (quat_xyzw[0] ** 2 + quat_xyzw[1] ** 2)
    measured_margin = np.minimum(joint_pos - joint_lower, joint_upper - joint_pos)

    bridge = simulator.bridge.robot_bridge
    command = np.asarray(getattr(bridge.low_cmd, "q_target", []), dtype=np.float64)
    if command.shape[0] >= joint_lower.shape[0]:
        command = command[: joint_lower.shape[0]]
        command_margin = np.minimum(command - joint_lower, joint_upper - command)
    else:
        command_margin = np.full_like(joint_lower, -np.inf)

    values = (root, joint_pos, joint_vel, torque, command)
    finite = all(np.isfinite(value).all() for value in values)
    stats["samples"] += 1
    stats["minimum_base_height_m"] = min(
        stats["minimum_base_height_m"], float(root[2])
    )
    stats["minimum_up_axis_z"] = min(stats["minimum_up_axis_z"], float(up_axis_z))
    stats["maximum_abs_joint_velocity_rad_s"] = max(
        stats["maximum_abs_joint_velocity_rad_s"], float(np.max(np.abs(joint_vel)))
    )
    stats["maximum_abs_torque_nm"] = max(
        stats["maximum_abs_torque_nm"], float(np.max(np.abs(torque)))
    )
    measured_min_index = int(np.argmin(measured_margin))
    measured_min = float(measured_margin[measured_min_index])
    if measured_min < stats["minimum_measured_joint_limit_margin_rad"]:
        stats["minimum_measured_joint_limit_margin_rad"] = measured_min
        stats["minimum_measured_joint_limit_index"] = measured_min_index
    commanded_min_index = int(np.argmin(command_margin))
    commanded_min = float(command_margin[commanded_min_index])
    if commanded_min < stats["minimum_commanded_joint_limit_margin_rad"]:
        stats["minimum_commanded_joint_limit_margin_rad"] = commanded_min
        stats["minimum_commanded_joint_limit_index"] = commanded_min_index
    stats["nonfinite_samples"] += int(not finite)
    stats["measured_joint_limit_violation_samples"] += int(
        np.any(measured_margin < -1.0e-6)
    )
    stats["commanded_joint_limit_violation_samples"] += int(
        np.any(command_margin < -1.0e-6)
    )
    stats["velocity_limit_violation_samples"] += int(
        np.any(np.abs(joint_vel) > velocity_limit + 1.0e-6)
    )
    stats["torque_limit_violation_samples"] += int(
        np.any(np.abs(torque) > torque_limit + 1.0e-5)
    )
    stats["torque_saturation_samples"] += int(
        np.any(np.abs(torque) >= torque_limit - 1.0e-3)
    )
    first_breaches = (
        ("first_base_height_violation_time_s", root[2] < minimum_base_height_m),
        ("first_up_axis_violation_time_s", up_axis_z < minimum_up_axis_z),
        ("first_measured_joint_limit_violation_time_s", measured_min < -1.0e-6),
        ("first_commanded_joint_limit_violation_time_s", commanded_min < -1.0e-6),
        (
            "first_velocity_limit_violation_time_s",
            np.any(np.abs(joint_vel) > velocity_limit + 1.0e-6),
        ),
        (
            "first_torque_limit_violation_time_s",
            np.any(np.abs(torque) > torque_limit + 1.0e-5),
        ),
    )
    for key, breached in first_breaches:
        if breached and stats[key] is None:
            stats[key] = float(elapsed_s)


def _finite_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in stats.items()
    }


def _send_key(process: subprocess.Popen, input_fd: int, key: str) -> None:
    if process.poll() is not None:
        raise RuntimeError(f"policy process exited before command {key!r}")
    os.write(input_fd, key.encode())


def _wait_for_log_marker(
    process: subprocess.Popen,
    log_path: pathlib.Path,
    marker: str,
    *,
    timeout_s: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"policy process exited before initialization marker {marker!r}"
            )
        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        if marker in text:
            return
        time.sleep(0.02)
    raise TimeoutError(f"policy initialization marker not observed: {marker!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--domain-id", type=int, default=70)
    parser.add_argument("--minimum-base-height-m", type=float, default=0.45)
    parser.add_argument("--minimum-up-axis-z", type=float, default=0.5)
    parser.add_argument("--release-length-m", type=float, default=2.0)
    parser.add_argument(
        "--safety-handoff",
        action="store_true",
        help="switch from WBT to Holosoma's default safety locomotion policy",
    )
    args = parser.parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if not INFERENCE_PYTHON.is_file() or not POLICY_ENTRY.is_file():
        raise FileNotFoundError("pinned Holosoma inference runtime is incomplete")
    if args.domain_id < 0 or args.minimum_base_height_m <= 0:
        raise ValueError("loopback domain and height threshold are invalid")
    if not -1.0 <= args.minimum_up_axis_z <= 1.0:
        raise ValueError("up-axis threshold must lie in [-1,1]")
    motion_end_timestep = 750 if args.safety_handoff else MOTION_END_TIMESTEP
    safe_hold_start_s = SAFETY_HANDOFF_S if args.safety_handoff else SAFE_HOLD_S

    import torch

    import holosoma.config_values.robot
    import holosoma.config_values.run_sim
    import holosoma.config_values.terrain
    from holosoma.config_types.run_sim import RunSimConfig, default_training_config
    from holosoma.config_types.simulator import BridgeConfig, MujocoBackend, VirtualGantryCfg
    from holosoma.utils.rate import RateLimiter
    from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment

    bridge = BridgeConfig(
        enabled=True,
        use_joystick=False,
        domain_id=args.domain_id,
        interface="lo",
        rate_limit_dt=None,
        use_ros=False,
    )
    gantry = VirtualGantryCfg(enabled=True)
    base_simulator = holosoma.config_values.run_sim.mujoco
    simulator = dataclasses.replace(
        base_simulator,
        config=dataclasses.replace(
            base_simulator.config,
            debug_viz=False,
            mujoco_backend=MujocoBackend.CLASSIC,
            bridge=bridge,
            virtual_gantry=gantry,
        ),
    )
    config = RunSimConfig(
        simulator=simulator,
        robot=holosoma.config_values.robot.g1_29dof,
        terrain=holosoma.config_values.terrain.terrain_locomotion_plane,
        training=dataclasses.replace(default_training_config(), headless=True),
        device="cpu",
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    policy_log = args.report.with_suffix(".policy.log")
    policy_env = os.environ.copy()
    policy_env.update(
        {
            "PYTHONNOUSERSITE": "0",
            "PYTHONPATH": "",
            "PYTHONUNBUFFERED": "1",
            "LOGURU_LEVEL": "INFO",
        }
    )
    policy_command = [
        str(INFERENCE_PYTHON),
        str(POLICY_ENTRY),
        "inference:g1-29dof-wbt",
        "--task.model-path",
        str(args.onnx.resolve()),
        "--task.no-use-joystick",
        "--task.use-sim-time",
        "--task.rl-rate",
        "50",
        "--task.interface",
        "lo",
        "--task.domain-id",
        str(args.domain_id),
        "--task.motion-end-timestep",
        str(motion_end_timestep),
    ]
    if not args.safety_handoff:
        policy_command.extend(["--secondary", "none"])

    joint_lower = np.asarray(config.robot.dof_pos_lower_limit_list, dtype=np.float64)
    joint_upper = np.asarray(config.robot.dof_pos_upper_limit_list, dtype=np.float64)
    velocity_limit = np.asarray(config.robot.dof_vel_limit_list, dtype=np.float64)
    torque_limit = np.asarray(config.robot.dof_effort_limit_list, dtype=np.float64)
    phases = {
        "gantry_assisted_initialization": _phase_stats(),
        "gantry_assisted_motion": _phase_stats(),
        "unassisted_motion": _phase_stats(),
        "safe_hold": _phase_stats(),
    }
    events: list[dict[str, Any]] = []
    policy_process: subprocess.Popen | None = None
    policy_input_fd: int | None = None
    policy_exit_code: int | None = None
    exception: str | None = None

    env, device, simulation_app = setup_simulation_environment(config, device="cpu")
    try:
        with DirectSimulation(config, env, device, simulation_app) as direct:
            sim = direct.simulator
            with policy_log.open("w") as log_handle:
                policy_input_fd, policy_slave_fd = pty.openpty()
                policy_process = subprocess.Popen(
                    policy_command,
                    cwd=HOLOSOMA_ROOT,
                    env=policy_env,
                    stdin=policy_slave_fd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                os.close(policy_slave_fd)
                # Do not advance physics while the production runtime is blocked
                # on its stiff-hold confirmation.  Otherwise the uncontrolled
                # arm transient is incorrectly attributed to the controller.
                _wait_for_log_marker(
                    policy_process, policy_log, "Press Enter to continue..."
                )
                _send_key(policy_process, policy_input_fd, "\n")
                _wait_for_log_marker(
                    policy_process, policy_log, "Policy initialized successfully"
                )
                events.append(
                    {
                        "event": "stiff_hold_confirmed_before_physics",
                        "time_s": 0.0,
                    }
                )
                rate = RateLimiter(SIMULATION_HZ)
                sample_every = SIMULATION_HZ // SAMPLE_HZ
                started = time.monotonic()
                step = 0
                sent_init = False
                released = False
                release_attempted = False
                sent_policy = False
                sent_motion = False
                sent_stop = False
                sent_handoff = False
                preflight_ok = False
                while True:
                    elapsed = time.monotonic() - started
                    if elapsed >= END_S:
                        break
                    if policy_process.poll() is not None:
                        policy_exit_code = policy_process.returncode
                        raise RuntimeError(
                            f"production policy exited early with {policy_exit_code}"
                        )

                    if elapsed >= FIRST_POSE_S and not sent_init:
                        _send_key(policy_process, policy_input_fd, "i")
                        sent_init = True
                        events.append({"event": "first_pose_transition", "time_s": elapsed})
                    if (
                        GANTRY_LOWER_START_S <= elapsed < GANTRY_RELEASE_S
                        and sim.virtual_gantry is not None
                    ):
                        progress = (elapsed - GANTRY_LOWER_START_S) / (
                            GANTRY_RELEASE_S - GANTRY_LOWER_START_S
                        )
                        sim.virtual_gantry.length = args.release_length_m * min(progress, 1.0)
                    if elapsed >= POLICY_START_S and not sent_policy:
                        root = sim.robot_root_states[0].detach().cpu().numpy()
                        quat = root[3:7]
                        up = 1.0 - 2.0 * (quat[0] ** 2 + quat[1] ** 2)
                        preflight_ok = bool(
                            root[2] >= args.minimum_base_height_m
                            and up >= args.minimum_up_axis_z
                        )
                        if preflight_ok:
                            _send_key(policy_process, policy_input_fd, "]")
                            events.append({"event": "policy_started", "time_s": elapsed})
                        else:
                            events.append(
                                {
                                    "event": "policy_start_blocked_by_preflight",
                                    "time_s": elapsed,
                                    "base_height_m": float(root[2]),
                                    "up_axis_z": float(up),
                                }
                            )
                        sent_policy = True
                    if elapsed >= MOTION_START_S and not sent_motion:
                        if preflight_ok:
                            _send_key(policy_process, policy_input_fd, "m")
                            events.append({"event": "motion_started", "time_s": elapsed})
                        sent_motion = True
                    if elapsed >= GANTRY_RELEASE_S and not release_attempted:
                        root = sim.robot_root_states[0].detach().cpu().numpy()
                        quat = root[3:7]
                        up = 1.0 - 2.0 * (quat[0] ** 2 + quat[1] ** 2)
                        release_ok = bool(
                            preflight_ok
                            and root[2] >= args.minimum_base_height_m
                            and up >= args.minimum_up_axis_z
                        )
                        if release_ok:
                            assert sim.virtual_gantry is not None
                            sim.virtual_gantry.length = args.release_length_m
                            sim.virtual_gantry.set_enable(False)
                            released = True
                            events.append({"event": "gantry_released", "time_s": elapsed})
                        else:
                            events.append(
                                {
                                    "event": "gantry_release_blocked_by_preflight",
                                    "time_s": elapsed,
                                    "base_height_m": float(root[2]),
                                    "up_axis_z": float(up),
                                }
                            )
                        release_attempted = True
                    if (
                        args.safety_handoff
                        and elapsed >= SAFETY_HANDOFF_S
                        and not sent_handoff
                    ):
                        _send_key(policy_process, policy_input_fd, "x")
                        sent_handoff = True
                        events.append(
                            {"event": "safety_handoff_requested", "time_s": elapsed}
                        )
                    if (
                        not args.safety_handoff
                        and elapsed >= SAFE_HOLD_S
                        and not sent_stop
                    ):
                        _send_key(policy_process, policy_input_fd, "o")
                        sent_stop = True
                        events.append({"event": "safe_hold_requested", "time_s": elapsed})

                    sim.simulate_at_each_physics_step()
                    if step % sample_every == 0:
                        if elapsed < MOTION_START_S:
                            phase = "gantry_assisted_initialization"
                        elif not released and elapsed < safe_hold_start_s:
                            phase = "gantry_assisted_motion"
                        elif elapsed < safe_hold_start_s:
                            phase = "unassisted_motion"
                        else:
                            phase = "safe_hold"
                        _update_phase(
                            phases[phase],
                            sim,
                            joint_lower,
                            joint_upper,
                            velocity_limit,
                            torque_limit,
                            elapsed_s=elapsed,
                            minimum_base_height_m=args.minimum_base_height_m,
                            minimum_up_axis_z=args.minimum_up_axis_z,
                        )
                    step += 1
                    rate.sleep()
    except Exception as caught:
        exception = f"{type(caught).__name__}: {caught}"
    finally:
        if policy_process is not None and policy_process.poll() is None:
            policy_process.send_signal(signal.SIGINT)
            try:
                policy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                policy_process.terminate()
                try:
                    policy_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    policy_process.kill()
                    policy_process.wait(timeout=3)
        if policy_process is not None:
            policy_exit_code = policy_process.returncode
        if policy_input_fd is not None:
            os.close(policy_input_fd)

    policy_text = policy_log.read_text(errors="replace") if policy_log.is_file() else ""
    marker_checks = {
        "policy_initialized": "Policy initialized successfully" in policy_text,
        "first_pose_requested": "Setting to init state" in policy_text,
        "policy_started": "Using policy actions" in policy_text,
        "motion_started": "Starting motion clip" in policy_text,
        "motion_end_reached": f"Reached end timestep {motion_end_timestep}" in policy_text,
        "safe_hold_requested": "Actions set to stiff startup command" in policy_text,
    }
    if args.safety_handoff:
        marker_checks.pop("motion_end_reached")
        marker_checks.pop("safe_hold_requested")
        marker_checks["safety_handoff"] = (
            "Switched to secondary policy" in policy_text
        )
    required_phases = ("gantry_assisted_motion", "unassisted_motion", "safe_hold")
    safety_checks = {
        "all_required_phases_sampled": all(phases[name]["samples"] for name in required_phases),
        "minimum_base_height": all(
            phases[name]["minimum_base_height_m"] >= args.minimum_base_height_m
            for name in required_phases
        ),
        "minimum_up_axis": all(
            phases[name]["minimum_up_axis_z"] >= args.minimum_up_axis_z
            for name in required_phases
        ),
        "all_finite": all(phases[name]["nonfinite_samples"] == 0 for name in phases),
        "no_measured_joint_limit_violation": all(
            phases[name]["measured_joint_limit_violation_samples"] == 0
            for name in phases
        ),
        "no_commanded_joint_limit_violation": all(
            phases[name]["commanded_joint_limit_violation_samples"] == 0
            for name in phases
        ),
        "no_velocity_limit_violation": all(
            phases[name]["velocity_limit_violation_samples"] == 0 for name in phases
        ),
        "no_torque_limit_violation": all(
            phases[name]["torque_limit_violation_samples"] == 0 for name in phases
        ),
    }
    passes = (
        exception is None
        and policy_exit_code in (0, -signal.SIGINT)
        and all(marker_checks.values())
        and all(safety_checks.values())
    )
    report = {
        "protocol": "E70 production WBT loopback CPU MuJoCo qualification v2",
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": sha256_file(args.onnx),
        "hardware_interface": None,
        "loopback_interface": "lo",
        "domain_id": args.domain_id,
        "physical_robot_commands_sent": 0,
        "simulator": "Holosoma MuJoCo Classic CPU",
        "runtime_variant": (
            "production WBT with default safety-policy handoff"
            if args.safety_handoff
            else "unmodified production WBT"
        ),
        "policy_entry_sha256": sha256_file(POLICY_ENTRY),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "schedule_s": {
            "stiff_hold_confirmation": "before first physics step",
            "first_pose_transition": FIRST_POSE_S,
            "policy_start_supported": POLICY_START_S,
            "motion_start_supported": MOTION_START_S,
            "gantry_lowering": [GANTRY_LOWER_START_S, GANTRY_RELEASE_S],
            "gantry_release_after_upright_preflight": GANTRY_RELEASE_S,
            "motion_end_timestep": motion_end_timestep,
            "registered_unassisted_motion_frames": 500,
            "safety_handoff": SAFETY_HANDOFF_S if args.safety_handoff else None,
            "safe_hold": safe_hold_start_s,
            "end": END_S,
        },
        "thresholds": {
            "minimum_base_height_m": args.minimum_base_height_m,
            "minimum_up_axis_z": args.minimum_up_axis_z,
            "hard_joint_limit_tolerance_rad": 1.0e-6,
            "torque_limit_tolerance_nm": 1.0e-5,
        },
        "events": events,
        "phases": {name: _finite_stats(stats) for name, stats in phases.items()},
        "joint_names": list(config.robot.dof_names),
        "policy_command": policy_command,
        "policy_log": str(policy_log.resolve()),
        "policy_log_sha256": sha256_file(policy_log) if policy_log.is_file() else None,
        "policy_exit_code": policy_exit_code,
        "marker_checks": marker_checks,
        "safety_checks": safety_checks,
        "exception": exception,
        "passes": passes,
        "interpretation": (
            "candidate passes the bounded 500-frame unassisted production loopback gate"
            if passes
            else "deployment remains blocked by loopback startup, survival, or safety evidence"
        ),
    }
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.report)
    print(json.dumps({"passes": passes, "events": events, "safety_checks": safety_checks}, indent=2))
    raise SystemExit(0 if passes else 3)


if __name__ == "__main__":
    main()
