#!/usr/bin/env python
"""Capture a sim2sim review rollout: loopback lifecycle plus state logging.

This is an engineering companion to ``run_e70_loopback_qualification.py`` for
the Phase E sim2sim campaign (see docs/ICRA_EXECUTION_PLAN_2026-08-11.md,
Phase E).  It reruns the identical safety-handoff loopback lifecycle — the
production Holosoma inference process driving CPU MuJoCo over ``lo`` — and
records the full MuJoCo ``qpos`` and the robot DOF vector at 200 Hz so a
separate script can replay-render a video and compute tracking error without
perturbing the physics loop.  It never enters the frozen paper-video pipeline
and is not a qualification gate; safety numbers come from the unmodified
qualification script.  Run with Holosoma's ``hsmujoco`` Python.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
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
SIMULATION_HZ = 2000
SAMPLE_HZ = 200
MOTION_END_TIMESTEP = 750

FIRST_POSE_S = 3.0
POLICY_START_S = 12.0
MOTION_START_S = 13.0
GANTRY_LOWER_START_S = 14.0
GANTRY_RELEASE_S = 17.0
SAFETY_HANDOFF_S = 27.0
END_S = 31.0


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--states-out", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--domain-id", type=int, default=71)
    parser.add_argument("--minimum-base-height-m", type=float, default=0.45)
    parser.add_argument("--minimum-up-axis-z", type=float, default=0.5)
    parser.add_argument("--release-length-m", type=float, default=2.0)
    args = parser.parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if not INFERENCE_PYTHON.is_file() or not POLICY_ENTRY.is_file():
        raise FileNotFoundError("pinned Holosoma inference runtime is incomplete")

    import mujoco

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

    args.states_out.parent.mkdir(parents=True, exist_ok=True)
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
        str(MOTION_END_TIMESTEP),
    ]

    events: list[dict[str, Any]] = []
    sample_times: list[float] = []
    qpos_log: list[np.ndarray] = []
    dof_log: list[np.ndarray] = []
    root_log: list[np.ndarray] = []
    policy_process: subprocess.Popen | None = None
    policy_input_fd: int | None = None
    policy_exit_code: int | None = None
    exception: str | None = None
    model_mjb_path = args.states_out.with_suffix(".model.mjb")

    env, device, simulation_app = setup_simulation_environment(config, device="cpu")
    try:
        with DirectSimulation(config, env, device, simulation_app) as direct:
            sim = direct.simulator
            try:
                mujoco.mj_saveModel(sim.root_model, str(model_mjb_path), None)
            except Exception as save_error:  # noqa: BLE001 - renderer falls back to stock scene
                print(f"model save failed (renderer will use fallback scene): {save_error}")
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
                _wait_for_log_marker(
                    policy_process, policy_log, "Press Enter to continue..."
                )
                _send_key(policy_process, policy_input_fd, "\n")
                _wait_for_log_marker(
                    policy_process, policy_log, "Policy initialized successfully"
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
                                {"event": "policy_start_blocked_by_preflight", "time_s": elapsed}
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
                                {"event": "gantry_release_blocked_by_preflight", "time_s": elapsed}
                            )
                        release_attempted = True
                    if elapsed >= SAFETY_HANDOFF_S and not sent_handoff:
                        _send_key(policy_process, policy_input_fd, "x")
                        sent_handoff = True
                        events.append(
                            {"event": "safety_handoff_requested", "time_s": elapsed}
                        )

                    sim.simulate_at_each_physics_step()
                    if step % sample_every == 0:
                        sample_times.append(elapsed)
                        qpos_log.append(np.array(sim.root_data.qpos, dtype=np.float64))
                        dof_log.append(
                            sim.dof_pos[0].detach().cpu().numpy().astype(np.float64)
                        )
                        root_log.append(
                            sim.robot_root_states[0].detach().cpu().numpy().astype(np.float64)
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

    np.savez_compressed(
        args.states_out,
        sample_times=np.asarray(sample_times, dtype=np.float64),
        qpos=np.stack(qpos_log) if qpos_log else np.zeros((0, 0)),
        dof_pos=np.stack(dof_log) if dof_log else np.zeros((0, 0)),
        root_states=np.stack(root_log) if root_log else np.zeros((0, 0)),
        dof_names=np.asarray(config.robot.dof_names, dtype="U64"),
    )
    report = {
        "protocol": "sim2sim review capture v1 (non-frozen engineering artifact)",
        "onnx": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "states_out": str(args.states_out),
        "model_mjb": str(model_mjb_path),
        "sample_hz": SAMPLE_HZ,
        "events": events,
        "samples": len(sample_times),
        "policy_exit_code": policy_exit_code,
        "exception": exception,
        "lifecycle": {
            "first_pose_s": FIRST_POSE_S,
            "policy_start_s": POLICY_START_S,
            "motion_start_s": MOTION_START_S,
            "gantry_release_s": GANTRY_RELEASE_S,
            "safety_handoff_s": SAFETY_HANDOFF_S,
            "end_s": END_S,
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"capture={args.states_out} samples={len(sample_times)} "
        f"exception={exception!r}"
    )


if __name__ == "__main__":
    main()
