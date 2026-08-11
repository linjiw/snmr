#!/usr/bin/env python
"""Exercise an E70 ONNX through Holosoma's real WBT policy without an SDK.

This deliberately bypasses ``WholeBodyTrackingPolicy.__init__`` so no Unitree
interface, DDS participant, input thread, or command sender can be created.  It
uses the production observation builder, Pinocchio reference-orientation path,
ONNX closure, and action-scale parser for a stationary-state fault injection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import pathlib
import platform
import time
import xml.etree.ElementTree as ET

import numpy as np


EXPECTED_TERMS = [
    "actions",
    "base_ang_vel",
    "dof_pos",
    "dof_vel",
    "motion_command",
    "motion_ref_ori_b",
]
EXPECTED_INPUTS = ["obs", "time_step"]
EXPECTED_OUTPUTS = ["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"]


def package_version(*distribution_names: str) -> str:
    """Return the first installed distribution version without failing a report."""
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def urdf_joint_limits(urdf: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    root = ET.fromstring(urdf)
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits[joint.attrib["name"]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
    missing = [name for name in names if name not in limits]
    if missing:
        raise ValueError(f"URDF lacks finite position limits for {missing}")
    return (
        np.asarray([limits[name][0] for name in names], dtype=np.float32),
        np.asarray([limits[name][1] for name in names], dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--deadline-ms", type=float, default=20.0)
    args = parser.parse_args()
    if args.steps < 1 or args.deadline_ms <= 0:
        raise ValueError("runtime validation steps and deadline must be positive")

    import onnx

    from holosoma_inference.config.config_values.inference import g1_29dof_wbt
    from holosoma_inference.policies.base import BasePolicy
    from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

    model = onnx.load(args.onnx)
    metadata = {item.key: json.loads(item.value) for item in model.metadata_props}
    names = [str(name) for name in metadata.get("dof_names", [])]
    urdf_lower, urdf_upper = urdf_joint_limits(str(metadata["robot_urdf"]), names)

    # Construct only the computation side of the production policy.  In particular,
    # do not call __init__, _init_communication_components, or create_interface.
    policy = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
    policy.config = g1_29dof_wbt
    BasePolicy._init_robot_config(policy, g1_29dof_wbt.robot)
    BasePolicy._init_obs_config(policy)
    policy.last_policy_action = np.zeros((1, policy.num_dofs), dtype=np.float32)
    policy.robot_yaw_offset = 0.0
    policy.motion_yaw_offset = 0.0
    policy.setup_policy(str(args.onnx))
    policy._configure_action_scales()

    runtime_names = list(policy.dof_names)
    metadata_scale = np.asarray(metadata.get("action_scale", []), dtype=np.float32)
    default = np.asarray(policy.default_dof_angles, dtype=np.float32)
    input_specs = [
        {"name": item.name, "shape": item.shape, "type": item.type}
        for item in policy.onnx_policy_session.get_inputs()
    ]
    output_specs = [
        {"name": item.name, "shape": item.shape, "type": item.type}
        for item in policy.onnx_policy_session.get_outputs()
    ]
    contract_checks = {
        "deployment_protocol_v2": str(metadata.get("protocol", "")).endswith(
            "deployment export v2"
        ),
        "input_names": policy.onnx_input_names == EXPECTED_INPUTS,
        "output_names": policy.onnx_output_names == EXPECTED_OUTPUTS,
        "observation_terms": policy.obs_terms_sorted.get("actor_obs") == EXPECTED_TERMS,
        "observation_dim_154": policy.obs_buf_dict["actor_obs"].shape == (1, 154),
        "joint_count_29": policy.num_dofs == len(names) == 29,
        "joint_order": names == runtime_names,
        "gain_lengths": len(metadata.get("kp", [])) == len(metadata.get("kd", [])) == 29,
        "action_scale_length": metadata_scale.shape == (29,),
        "runtime_action_scale": np.allclose(
            policy.per_joint_policy_action_scale.reshape(-1), metadata_scale, atol=1.0e-7
        ),
    }

    metadata_lower = np.asarray(metadata.get("joint_pos_lower", []), dtype=np.float32)
    metadata_upper = np.asarray(metadata.get("joint_pos_upper", []), dtype=np.float32)
    metadata_default = np.asarray(metadata.get("default_joint_pos", []), dtype=np.float32)
    fraction = float(metadata.get("safety_limit_fraction", 0.0))
    action_lower = np.asarray(metadata.get("action_lower", []), dtype=np.float32)
    action_upper = np.asarray(metadata.get("action_upper", []), dtype=np.float32)
    contract_checks.update(
        {
            "hard_limits_match_urdf": metadata_lower.shape == (29,)
            and metadata_upper.shape == (29,)
            and np.allclose(metadata_lower, urdf_lower, atol=1.0e-5)
            and np.allclose(metadata_upper, urdf_upper, atol=1.0e-5),
            "default_pose_matches_runtime": metadata_default.shape == (29,)
            and np.allclose(metadata_default, default, atol=1.0e-7),
            "safety_fraction": 0.0 < fraction <= 1.0,
            "action_envelope_shape": action_lower.shape == action_upper.shape == (29,),
        }
    )

    state = np.zeros((1, 7 + policy.num_dofs + 6 + policy.num_dofs), dtype=np.float32)
    state[:, 3] = 1.0  # base quaternion wxyz
    state[:, 7 : 7 + policy.num_dofs] = default
    steps = min(args.steps, int(metadata.get("motion_frames", args.steps)))
    latencies = []
    hard_violation_steps = 0
    safe_violation_steps = 0
    action_envelope_violation_steps = 0
    nonfinite_steps = 0
    runtime_clip_steps = 0
    minimum_hard_margin = float("inf")
    minimum_safe_margin = float("inf")
    maximum_abs_raw_action = 0.0
    center = 0.5 * (urdf_lower + urdf_upper)
    safe_lower = center - 0.5 * (urdf_upper - urdf_lower) * fraction
    safe_upper = center + 0.5 * (urdf_upper - urdf_lower) * fraction

    # Warm up ONNX and Pinocchio before measuring steady-state deadlines.
    policy.prepare_obs_for_rl(state)
    for step in range(steps):
        started = time.perf_counter_ns()
        observation = policy.prepare_obs_for_rl(state)["actor_obs"]
        raw_action, motion_command, ref_quat = policy.policy(
            {
                "obs": observation,
                "time_step": np.asarray([[step]], dtype=np.float32),
            }
        )
        latencies.append((time.perf_counter_ns() - started) / 1.0e6)
        clipped_action = np.clip(raw_action, -100.0, 100.0)
        runtime_clip_steps += int(not np.array_equal(raw_action, clipped_action))
        target = default + clipped_action.reshape(-1) * metadata_scale
        finite = all(
            np.isfinite(value).all()
            for value in (observation, raw_action, motion_command, ref_quat, target)
        )
        nonfinite_steps += int(not finite)
        hard_margin = np.minimum(target - urdf_lower, urdf_upper - target)
        safe_margin = np.minimum(target - safe_lower, safe_upper - target)
        minimum_hard_margin = min(minimum_hard_margin, float(hard_margin.min()))
        minimum_safe_margin = min(minimum_safe_margin, float(safe_margin.min()))
        hard_violation_steps += int(np.any(hard_margin < -1.0e-6))
        safe_violation_steps += int(np.any(safe_margin < -1.0e-6))
        if action_lower.shape == (29,) and action_upper.shape == (29,):
            action_envelope_violation_steps += int(
                np.any(raw_action.reshape(-1) < action_lower - 1.0e-6)
                or np.any(raw_action.reshape(-1) > action_upper + 1.0e-6)
            )
        maximum_abs_raw_action = max(
            maximum_abs_raw_action, float(np.max(np.abs(raw_action)))
        )
        policy.last_policy_action = clipped_action.copy()
        policy.motion_command_t = motion_command
        policy.ref_quat_xyzw_t = ref_quat

    latency = np.asarray(latencies)
    fault_checks = {
        "all_finite": nonfinite_steps == 0,
        "no_20ms_deadline_miss": int(np.sum(latency > args.deadline_ms)) == 0,
        "no_runtime_action_clip": runtime_clip_steps == 0,
        "no_hard_limit_violation": hard_violation_steps == 0,
        "no_safety_envelope_violation": safe_violation_steps == 0,
        "onnx_action_envelope_enforced": action_envelope_violation_steps == 0,
    }
    passes = all(contract_checks.values()) and all(fault_checks.values())
    runtime_files = {
        "wbt_policy": pathlib.Path(inspect.getsourcefile(WholeBodyTrackingPolicy)).resolve(),
        "base_policy": pathlib.Path(inspect.getsourcefile(BasePolicy)).resolve(),
    }
    report = {
        "protocol": "E70 production WBT no-command validation v1",
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": sha256_file(args.onnx),
        "sdk_initialized": False,
        "commands_sent": 0,
        "python": platform.python_version(),
        "packages": {
            "holosoma-inference": package_version("holosoma-inference"),
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime", "onnxruntime-gpu"),
            "pinocchio": package_version("pin", "pinocchio"),
        },
        "runtime_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in runtime_files.items()
        },
        "input_specs": input_specs,
        "output_specs": output_specs,
        "observation_terms": policy.obs_terms_sorted["actor_obs"],
        "contract_checks": contract_checks,
        "stationary_fault_injection": {
            "steps": steps,
            "deadline_ms": args.deadline_ms,
            "latency_ms": {
                "mean": float(latency.mean()),
                "p50": float(np.percentile(latency, 50)),
                "p95": float(np.percentile(latency, 95)),
                "p99": float(np.percentile(latency, 99)),
                "maximum": float(latency.max()),
            },
            "deadline_misses": int(np.sum(latency > args.deadline_ms)),
            "nonfinite_steps": nonfinite_steps,
            "runtime_action_clip_steps": runtime_clip_steps,
            "action_envelope_violation_steps": action_envelope_violation_steps,
            "hard_limit_violation_steps": hard_violation_steps,
            "safety_envelope_violation_steps": safe_violation_steps,
            "minimum_hard_limit_margin_rad": minimum_hard_margin,
            "minimum_safety_limit_margin_rad": minimum_safe_margin,
            "maximum_abs_raw_action": maximum_abs_raw_action,
            "checks": fault_checks,
        },
        "passes": passes,
        "interpretation": (
            "production computation contract and stationary-state safety fault pass"
            if passes
            else "deployment blocked by a production-contract or stationary-state safety failure"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.report)
    print(json.dumps({"passes": passes, **report["stationary_fault_injection"]}, indent=2))
    raise SystemExit(0 if passes else 3)


if __name__ == "__main__":
    main()
