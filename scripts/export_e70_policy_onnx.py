#!/usr/bin/env python
"""Export a frozen E70 SNMR student for Holosoma's offline G1 WBT runtime.

Run this with the pinned Holosoma WBT Python environment, which supplies ONNX and
ONNX Runtime.  The resulting model takes the standard raw 154-d WBT observation
plus ``time_step`` and emits ``actions``, ``joint_pos``, ``joint_vel``, and
``ref_quat_xyzw``.  It embeds one preplanned motion's frozen SNMR trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch

from snmr.deployment import (
    OfflineExplicitMotionPolicy,
    OfflineSnmrMotionPolicy,
    action_bounds_from_joint_limits,
)
from snmr.integration.distillation import CommandStudent


PROPRIO_DIM = 90
OBS_DIM = 154
CMD_DIM = 64
Z_WINDOW_DIM = 256
Z_CMD_DIM = 64


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pattern_values(names: list[str], values: dict[str, float]) -> list[float]:
    result = []
    for name in names:
        matches = [pattern for pattern in values if pattern in name]
        if len(matches) != 1:
            raise ValueError(f"joint {name!r} matches {matches} in gain table")
        result.append(float(values[matches[0]]))
    return result


def load_export_policy(
    student_path: pathlib.Path,
    motion_path: pathlib.Path,
    *,
    safety_limit_fraction: float = 0.95,
) -> tuple[OfflineExplicitMotionPolicy | OfflineSnmrMotionPolicy, dict]:
    checkpoint = torch.load(student_path, map_location="cpu", weights_only=False)
    arm = checkpoint.get("arm")
    prior_goal_by_arm = {
        "a_prior_snmr": "snmr",
        "c_prior_explicit": "explicit",
    }
    if arm not in prior_goal_by_arm:
        raise ValueError("deployment export requires the SNMR or explicit E70 arm")
    prior_goal = prior_goal_by_arm[arm]
    config = checkpoint.get("config", {})
    if config.get("deterministic") is not True:
        raise ValueError("deployment export requires the deterministic student")
    if config.get("time_index") or config.get("shuffle_latent"):
        raise ValueError("refusing to export a time/shuffled control as SNMR")
    teacher_paths = [pathlib.Path(path) for path in config.get("teacher_ckpts", [])]
    if not teacher_paths or not all(path.is_file() for path in teacher_paths):
        raise FileNotFoundError("student does not reference complete teacher checkpoints")
    teacher = torch.load(teacher_paths[0], map_location="cpu", weights_only=False)
    normalizer = teacher["actor_obs_normalizer_state_dict"]
    robot = teacher["experiment_config"]["robot"]

    prior_in = checkpoint["student"]["prior.0.weight"].shape[1]
    posterior_in = checkpoint["student"]["posterior.0.weight"].shape[1]
    priv_dim = posterior_in - prior_in
    num_actions = checkpoint["student"]["decoder.6.weight"].shape[0]
    student = CommandStudent(
        PROPRIO_DIM,
        priv_dim,
        num_actions,
        prior_goal,
        CMD_DIM,
        z_window_dim=Z_WINDOW_DIM,
        z_cmd_dim=Z_CMD_DIM,
    )
    student.load_state_dict(checkpoint["student"])
    student.eval()

    with np.load(motion_path) as motion:
        joint_names = [str(name) for name in motion["joint_names"].tolist()]
        expected_names = [str(name) for name in robot["dof_names"]]
        if set(joint_names) != set(expected_names):
            raise ValueError("motion and teacher joint-name sets differ")
        joint_index = [joint_names.index(name) for name in expected_names]
        body_names = [str(name) for name in motion["body_names"].tolist()]
        ref_name = str(robot["torso_name"])
        if ref_name not in body_names:
            raise ValueError(f"reference body {ref_name!r} is absent from motion")
        ref_index = body_names.index(ref_name)
        latent_z = torch.from_numpy(motion["latent_z"].copy())
        joint_pos = torch.from_numpy(motion["joint_pos"][:, 7:][:, joint_index].copy())
        joint_vel = torch.from_numpy(motion["joint_vel"][:, 6:][:, joint_index].copy())
        ref_quat_wxyz = torch.from_numpy(motion["body_quat_w"][:, ref_index].copy())
        ref_quat_xyzw = ref_quat_wxyz[:, [1, 2, 3, 0]]
        fps = int(np.asarray(motion["fps"]).reshape(-1)[0])

    control = robot["control"]
    kp = _pattern_values(expected_names, control["stiffness"])
    kd = _pattern_values(expected_names, control["damping"])
    if control.get("action_scales_by_effort_limit_over_p_gain"):
        action_scale = [
            float(control["action_scale"]) * float(effort) / stiffness
            for effort, stiffness in zip(robot["dof_effort_limit_list"], kp)
        ]
    else:
        action_scale = [float(control["action_scale"])] * len(expected_names)
    joint_lower = [float(value) for value in robot["dof_pos_lower_limit_list"]]
    joint_upper = [float(value) for value in robot["dof_pos_upper_limit_list"]]
    default_by_name = robot["init_state"]["default_joint_angles"]
    default_joint_pos = [float(default_by_name[name]) for name in expected_names]
    action_lower, action_upper = action_bounds_from_joint_limits(
        torch.tensor(default_joint_pos),
        torch.tensor(action_scale),
        torch.tensor(joint_lower),
        torch.tensor(joint_upper),
        limit_fraction=safety_limit_fraction,
    )

    common_policy_args = {
        "observation_mean": normalizer["_mean"],
        "observation_std": normalizer["_std"],
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "ref_quat_xyzw": ref_quat_xyzw,
        "action_lower": action_lower,
        "action_upper": action_upper,
    }
    if prior_goal == "snmr":
        policy = OfflineSnmrMotionPolicy(
            student,
            latent_mean=checkpoint["z_mean"],
            latent_std=checkpoint["z_std"],
            latent_z=latent_z,
            future_offset=int(config["offsets"][1]),
            **common_policy_args,
        ).eval()
    else:
        policy = OfflineExplicitMotionPolicy(
            student,
            **common_policy_args,
        ).eval()

    asset_root = pathlib.Path("/home/robotixx/holosoma/src/holosoma/holosoma/data/robots")
    urdf_path = asset_root / str(robot["asset"]["urdf_file"])
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    metadata = {
        "protocol": f"E70 offline {prior_goal} deployment export v2",
        "arm": arm,
        "prior_goal": prior_goal,
        "student_checkpoint": str(student_path.resolve()),
        "student_checkpoint_sha256": sha256_file(student_path),
        "teacher_checkpoint": str(teacher_paths[0].resolve()),
        "teacher_checkpoint_sha256": sha256_file(teacher_paths[0]),
        "motion": str(motion_path.resolve()),
        "motion_sha256": sha256_file(motion_path),
        "motion_fps": fps,
        "motion_frames": int(latent_z.shape[0]),
        "dof_names": expected_names,
        "kp": kp,
        "kd": kd,
        "action_scale": action_scale,
        "joint_pos_lower": joint_lower,
        "joint_pos_upper": joint_upper,
        "default_joint_pos": default_joint_pos,
        "safety_limit_fraction": safety_limit_fraction,
        "action_lower": action_lower.squeeze(0).tolist(),
        "action_upper": action_upper.squeeze(0).tolist(),
        "robot_urdf": urdf_path.read_text(),
        "robot_urdf_path": str(urdf_path.resolve()),
        "input_contract": {
            "obs": "raw 154-d WBT actor_obs in alphabetical term order",
            "time_step": "zero-based motion frame at 50 Hz",
            "proprio_slice": [0, PROPRIO_DIM],
            "latent_offsets": list(config["offsets"]) if prior_goal == "snmr" else None,
            "scope": "fixed preplanned motion; not an online teleoperation encoder",
        },
    }
    return policy, metadata


def export_onnx(
    policy: OfflineExplicitMotionPolicy | OfflineSnmrMotionPolicy,
    metadata: dict,
    output: pathlib.Path,
) -> dict:
    import onnx
    import onnxruntime

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    time_step = torch.zeros(1, 1, dtype=torch.float32)
    torch.onnx.export(
        policy,
        (obs, time_step),
        temporary,
        export_params=True,
        opset_version=13,
        verbose=False,
        input_names=["obs", "time_step"],
        output_names=["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"],
        dynamo=False,
    )
    model = onnx.load(temporary)
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value)
    onnx.checker.check_model(model)
    onnx.save(model, temporary)

    session = onnxruntime.InferenceSession(
        str(temporary), providers=["CPUExecutionProvider"]
    )
    generator = torch.Generator().manual_seed(70)
    cases = [
        (obs, time_step),
        (
            torch.randn(1, OBS_DIM, generator=generator),
            torch.tensor([[policy.joint_pos.shape[0] // 2]], dtype=torch.float32),
        ),
        (
            torch.randn(1, OBS_DIM, generator=generator),
            torch.tensor([[policy.joint_pos.shape[0] - 1]], dtype=torch.float32),
        ),
    ]
    output_names = ("actions", "joint_pos", "joint_vel", "ref_quat_xyzw")
    per_output = {name: 0.0 for name in output_names}
    with torch.no_grad():
        for case_obs, case_time in cases:
            targets = policy(case_obs, case_time)
            observed = session.run(
                None,
                {"obs": case_obs.numpy(), "time_step": case_time.numpy()},
            )
            for name, actual, target in zip(output_names, observed, targets):
                per_output[name] = max(
                    per_output[name],
                    float(np.max(np.abs(actual - target.numpy()))),
                )
    maximum = max(per_output.values())
    if maximum > 1.0e-5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX parity failed: max_abs={maximum:.3g}")
    temporary.replace(output)
    return {
        "protocol": "E70 ONNX export validation v2",
        "onnx": str(output.resolve()),
        "onnx_sha256": sha256_file(output),
        "onnx_checker": "pass",
        "onnxruntime_provider": "CPUExecutionProvider",
        "parity_cases": len(cases),
        "max_abs_error": maximum,
        "max_abs_error_by_output": per_output,
        "parity_threshold": 1.0e-5,
        "passes": True,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=pathlib.Path, required=True)
    parser.add_argument("--motion", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--safety-limit-fraction", type=float, default=0.95)
    args = parser.parse_args()

    policy, metadata = load_export_policy(
        args.student,
        args.motion,
        safety_limit_fraction=args.safety_limit_fraction,
    )
    report = export_onnx(policy, metadata, args.out)
    report_path = args.report or args.out.with_suffix(".validation.json")
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(report_path)
    print(
        f"onnx={args.out} sha256={report['onnx_sha256']} "
        f"max_abs_error={report['max_abs_error']:.3g}"
    )


if __name__ == "__main__":
    main()
