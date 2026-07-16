#!/usr/bin/env python3
"""Describe aligned GMR and SNMR WBT references before policy evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import numpy as np


REQUIRED_ARRAYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
)
DEFAULT_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _quaternion_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    dot = np.abs(np.sum(a * b, axis=-1)) / np.maximum(denominator, 1e-12)
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def _height_hysteresis(
    foot_pos: np.ndarray,
    *,
    enter_height: float = 0.03,
    exit_height: float = 0.05,
) -> np.ndarray:
    height = foot_pos[..., 2] - foot_pos[..., 2].min(axis=0, keepdims=True)
    contact = np.zeros(height.shape, dtype=bool)
    if not len(height):
        return contact
    state = height[0] <= enter_height
    contact[0] = state
    for frame in range(1, len(height)):
        state = np.where(
            state,
            height[frame] < exit_height,
            height[frame] <= enter_height,
        )
        contact[frame] = state
    return contact


def _contact_metrics(
    foot_pos: np.ndarray,
    mask: np.ndarray,
    *,
    fps: float,
    reference_foot_pos: np.ndarray,
) -> dict[str, Any]:
    speed = np.zeros(mask.shape, dtype=np.float64)
    if len(speed) > 1:
        speed[1:] = np.linalg.norm(
            foot_pos[1:, :, :2] - foot_pos[:-1, :, :2],
            axis=-1,
        ) * fps
        speed[0] = speed[1]
    active_count = int(mask.sum())
    denominator = max(active_count, 1)
    height_excess = foot_pos[..., 2] - reference_foot_pos[..., 2]
    return {
        "stance_speed_ms": (
            float(speed[mask].sum() / denominator) if active_count else None
        ),
        "slide_fraction_gt_0_3_ms": (
            float(((speed > 0.3) & mask).sum() / denominator)
            if active_count
            else None
        ),
        "floating_mean_m": (
            float(np.maximum(height_excess, 0.0)[mask].sum() / denominator)
            if active_count
            else None
        ),
        "floating_fraction_gt_0_03_m": (
            float(((height_excess > 0.03) & mask).sum() / denominator)
            if active_count
            else None
        ),
        "contact_prevalence": float(mask.mean()) if mask.size else 0.0,
        "contact_samples": active_count,
        "per_foot_stance_speed_ms": [
            float(speed[:, index][mask[:, index]].mean())
            if mask[:, index].any()
            else None
            for index in range(mask.shape[1])
        ],
    }


def _load(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def analyze(
    gmr_path: pathlib.Path,
    snmr_path: pathlib.Path,
    *,
    foot_body_names: tuple[str, ...] = DEFAULT_FEET,
) -> dict[str, Any]:
    errors = []
    try:
        gmr = _load(gmr_path)
        snmr = _load(snmr_path)
    except (OSError, ValueError) as exc:
        return {
            "passed": False,
            "protocol_errors": [f"cannot load references: {exc}"],
        }

    for source, arrays in (("gmr", gmr), ("snmr", snmr)):
        missing = sorted(set(REQUIRED_ARRAYS) - set(arrays))
        if missing:
            errors.append(f"{source} missing arrays: {missing}")
        for name in set(REQUIRED_ARRAYS) & set(arrays):
            if name in ("joint_names", "body_names"):
                continue
            if not np.issubdtype(arrays[name].dtype, np.number):
                errors.append(f"{source} {name} is not numeric")
            elif not np.isfinite(arrays[name]).all():
                errors.append(f"{source} {name} contains nonfinite values")
    if errors:
        return {"passed": False, "protocol_errors": errors}

    if set(gmr) != set(snmr):
        errors.append("GMR and SNMR archive fields differ")
    for name in REQUIRED_ARRAYS:
        if gmr[name].shape != snmr[name].shape:
            errors.append(
                f"{name} shape differs: {gmr[name].shape} vs {snmr[name].shape}"
            )
    for name in ("joint_names", "body_names"):
        if not np.array_equal(gmr[name], snmr[name]):
            errors.append(f"{name} differ")
    gmr_fps = np.asarray(gmr["fps"]).reshape(-1)
    snmr_fps = np.asarray(snmr["fps"]).reshape(-1)
    if (
        len(gmr_fps) != 1
        or len(snmr_fps) != 1
        or float(gmr_fps[0]) <= 0
        or float(gmr_fps[0]) != float(snmr_fps[0])
    ):
        errors.append("fps must be one matching positive value")
    if gmr["joint_pos"].ndim != 2 or gmr["joint_pos"].shape[1] < 8:
        errors.append("joint_pos must contain root position, root quaternion, and DOFs")
    if gmr["body_pos_w"].ndim != 3 or gmr["body_pos_w"].shape[-1] != 3:
        errors.append("body_pos_w must have shape (T, B, 3)")
    if gmr["body_quat_w"].shape != (*gmr["body_pos_w"].shape[:2], 4):
        errors.append("body_quat_w must have shape (T, B, 4)")
    body_names = list(gmr["body_names"].tolist())
    missing_feet = [name for name in foot_body_names if name not in body_names]
    if missing_feet:
        errors.append(f"missing foot bodies: {missing_feet}")
    if errors:
        return {"passed": False, "protocol_errors": errors}

    fps = float(gmr_fps[0])
    foot_indices = [body_names.index(name) for name in foot_body_names]
    gmr_feet = gmr["body_pos_w"][:, foot_indices]
    snmr_feet = snmr["body_pos_w"][:, foot_indices]
    gmr_mask = _height_hysteresis(gmr_feet)
    snmr_mask = _height_hysteresis(snmr_feet)
    union = int((gmr_mask | snmr_mask).sum())

    errors_by_frame = {
        "root_position_error_m": np.linalg.norm(
            snmr["joint_pos"][:, :3] - gmr["joint_pos"][:, :3],
            axis=-1,
        ),
        "root_rotation_error_rad": _quaternion_angle(
            snmr["joint_pos"][:, 3:7], gmr["joint_pos"][:, 3:7]
        ),
        "dof_absolute_error_rad": np.abs(
            snmr["joint_pos"][:, 7:] - gmr["joint_pos"][:, 7:]
        ).mean(axis=-1),
        "body_mpjpe_m": np.linalg.norm(
            snmr["body_pos_w"] - gmr["body_pos_w"], axis=-1
        ).mean(axis=-1),
        "body_rotation_error_rad": _quaternion_angle(
            snmr["body_quat_w"], gmr["body_quat_w"]
        ).mean(axis=-1),
        "joint_velocity_absolute_error": np.abs(
            snmr["joint_vel"] - gmr["joint_vel"]
        ).mean(axis=-1),
        "body_linear_velocity_error_ms": np.linalg.norm(
            snmr["body_lin_vel_w"] - gmr["body_lin_vel_w"], axis=-1
        ).mean(axis=-1),
        "body_angular_velocity_error_rads": np.linalg.norm(
            snmr["body_ang_vel_w"] - gmr["body_ang_vel_w"], axis=-1
        ).mean(axis=-1),
    }
    return {
        "passed": True,
        "protocol_errors": [],
        "interpretation": (
            "descriptive aligned-reference context; no policy outcome or promotion "
            "decision is computed"
        ),
        "artifacts": {
            "gmr": {"path": str(gmr_path.resolve()), "sha256": _sha256(gmr_path)},
            "snmr": {"path": str(snmr_path.resolve()), "sha256": _sha256(snmr_path)},
        },
        "schema": {
            "frames": int(gmr["joint_pos"].shape[0]),
            "fps": fps,
            "joint_count": int(len(gmr["joint_names"])),
            "body_count": int(len(body_names)),
            "foot_body_names": list(foot_body_names),
        },
        "aligned_errors": {
            name: _stats(values) for name, values in errors_by_frame.items()
        },
        "gmr_height_mask_context": {
            "definition": (
                "per-foot clip-minimum height hysteresis, enter 0.03 m, exit 0.05 m"
            ),
            "gmr": _contact_metrics(
                gmr_feet, gmr_mask, fps=fps, reference_foot_pos=gmr_feet
            ),
            "snmr": _contact_metrics(
                snmr_feet, gmr_mask, fps=fps, reference_foot_pos=gmr_feet
            ),
        },
        "decoded_height_mask_context": {
            "gmr_prevalence": float(gmr_mask.mean()),
            "snmr_prevalence": float(snmr_mask.mean()),
            "agreement": float((gmr_mask == snmr_mask).mean()),
            "intersection_over_union": (
                float((gmr_mask & snmr_mask).sum() / union) if union else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmr", type=pathlib.Path, required=True)
    parser.add_argument("--snmr", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--foot-body-names",
        nargs="+",
        default=list(DEFAULT_FEET),
    )
    args = parser.parse_args()
    result = analyze(
        args.gmr,
        args.snmr,
        foot_body_names=tuple(args.foot_body_names),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for error in result["protocol_errors"]:
        print(f"ERROR: {error}")
    if result["passed"]:
        aligned = result["aligned_errors"]
        contact = result["gmr_height_mask_context"]
        print(f"body MPJPE: {aligned['body_mpjpe_m']['mean']:.6f} m")
        print(
            "stance speed under GMR mask: "
            f"{contact['gmr']['stance_speed_ms']:.6f} -> "
            f"{contact['snmr']['stance_speed_ms']:.6f} m/s"
        )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
