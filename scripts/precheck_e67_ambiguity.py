#!/usr/bin/env python
"""Pre-register and measure reference-state ambiguity for E67 candidate clip pairs.

The check is intentionally simulator-free.  At the perfect-tracking reference, a proxy for
the 90-d actor proprioception is ``[previous target, root angular velocity, joint position,
joint velocity]``.  Candidate frames must share a normalized-time bin, have nearby globally
standardized current states, and have divergent globally standardized robot-goal trajectories
over the next second.

This script selects non-overlapping cross-clip window pairs and writes every threshold, input
hash, distance quantile, and selected frame to JSON.  It never uses SNMR latents, teacher
actions, or a learned policy, so it cannot tune the test to favor an E67 arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.paths import data_root  # noqa: E402


@dataclass(frozen=True)
class ReferenceFeatures:
    state: np.ndarray
    goal: np.ndarray
    fps: float


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_angular_velocity(quaternion: np.ndarray, fps: float) -> np.ndarray:
    """Finite-difference angular velocity from wxyz quaternions."""
    previous = quaternion[:-1]
    current = quaternion[1:]
    # q_delta = current * conjugate(previous), wxyz.
    aw, ax, ay, az = np.moveaxis(current, -1, 0)
    bw, bx, by, bz = np.moveaxis(previous * np.array([1, -1, -1, -1]), -1, 0)
    delta = np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )
    delta = np.where(delta[:, :1] < 0, -delta, delta)
    xyz = delta[:, 1:]
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(delta[:, :1], 1.0e-12, None))
    omega = xyz / np.clip(norm, 1.0e-12, None) * angle * fps
    return np.concatenate((omega[:1], omega), axis=0)


def reference_features(qpos: np.ndarray, fps: float) -> ReferenceFeatures:
    if qpos.ndim != 2 or qpos.shape[1] < 8:
        raise ValueError(f"qpos must have shape (T,7+D), got {qpos.shape}")
    if not np.isfinite(qpos).all() or fps <= 0:
        raise ValueError("reference contains nonfinite values or invalid fps")
    dof = np.asarray(qpos[:, 7:], dtype=np.float64)
    dof_velocity = np.gradient(dof, 1.0 / fps, axis=0)
    angular_velocity = _root_angular_velocity(qpos[:, 3:7], fps)
    previous_action = np.concatenate((dof[:1], dof[:-1]), axis=0)
    state = np.concatenate((previous_action, angular_velocity, dof, dof_velocity), axis=-1)
    goal = np.concatenate((dof, dof_velocity), axis=-1)
    return ReferenceFeatures(state=state, goal=goal, fps=fps)


def _standardize_pooled(arrays: list[np.ndarray]) -> list[np.ndarray]:
    pooled = np.concatenate(arrays, axis=0)
    mean = pooled.mean(axis=0, keepdims=True)
    std = pooled.std(axis=0, keepdims=True)
    std = np.where(std < 1.0e-6, 1.0, std)
    return [(array - mean) / std for array in arrays]


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p10", "median", "p90", "max")}
    array = np.asarray(values)
    return {
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def ambiguity_windows(
    first: ReferenceFeatures,
    second: ReferenceFeatures,
    *,
    time_bins: int = 100,
    future_seconds: float = 1.0,
    rollout_seconds: float = 10.0,
    future_samples: int = 11,
    max_state_distance: float = 0.75,
    min_future_distance: float = 0.75,
    min_spacing_seconds: float = 0.5,
    max_windows: int = 200,
) -> dict:
    if abs(first.fps - second.fps) > 1.0e-6:
        raise ValueError("candidate references must use the same fps")
    fps = first.fps
    future_frames = int(round(future_seconds * fps))
    rollout_frames = int(round(rollout_seconds * fps))
    if future_frames < 1 or rollout_frames < future_frames or future_samples < 2:
        raise ValueError(
            "rollout horizon must be positive and at least as long as the future horizon"
        )
    state_first, state_second = _standardize_pooled([first.state, second.state])
    goal_first, goal_second = _standardize_pooled([first.goal, second.goal])
    # A selected frame must support the complete downstream rollout without wrapping into
    # another clip.  This is stricter than what is needed for the one-second ambiguity
    # calculation and prevents end-of-clip resets from contaminating completion.
    valid_lengths = [len(state_first) - rollout_frames, len(state_second) - rollout_frames]
    if min(valid_lengths) < time_bins:
        raise ValueError("clips are too short for the requested bins/horizon")
    bins = [
        np.minimum(
            (np.arange(length) / max(length - 1, 1) * time_bins).astype(int),
            time_bins - 1,
        )
        for length in valid_lengths
    ]
    future_offsets = np.linspace(0, future_frames, future_samples).round().astype(int)

    nearest_records: list[dict] = []
    for time_bin in range(time_bins):
        idx_first = np.flatnonzero(bins[0] == time_bin)
        idx_second = np.flatnonzero(bins[1] == time_bin)
        if not len(idx_first) or not len(idx_second):
            continue
        a = state_first[idx_first]
        b = state_second[idx_second]
        distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).mean(axis=-1))
        nearest = distances.argmin(axis=1)
        for row, column in enumerate(nearest):
            frame_first = int(idx_first[row])
            frame_second = int(idx_second[column])
            future_a = goal_first[frame_first + future_offsets]
            future_b = goal_second[frame_second + future_offsets]
            future_distance = float(np.sqrt(((future_a - future_b) ** 2).mean()))
            nearest_records.append(
                {
                    "frame_first": frame_first,
                    "frame_second": frame_second,
                    "time_seconds_first": frame_first / fps,
                    "time_seconds_second": frame_second / fps,
                    "time_bin": time_bin,
                    "normalized_time_first": frame_first / max(valid_lengths[0] - 1, 1),
                    "normalized_time_second": frame_second / max(valid_lengths[1] - 1, 1),
                    "state_distance": float(distances[row, column]),
                    "future_distance": future_distance,
                }
            )

    eligible = [
        record
        for record in nearest_records
        if record["state_distance"] <= max_state_distance
        and record["future_distance"] >= min_future_distance
    ]
    eligible.sort(
        key=lambda record: (
            -(record["future_distance"] - record["state_distance"]),
            record["frame_first"],
        )
    )
    spacing = max(1, int(round(min_spacing_seconds * fps)))
    selected: list[dict] = []
    for record in eligible:
        if all(
            abs(record["frame_first"] - other["frame_first"]) >= spacing
            and abs(record["frame_second"] - other["frame_second"]) >= spacing
            for other in selected
        ):
            selected.append(record)
            if len(selected) >= max_windows:
                break
    selected.sort(key=lambda record: record["frame_first"])
    return {
        "num_nearest_candidates": len(nearest_records),
        "num_threshold_eligible": len(eligible),
        "num_selected_windows": len(selected),
        "source_fps": fps,
        "rollout_frames": rollout_frames,
        "nearest_state_distance": _quantiles(
            [record["state_distance"] for record in nearest_records]
        ),
        "nearest_future_distance": _quantiles(
            [record["future_distance"] for record in nearest_records]
        ),
        "eligible_state_distance": _quantiles(
            [record["state_distance"] for record in eligible]
        ),
        "eligible_future_distance": _quantiles(
            [record["future_distance"] for record in eligible]
        ),
        "windows": selected,
    }


def _load_clip(path: pathlib.Path) -> ReferenceFeatures:
    with np.load(path, allow_pickle=False) as data:
        return reference_features(np.asarray(data["qpos"]), float(data["fps"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate_pairs",
        nargs="+",
        default=["walk1_subject5,walk3_subject1", "walk1_subject5,run1_subject2"],
        help="comma-separated cross-clip pairs",
    )
    parser.add_argument("--pairs_dir", default=str(data_root() / "pairs" / "unitree_g1"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--time_bins", type=int, default=100)
    parser.add_argument("--future_seconds", type=float, default=1.0)
    parser.add_argument("--rollout_seconds", type=float, default=10.0)
    parser.add_argument("--future_samples", type=int, default=11)
    parser.add_argument("--max_state_distance", type=float, default=0.75)
    parser.add_argument("--min_future_distance", type=float, default=0.75)
    parser.add_argument("--min_spacing_seconds", type=float, default=0.5)
    parser.add_argument("--min_windows", type=int, default=20)
    args = parser.parse_args()

    pairs_dir = pathlib.Path(args.pairs_dir)
    report = {
        "protocol": "E67 reference-only ambiguity precheck v1",
        "thresholds": {
            "time_bins": args.time_bins,
            "future_seconds": args.future_seconds,
            "rollout_seconds": args.rollout_seconds,
            "future_samples": args.future_samples,
            "max_state_distance_rms_z": args.max_state_distance,
            "min_future_distance_rms_z": args.min_future_distance,
            "min_spacing_seconds": args.min_spacing_seconds,
            "min_windows": args.min_windows,
        },
        "state_proxy": "previous_dof_target + root_angular_velocity + dof_pos + dof_vel",
        "future_goal": "dof_pos + dof_vel, globally standardized across each candidate pair",
        "pairs": {},
    }
    for specification in args.candidate_pairs:
        clips = specification.split(",")
        if len(clips) != 2 or clips[0] == clips[1]:
            parser.error(f"invalid candidate pair {specification!r}")
        paths = [pairs_dir / f"{clip}.npz" for clip in clips]
        if not all(path.is_file() for path in paths):
            missing = [str(path) for path in paths if not path.is_file()]
            parser.error(f"missing pair files: {missing}")
        result = ambiguity_windows(
            _load_clip(paths[0]),
            _load_clip(paths[1]),
            time_bins=args.time_bins,
            future_seconds=args.future_seconds,
            rollout_seconds=args.rollout_seconds,
            future_samples=args.future_samples,
            max_state_distance=args.max_state_distance,
            min_future_distance=args.min_future_distance,
            min_spacing_seconds=args.min_spacing_seconds,
        )
        result["clips"] = clips
        result["inputs"] = [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
        ]
        result["passes_floor"] = result["num_selected_windows"] >= args.min_windows
        report["pairs"][specification] = result
        print(
            f"{specification}: {result['num_selected_windows']} selected windows "
            f"(floor {args.min_windows}) -> "
            f"{'PASS' if result['passes_floor'] else 'FAIL'}"
        )

    passing = [
        (name, value)
        for name, value in report["pairs"].items()
        if value["passes_floor"]
    ]
    passing.sort(
        key=lambda item: (
            -item[1]["num_selected_windows"],
            -float(item[1]["eligible_future_distance"]["median"] or 0.0),
            item[0],
        )
    )
    report["preferred_pair"] = passing[0][0] if passing else None
    report["gate_passed"] = bool(passing)
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"preferred_pair={report['preferred_pair']}; report={output}")
    if not report["gate_passed"]:
        raise SystemExit("no candidate pair passed the preregistered ambiguity floor")


if __name__ == "__main__":
    main()
