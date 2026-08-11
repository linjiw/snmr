#!/usr/bin/env python
"""Reference-only exhaustive pair screen registered in the E69 protocol."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from precheck_e67_ambiguity import (  # noqa: E402
    _load_clip,
    ambiguity_windows,
    sha256_file,
)
from snmr.paths import data_root  # noqa: E402


def reference_difficulty(features) -> dict[str, float]:
    """Return robust kinematic difficulty statistics from reference state features."""
    goal_dim = features.goal.shape[1]
    if goal_dim % 2:
        raise ValueError("goal must concatenate equal-width position and velocity")
    dof = goal_dim // 2
    velocity = features.goal[:, dof:]
    acceleration = np.gradient(velocity, 1.0 / features.fps, axis=0)
    root_angular_velocity = features.state[:, dof : dof + 3]

    def p95_rms(array: np.ndarray) -> float:
        return float(np.quantile(np.sqrt(np.mean(array * array, axis=-1)), 0.95))

    return {
        "joint_speed_p95_rms": p95_rms(velocity),
        "joint_acceleration_p95_rms": p95_rms(acceleration),
        "root_angular_speed_p95_rms": p95_rms(root_angular_velocity),
    }


def difficulty_ratio(candidate: dict[str, float], anchor: dict[str, float]) -> float:
    return max(candidate[key] / max(anchor[key], 1.0e-6) for key in anchor)


def select_candidate(records: list[dict], max_difficulty_ratio: float) -> dict | None:
    eligible = [
        record
        for record in records
        if record.get("passes_ambiguity")
        and record.get("difficulty_ratio", float("inf")) <= max_difficulty_ratio
    ]
    eligible.sort(
        key=lambda record: (
            record["difficulty_ratio"],
            -record["ambiguity"]["num_selected_windows"],
            -float(record["ambiguity"]["eligible_future_distance"]["median"] or 0.0),
            record["clip"],
        )
    )
    return eligible[0] if eligible else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", default="walk1_subject5")
    parser.add_argument("--exclude", nargs="*", default=["walk3_subject1"])
    parser.add_argument("--pairs_dir", default=str(data_root() / "pairs" / "unitree_g1"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_windows", type=int, default=20)
    parser.add_argument("--max_difficulty_ratio", type=float, default=1.25)
    args = parser.parse_args()

    pairs_dir = pathlib.Path(args.pairs_dir)
    anchor_path = pairs_dir / f"{args.anchor}.npz"
    if not anchor_path.is_file():
        parser.error(f"missing anchor {anchor_path}")
    anchor_features = _load_clip(anchor_path)
    anchor_difficulty = reference_difficulty(anchor_features)
    excluded = set(args.exclude) | {args.anchor}
    records = []
    for path in sorted(pairs_dir.glob("*.npz")):
        clip = path.stem
        if clip in excluded:
            continue
        record = {
            "clip": clip,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        try:
            features = _load_clip(path)
            difficulty = reference_difficulty(features)
            ambiguity = ambiguity_windows(anchor_features, features)
            record.update(
                {
                    "difficulty": difficulty,
                    "difficulty_ratio": difficulty_ratio(difficulty, anchor_difficulty),
                    "ambiguity": ambiguity,
                    "passes_ambiguity": ambiguity["num_selected_windows"] >= args.min_windows,
                }
            )
        except ValueError as error:
            record.update({"passes_ambiguity": False, "error": str(error)})
        records.append(record)
        if "ambiguity" in record:
            print(
                f"{clip}: windows={record['ambiguity']['num_selected_windows']} "
                f"difficulty={record['difficulty_ratio']:.3f}"
            )
        else:
            print(f"{clip}: ineligible ({record['error']})")

    selected = select_candidate(records, args.max_difficulty_ratio)
    report = {
        "protocol": "E69 exhaustive reference-only pair screen v1",
        "anchor": args.anchor,
        "excluded": sorted(excluded),
        "thresholds": {
            "time_bins": 100,
            "future_seconds": 1.0,
            "rollout_seconds": 10.0,
            "future_samples": 11,
            "max_state_distance_rms_z": 0.75,
            "min_future_distance_rms_z": 0.75,
            "min_spacing_seconds": 0.5,
            "min_windows": args.min_windows,
            "max_difficulty_ratio": args.max_difficulty_ratio,
        },
        "selection_order": [
            "difficulty_ratio ascending",
            "num_selected_windows descending",
            "eligible_future_distance.median descending",
            "clip ascending",
        ],
        "anchor_input": {
            "path": str(anchor_path.resolve()),
            "sha256": sha256_file(anchor_path),
            "difficulty": anchor_difficulty,
        },
        "candidates": records,
        "selected_clip": selected["clip"] if selected is not None else None,
        "gate_passed": selected is not None,
    }
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(f"selected_clip={report['selected_clip']}; report={output}")
    if selected is None:
        raise SystemExit("no candidate passed the frozen E69 screen")


if __name__ == "__main__":
    main()

