#!/usr/bin/env python3
"""Validate and decide the Gate 1 seed-0 screen."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any


TRAIN_REVISION = "4929924d6147b131f601ab5f6ac238a0fa45bae9"
ARMS = (
    "c0_seed0",
    "c1_bce_seed0",
    "c3_stance_seed0",
    "c4_teacher_velocity_seed0",
)
CANDIDATE_ARMS = (
    "c3_stance_seed0",
    "c4_teacher_velocity_seed0",
)
CLIPS = (
    "walk1_subject5",
    "dance2_subject4",
    "fight1_subject3",
    "run2_subject1",
    "jumps1_subject2",
    "sprint1_subject4",
    "aiming2_subject3",
)
OBJECTIVE_CONFIG = {
    "c0_seed0": {
        "contact_bce_weight": 0.0,
        "stance_velocity_weight": 0.0,
        "teacher_velocity_weight": 0.0,
        "phase_balanced_velocity": False,
    },
    "c1_bce_seed0": {
        "contact_bce_weight": 0.25,
        "stance_velocity_weight": 0.0,
        "teacher_velocity_weight": 0.0,
        "phase_balanced_velocity": False,
    },
    "c3_stance_seed0": {
        "contact_bce_weight": 0.0,
        "stance_velocity_weight": 0.03,
        "teacher_velocity_weight": 0.0,
        "phase_balanced_velocity": False,
    },
    "c4_teacher_velocity_seed0": {
        "contact_bce_weight": 0.0,
        "stance_velocity_weight": 0.0,
        "teacher_velocity_weight": 0.05,
        "phase_balanced_velocity": True,
    },
}
BASE_CONFIG = {
    "robot": "unitree_g1",
    "window": 64,
    "lr": 0.0003,
    "min_lr": 0.00001,
    "latent_dim": 128,
    "enc_hidden": 256,
    "dec_hidden": 256,
    "contact_mask": "teacher_height",
    "contact_weight": 0.0,
    "foot_vel_weight": 0.0,
    "edge_contact": False,
    "edge_velocity_weight": 0.0,
    "penetration_weight": 0.0,
    "no_temporal": False,
    "steps": 50000,
    "eval_every": 5000,
    "ckpt_every": 5000,
    "diag_every": 5000,
    "seed": 0,
}
REQUIRED_METRICS = (
    "teacher_height_stance_speed_ms",
    "source_contact_stance_speed_ms",
    "mpjpe_m",
    "dof_jerk",
    "limit_violation_fraction",
    "penetration_mean_m",
    "penetration_fraction",
)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return _finite(actual) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    return actual == expected


def _validate_manifest(arm: str, manifest: dict[str, Any]) -> list[str]:
    errors = []
    if manifest.get("status") != "completed":
        errors.append(f"manifest status is {manifest.get('status')!r}")
    progress = manifest.get("progress", {})
    if progress.get("completion_state") != "completed" or progress.get("step") != 50000:
        errors.append("manifest progress is not completed at step 50000")
    git = manifest.get("git", {})
    if git.get("sha") != TRAIN_REVISION or git.get("dirty") is not False:
        errors.append("manifest does not use the clean frozen trainer revision")

    config = manifest.get("config", {})
    expected_config = {**BASE_CONFIG, **OBJECTIVE_CONFIG[arm]}
    for key, expected in expected_config.items():
        actual = config.get(key)
        if not _same_value(actual, expected):
            errors.append(f"config {key}={actual!r}, expected {expected!r}")
    return errors


def _validate_benchmark(
    arm: str,
    benchmark: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors = []
    protocol = benchmark.get("_protocol", {})
    checks = {
        "window_frames": (protocol.get("window_frames"), 192),
        "windows_per_clip_max": (protocol.get("windows_per_clip_max"), 6),
        "clips": (protocol.get("clips"), list(CLIPS)),
        "bootstrap.samples": (protocol.get("bootstrap", {}).get("samples"), 2000),
        "bootstrap.seed": (protocol.get("bootstrap", {}).get("seed"), 0),
        "checkpoint.step": (protocol.get("checkpoint", {}).get("step"), 50000),
        "checkpoint.complete": (protocol.get("checkpoint", {}).get("complete"), True),
        "git.tracked_dirty": (protocol.get("git", {}).get("tracked_dirty"), False),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"benchmark {field}={actual!r}, expected {expected!r}")

    payload = benchmark.get("unitree_g1", {})
    if payload.get("num_windows") != 42:
        errors.append(f"benchmark num_windows={payload.get('num_windows')!r}, expected 42")
    metrics = payload.get("snmr", {})
    for name in REQUIRED_METRICS:
        value = metrics.get(name)
        if not _finite(value):
            errors.append(f"benchmark metric {name} is missing or nonfinite")

    per_clip = payload.get("per_clip", {}).get("snmr", {})
    if set(per_clip) != set(CLIPS):
        errors.append(
            f"benchmark per-clip set mismatch: "
            f"missing={sorted(set(CLIPS) - set(per_clip))}, "
            f"unexpected={sorted(set(per_clip) - set(CLIPS))}"
        )
    for clip in CLIPS:
        clip_metrics = per_clip.get(clip, {})
        contact_samples = clip_metrics.get("teacher_height_contact_samples")
        if "teacher_height_stance_speed_ms" not in clip_metrics:
            if _finite(contact_samples) and float(contact_samples) == 0.0:
                continue
            errors.append(
                f"{clip}: teacher-height stance speed is missing without zero support"
            )
            continue
        value = clip_metrics["teacher_height_stance_speed_ms"]
        if value is None and _finite(contact_samples) and float(contact_samples) == 0.0:
            continue
        if not _finite(value):
            errors.append(f"{clip}: teacher-height stance speed is malformed or nonfinite")

    checkpoint_path = protocol.get("checkpoint", {}).get("path")
    if isinstance(checkpoint_path, str):
        expected_suffix = f"/{arm}/ckpt.pt"
        if not checkpoint_path.endswith(expected_suffix):
            errors.append(
                f"benchmark checkpoint {checkpoint_path!r} does not end with {expected_suffix!r}"
            )
    else:
        errors.append("benchmark checkpoint path is missing")

    summary = {
        name: metrics.get(name)
        for name in REQUIRED_METRICS
    }
    summary["contact_head_vs_teacher_height_f1"] = metrics.get(
        "contact_head_vs_teacher_height_f1"
    )
    summary["per_clip_teacher_height_stance_speed_ms"] = {
        clip: per_clip.get(clip, {}).get("teacher_height_stance_speed_ms")
        for clip in CLIPS
    }
    summary["evaluator_sha256"] = protocol.get("evaluator_sha256")
    summary["checkpoint_sha256"] = protocol.get("checkpoint", {}).get("sha256")
    return errors, summary


def promotion_decision(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    control_speed = control["teacher_height_stance_speed_ms"]
    candidate_speed = candidate["teacher_height_stance_speed_ms"]
    comparable_clips = [
        clip
        for clip in CLIPS
        if _finite(control["per_clip_teacher_height_stance_speed_ms"][clip])
        and _finite(candidate["per_clip_teacher_height_stance_speed_ms"][clip])
    ]
    unavailable_clips = [clip for clip in CLIPS if clip not in comparable_clips]
    clip_improvements = sum(
        candidate["per_clip_teacher_height_stance_speed_ms"][clip]
        < control["per_clip_teacher_height_stance_speed_ms"][clip]
        for clip in comparable_clips
    )
    checks = {
        "teacher_height_reduction_ge_25pct": candidate_speed <= 0.75 * control_speed,
        "at_least_5_of_7_clips_improve": clip_improvements >= 5,
        "mpjpe_delta_le_0.005": candidate["mpjpe_m"] <= control["mpjpe_m"] + 0.005,
        "source_contact_not_over_1.1x": (
            candidate["source_contact_stance_speed_ms"]
            <= 1.1 * control["source_contact_stance_speed_ms"]
        ),
        "dof_jerk_le_1.2x": candidate["dof_jerk"] <= 1.2 * control["dof_jerk"],
        "zero_limit_violations": candidate["limit_violation_fraction"] == 0.0,
        "penetration_mean_guard": (
            candidate["penetration_mean_m"] <= control["penetration_mean_m"] + 0.002
        ),
        "penetration_fraction_guard": (
            candidate["penetration_fraction"] <= control["penetration_fraction"] + 0.02
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "clip_improvements": clip_improvements,
        "comparable_clips": comparable_clips,
        "unavailable_clips": unavailable_clips,
        "teacher_height_relative_reduction": (
            1.0 - candidate_speed / control_speed if control_speed > 0 else None
        ),
        "mpjpe_delta_m": candidate["mpjpe_m"] - control["mpjpe_m"],
        "source_contact_relative_change": (
            candidate["source_contact_stance_speed_ms"]
            / control["source_contact_stance_speed_ms"]
            - 1.0
            if control["source_contact_stance_speed_ms"] > 0
            else None
        ),
        "dof_jerk_ratio": (
            candidate["dof_jerk"] / control["dof_jerk"]
            if control["dof_jerk"] > 0
            else None
        ),
    }


def analyze_screen(root: pathlib.Path) -> dict[str, Any]:
    protocol_errors = []
    arms = {}
    existing_arm_dirs = {
        path.name for path in root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    unexpected = existing_arm_dirs - set(ARMS)
    missing = set(ARMS) - existing_arm_dirs
    if unexpected:
        protocol_errors.append(f"unexpected arm directories: {sorted(unexpected)}")
    if missing:
        protocol_errors.append(f"missing arm directories: {sorted(missing)}")

    evaluator_hashes = set()
    for arm in ARMS:
        arm_dir = root / arm
        errors = []
        manifest_path = arm_dir / "manifest.json"
        benchmark_path = arm_dir / "benchmark.json"
        if not manifest_path.is_file():
            errors.append("missing manifest.json")
        if not benchmark_path.is_file():
            errors.append("missing benchmark.json")
        if errors:
            arms[arm] = {"passed": False, "errors": errors}
            continue
        manifest = _read_json(manifest_path)
        benchmark = _read_json(benchmark_path)
        errors.extend(_validate_manifest(arm, manifest))
        benchmark_errors, summary = _validate_benchmark(arm, benchmark)
        errors.extend(benchmark_errors)
        if summary.get("evaluator_sha256"):
            evaluator_hashes.add(summary["evaluator_sha256"])
        arms[arm] = {
            "passed": not errors,
            "errors": errors,
            "summary": summary,
        }

    if len(evaluator_hashes) != 1:
        protocol_errors.append(
            f"benchmarks use {len(evaluator_hashes)} evaluator hashes: "
            f"{sorted(evaluator_hashes)}"
        )
    protocol_passed = (
        not protocol_errors
        and len(arms) == len(ARMS)
        and all(arm["passed"] for arm in arms.values())
    )

    decisions = {}
    promoted_arms = []
    if protocol_passed:
        control = arms["c0_seed0"]["summary"]
        decisions = {
            arm: promotion_decision(control, arms[arm]["summary"])
            for arm in CANDIDATE_ARMS
        }
        promoted_arms = sorted(
            (arm for arm, decision in decisions.items() if decision["passed"]),
            key=lambda arm: (
                arms[arm]["summary"]["teacher_height_stance_speed_ms"],
                arms[arm]["summary"]["mpjpe_m"],
                arms[arm]["summary"]["source_contact_stance_speed_ms"],
            ),
        )[:2]

    return {
        "root": str(root),
        "passed": protocol_passed,
        "interpretation": (
            "seed-0 causal screen; promotion requires replication with seeds 1 and 2"
        ),
        "protocol_errors": protocol_errors,
        "evaluator_sha256": (
            next(iter(evaluator_hashes)) if len(evaluator_hashes) == 1 else None
        ),
        "retained_negative_control": "c1_bce_seed0",
        "candidate_decisions": decisions,
        "promoted_arms": promoted_arms,
        "arms": arms,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(
        "arm\tteacher_height_speed\tmpjpe_m\tsource_contact_speed\t"
        "clip_improvements\tresult"
    )
    for arm in ARMS:
        result = report["arms"].get(arm, {})
        summary = result.get("summary", {})
        decision = report["candidate_decisions"].get(arm)
        values = [
            arm,
            summary.get("teacher_height_stance_speed_ms"),
            summary.get("mpjpe_m"),
            summary.get("source_contact_stance_speed_ms"),
            decision.get("clip_improvements") if decision else None,
            (
                "PROMOTE" if decision and decision["passed"]
                else "RETAIN_CONTROL" if arm == "c1_bce_seed0"
                else "PASS" if result.get("passed") and decision is None
                else "REJECT" if decision
                else "INVALID"
            ),
        ]
        print("\t".join("-" if value is None else str(value) for value in values))
    for error in report["protocol_errors"]:
        print(f"ERROR: {error}")
    for arm, result in report["arms"].items():
        for error in result["errors"]:
            print(f"ERROR: {arm}: {error}")
    print(f"promoted: {report['promoted_arms']}")
    print(f"overall: {'PASS' if report['passed'] else 'INVALID'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    report = analyze_screen(args.root)
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
