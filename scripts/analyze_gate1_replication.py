#!/usr/bin/env python3
"""Validate and decide the three-seed Gate 1 replication."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from statistics import fmean
from typing import Any


TRAIN_REVISION = "4929924d6147b131f601ab5f6ac238a0fa45bae9"
EVALUATOR_SHA256 = "76f736449b67bc8df4150bb8cfb2da44191a673abbd0d93c3af886cd46128cdd"
SEEDS = (0, 1, 2)
REPLICATION_SEEDS = (1, 2)
FAMILIES = ("c0", "c3_stance", "c4_teacher_velocity")
CANDIDATE_FAMILIES = ("c3_stance", "c4_teacher_velocity")
SCREEN_ARMS = {
    "c0": "c0_seed0",
    "c3_stance": "c3_stance_seed0",
    "c4_teacher_velocity": "c4_teacher_velocity_seed0",
}
OBJECTIVE_CONFIG = {
    "c0": {
        "contact_bce_weight": 0.0,
        "stance_velocity_weight": 0.0,
        "teacher_velocity_weight": 0.0,
        "phase_balanced_velocity": False,
    },
    "c3_stance": {
        "contact_bce_weight": 0.0,
        "stance_velocity_weight": 0.03,
        "teacher_velocity_weight": 0.0,
        "phase_balanced_velocity": False,
    },
    "c4_teacher_velocity": {
        "contact_bce_weight": 0.0,
        "stance_velocity_weight": 0.0,
        "teacher_velocity_weight": 0.05,
        "phase_balanced_velocity": True,
    },
}
CLIPS = (
    "walk1_subject5",
    "dance2_subject4",
    "fight1_subject3",
    "run2_subject1",
    "jumps1_subject2",
    "sprint1_subject4",
    "aiming2_subject3",
)
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


def _arm_name(family: str, seed: int) -> str:
    return f"{family}_seed{seed}"


def _validate_manifest(
    family: str,
    seed: int,
    manifest: dict[str, Any],
) -> list[str]:
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
    expected_config = {
        **BASE_CONFIG,
        **OBJECTIVE_CONFIG[family],
        "seed": seed,
    }
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
        "evaluator_sha256": (protocol.get("evaluator_sha256"), EVALUATOR_SHA256),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"benchmark {field}={actual!r}, expected {expected!r}")

    payload = benchmark.get("unitree_g1", {})
    if payload.get("num_windows") != 42:
        errors.append(f"benchmark num_windows={payload.get('num_windows')!r}, expected 42")
    metrics = payload.get("snmr", {})
    for name in REQUIRED_METRICS:
        if not _finite(metrics.get(name)):
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
        value = clip_metrics.get("teacher_height_stance_speed_ms")
        if _finite(value):
            continue
        if value is None and _finite(contact_samples) and float(contact_samples) == 0.0:
            continue
        errors.append(f"{clip}: invalid teacher-height stance speed/support")

    checkpoint_path = protocol.get("checkpoint", {}).get("path")
    expected_suffix = f"/{arm}/ckpt.pt"
    if not isinstance(checkpoint_path, str) or not checkpoint_path.endswith(expected_suffix):
        errors.append(
            f"benchmark checkpoint {checkpoint_path!r} does not end with {expected_suffix!r}"
        )

    summary = {name: metrics.get(name) for name in REQUIRED_METRICS}
    summary["per_clip_teacher_height_stance_speed_ms"] = {
        clip: per_clip.get(clip, {}).get("teacher_height_stance_speed_ms")
        for clip in CLIPS
    }
    summary["evaluator_sha256"] = protocol.get("evaluator_sha256")
    summary["checkpoint_sha256"] = protocol.get("checkpoint", {}).get("sha256")
    return errors, summary


def _means(seed_summaries: dict[int, dict[str, Any]]) -> dict[str, float]:
    return {
        metric: fmean(float(seed_summaries[seed][metric]) for seed in SEEDS)
        for metric in REQUIRED_METRICS
    }


def _gate_decision(
    control: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    control_mean = _means(control)
    candidate_mean = _means(candidate)
    endpoint_pass_seeds = [
        seed
        for seed in SEEDS
        if candidate[seed]["teacher_height_stance_speed_ms"] <= 0.08
    ]
    improvement_seeds = [
        seed
        for seed in SEEDS
        if candidate[seed]["teacher_height_stance_speed_ms"]
        < control[seed]["teacher_height_stance_speed_ms"]
    ]
    checks = {
        "endpoint_le_0.08_in_at_least_2_seeds": len(endpoint_pass_seeds) >= 2,
        "mean_mpjpe_delta_le_0.005": (
            candidate_mean["mpjpe_m"] <= control_mean["mpjpe_m"] + 0.005
        ),
        "mean_source_contact_not_over_1.1x": (
            candidate_mean["source_contact_stance_speed_ms"]
            <= 1.1 * control_mean["source_contact_stance_speed_ms"]
        ),
        "mean_dof_jerk_le_1.2x": (
            candidate_mean["dof_jerk"] <= 1.2 * control_mean["dof_jerk"]
        ),
        "zero_limit_violations_all_seeds": all(
            candidate[seed]["limit_violation_fraction"] == 0.0 for seed in SEEDS
        ),
        "mean_penetration_guard": (
            candidate_mean["penetration_mean_m"]
            <= control_mean["penetration_mean_m"] + 0.002
        ),
        "mean_penetration_fraction_guard": (
            candidate_mean["penetration_fraction"]
            <= control_mean["penetration_fraction"] + 0.02
        ),
        "improves_in_at_least_2_seeds": len(improvement_seeds) >= 2,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "endpoint_pass_seeds": endpoint_pass_seeds,
        "improvement_seeds": improvement_seeds,
        "control_mean": control_mean,
        "candidate_mean": candidate_mean,
        "mean_mpjpe_delta_m": (
            candidate_mean["mpjpe_m"] - control_mean["mpjpe_m"]
        ),
        "mean_teacher_height_relative_reduction": (
            1.0
            - candidate_mean["teacher_height_stance_speed_ms"]
            / control_mean["teacher_height_stance_speed_ms"]
        ),
        "absolute_product_mpjpe_mean_le_0.04": candidate_mean["mpjpe_m"] <= 0.04,
    }


def analyze_replication(
    replication_root: pathlib.Path,
    screen_root: pathlib.Path,
) -> dict[str, Any]:
    protocol_errors = []
    screen_analysis_path = screen_root / "analysis.json"
    if not screen_analysis_path.is_file():
        screen_analysis = {}
        protocol_errors.append("missing screen analysis.json")
    else:
        screen_analysis = _read_json(screen_analysis_path)
    if screen_analysis.get("passed") is not True:
        protocol_errors.append("seed-0 screen analysis did not pass")
    if screen_analysis.get("evaluator_sha256") != EVALUATOR_SHA256:
        protocol_errors.append("seed-0 screen evaluator hash mismatch")
    if screen_analysis.get("promoted_arms") != [
        "c4_teacher_velocity_seed0",
        "c3_stance_seed0",
    ]:
        protocol_errors.append("seed-0 promoted-arm set or order mismatch")

    expected_arms = {
        _arm_name(family, seed)
        for family in FAMILIES
        for seed in REPLICATION_SEEDS
    }
    existing_arms = {
        path.name
        for path in replication_root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    if existing_arms - expected_arms:
        protocol_errors.append(
            f"unexpected arm directories: {sorted(existing_arms - expected_arms)}"
        )
    if expected_arms - existing_arms:
        protocol_errors.append(
            f"missing arm directories: {sorted(expected_arms - existing_arms)}"
        )

    arms = {}
    for family in FAMILIES:
        for seed in REPLICATION_SEEDS:
            arm = _arm_name(family, seed)
            arm_dir = replication_root / arm
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
            errors.extend(
                _validate_manifest(family, seed, _read_json(manifest_path))
            )
            benchmark_errors, summary = _validate_benchmark(
                arm, _read_json(benchmark_path)
            )
            errors.extend(benchmark_errors)
            arms[arm] = {
                "passed": not errors,
                "errors": errors,
                "summary": summary,
            }

    screen_arms = screen_analysis.get("arms", {})
    for family, arm in SCREEN_ARMS.items():
        result = screen_arms.get(arm, {})
        if result.get("passed") is not True or not isinstance(result.get("summary"), dict):
            protocol_errors.append(f"invalid seed-0 screen arm {arm}")

    protocol_passed = (
        not protocol_errors
        and len(arms) == len(expected_arms)
        and all(result["passed"] for result in arms.values())
    )
    seed_summaries: dict[str, dict[int, dict[str, Any]]] = {}
    decisions = {}
    passing_families = []
    if protocol_passed:
        for family in FAMILIES:
            seed_summaries[family] = {
                0: screen_arms[SCREEN_ARMS[family]]["summary"],
                **{
                    seed: arms[_arm_name(family, seed)]["summary"]
                    for seed in REPLICATION_SEEDS
                },
            }
        decisions = {
            family: _gate_decision(seed_summaries["c0"], seed_summaries[family])
            for family in CANDIDATE_FAMILIES
        }
        passing_families = sorted(
            (family for family, decision in decisions.items() if decision["passed"]),
            key=lambda family: (
                decisions[family]["candidate_mean"]["teacher_height_stance_speed_ms"],
                decisions[family]["candidate_mean"]["mpjpe_m"],
                decisions[family]["candidate_mean"]["source_contact_stance_speed_ms"],
            ),
        )

    return {
        "replication_root": str(replication_root),
        "screen_root": str(screen_root),
        "passed": protocol_passed,
        "gate1_passed": bool(passing_families),
        "interpretation": (
            "protocol validity is separate from the preregistered three-seed Gate 1 decision"
        ),
        "protocol_errors": protocol_errors,
        "evaluator_sha256": EVALUATOR_SHA256,
        "candidate_decisions": decisions,
        "gate1_passing_families": passing_families,
        "seed_summaries": seed_summaries,
        "replication_arms": arms,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(
        "family\tseed\tteacher_height_speed\tmpjpe_m\t"
        "source_contact_speed\tdof_jerk"
    )
    for family, summaries in report["seed_summaries"].items():
        for seed, summary in sorted(summaries.items()):
            print(
                "\t".join(
                    str(value)
                    for value in (
                        family,
                        seed,
                        summary["teacher_height_stance_speed_ms"],
                        summary["mpjpe_m"],
                        summary["source_contact_stance_speed_ms"],
                        summary["dof_jerk"],
                    )
                )
            )
    for family, decision in report["candidate_decisions"].items():
        print(
            f"{family}: {'PASS' if decision['passed'] else 'FAIL'} "
            f"endpoint_seeds={decision['endpoint_pass_seeds']} "
            f"improvement_seeds={decision['improvement_seeds']} "
            f"mean_speed={decision['candidate_mean']['teacher_height_stance_speed_ms']}"
        )
    for error in report["protocol_errors"]:
        print(f"ERROR: {error}")
    for arm, result in report["replication_arms"].items():
        for error in result["errors"]:
            print(f"ERROR: {arm}: {error}")
    print(f"gate1_passing_families: {report['gate1_passing_families']}")
    print(f"protocol: {'PASS' if report['passed'] else 'INVALID'}")
    print(f"gate1: {'PASS' if report['gate1_passed'] else 'FAIL_OR_UNDECIDED'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replication_root", type=pathlib.Path)
    parser.add_argument("--screen-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    report = analyze_replication(args.replication_root, args.screen_root)
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
