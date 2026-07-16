#!/usr/bin/env python3
"""Analyze the Phase-2 multi-seed replication of the WBT latent arms.

Reads a manifest TSV with columns:
    arm  train_seed  eval_seed  training_name  checkpoint_path  report_path

For every (arm != base) cell, computes paired deltas against the base cell with the same
(train_seed, eval_seed). Replication rules frozen from docs/WBT_LATENT_PLAN_v2.md section 5.3:

- baseline-relative arms (s3): median cell must meet the frozen Phase-0 promotion rule
  (completion delta >= -5pp floor AND >=5% relative improvement on joint RMSE or survival,
  or >=5pp completion) and NO cell may violate the -5pp completion floor.
- absolute-floor arms (l1, min_completion=0.70): median cell completion >= floor and no cell
  below the floor.
- screening arms (c3, single seed): descriptive only, no verdict; reported for the
  latent-vs-explicit critic-preview contrast against s3.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import Counter
from typing import Any, NamedTuple

from analyze_wbt_latent_pilot import (
    COMPLETION_FLOOR_DELTA,
    COMPLETION_IMPROVEMENT,
    METRICS,
    PRIMARY_JOINT_METRIC,
    RELATIVE_IMPROVEMENT,
    _finite_number,
    _validate_rollout_report,
)


class ManifestRow(NamedTuple):
    arm: str
    train_seed: int
    eval_seed: int
    training_name: str
    checkpoint_path: pathlib.Path
    report_path: pathlib.Path


EXPECTED_TRAIN_SEEDS = {
    "base": (0, 1, 2),
    "s3": (0, 1, 2),
    "l1": (0, 1, 2),
    "c3": (0,),
}
EXPECTED_EVAL_SEEDS = (404, 405)


def read_manifest(path: pathlib.Path) -> list[ManifestRow]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(
                f"{path}:{line_number}: expected 6 fields, found {len(fields)}"
            )
        rows.append(
            ManifestRow(
                arm=fields[0],
                train_seed=int(fields[1]),
                eval_seed=int(fields[2]),
                training_name=fields[3],
                checkpoint_path=pathlib.Path(fields[4]),
                report_path=pathlib.Path(fields[5]),
            )
        )
    return rows


def _cell_metrics(report: dict[str, Any]) -> dict[str, float]:
    rollouts = report["rollouts"]
    n = len(rollouts)
    out = {
        "completion": sum(float(r["completed"]) for r in rollouts) / n,
        "survival_s": sum(float(r["survival_s"]) for r in rollouts) / n,
    }
    for metric in METRICS:
        out[metric] = sum(float(r["metrics"][metric]) for r in rollouts) / n
    return out


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def decide_baseline_relative(cells: list[dict[str, float]]) -> dict[str, Any]:
    """Frozen Phase-0 rule applied to the median cell; floor applied to every cell."""
    completion_deltas = [c["completion_delta"] for c in cells]
    median_completion_delta = _median(completion_deltas)
    median_rmse_rel = _median([c["joint_rmse_rel"] for c in cells])
    median_survival_rel = _median([c["survival_rel"] for c in cells])
    floor_violations = [d for d in completion_deltas if d < COMPLETION_FLOOR_DELTA]
    improvements = {
        "completion_at_least_5pp": median_completion_delta >= COMPLETION_IMPROVEMENT,
        "survival_at_least_5pct": median_survival_rel >= RELATIVE_IMPROVEMENT,
        "joint_rmse_at_least_5pct_lower": median_rmse_rel <= -RELATIVE_IMPROVEMENT,
    }
    return {
        "rule": "baseline_relative_frozen_median",
        "median_completion_delta": median_completion_delta,
        "median_joint_rmse_rel": median_rmse_rel,
        "median_survival_rel": median_survival_rel,
        "floor_violation_count": len(floor_violations),
        "improvement_checks": improvements,
        "replicated": (
            not floor_violations
            and median_completion_delta >= COMPLETION_FLOOR_DELTA
            and any(improvements.values())
        ),
    }


def decide_absolute_floor(
    cells: list[dict[str, float]], min_completion: float
) -> dict[str, Any]:
    completions = [c["completion"] for c in cells]
    below = [c for c in completions if c < min_completion]
    return {
        "rule": "absolute_completion_floor_all_cells",
        "min_completion": min_completion,
        "median_completion": _median(completions),
        "cells_below_floor": len(below),
        "replicated": not below,
    }


def analyze(
    manifest_rows: list[ManifestRow],
    arm_rules: dict[str, dict[str, Any]],
    motion_files: dict[str, pathlib.Path],
) -> dict[str, Any]:
    errors = []
    row_keys = [
        (row.arm, row.train_seed, row.eval_seed) for row in manifest_rows
    ]
    duplicate_keys = sorted(
        key for key, count in Counter(row_keys).items() if count > 1
    )
    if duplicate_keys:
        errors.append(f"duplicate manifest cells: {duplicate_keys}")

    if set(arm_rules) == set(EXPECTED_TRAIN_SEEDS):
        expected_keys = {
            (arm, train_seed, eval_seed)
            for arm, train_seeds in EXPECTED_TRAIN_SEEDS.items()
            for train_seed in train_seeds
            for eval_seed in EXPECTED_EVAL_SEEDS
        }
        actual_keys = set(row_keys)
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        if missing:
            errors.append(f"missing manifest cells: {missing}")
        if unexpected:
            errors.append(f"unexpected manifest cells: {unexpected}")

    cells: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in manifest_rows:
        if row.arm not in arm_rules:
            errors.append(f"unknown arm {row.arm!r} in manifest")
            continue
        if not row.checkpoint_path.is_file():
            errors.append(f"{row.arm} seed{row.train_seed}: missing checkpoint")
        try:
            report = json.loads(row.report_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                f"{row.arm} s{row.train_seed} e{row.eval_seed}: cannot load report: {exc}"
            )
            continue
        report_errors = _validate_rollout_report(
            report,
            expected_training_name=row.training_name,
            expected_motion_file=motion_files[row.arm],
            expected_seed=row.eval_seed,
        )
        errors.extend(
            f"{row.arm} s{row.train_seed} e{row.eval_seed}: {e}" for e in report_errors
        )
        if not report_errors:
            cells[(row.arm, row.train_seed, row.eval_seed)] = _cell_metrics(report)

    base_cells = {
        (seed, eval_seed): metrics
        for (arm, seed, eval_seed), metrics in cells.items()
        if arm == "base"
    }
    arm_summaries: dict[str, Any] = {}
    for arm, rule in arm_rules.items():
        if arm == "base":
            continue
        paired = []
        for (cell_arm, seed, eval_seed), metrics in sorted(cells.items()):
            if cell_arm != arm:
                continue
            base = base_cells.get((seed, eval_seed))
            if base is None:
                errors.append(f"{arm} s{seed} e{eval_seed}: no matching base cell")
                continue
            paired.append(
                {
                    "train_seed": seed,
                    "eval_seed": eval_seed,
                    "completion": metrics["completion"],
                    "completion_delta": metrics["completion"] - base["completion"],
                    "survival_rel": metrics["survival_s"] / base["survival_s"] - 1.0,
                    "joint_rmse_rel": (
                        metrics[PRIMARY_JOINT_METRIC] / base[PRIMARY_JOINT_METRIC]
                        - 1.0
                    ),
                    "metrics": metrics,
                }
            )
        summary: dict[str, Any] = {"cells": paired}
        seed_summaries = []
        for seed in sorted({cell["train_seed"] for cell in paired}):
            seed_cells = [
                cell for cell in paired if cell["train_seed"] == seed
            ]
            seed_summaries.append(
                {
                    "train_seed": seed,
                    "eval_seeds": [
                        cell["eval_seed"] for cell in seed_cells
                    ],
                    "completion": float(
                        statistics.mean(
                            cell["completion"] for cell in seed_cells
                        )
                    ),
                    "completion_delta": float(
                        statistics.mean(
                            cell["completion_delta"] for cell in seed_cells
                        )
                    ),
                    "survival_rel": float(
                        statistics.mean(
                            cell["survival_rel"] for cell in seed_cells
                        )
                    ),
                    "joint_rmse_rel": float(
                        statistics.mean(
                            cell["joint_rmse_rel"] for cell in seed_cells
                        )
                    ),
                }
            )
        summary["seed_summaries"] = seed_summaries
        if paired and not errors:
            if rule.get("screening"):
                summary["decision"] = {"rule": "screening_descriptive_only"}
            elif "min_completion" in rule:
                summary["decision"] = decide_absolute_floor(
                    paired, rule["min_completion"]
                )
            else:
                summary["decision"] = decide_baseline_relative(seed_summaries)
        arm_summaries[arm] = summary

    contrasts: dict[str, Any] = {}
    if "s3" in arm_summaries and "c3" in arm_summaries:
        s3_cells = {
            (cell["train_seed"], cell["eval_seed"]): cell
            for cell in arm_summaries["s3"]["cells"]
        }
        c3_cells = {
            (cell["train_seed"], cell["eval_seed"]): cell
            for cell in arm_summaries["c3"]["cells"]
        }
        common = sorted(set(s3_cells) & set(c3_cells))
        contrasts["s3_minus_c3"] = {
            "interpretation": (
                "descriptive seed-matched contrast; negative RMSE is favorable "
                "to S3"
            ),
            "cells": [
                {
                    "train_seed": seed,
                    "eval_seed": eval_seed,
                    "completion_delta": (
                        s3_cells[(seed, eval_seed)]["completion"]
                        - c3_cells[(seed, eval_seed)]["completion"]
                    ),
                    "joint_rmse_rel": (
                        s3_cells[(seed, eval_seed)]["metrics"][
                            PRIMARY_JOINT_METRIC
                        ]
                        / c3_cells[(seed, eval_seed)]["metrics"][
                            PRIMARY_JOINT_METRIC
                        ]
                        - 1.0
                    ),
                    "survival_rel": (
                        s3_cells[(seed, eval_seed)]["metrics"]["survival_s"]
                        / c3_cells[(seed, eval_seed)]["metrics"]["survival_s"]
                        - 1.0
                    ),
                }
                for seed, eval_seed in common
            ],
        }

    replicated = [
        arm
        for arm, summary in arm_summaries.items()
        if summary.get("decision", {}).get("replicated")
    ]
    return {
        "passed": not errors,
        "verdict": (
            "invalid_phase2_artifacts"
            if errors
            else (
                "replicated:" + ",".join(sorted(replicated))
                if replicated
                else "no_arm_replicates"
            )
        ),
        "interpretation": (
            "multi-seed single-clip replication; establishes seed robustness on "
            "walk1_subject5 only, not cross-motion benefit"
        ),
        "protocol_errors": errors,
        "base_cells": {
            f"s{seed}_e{eval_seed}": metrics
            for (seed, eval_seed), metrics in sorted(base_cells.items())
        },
        "arms": arm_summaries,
        "contrasts": contrasts,
        "replicated_arms": sorted(replicated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--arm-rules", required=True, type=pathlib.Path)
    parser.add_argument("--motion-files", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    arm_rules = json.loads(args.arm_rules.read_text())
    motion_files = {
        arm: pathlib.Path(path)
        for arm, path in json.loads(args.motion_files.read_text()).items()
    }
    report = analyze(read_manifest(args.manifest), arm_rules, motion_files)
    for error in report["protocol_errors"]:
        print(f"ERROR: {error}")
    for arm, summary in report["arms"].items():
        decision = summary.get("decision", {})
        for cell in summary["cells"]:
            print(
                f"{arm} s{cell['train_seed']} e{cell['eval_seed']}: "
                f"completion={cell['completion']:.1%} "
                f"(delta {cell['completion_delta']:+.1%}), "
                f"rmse_rel={cell['joint_rmse_rel']:+.1%}, "
                f"survival_rel={cell['survival_rel']:+.1%}"
            )
        print(f"{arm} decision: {decision}")
    print(f"verdict: {report['verdict']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
