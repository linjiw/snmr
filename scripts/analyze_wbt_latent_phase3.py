#!/usr/bin/env python3
"""Analyze the Phase-3 multi-clip WBT screen (docs/WBT_LATENT_PHASE3_PROTOCOL.md section 5).

Manifest TSV columns:
    arm  clip  split  eval_seed  training_name  checkpoint_path  report_path

`split` is "heldout" or "trained". The baseline arm is `b_multi`; every other arm's cells are
paired per (clip, eval_seed) against the baseline cell. Promotion rules:

- arms with rule "phase0_heldout" (s3_multi, c3_multi): the HELD-OUT mean must meet the frozen
  Phase-0 rule vs b_multi (completion delta >= -5pp floor AND >=5% relative improvement on
  joint RMSE or survival, or >=5pp completion).
- arms with rule "descriptive" (l1_multi): no promotion; report the held-out completion gap and
  the pre-registered collapse flag (held-out completion < 0.40).
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from typing import Any, NamedTuple

from analyze_wbt_latent_pilot import (
    COMPLETION_FLOOR_DELTA,
    COMPLETION_IMPROVEMENT,
    PRIMARY_JOINT_METRIC,
    RELATIVE_IMPROVEMENT,
    _validate_rollout_report,
)
from analyze_wbt_latent_phase2 import _cell_metrics

BASELINE_ARM = "b_multi"
COLLAPSE_COMPLETION = 0.40


class ManifestRow(NamedTuple):
    arm: str
    clip: str
    split: str
    eval_seed: int
    training_name: str
    checkpoint_path: pathlib.Path
    report_path: pathlib.Path


def read_manifest(path: pathlib.Path) -> list[ManifestRow]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(
                f"{path}:{line_number}: expected 7 fields, found {len(fields)}"
            )
        if fields[2] not in ("heldout", "trained"):
            raise ValueError(f"{path}:{line_number}: bad split {fields[2]!r}")
        rows.append(
            ManifestRow(
                arm=fields[0],
                clip=fields[1],
                split=fields[2],
                eval_seed=int(fields[3]),
                training_name=fields[4],
                checkpoint_path=pathlib.Path(fields[5]),
                report_path=pathlib.Path(fields[6]),
            )
        )
    return rows


def _phase0_rule(completion_delta: float, rmse_rel: float, survival_rel: float) -> dict:
    improvements = {
        "completion_at_least_5pp": completion_delta >= COMPLETION_IMPROVEMENT,
        "survival_at_least_5pct": survival_rel >= RELATIVE_IMPROVEMENT,
        "joint_rmse_at_least_5pct_lower": rmse_rel <= -RELATIVE_IMPROVEMENT,
    }
    return {
        "completion_floor_passed": completion_delta >= COMPLETION_FLOOR_DELTA,
        "improvement_checks": improvements,
        "promoted": (
            completion_delta >= COMPLETION_FLOOR_DELTA and any(improvements.values())
        ),
    }


def analyze(
    manifest_rows: list[ManifestRow],
    arm_rules: dict[str, str],
    clip_motion_files: dict[str, pathlib.Path],
) -> dict[str, Any]:
    errors = []
    row_keys = [(row.arm, row.clip, row.eval_seed) for row in manifest_rows]
    duplicate_keys = sorted(
        key for key, count in Counter(row_keys).items() if count > 1
    )
    if duplicate_keys:
        errors.append(f"duplicate manifest cells: {duplicate_keys}")

    expected_keys = {
        (arm, clip, 404)
        for arm in arm_rules
        for clip in clip_motion_files
    }
    actual_keys = set(row_keys)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"missing manifest cells: {missing}")
    if unexpected:
        errors.append(f"unexpected manifest cells: {unexpected}")
    if arm_rules.get(BASELINE_ARM) != "baseline":
        errors.append(f"{BASELINE_ARM} must use the baseline rule")

    cells: dict[tuple[str, str, int], dict[str, float]] = {}
    splits: dict[str, str] = {}
    for row in manifest_rows:
        if row.arm not in arm_rules:
            errors.append(f"unknown arm {row.arm!r}")
            continue
        if row.clip not in clip_motion_files:
            errors.append(f"unknown clip {row.clip!r}")
            continue
        prior_split = splits.get(row.clip)
        if prior_split is not None and prior_split != row.split:
            errors.append(
                f"clip {row.clip!r} has inconsistent splits: "
                f"{prior_split!r} and {row.split!r}"
            )
        splits[row.clip] = row.split
        if not row.checkpoint_path.is_file():
            errors.append(f"{row.arm}/{row.clip}: missing checkpoint")
        try:
            report = json.loads(row.report_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{row.arm}/{row.clip}: cannot load report: {exc}")
            continue
        report_errors = _validate_rollout_report(
            report,
            expected_training_name=row.training_name,
            expected_motion_file=clip_motion_files[row.clip],
            expected_seed=row.eval_seed,
        )
        errors.extend(f"{row.arm}/{row.clip}: {e}" for e in report_errors)
        if not report_errors:
            cells[(row.arm, row.clip, row.eval_seed)] = _cell_metrics(report)

    base_cells = {
        (clip, eval_seed): metrics
        for (arm, clip, eval_seed), metrics in cells.items()
        if arm == BASELINE_ARM
    }
    heldout_clips = sorted(
        clip for clip, split in splits.items() if split == "heldout"
    )
    if len(heldout_clips) < 2:
        errors.append("Phase 3 requires at least two held-out clips")
    arm_summaries: dict[str, Any] = {}
    for arm, rule in arm_rules.items():
        if arm == BASELINE_ARM:
            continue
        clip_rows = []
        for (cell_arm, clip, eval_seed), metrics in sorted(cells.items()):
            if cell_arm != arm:
                continue
            base = base_cells.get((clip, eval_seed))
            if base is None:
                errors.append(f"{arm}/{clip}: no baseline cell")
                continue
            clip_rows.append(
                {
                    "clip": clip,
                    "split": splits[clip],
                    "eval_seed": eval_seed,
                    "completion": metrics["completion"],
                    "baseline_completion": base["completion"],
                    "completion_delta": metrics["completion"] - base["completion"],
                    "survival_rel": metrics["survival_s"] / base["survival_s"] - 1.0,
                    "joint_rmse": metrics[PRIMARY_JOINT_METRIC],
                    "joint_rmse_rel": (
                        metrics[PRIMARY_JOINT_METRIC] / base[PRIMARY_JOINT_METRIC] - 1.0
                    ),
                }
            )
        summary: dict[str, Any] = {"cells": clip_rows}
        heldout = [c for c in clip_rows if c["split"] == "heldout"]
        if heldout and not errors:
            mean = lambda key: sum(c[key] for c in heldout) / len(heldout)  # noqa: E731
            heldout_summary = {
                "clips": [c["clip"] for c in heldout],
                "completion": mean("completion"),
                "completion_delta": mean("completion_delta"),
                "joint_rmse_rel": mean("joint_rmse_rel"),
                "survival_rel": mean("survival_rel"),
            }
            summary["heldout_mean"] = heldout_summary
            if rule == "phase0_heldout":
                summary["decision"] = _phase0_rule(
                    heldout_summary["completion_delta"],
                    heldout_summary["joint_rmse_rel"],
                    heldout_summary["survival_rel"],
                )
            elif rule == "descriptive":
                summary["decision"] = {
                    "rule": "descriptive",
                    "heldout_completion": heldout_summary["completion"],
                    "collapsed": heldout_summary["completion"] < COLLAPSE_COMPLETION,
                    "singleclip_reference_gap_pp": -16.0,
                    "heldout_gap_pp": heldout_summary["completion_delta"] * 100.0,
                }
            else:
                errors.append(f"{arm}: unknown rule {rule!r}")
        arm_summaries[arm] = summary

    promoted = [
        arm
        for arm, summary in arm_summaries.items()
        if summary.get("decision", {}).get("promoted")
    ]
    return {
        "passed": not errors,
        "verdict": (
            "invalid_phase3_artifacts"
            if errors
            else ("promoted:" + ",".join(sorted(promoted)) if promoted else "no_arm_promotes")
        ),
        "interpretation": (
            "multi-clip single-training-seed screen; held-out-clip endpoints test "
            "generalization, not cross-seed robustness"
        ),
        "protocol_errors": errors,
        "baseline_cells": {
            f"{clip}_e{eval_seed}": metrics
            for (clip, eval_seed), metrics in sorted(base_cells.items())
        },
        "heldout_clips": heldout_clips,
        "arms": arm_summaries,
        "promoted_arms": sorted(promoted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--arm-rules", required=True, type=pathlib.Path)
    parser.add_argument("--clip-motion-files", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    arm_rules = json.loads(args.arm_rules.read_text())
    clip_motion_files = {
        clip: pathlib.Path(path)
        for clip, path in json.loads(args.clip_motion_files.read_text()).items()
    }
    report = analyze(read_manifest(args.manifest), arm_rules, clip_motion_files)
    for error in report["protocol_errors"]:
        print(f"ERROR: {error}")
    for arm, summary in report["arms"].items():
        for cell in summary["cells"]:
            print(
                f"{arm} {cell['clip']} ({cell['split']}): "
                f"completion={cell['completion']:.1%} "
                f"(delta {cell['completion_delta']:+.1%}), "
                f"rmse_rel={cell['joint_rmse_rel']:+.1%}"
            )
        if "heldout_mean" in summary:
            h = summary["heldout_mean"]
            print(
                f"{arm} HELD-OUT MEAN: completion={h['completion']:.1%} "
                f"(delta {h['completion_delta']:+.1%}), rmse_rel={h['joint_rmse_rel']:+.1%}"
            )
        print(f"{arm} decision: {summary.get('decision')}")
    print(f"verdict: {report['verdict']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
