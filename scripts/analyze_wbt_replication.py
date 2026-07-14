#!/usr/bin/env python3
"""Validate and summarize the three-seed paired WBT training pilot."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import pathlib
import re
from typing import Any

import numpy as np


_PILOT_PATH = pathlib.Path(__file__).with_name("analyze_wbt_pilot.py")
_PILOT_SPEC = importlib.util.spec_from_file_location("analyze_wbt_pilot", _PILOT_PATH)
assert _PILOT_SPEC is not None and _PILOT_SPEC.loader is not None
pilot = importlib.util.module_from_spec(_PILOT_SPEC)
_PILOT_SPEC.loader.exec_module(pilot)

CLIPS = pilot.CLIPS
SOURCES = pilot.SOURCES
PRIMARY_TAGS = pilot.PRIMARY_TAGS
REQUIRED_TAGS = pilot.REQUIRED_TAGS
LOWER_IS_BETTER = pilot.LOWER_IS_BETTER
SEEDS = (0, 1, 2)
RUN_PATTERN = re.compile(
    r"^pilot_(gmr|snmr)_(walk1|dance2|fight1)_seed([0-2])$"
)


def _run_name(source: str, clip: str, seed: int) -> str:
    return f"pilot_{source}_{clip}_seed{seed}"


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(pilot._normalized_config(config))
    normalized["training"]["seed"] = "<SEED>"
    return normalized


def _validate_protocol(
    runs: dict[str, dict[str, Any]],
    *,
    expected_events: int,
    expected_envs: int,
) -> list[str]:
    errors = []
    expected_names = {
        _run_name(source, clip, seed)
        for source in SOURCES
        for clip in CLIPS
        for seed in SEEDS
    }
    if set(runs) != expected_names:
        errors.append(
            f"run set mismatch: missing={sorted(expected_names - set(runs))}, "
            f"unexpected={sorted(set(runs) - expected_names)}"
        )

    canonical = None
    for name in sorted(expected_names & set(runs)):
        run = runs[name]
        match = RUN_PATTERN.fullmatch(name)
        if match is None:
            errors.append(f"{name}: invalid run name")
            continue
        source, clip, seed_text = match.groups()
        seed = int(seed_text)
        config = run["config"]
        checks = {
            "training.name": (pilot._get(config, "training", "name"), name),
            "training.num_envs": (
                pilot._get(config, "training", "num_envs"),
                expected_envs,
            ),
            "training.seed": (pilot._get(config, "training", "seed"), seed),
            "algo.config.num_learning_iterations": (
                pilot._get(config, "algo", "config", "num_learning_iterations"),
                expected_events,
            ),
        }
        for field, (actual, expected) in checks.items():
            if actual != expected:
                errors.append(f"{name}: {field}={actual!r}, expected {expected!r}")

        motion_file = pilot._get(
            config,
            "command",
            "setup_terms",
            "motion_command",
            "params",
            "motion_config",
            "motion_file",
        )
        motion_path = pathlib.Path(motion_file) if isinstance(motion_file, str) else None
        if (
            motion_path is None
            or motion_path.parent.name != source
            or not motion_path.name.startswith(clip + "_")
        ):
            errors.append(f"{name}: unexpected motion file {motion_file!r}")

        checkpoint = run.get("checkpoint")
        if not isinstance(checkpoint, str) or pathlib.Path(checkpoint).name != "model_00999.pt":
            errors.append(f"{name}: final checkpoint is {checkpoint!r}")

        encoded = json.dumps(
            _normalized_config(config),
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical is None:
            canonical = encoded
        elif encoded != canonical:
            errors.append(
                f"{name}: resolved config differs beyond run name, motion file, and seed"
            )
    return errors


def _paired_effects(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons = {}
    for clip in CLIPS:
        clip_result = {}
        for seed in SEEDS:
            gmr = runs.get(_run_name("gmr", clip, seed), {})
            snmr = runs.get(_run_name("snmr", clip, seed), {})
            seed_result = {}
            for tag in PRIMARY_TAGS:
                gmr_summary = gmr.get("metrics", {}).get(tag)
                snmr_summary = snmr.get("metrics", {}).get(tag)
                if not gmr_summary or not snmr_summary:
                    continue
                statistics = {}
                for statistic in ("final_window_mean", "normalized_auc"):
                    gmr_value = gmr_summary[statistic]
                    snmr_value = snmr_summary[statistic]
                    delta = snmr_value - gmr_value
                    denominator = abs(gmr_value)
                    statistics[statistic] = {
                        "gmr": gmr_value,
                        "snmr": snmr_value,
                        "snmr_minus_gmr": delta,
                        "relative_delta": (
                            delta / denominator if denominator > 0 else None
                        ),
                        "favorable_effect": (
                            -delta if tag in LOWER_IS_BETTER else delta
                        ),
                    }
                seed_result[tag] = statistics
            clip_result[str(seed)] = seed_result
        comparisons[clip] = clip_result
    return comparisons


def _aggregate_effects(paired_effects: dict[str, Any]) -> dict[str, Any]:
    aggregates = {}
    for tag in PRIMARY_TAGS:
        tag_result = {}
        for statistic in ("final_window_mean", "normalized_auc"):
            rows = []
            for clip in CLIPS:
                for seed in SEEDS:
                    effect = (
                        paired_effects.get(clip, {})
                        .get(str(seed), {})
                        .get(tag, {})
                        .get(statistic)
                    )
                    if effect is not None:
                        rows.append(
                            {
                                "clip": clip,
                                "seed": seed,
                                **effect,
                            }
                        )
            favorable = np.asarray(
                [row["favorable_effect"] for row in rows],
                dtype=np.float64,
            )
            relative = np.asarray(
                [
                    row["relative_delta"]
                    for row in rows
                    if row["relative_delta"] is not None
                ],
                dtype=np.float64,
            )
            tag_result[statistic] = {
                "pair_count": len(rows),
                "mean_favorable_effect": (
                    float(np.mean(favorable)) if favorable.size else None
                ),
                "median_favorable_effect": (
                    float(np.median(favorable)) if favorable.size else None
                ),
                "favorable_pair_count": (
                    int(np.sum(favorable > 0.0)) if favorable.size else 0
                ),
                "mean_relative_delta": (
                    float(np.mean(relative)) if relative.size else None
                ),
                "pairs": rows,
            }
        aggregates[tag] = tag_result
    return aggregates


def analyze_replication(
    run_map: dict[str, pathlib.Path],
    *,
    expected_events: int = 1000,
    expected_envs: int = 1024,
    final_window: int = 100,
) -> dict[str, Any]:
    runs = {}
    for name, run_dir in sorted(run_map.items()):
        run = pilot.analyze_run(
            name,
            run_dir,
            expected_events=expected_events,
            final_window=final_window,
        )
        if "config" in run:
            run["normalized_config"] = _normalized_config(run["config"])
        runs[name] = run
    protocol_errors = _validate_protocol(
        runs,
        expected_events=expected_events,
        expected_envs=expected_envs,
    )
    paired_effects = _paired_effects(runs)
    return {
        "passed": (
            not protocol_errors
            and len(runs) == len(CLIPS) * len(SOURCES) * len(SEEDS)
            and all(run["passed"] for run in runs.values())
        ),
        "interpretation": (
            "three-seed paired training-curve pilot; independent policy rollouts "
            "are required before any non-inferiority or benefit claim"
        ),
        "protocol_errors": protocol_errors,
        "runs": {
            name: {
                key: value
                for key, value in run.items()
                if key not in {"config", "normalized_config"}
            }
            for name, run in runs.items()
        },
        "paired_effects": paired_effects,
        "aggregate_effects": _aggregate_effects(paired_effects),
    }


def _print_report(report: dict[str, Any]) -> None:
    print("clip\tseed\tmetric\tgmr_final100\tsnmr_final100\tfavorable_effect")
    for clip, seeds in report["paired_effects"].items():
        for seed, metrics in seeds.items():
            for tag, statistics in metrics.items():
                final = statistics["final_window_mean"]
                print(
                    f"{clip}\t{seed}\t{tag}\t{final['gmr']:.8g}\t"
                    f"{final['snmr']:.8g}\t{final['favorable_effect']:.8g}"
                )
    for error in report["protocol_errors"]:
        print(f"ERROR: {error}")
    for name, run in report["runs"].items():
        for error in run["errors"]:
            print(f"ERROR: {name}: {error}")
    print(f"overall: {'PASS' if report['passed'] else 'FAIL'}")
    print(report["interpretation"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_map",
        type=pathlib.Path,
        help="TSV with run name and Holosoma run directory",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--expected-events", type=int, default=1000)
    parser.add_argument("--expected-envs", type=int, default=1024)
    parser.add_argument("--final-window", type=int, default=100)
    args = parser.parse_args()

    report = analyze_replication(
        pilot.read_run_map(args.run_map),
        expected_events=args.expected_events,
        expected_envs=args.expected_envs,
        final_window=args.final_window,
    )
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
