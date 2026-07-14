#!/usr/bin/env python3
"""Validate and summarize the paired SNMR/GMR WBT pilot."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
from typing import Any

import numpy as np


CLIPS = ("walk1", "dance2", "fight1")
SOURCES = ("gmr", "snmr")
PRIMARY_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Env/average_episode_length",
    "Env/motion/error_ref_pos",
    "Env/motion/error_ref_rot",
    "Env/motion/error_ref_lin_vel",
    "Env/motion/error_ref_ang_vel",
    "Env/motion/error_body_pos",
    "Env/motion/error_body_rot",
    "Env/motion/error_body_lin_vel",
    "Env/motion/error_body_ang_vel",
    "Env/motion/error_joint_pos",
    "Env/motion/error_joint_vel",
)
REQUIRED_TAGS = PRIMARY_TAGS + (
    "Loss/Value",
    "Loss/Surrogate",
    "Train/num_samples",
)
LOWER_IS_BETTER = {
    tag for tag in PRIMARY_TAGS
    if tag.startswith("Env/motion/error_")
}


def _run_name(source: str, clip: str) -> str:
    return f"pilot_{source}_{clip}_seed0"


def read_run_map(path: pathlib.Path) -> dict[str, pathlib.Path]:
    result = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected NAME<TAB>RUN_DIR")
            name, run_dir = fields
            if name in result:
                raise ValueError(f"{path}:{line_number}: duplicate run {name}")
            result[name] = pathlib.Path(run_dir)
    return result


def _load_config(run_dir: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; run this script in the WBT environment") from exc

    path = run_dir / "holosoma_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping")
    return config


def _load_scalars(run_dir: pathlib.Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("tensorboard is required; run this script in the WBT environment") from exc

    event_files = list(run_dir.glob("events.out.tfevents*"))
    if len(event_files) != 1:
        raise ValueError(f"{run_dir}: expected one TensorBoard event file, found {len(event_files)}")
    accumulator = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }


def _get(config: dict[str, Any], *path: str) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized["training"]["name"] = "<RUN_NAME>"
    normalized["command"]["setup_terms"]["motion_command"]["params"]["motion_config"][
        "motion_file"
    ] = "<MOTION_FILE>"
    return normalized


def _summarize_series(
    series: list[tuple[int, float]],
    *,
    expected_events: int,
    final_window: int,
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    steps = [step for step, _ in series]
    values = np.asarray([value for _, value in series], dtype=np.float64)
    if len(series) != expected_events:
        errors.append(f"expected {expected_events} events, found {len(series)}")
    if steps != list(range(expected_events)):
        errors.append(
            f"expected steps 0..{expected_events - 1}, found "
            f"{steps[0] if steps else None}..{steps[-1] if steps else None}"
        )
    if values.size == 0 or not np.isfinite(values).all():
        errors.append("series is empty or nonfinite")
        return {"event_count": len(series)}, errors
    window = min(final_window, len(values))
    curve_mean = float(np.mean(values))
    if len(values) > 1 and steps[-1] != steps[0]:
        step_widths = np.diff(np.asarray(steps, dtype=np.float64))
        area = np.sum((values[:-1] + values[1:]) * 0.5 * step_widths)
        curve_auc = float(area / (steps[-1] - steps[0]))
    else:
        curve_auc = curve_mean
    return {
        "event_count": len(series),
        "first_step": steps[0],
        "last_step": steps[-1],
        "initial_window_mean": float(np.mean(values[:window])),
        "final_window_mean": float(np.mean(values[-window:])),
        "curve_mean": curve_mean,
        "normalized_auc": curve_auc,
        "final_value": float(values[-1]),
    }, errors


def analyze_run(
    name: str,
    run_dir: pathlib.Path,
    *,
    expected_events: int,
    final_window: int,
) -> dict[str, Any]:
    errors = []
    if not run_dir.is_dir():
        return {"name": name, "run_dir": str(run_dir), "passed": False, "errors": ["missing run directory"]}
    config = _load_config(run_dir)
    scalars = _load_scalars(run_dir)
    metric_summaries = {}
    for tag in REQUIRED_TAGS:
        if tag not in scalars:
            errors.append(f"missing scalar tag {tag}")
            continue
        summary, series_errors = _summarize_series(
            scalars[tag],
            expected_events=expected_events,
            final_window=final_window,
        )
        metric_summaries[tag] = summary
        errors.extend(f"{tag}: {message}" for message in series_errors)

    checkpoints = sorted(run_dir.glob("model_*.pt"))
    if not checkpoints:
        errors.append("no policy checkpoint found")
    return {
        "name": name,
        "run_dir": str(run_dir.resolve()),
        "passed": not errors,
        "errors": errors,
        "config": config,
        "normalized_config": _normalized_config(config),
        "metrics": metric_summaries,
        "checkpoint": str(checkpoints[-1].resolve()) if checkpoints else None,
    }


def _validate_protocol(
    runs: dict[str, dict[str, Any]],
    *,
    expected_events: int,
    expected_envs: int,
    expected_seed: int,
) -> list[str]:
    errors = []
    expected_names = {
        _run_name(source, clip)
        for source in SOURCES
        for clip in CLIPS
    }
    if set(runs) != expected_names:
        errors.append(
            f"run set mismatch: missing={sorted(expected_names - set(runs))}, "
            f"unexpected={sorted(set(runs) - expected_names)}"
        )

    canonical = None
    for name in sorted(expected_names & set(runs)):
        run = runs[name]
        config = run["config"]
        source = name.split("_")[1]
        clip = name.split("_")[2]
        checks = {
            "training.name": (_get(config, "training", "name"), name),
            "training.num_envs": (_get(config, "training", "num_envs"), expected_envs),
            "training.seed": (_get(config, "training", "seed"), expected_seed),
            "algo.config.num_learning_iterations": (
                _get(config, "algo", "config", "num_learning_iterations"),
                expected_events,
            ),
        }
        for field, (actual, expected) in checks.items():
            if actual != expected:
                errors.append(f"{name}: {field}={actual!r}, expected {expected!r}")

        motion_file = _get(
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

        normalized = run["normalized_config"]
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        if canonical is None:
            canonical = encoded
        elif encoded != canonical:
            errors.append(f"{name}: resolved config differs beyond run name and motion file")
    return errors


def _paired_effects(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    comparisons = {}
    for clip in CLIPS:
        gmr = runs.get(_run_name("gmr", clip), {})
        snmr = runs.get(_run_name("snmr", clip), {})
        clip_result = {}
        for tag in PRIMARY_TAGS:
            gmr_summary = gmr.get("metrics", {}).get(tag)
            snmr_summary = snmr.get("metrics", {}).get(tag)
            if not gmr_summary or not snmr_summary:
                continue
            metrics = {}
            for statistic in ("final_window_mean", "normalized_auc"):
                gmr_value = gmr_summary[statistic]
                snmr_value = snmr_summary[statistic]
                delta = snmr_value - gmr_value
                denominator = abs(gmr_value)
                metrics[statistic] = {
                    "gmr": gmr_value,
                    "snmr": snmr_value,
                    "snmr_minus_gmr": delta,
                    "relative_delta": delta / denominator if denominator > 0 else None,
                    "favorable_effect": -delta if tag in LOWER_IS_BETTER else delta,
                }
            clip_result[tag] = metrics
        comparisons[clip] = clip_result
    return comparisons


def analyze_pilot(
    run_map: dict[str, pathlib.Path],
    *,
    expected_events: int = 1000,
    expected_envs: int = 1024,
    expected_seed: int = 0,
    final_window: int = 100,
) -> dict[str, Any]:
    runs = {
        name: analyze_run(
            name,
            run_dir,
            expected_events=expected_events,
            final_window=final_window,
        )
        for name, run_dir in sorted(run_map.items())
    }
    protocol_errors = _validate_protocol(
        runs,
        expected_events=expected_events,
        expected_envs=expected_envs,
        expected_seed=expected_seed,
    )
    return {
        "passed": (
            not protocol_errors
            and len(runs) == len(CLIPS) * len(SOURCES)
            and all(run["passed"] for run in runs.values())
        ),
        "interpretation": "single-seed descriptive pilot; no inferential tracking claim",
        "protocol_errors": protocol_errors,
        "runs": {
            name: {
                key: value
                for key, value in run.items()
                if key not in {"config", "normalized_config"}
            }
            for name, run in runs.items()
        },
        "paired_effects": _paired_effects(runs),
    }


def _print_report(report: dict[str, Any]) -> None:
    print("clip\tmetric\tgmr_final100\tsnmr_final100\tfavorable_effect")
    for clip, metrics in report["paired_effects"].items():
        for tag, statistics in metrics.items():
            final = statistics["final_window_mean"]
            print(
                f"{clip}\t{tag}\t{final['gmr']:.8g}\t{final['snmr']:.8g}\t"
                f"{final['favorable_effect']:.8g}"
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
    parser.add_argument("run_map", type=pathlib.Path, help="TSV with run name and Holosoma run directory")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--expected-events", type=int, default=1000)
    parser.add_argument("--expected-envs", type=int, default=1024)
    parser.add_argument("--expected-seed", type=int, default=0)
    parser.add_argument("--final-window", type=int, default=100)
    args = parser.parse_args()

    report = analyze_pilot(
        read_run_map(args.run_map),
        expected_events=args.expected_events,
        expected_envs=args.expected_envs,
        expected_seed=args.expected_seed,
        final_window=args.final_window,
    )
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
