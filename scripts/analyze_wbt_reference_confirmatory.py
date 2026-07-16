#!/usr/bin/env python3
"""Validate and analyze the confirmatory GMR-vs-SNMR WBT source matrix."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from typing import Any, NamedTuple

import numpy as np

SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_wbt_horizon import (  # noqa: E402
    REQUIRED_TAGS,
    _checkpoint_config_matches,
    _get,
    _load_checkpoint,
    _load_config,
    _load_scalars,
    _sha256,
)
from analyze_wbt_latent_pilot import (  # noqa: E402
    BASE_COMMAND_FUNC,
    METRICS,
    _report_arrays,
    _validate_rollout_report,
)


SOURCES = ("gmr", "snmr")
TRAINING_SEEDS = (0, 1, 2)
EVALUATION_SEEDS = (404, 405)
TRAINING_ITERATIONS = 8000
SAVE_INTERVAL = 2000
NUM_ENVS = 1024
ROLLOUTS = 100
COMPLETION_MARGIN = -0.05
JOINT_RMSE_MARGIN = 0.10
GMR_COMPLETION_FLOOR = 0.50
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260716
PRIMARY_JOINT_METRIC = "joint_position_rmse_rad"


class EvaluationRow(NamedTuple):
    source: str
    training_seed: int
    evaluation_seed: int
    training_name: str
    run_dir: pathlib.Path
    checkpoint_path: pathlib.Path
    checkpoint_sha256: str
    report_path: pathlib.Path


def read_manifest(path: pathlib.Path) -> list[EvaluationRow]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number}: expected 8 fields, found {len(fields)}"
                )
            rows.append(
                EvaluationRow(
                    source=fields[0],
                    training_seed=int(fields[1]),
                    evaluation_seed=int(fields[2]),
                    training_name=fields[3],
                    run_dir=pathlib.Path(fields[4]),
                    checkpoint_path=pathlib.Path(fields[5]),
                    checkpoint_sha256=fields[6],
                    report_path=pathlib.Path(fields[7]),
                )
            )
    return rows


def _training_name(source: str, seed: int) -> str:
    return f"reference_confirm_{source}_walk1_seed{seed}_to8000"


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    training = normalized.get("training", {})
    training["name"] = "<TRAINING_NAME>"
    training["seed"] = "<TRAINING_SEED>"
    motion = _get(
        normalized,
        "command",
        "setup_terms",
        "motion_command",
        "params",
        "motion_config",
    )
    if isinstance(motion, dict):
        motion["motion_file"] = "<MOTION_FILE>"
    return normalized


def _validate_training(
    row: EvaluationRow,
    reference_path: pathlib.Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = []
    try:
        config = _load_config(row.run_dir / "holosoma_config.yaml")
        scalars = _load_scalars(row.run_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, [str(exc)]
    checks = {
        "training.name": (
            _get(config, "training", "name"),
            _training_name(row.source, row.training_seed),
        ),
        "training.num_envs": (_get(config, "training", "num_envs"), NUM_ENVS),
        "training.seed": (_get(config, "training", "seed"), row.training_seed),
        "training.checkpoint": (_get(config, "training", "checkpoint"), None),
        "iterations": (
            _get(config, "algo", "config", "num_learning_iterations"),
            TRAINING_ITERATIONS,
        ),
        "save_interval": (
            _get(config, "algo", "config", "save_interval"),
            SAVE_INTERVAL,
        ),
        "actor command": (
            _get(
                config,
                "observation",
                "groups",
                "actor_obs",
                "terms",
                "motion_command",
                "func",
            ),
            BASE_COMMAND_FUNC,
        ),
        "critic command": (
            _get(
                config,
                "observation",
                "groups",
                "critic_obs",
                "terms",
                "motion_command",
                "func",
            ),
            BASE_COMMAND_FUNC,
        ),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{field}={actual!r}, expected {expected!r}")
    motion_file = _get(
        config,
        "command",
        "setup_terms",
        "motion_command",
        "params",
        "motion_config",
        "motion_file",
    )
    if (
        not isinstance(motion_file, str)
        or pathlib.Path(motion_file).resolve() != reference_path.resolve()
    ):
        errors.append(f"unexpected motion_file={motion_file!r}")

    expected_steps = list(range(TRAINING_ITERATIONS))
    for tag in REQUIRED_TAGS:
        series = scalars.get(tag)
        if series is None:
            errors.append(f"missing scalar tag {tag}")
            continue
        steps = [step for step, _ in series]
        values = np.asarray([value for _, value in series], dtype=np.float64)
        if steps != expected_steps:
            errors.append(f"{tag}: training steps are not exactly 0..7999")
        if values.size != TRAINING_ITERATIONS or not np.isfinite(values).all():
            errors.append(f"{tag}: values are incomplete or nonfinite")

    if (
        row.checkpoint_path.parent.resolve() != row.run_dir.resolve()
        or row.checkpoint_path.name != "model_07999.pt"
        or not row.checkpoint_path.is_file()
    ):
        return config, [*errors, f"invalid checkpoint {row.checkpoint_path}"]
    if _sha256(row.checkpoint_path) != row.checkpoint_sha256:
        errors.append("checkpoint SHA-256 mismatch")
    try:
        checkpoint = _load_checkpoint(row.checkpoint_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return config, [*errors, str(exc)]
    if checkpoint.get("iter") != 7999 or checkpoint.get("iteration") != 7999:
        errors.append("checkpoint is not iteration 7999")
    if not _checkpoint_config_matches(checkpoint.get("experiment_config"), config):
        errors.append("checkpoint experiment_config differs from run config")
    for key in (
        "actor_model_state_dict",
        "critic_model_state_dict",
        "actor_optimizer_state_dict",
        "critic_optimizer_state_dict",
        "actor_obs_normalizer_state_dict",
        "critic_obs_normalizer_state_dict",
    ):
        value = checkpoint.get(key)
        if not isinstance(value, dict) or not value:
            errors.append(f"checkpoint is missing {key}")
    return config, errors


def hierarchical_effects(
    completion: np.ndarray,
    joint_rmse: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap paired train seeds, evaluation seeds, and phase windows."""
    expected_shape = (
        len(SOURCES),
        len(TRAINING_SEEDS),
        len(EVALUATION_SEEDS),
        ROLLOUTS,
    )
    if completion.shape != expected_shape or joint_rmse.shape != expected_shape:
        raise ValueError(
            f"expected source/train/eval/window shape {expected_shape}, "
            f"got {completion.shape} and {joint_rmse.shape}"
        )
    gmr_completion = float(completion[0].mean())
    snmr_completion = float(completion[1].mean())
    gmr_joint = float(joint_rmse[0].mean())
    snmr_joint = float(joint_rmse[1].mean())
    rng = np.random.default_rng(seed)
    completion_samples = np.empty(replicates, dtype=np.float64)
    joint_samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        train_indices = rng.integers(0, len(TRAINING_SEEDS), len(TRAINING_SEEDS))
        sampled_completion = [[], []]
        sampled_joint = [[], []]
        for train_index in train_indices:
            eval_indices = rng.integers(
                0, len(EVALUATION_SEEDS), len(EVALUATION_SEEDS)
            )
            for eval_index in eval_indices:
                rollout_indices = rng.integers(0, ROLLOUTS, ROLLOUTS)
                for source_index in range(len(SOURCES)):
                    sampled_completion[source_index].append(
                        completion[
                            source_index,
                            train_index,
                            eval_index,
                            rollout_indices,
                        ]
                    )
                    sampled_joint[source_index].append(
                        joint_rmse[
                            source_index,
                            train_index,
                            eval_index,
                            rollout_indices,
                        ]
                    )
        completion_means = [
            float(np.concatenate(values).mean()) for values in sampled_completion
        ]
        joint_means = [
            float(np.concatenate(values).mean()) for values in sampled_joint
        ]
        completion_samples[replicate] = completion_means[1] - completion_means[0]
        joint_samples[replicate] = joint_means[1] / joint_means[0] - 1.0

    completion_ci = [
        float(value) for value in np.quantile(completion_samples, [0.025, 0.975])
    ]
    joint_ci = [
        float(value) for value in np.quantile(joint_samples, [0.025, 0.975])
    ]
    checks = {
        "gmr_completion_floor": gmr_completion >= GMR_COMPLETION_FLOOR,
        "completion_noninferiority": completion_ci[0] >= COMPLETION_MARGIN,
        "joint_rmse_noninferiority": joint_ci[1] <= JOINT_RMSE_MARGIN,
    }
    return {
        "gmr_completion": gmr_completion,
        "snmr_completion": snmr_completion,
        "completion_difference": snmr_completion - gmr_completion,
        "completion_difference_ci95": completion_ci,
        "gmr_joint_rmse_rad": gmr_joint,
        "snmr_joint_rmse_rad": snmr_joint,
        "joint_rmse_relative_effect": snmr_joint / gmr_joint - 1.0,
        "joint_rmse_relative_effect_ci95": joint_ci,
        "checks": checks,
        "noninferior": all(checks.values()),
    }


def analyze(
    rows: list[EvaluationRow],
    references: dict[str, pathlib.Path],
) -> dict[str, Any]:
    errors = []
    expected_cells = {
        (source, train_seed, eval_seed)
        for source in SOURCES
        for train_seed in TRAINING_SEEDS
        for eval_seed in EVALUATION_SEEDS
    }
    actual_cells = [
        (row.source, row.training_seed, row.evaluation_seed) for row in rows
    ]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        errors.append(
            "manifest must contain exactly the 12 source/train/evaluation cells"
        )

    configs = {}
    reports = {}
    checkpoint_records = {}
    for row in rows:
        cell = (row.source, row.training_seed, row.evaluation_seed)
        if row.source not in references:
            errors.append(f"{cell}: unexpected source")
            continue
        expected_name = _training_name(row.source, row.training_seed)
        if row.training_name != expected_name:
            errors.append(
                f"{cell}: training_name={row.training_name!r}, expected {expected_name!r}"
            )
        policy_key = (row.source, row.training_seed)
        if policy_key not in configs:
            config, training_errors = _validate_training(
                row, references[row.source]
            )
            configs[policy_key] = config
            errors.extend(
                f"{policy_key}: {error}" for error in training_errors
            )
            checkpoint_records[policy_key] = {
                "path": str(row.checkpoint_path.resolve()),
                "sha256": row.checkpoint_sha256,
            }
        else:
            record = checkpoint_records[policy_key]
            if (
                str(row.checkpoint_path.resolve()) != record["path"]
                or row.checkpoint_sha256 != record["sha256"]
            ):
                errors.append(f"{policy_key}: evaluation rows use different checkpoints")
        try:
            report = json.loads(row.report_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{cell}: cannot load report: {exc}")
            continue
        errors.extend(
            f"{cell}: {error}"
            for error in _validate_rollout_report(
                report,
                expected_training_name=expected_name,
                expected_motion_file=references[row.source],
                expected_seed=row.evaluation_seed,
            )
        )
        reports[cell] = report

    normalized_configs = {
        policy: _normalized_config(config)
        for policy, config in configs.items()
        if config is not None
    }
    if normalized_configs:
        reference_policy = sorted(normalized_configs)[0]
        reference_config = normalized_configs[reference_policy]
        for policy, config in sorted(normalized_configs.items()):
            if config != reference_config:
                errors.append(
                    f"{policy}: config differs from {reference_policy} beyond "
                    "source, name, and training seed"
                )

    for eval_seed in EVALUATION_SEEDS:
        report_cells = {
            (source, seed): reports.get((source, seed, eval_seed))
            for source in SOURCES
            for seed in TRAINING_SEEDS
        }
        available = {
            policy: report
            for policy, report in report_cells.items()
            if report is not None
        }
        if not available:
            continue
        reference_policy = sorted(available)[0]
        reference_report = available[reference_policy]
        reference_starts = [
            row["start_step"] for row in reference_report["rollouts"]
        ]
        reference_motion_steps = reference_report.get("motion_steps")
        for policy, report in sorted(available.items()):
            starts = [row["start_step"] for row in report["rollouts"]]
            if starts != reference_starts:
                errors.append(
                    f"{policy} eval {eval_seed}: phase starts differ from "
                    f"{reference_policy}"
                )
            if report.get("motion_steps") != reference_motion_steps:
                errors.append(
                    f"{policy} eval {eval_seed}: motion length differs from "
                    f"{reference_policy}"
                )

    if errors:
        return {
            "passed": False,
            "verdict": "invalid_confirmatory_artifacts",
            "protocol_errors": errors,
        }

    shape = (
        len(SOURCES),
        len(TRAINING_SEEDS),
        len(EVALUATION_SEEDS),
        ROLLOUTS,
    )
    arrays = {
        metric: np.empty(shape, dtype=np.float64)
        for metric in ("completion", "survival_s", *METRICS)
    }
    for source_index, source in enumerate(SOURCES):
        for train_index, train_seed in enumerate(TRAINING_SEEDS):
            for eval_index, eval_seed in enumerate(EVALUATION_SEEDS):
                report_arrays = _report_arrays(
                    reports[(source, train_seed, eval_seed)]
                )
                for metric in arrays:
                    arrays[metric][source_index, train_index, eval_index] = (
                        report_arrays[metric]
                    )

    primary = hierarchical_effects(
        arrays["completion"], arrays[PRIMARY_JOINT_METRIC]
    )
    source_means = {
        source: {
            metric: float(arrays[metric][source_index].mean())
            for metric in arrays
        }
        for source_index, source in enumerate(SOURCES)
    }
    per_training_seed = {
        str(seed): {
            source: {
                metric: float(
                    arrays[metric][source_index, train_index].mean()
                )
                for metric in ("completion", "survival_s", PRIMARY_JOINT_METRIC)
            }
            for source_index, source in enumerate(SOURCES)
        }
        for train_index, seed in enumerate(TRAINING_SEEDS)
    }
    return {
        "passed": True,
        "verdict": (
            "noninferior"
            if primary["noninferior"]
            else "noninferiority_not_established"
        ),
        "interpretation": (
            "single-clip confirmatory source comparison across three independently "
            "trained policies per source; evaluation seeds are rollout variability, "
            "not additional trained policies"
        ),
        "protocol_errors": [],
        "protocol": {
            "sources": list(SOURCES),
            "training_seeds": list(TRAINING_SEEDS),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "training_iterations": TRAINING_ITERATIONS,
            "num_envs": NUM_ENVS,
            "rollouts_per_cell": ROLLOUTS,
            "completion_margin": COMPLETION_MARGIN,
            "joint_rmse_relative_margin": JOINT_RMSE_MARGIN,
            "gmr_completion_floor": GMR_COMPLETION_FLOOR,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "references": {
            source: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for source, path in references.items()
        },
        "checkpoints": {
            f"{source}_seed{seed}": checkpoint_records[(source, seed)]
            for source in SOURCES
            for seed in TRAINING_SEEDS
        },
        "source_means": source_means,
        "per_training_seed": per_training_seed,
        "primary": primary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--gmr-reference", type=pathlib.Path, required=True)
    parser.add_argument("--snmr-reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    result = analyze(
        read_manifest(args.manifest),
        {"gmr": args.gmr_reference, "snmr": args.snmr_reference},
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for error in result.get("protocol_errors", []):
        print(f"ERROR: {error}")
    if result["passed"]:
        primary = result["primary"]
        print(
            f"completion: {primary['gmr_completion']:.1%} -> "
            f"{primary['snmr_completion']:.1%}; "
            f"delta={primary['completion_difference']:+.1%}, "
            f"CI={primary['completion_difference_ci95']}"
        )
        print(
            f"joint RMSE effect={primary['joint_rmse_relative_effect']:+.2%}, "
            f"CI={primary['joint_rmse_relative_effect_ci95']}"
        )
    print(f"verdict: {result['verdict']}")
    print(result.get("interpretation", "invalid artifacts"))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
