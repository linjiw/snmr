#!/usr/bin/env python3
"""Validate and analyze the GMR-only WBT training-horizon calibration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
from typing import Any, NamedTuple

import numpy as np


CLIPS = ("walk1", "dance2", "fight1")
HORIZONS = (2000, 4000, 8000)
TRAINING_SEED = 0
EVALUATION_SEED = 404
SOURCE_ITERATIONS = 1000
ADDITIONAL_ITERATIONS = 7000
SAVE_INTERVAL = 2000
NUM_ENVS = 1024
ROLLOUTS_PER_EVALUATION = 100
HORIZON_STEPS = 500
HORIZON_S = 10.0
POOLED_COMPLETION_FLOOR = 0.50
PER_CLIP_COMPLETION_FLOOR = 0.25
METRICS = (
    "root_position_error_m",
    "root_orientation_error_rad",
    "root_linear_velocity_error_mps",
    "root_angular_velocity_error_radps",
    "body_position_error_m",
    "body_orientation_error_rad",
    "body_linear_velocity_error_mps",
    "body_angular_velocity_error_radps",
    "joint_position_error_l2_rad",
    "joint_velocity_error_l2_radps",
    "joint_position_rmse_rad",
    "torque_rms_nm",
    "mechanical_power_abs_w",
    "joint_limit_violation_rad",
    "undesired_contact_count",
)
REQUIRED_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Env/motion/error_ref_pos",
    "Env/motion/error_joint_pos",
    "Loss/Value",
    "Loss/Surrogate",
    "Train/num_samples",
)


class TrainingRow(NamedTuple):
    train_name: str
    clip: str
    run_dir: pathlib.Path
    source_checkpoint: pathlib.Path
    source_sha256: str


class EvaluationRow(NamedTuple):
    name: str
    train_name: str
    clip: str
    total_iterations: int
    report_path: pathlib.Path
    checkpoint_path: pathlib.Path
    checkpoint_sha256: str


def _training_name(clip: str) -> str:
    return f"horizon_gmr_{clip}_seed0_to8000"


def _evaluation_name(clip: str, total_iterations: int) -> str:
    return f"{_training_name(clip)}_iter{total_iterations}_eval404"


def _read_tsv(path: pathlib.Path, field_count: int) -> list[list[str]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != field_count:
                raise ValueError(
                    f"{path}:{line_number}: expected {field_count} fields, "
                    f"found {len(fields)}"
                )
            rows.append(fields)
    return rows


def read_training_map(path: pathlib.Path) -> list[TrainingRow]:
    return [
        TrainingRow(
            train_name=name,
            clip=clip,
            run_dir=pathlib.Path(run_dir),
            source_checkpoint=pathlib.Path(source_checkpoint),
            source_sha256=source_sha256,
        )
        for name, clip, run_dir, source_checkpoint, source_sha256 in _read_tsv(
            path, 5
        )
    ]


def read_evaluation_map(path: pathlib.Path) -> list[EvaluationRow]:
    return [
        EvaluationRow(
            name=name,
            train_name=train_name,
            clip=clip,
            total_iterations=int(total_iterations),
            report_path=pathlib.Path(report_path),
            checkpoint_path=pathlib.Path(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
        )
        for (
            name,
            train_name,
            clip,
            total_iterations,
            report_path,
            checkpoint_path,
            checkpoint_sha256,
        ) in _read_tsv(path, 7)
    ]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _load_config(run_dir: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required; run this script in the WBT environment"
        ) from exc

    config_path = run_dir / "holosoma_config.yaml"
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    return config


def _load_scalars(run_dir: pathlib.Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise RuntimeError(
            "tensorboard is required; run this script in the WBT environment"
        ) from exc

    event_files = list(run_dir.glob("events.out.tfevents*"))
    if len(event_files) != 1:
        raise ValueError(
            f"{run_dir}: expected one TensorBoard event file, "
            f"found {len(event_files)}"
        )
    accumulator = EventAccumulator(
        str(event_files[0]), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    return {
        tag: [
            (int(event.step), float(event.value))
            for event in accumulator.Scalars(tag)
        ]
        for tag in accumulator.Tags().get("scalars", [])
    }


def _load_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required; run this script in the WBT environment"
        ) from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} must contain a checkpoint mapping")
    return checkpoint


def _checkpoint_config_matches(
    checkpoint_config: Any,
    run_config: dict[str, Any],
    *,
    robot_action_dim: int = 29,
) -> bool:
    """Compare config serializations while resolving Holosoma's action-dim token."""
    if not isinstance(checkpoint_config, dict):
        return False
    checkpoint_normalized = copy.deepcopy(checkpoint_config)
    run_normalized = copy.deepcopy(run_config)
    for config in (checkpoint_normalized, run_normalized):
        actor_output_dim = _get(
            config,
            "algo",
            "config",
            "module_dict",
            "actor",
            "output_dim",
        )
        if actor_output_dim in (["robot_action_dim"], [robot_action_dim]):
            module_dict = _get(config, "algo", "config", "module_dict")
            module_dict["actor"]["output_dim"] = ["<ROBOT_ACTION_DIM>"]
    return checkpoint_normalized == run_normalized


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    training = normalized.get("training", {})
    training["name"] = "<TRAINING_NAME>"
    training["checkpoint"] = "<CHECKPOINT>"
    algo_config = _get(normalized, "algo", "config")
    if isinstance(algo_config, dict):
        algo_config["num_learning_iterations"] = "<ITERATIONS>"
        algo_config["save_interval"] = "<SAVE_INTERVAL>"
    motion_config = _get(
        normalized,
        "command",
        "setup_terms",
        "motion_command",
        "params",
        "motion_config",
    )
    if isinstance(motion_config, dict):
        motion_config["motion_file"] = "<MOTION_FILE>"
    return normalized


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _validate_training_row(
    row: TrainingRow,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = []
    expected_name = _training_name(row.clip)
    if row.clip not in CLIPS:
        errors.append(f"unexpected clip={row.clip!r}")
    if row.train_name != expected_name:
        errors.append(
            f"train_name={row.train_name!r}, expected {expected_name!r}"
        )
    if not row.run_dir.is_dir():
        return None, [*errors, f"missing run directory {row.run_dir}"]
    if (
        row.source_checkpoint.name != "model_00999.pt"
        or not row.source_checkpoint.is_file()
    ):
        errors.append(
            f"invalid source checkpoint {row.source_checkpoint}"
        )
    elif _sha256(row.source_checkpoint) != row.source_sha256:
        errors.append("source checkpoint SHA-256 mismatch")

    try:
        config = _load_config(row.run_dir)
        scalars = _load_scalars(row.run_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, [*errors, str(exc)]

    checks = {
        "training.name": (
            _get(config, "training", "name"),
            row.train_name,
        ),
        "training.num_envs": (
            _get(config, "training", "num_envs"),
            NUM_ENVS,
        ),
        "training.seed": (
            _get(config, "training", "seed"),
            TRAINING_SEED,
        ),
        "training.checkpoint": (
            _get(config, "training", "checkpoint"),
            str(row.source_checkpoint),
        ),
        "algo.config.num_learning_iterations": (
            _get(config, "algo", "config", "num_learning_iterations"),
            ADDITIONAL_ITERATIONS,
        ),
        "algo.config.save_interval": (
            _get(config, "algo", "config", "save_interval"),
            SAVE_INTERVAL,
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
    motion_path = (
        pathlib.Path(motion_file) if isinstance(motion_file, str) else None
    )
    if (
        motion_path is None
        or motion_path.parent.name != "gmr"
        or not motion_path.name.startswith(row.clip + "_")
    ):
        errors.append(f"unexpected motion_file={motion_file!r}")

    expected_steps = list(
        range(
            SOURCE_ITERATIONS,
            SOURCE_ITERATIONS + ADDITIONAL_ITERATIONS,
        )
    )
    for tag in REQUIRED_TAGS:
        series = scalars.get(tag)
        if series is None:
            errors.append(f"missing scalar tag {tag}")
            continue
        steps = [step for step, _ in series]
        values = np.asarray([value for _, value in series])
        if steps != expected_steps:
            errors.append(
                f"{tag}: expected steps 1000..7999, found "
                f"{steps[0] if steps else None}.."
                f"{steps[-1] if steps else None} ({len(steps)} events)"
            )
        if values.size != ADDITIONAL_ITERATIONS or not np.isfinite(values).all():
            errors.append(f"{tag}: values are incomplete or nonfinite")

    try:
        source_checkpoint = _load_checkpoint(row.source_checkpoint)
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(str(exc))
    else:
        if source_checkpoint.get("iter") != SOURCE_ITERATIONS - 1:
            errors.append("source checkpoint iter is not 999")
        source_config = source_checkpoint.get("experiment_config")
        if not isinstance(source_config, dict):
            errors.append("source checkpoint is missing experiment_config")
        elif _normalized_config(source_config) != _normalized_config(config):
            errors.append(
                "continued config differs from source beyond allowed fields"
            )

    return config, errors


def _validate_checkpoint(
    row: EvaluationRow,
    training: TrainingRow,
    training_config: dict[str, Any],
) -> list[str]:
    errors = []
    expected_name = f"model_{row.total_iterations - 1:05d}.pt"
    if (
        row.checkpoint_path.parent.resolve() != training.run_dir.resolve()
        or row.checkpoint_path.name != expected_name
        or not row.checkpoint_path.is_file()
    ):
        return [f"invalid checkpoint {row.checkpoint_path}"]
    if _sha256(row.checkpoint_path) != row.checkpoint_sha256:
        errors.append("checkpoint SHA-256 mismatch")
    try:
        checkpoint = _load_checkpoint(row.checkpoint_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return [*errors, str(exc)]

    expected_iteration = row.total_iterations - 1
    if checkpoint.get("iter") != expected_iteration:
        errors.append(
            f"checkpoint iter={checkpoint.get('iter')!r}, "
            f"expected {expected_iteration}"
        )
    if checkpoint.get("iteration") != expected_iteration:
        errors.append(
            f"checkpoint iteration={checkpoint.get('iteration')!r}, "
            f"expected {expected_iteration}"
        )
    checkpoint_config = checkpoint.get("experiment_config")
    if not _checkpoint_config_matches(checkpoint_config, training_config):
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
    return errors


def _validate_report(
    row: EvaluationRow,
    report: dict[str, Any],
) -> list[str]:
    errors = []
    checks = {
        "schema_version": (report.get("schema_version"), 1),
        "passed": (report.get("passed"), True),
        "seed": (report.get("seed"), EVALUATION_SEED),
        "training_name": (report.get("training_name"), row.train_name),
        "num_rollouts": (
            report.get("num_rollouts"),
            ROLLOUTS_PER_EVALUATION,
        ),
        "horizon_steps": (report.get("horizon_steps"), HORIZON_STEPS),
        "horizon_s": (report.get("horizon_s"), HORIZON_S),
        "policy_dt": (
            report.get("policy_dt"),
            HORIZON_S / HORIZON_STEPS,
        ),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{field}={actual!r}, expected {expected!r}")

    motion_file = report.get("motion_file")
    motion_path = (
        pathlib.Path(motion_file) if isinstance(motion_file, str) else None
    )
    if (
        motion_path is None
        or motion_path.parent.name != "gmr"
        or not motion_path.name.startswith(row.clip + "_")
    ):
        errors.append(f"unexpected motion_file={motion_file!r}")

    rollouts = report.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != ROLLOUTS_PER_EVALUATION:
        errors.append("rollouts must contain exactly 100 rows")
        return errors

    env_ids = []
    start_steps = []
    completions = []
    survival_values = []
    for index, rollout in enumerate(rollouts):
        if not isinstance(rollout, dict):
            errors.append(f"rollout {index} is not an object")
            continue
        env_id = rollout.get("env_id")
        start_step = rollout.get("start_step")
        completed = rollout.get("completed")
        failed = rollout.get("failed")
        survival_steps = rollout.get("survival_steps")
        survival_s = rollout.get("survival_s")
        env_ids.append(env_id)
        start_steps.append(start_step)
        if (
            not isinstance(completed, bool)
            or not isinstance(failed, bool)
            or completed == failed
        ):
            errors.append(
                f"rollout {index}: completed/failed are inconsistent"
            )
        if (
            not isinstance(survival_steps, int)
            or not 1 <= survival_steps <= HORIZON_STEPS
        ):
            errors.append(
                f"rollout {index}: invalid survival_steps={survival_steps!r}"
            )
        elif completed and survival_steps != HORIZON_STEPS:
            errors.append(
                f"rollout {index}: completion before the horizon"
            )
        elif not _finite_number(survival_s) or not np.isclose(
            survival_s,
            survival_steps * HORIZON_S / HORIZON_STEPS,
            atol=1e-12,
        ):
            errors.append(
                f"rollout {index}: survival_s does not match steps"
            )
        metrics = rollout.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
            errors.append(f"rollout {index}: metric set mismatch")
        elif not all(
            _finite_number(metrics[name]) and metrics[name] >= 0.0
            for name in METRICS
        ):
            errors.append(
                f"rollout {index}: metrics must be finite and nonnegative"
            )
        if isinstance(completed, bool):
            completions.append(float(completed))
        if isinstance(survival_s, (int, float)):
            survival_values.append(float(survival_s))

    if env_ids != list(range(ROLLOUTS_PER_EVALUATION)):
        errors.append("env_ids must be exactly 0..99 in order")
    motion_steps = report.get("motion_steps")
    if isinstance(motion_steps, int) and motion_steps > HORIZON_STEPS:
        expected_starts = (
            np.linspace(
                0,
                motion_steps - HORIZON_STEPS - 1,
                ROLLOUTS_PER_EVALUATION,
            )
            .round()
            .astype(int)
            .tolist()
        )
        if start_steps != expected_starts:
            errors.append(
                "start_steps do not match the phase-stratified grid"
            )
    else:
        errors.append(f"invalid motion_steps={motion_steps!r}")

    completion_rate = report.get("completion_rate")
    if completions and (
        not _finite_number(completion_rate)
        or not np.isclose(
            np.mean(completions),
            completion_rate,
            atol=1e-12,
        )
    ):
        errors.append("completion_rate does not match rollout rows")
    mean_survival_s = report.get("mean_survival_s")
    if survival_values and (
        not _finite_number(mean_survival_s)
        or not np.isclose(
            np.mean(survival_values),
            mean_survival_s,
            atol=1e-9,
        )
    ):
        errors.append("mean_survival_s does not match rollout rows")
    return errors


def _select_horizon(
    summaries: dict[str, dict[str, Any]],
) -> int | None:
    for horizon in HORIZONS:
        summary = summaries[str(horizon)]
        if (
            summary["pooled_completion"] >= POOLED_COMPLETION_FLOOR
            and all(
                completion >= PER_CLIP_COMPLETION_FLOOR
                for completion in summary["per_clip_completion"].values()
            )
        ):
            return horizon
    return None


def analyze(
    training_rows: list[TrainingRow],
    evaluation_rows: list[EvaluationRow],
) -> dict[str, Any]:
    errors = []
    expected_training_names = {_training_name(clip) for clip in CLIPS}
    actual_training_names = [row.train_name for row in training_rows]
    if (
        len(actual_training_names) != len(set(actual_training_names))
        or set(actual_training_names) != expected_training_names
    ):
        errors.append("training map must contain exactly one row per clip")

    training_by_name = {}
    training_configs = {}
    for row in training_rows:
        config, row_errors = _validate_training_row(row)
        errors.extend(f"{row.train_name}: {error}" for error in row_errors)
        training_by_name[row.train_name] = row
        if config is not None:
            training_configs[row.train_name] = config

    expected_evaluation_names = {
        _evaluation_name(clip, horizon)
        for clip in CLIPS
        for horizon in HORIZONS
    }
    actual_evaluation_names = [row.name for row in evaluation_rows]
    if (
        len(actual_evaluation_names) != len(set(actual_evaluation_names))
        or set(actual_evaluation_names) != expected_evaluation_names
    ):
        errors.append(
            "evaluation map must contain all nine unique clip-horizon rows"
        )

    reports = {}
    for row in evaluation_rows:
        expected_name = _evaluation_name(row.clip, row.total_iterations)
        if row.name != expected_name:
            errors.append(
                f"{row.name}: expected evaluation name {expected_name}"
            )
        expected_train_name = _training_name(row.clip)
        if row.train_name != expected_train_name:
            errors.append(
                f"{row.name}: train_name={row.train_name!r}, "
                f"expected {expected_train_name!r}"
            )
        if row.total_iterations not in HORIZONS:
            errors.append(
                f"{row.name}: unexpected horizon {row.total_iterations}"
            )
        training = training_by_name.get(row.train_name)
        config = training_configs.get(row.train_name)
        if training is None or config is None:
            errors.append(f"{row.name}: missing valid training row")
        else:
            checkpoint_errors = _validate_checkpoint(
                row, training, config
            )
            errors.extend(
                f"{row.name}: {error}" for error in checkpoint_errors
            )
        if not row.report_path.is_file():
            errors.append(f"{row.name}: missing report {row.report_path}")
            continue
        try:
            report = json.loads(row.report_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{row.name}: cannot read report: {exc}")
            continue
        report_errors = _validate_report(row, report)
        errors.extend(f"{row.name}: {error}" for error in report_errors)
        reports[row.name] = report

    if errors:
        return {
            "passed": False,
            "verdict": "invalid_horizon_calibration_artifacts",
            "interpretation": (
                "invalid GMR-only development calibration; no horizon decision"
            ),
            "protocol_errors": errors,
        }

    summaries = {}
    for horizon in HORIZONS:
        horizon_reports = {
            clip: reports[_evaluation_name(clip, horizon)]
            for clip in CLIPS
        }
        per_clip_completion = {
            clip: float(report["completion_rate"])
            for clip, report in horizon_reports.items()
        }
        rollouts = [
            rollout
            for report in horizon_reports.values()
            for rollout in report["rollouts"]
        ]
        summaries[str(horizon)] = {
            "pooled_completion": float(
                np.mean([rollout["completed"] for rollout in rollouts])
            ),
            "per_clip_completion": per_clip_completion,
            "mean_survival_s": float(
                np.mean([rollout["survival_s"] for rollout in rollouts])
            ),
            "mean_joint_position_rmse_rad": float(
                np.mean(
                    [
                        rollout["metrics"]["joint_position_rmse_rad"]
                        for rollout in rollouts
                    ]
                )
            ),
            "mean_root_position_error_m": float(
                np.mean(
                    [
                        rollout["metrics"]["root_position_error_m"]
                        for rollout in rollouts
                    ]
                )
            ),
            "passes_promotion_rule": (
                float(np.mean([rollout["completed"] for rollout in rollouts]))
                >= POOLED_COMPLETION_FLOOR
                and all(
                    completion >= PER_CLIP_COMPLETION_FLOOR
                    for completion in per_clip_completion.values()
                )
            ),
        }

    selected_horizon = _select_horizon(summaries)
    verdict = (
        f"promote_{selected_horizon}_iteration_horizon"
        if selected_horizon is not None
        else "no_viable_horizon_through_8000"
    )
    return {
        "passed": True,
        "verdict": verdict,
        "interpretation": (
            "GMR-only development horizon calibration; not a source "
            "comparison or tracking-benefit claim"
        ),
        "protocol_errors": [],
        "protocol": {
            "clips": list(CLIPS),
            "training_seed": TRAINING_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "source_iterations": SOURCE_ITERATIONS,
            "additional_iterations": ADDITIONAL_ITERATIONS,
            "candidate_total_iterations": list(HORIZONS),
            "rollouts_per_clip_horizon": ROLLOUTS_PER_EVALUATION,
            "horizon_s": HORIZON_S,
            "pooled_completion_floor": POOLED_COMPLETION_FLOOR,
            "per_clip_completion_floor": PER_CLIP_COMPLETION_FLOOR,
        },
        "selected_horizon_iterations": selected_horizon,
        "stop_before_matched_extension": selected_horizon is None,
        "horizons": summaries,
    }


def _print_report(report: dict[str, Any]) -> None:
    for error in report.get("protocol_errors", []):
        print(f"ERROR: {error}")
    for horizon, summary in report.get("horizons", {}).items():
        clips = ", ".join(
            f"{clip}={completion:.1%}"
            for clip, completion in summary["per_clip_completion"].items()
        )
        print(
            f"{horizon}: pooled={summary['pooled_completion']:.1%}; "
            f"{clips}; pass={summary['passes_promotion_rule']}"
        )
    print(f"verdict: {report['verdict']}")
    print(report["interpretation"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_map", type=pathlib.Path)
    parser.add_argument("evaluation_map", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    report = analyze(
        read_training_map(args.training_map),
        read_evaluation_map(args.evaluation_map),
    )
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
