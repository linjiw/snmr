#!/usr/bin/env python3
"""Validate and analyze the controlled GMR-reference + SNMR-latent WBT screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, NamedTuple

import numpy as np


TRAINING_SEED = 0
EVALUATION_SEED = 404
TRAINING_ITERATIONS = 8000
SAVE_INTERVAL = 2000
NUM_ENVS = 1024
ROLLOUTS = 100
HORIZON_STEPS = 500
HORIZON_S = 10.0
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260714
COMPLETION_FLOOR_DELTA = -0.05
COMPLETION_IMPROVEMENT = 0.05
RELATIVE_IMPROVEMENT = 0.05
PRIMARY_JOINT_METRIC = "joint_position_rmse_rad"
BASE_COMMAND_FUNC = "holosoma.managers.observation.terms.wbt:motion_command"
CURRENT_LATENT_FUNC = (
    "snmr.integration.wbt_latent:motion_command_with_current_latent"
)
PREVIEW_LATENT_FUNC = (
    "snmr.integration.wbt_latent:motion_command_with_latent_preview"
)
ARMS = {
    "s1_current_ac": {
        "training_name": "latent_s1_current_ac_walk1_seed0_to8000",
        "actor_func": CURRENT_LATENT_FUNC,
        "critic_func": CURRENT_LATENT_FUNC,
    },
    "s2_preview_ac": {
        "training_name": "latent_s2_preview_ac_walk1_seed0_to8000",
        "actor_func": PREVIEW_LATENT_FUNC,
        "critic_func": PREVIEW_LATENT_FUNC,
    },
    "s3_preview_critic": {
        "training_name": "latent_s3_preview_critic_walk1_seed0_to8000",
        "actor_func": BASE_COMMAND_FUNC,
        "critic_func": PREVIEW_LATENT_FUNC,
    },
}
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
    PRIMARY_JOINT_METRIC,
    "torque_rms_nm",
    "mechanical_power_abs_w",
    "joint_limit_violation_rad",
    "undesired_contact_count",
)


class EvaluationRow(NamedTuple):
    arm: str
    training_name: str
    report_path: pathlib.Path
    checkpoint_path: pathlib.Path
    checkpoint_sha256: str
    actor_func: str
    critic_func: str


def read_evaluation_map(path: pathlib.Path) -> list[EvaluationRow]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 7:
                raise ValueError(
                    f"{path}:{line_number}: expected 7 tab-separated fields, "
                    f"found {len(fields)}"
                )
            rows.append(
                EvaluationRow(
                    arm=fields[0],
                    training_name=fields[1],
                    report_path=pathlib.Path(fields[2]),
                    checkpoint_path=pathlib.Path(fields[3]),
                    checkpoint_sha256=fields[4],
                    actor_func=fields[5],
                    critic_func=fields[6],
                )
            )
    return rows


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _resolve(path: pathlib.Path) -> pathlib.Path:
    return path.expanduser().resolve()


def _validate_augmented_reference(
    reference_path: pathlib.Path,
    augmented_path: pathlib.Path,
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    try:
        with np.load(reference_path, allow_pickle=True) as reference:
            reference_arrays = {
                key: np.asarray(reference[key]) for key in reference.files
            }
        with np.load(augmented_path, allow_pickle=True) as augmented:
            augmented_arrays = {
                key: np.asarray(augmented[key]) for key in augmented.files
            }
    except (OSError, ValueError) as exc:
        return {}, [f"cannot load reference artifacts: {exc}"]

    expected_keys = set(reference_arrays) | {"latent_z"}
    if set(augmented_arrays) != expected_keys:
        errors.append(
            "augmented reference fields differ from reference + latent_z"
        )
    changed_fields = []
    for key, reference_value in reference_arrays.items():
        augmented_value = augmented_arrays.get(key)
        if augmented_value is None or not np.array_equal(
            reference_value, augmented_value
        ):
            changed_fields.append(key)
    if changed_fields:
        errors.append(
            f"standard GMR fields changed: {sorted(changed_fields)}"
        )

    latent = augmented_arrays.get("latent_z")
    joint_pos = reference_arrays.get("joint_pos")
    if (
        latent is None
        or latent.ndim != 2
        or latent.shape[1] != 128
        or joint_pos is None
        or latent.shape[0] != joint_pos.shape[0]
        or not np.isfinite(latent).all()
    ):
        errors.append(
            "latent_z must be finite (T,128) and aligned to reference joint_pos"
        )
    summary = {
        "reference_path": str(_resolve(reference_path)),
        "augmented_path": str(_resolve(augmented_path)),
        "standard_field_count": len(reference_arrays),
        "standard_fields_unchanged": not changed_fields,
        "latent_shape": list(latent.shape) if latent is not None else None,
    }
    return summary, errors


def _validate_rollout_report(
    report: dict[str, Any],
    *,
    expected_training_name: str,
    expected_motion_file: pathlib.Path,
) -> list[str]:
    errors = []
    checks = {
        "schema_version": (report.get("schema_version"), 1),
        "passed": (report.get("passed"), True),
        "seed": (report.get("seed"), EVALUATION_SEED),
        "training_name": (
            report.get("training_name"),
            expected_training_name,
        ),
        "num_rollouts": (report.get("num_rollouts"), ROLLOUTS),
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
    if not isinstance(motion_file, str) or _resolve(
        pathlib.Path(motion_file)
    ) != _resolve(expected_motion_file):
        errors.append(
            f"motion_file={motion_file!r}, expected {str(expected_motion_file)!r}"
        )

    rollouts = report.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != ROLLOUTS:
        errors.append(
            f"rollouts length is "
            f"{len(rollouts) if isinstance(rollouts, list) else None}, "
            f"expected {ROLLOUTS}"
        )
        return errors

    env_ids = []
    start_steps = []
    completions = []
    survival_values = []
    for index, rollout in enumerate(rollouts):
        if not isinstance(rollout, dict):
            errors.append(f"rollout {index} is not an object")
            continue
        env_ids.append(rollout.get("env_id"))
        start_steps.append(rollout.get("start_step"))
        completed = rollout.get("completed")
        failed = rollout.get("failed")
        survival_steps = rollout.get("survival_steps")
        survival_s = rollout.get("survival_s")
        if (
            not isinstance(completed, bool)
            or not isinstance(failed, bool)
            or completed == failed
        ):
            errors.append(
                f"rollout {index}: completed/failed are not complementary"
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
                f"rollout {index}: completion before the fixed horizon"
            )
        elif not _finite_number(survival_s) or not np.isclose(
            survival_s,
            survival_steps * HORIZON_S / HORIZON_STEPS,
            atol=1e-12,
        ):
            errors.append(
                f"rollout {index}: survival_s does not match survival_steps"
            )

        metrics = rollout.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
            errors.append(f"rollout {index}: official metric set mismatch")
        elif not all(
            _finite_number(metrics[metric]) and metrics[metric] >= 0.0
            for metric in METRICS
        ):
            errors.append(
                f"rollout {index}: metrics must be finite and nonnegative"
            )
        if isinstance(completed, bool):
            completions.append(float(completed))
        if _finite_number(survival_s):
            survival_values.append(float(survival_s))

    if env_ids != list(range(ROLLOUTS)):
        errors.append("env_ids must be exactly 0..99 in order")
    motion_steps = report.get("motion_steps")
    if isinstance(motion_steps, int) and motion_steps > HORIZON_STEPS:
        expected_starts = (
            np.linspace(
                0,
                motion_steps - HORIZON_STEPS - 1,
                ROLLOUTS,
            )
            .round()
            .astype(int)
            .tolist()
        )
        if start_steps != expected_starts:
            errors.append(
                "start_steps do not match the frozen phase-stratified grid"
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


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required; run analysis in the WBT environment"
        ) from exc
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping")
    return config


def _validate_training_config(
    row: EvaluationRow,
    augmented_path: pathlib.Path,
) -> list[str]:
    errors = []
    config_path = row.checkpoint_path.parent / "holosoma_config.yaml"
    try:
        config = _load_config(config_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return [str(exc)]
    checks = {
        "training.name": (
            _get(config, "training", "name"),
            row.training_name,
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
            None,
        ),
        "algo.config.num_learning_iterations": (
            _get(config, "algo", "config", "num_learning_iterations"),
            TRAINING_ITERATIONS,
        ),
        "algo.config.save_interval": (
            _get(config, "algo", "config", "save_interval"),
            SAVE_INTERVAL,
        ),
        "actor motion_command func": (
            _get(
                config,
                "observation",
                "groups",
                "actor_obs",
                "terms",
                "motion_command",
                "func",
            ),
            row.actor_func,
        ),
        "critic motion_command func": (
            _get(
                config,
                "observation",
                "groups",
                "critic_obs",
                "terms",
                "motion_command",
                "func",
            ),
            row.critic_func,
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
    if not isinstance(motion_file, str) or _resolve(
        pathlib.Path(motion_file)
    ) != _resolve(augmented_path):
        errors.append(
            f"training motion_file={motion_file!r}, "
            f"expected {str(augmented_path)!r}"
        )
    return errors


def _report_arrays(report: dict[str, Any]) -> dict[str, np.ndarray]:
    rollouts = report["rollouts"]
    return {
        "completion": np.asarray(
            [float(rollout["completed"]) for rollout in rollouts],
            dtype=np.float64,
        ),
        "survival_s": np.asarray(
            [rollout["survival_s"] for rollout in rollouts],
            dtype=np.float64,
        ),
        **{
            metric: np.asarray(
                [rollout["metrics"][metric] for rollout in rollouts],
                dtype=np.float64,
            )
            for metric in METRICS
        },
    }


def _confidence_interval(samples: np.ndarray) -> list[float]:
    return [
        float(value) for value in np.quantile(samples, [0.025, 0.975])
    ]


def _effect_summary(
    baseline: np.ndarray,
    arm: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    baseline_mean = float(np.mean(baseline))
    arm_mean = float(np.mean(arm))
    baseline_samples = baseline[bootstrap_indices].mean(axis=1)
    arm_samples = arm[bootstrap_indices].mean(axis=1)
    absolute_samples = arm_samples - baseline_samples
    relative_effect = (
        arm_mean / baseline_mean - 1.0 if baseline_mean > 0.0 else None
    )
    relative_ci = None
    if np.all(baseline_samples > 0.0):
        relative_ci = _confidence_interval(
            arm_samples / baseline_samples - 1.0
        )
    return {
        "baseline_mean": baseline_mean,
        "arm_mean": arm_mean,
        "arm_minus_baseline": arm_mean - baseline_mean,
        "arm_minus_baseline_ci95": _confidence_interval(absolute_samples),
        "arm_relative_to_baseline": relative_effect,
        "arm_relative_to_baseline_ci95": relative_ci,
    }


def _promotion_decision(metric_effects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    completion_delta = metric_effects["completion"]["arm_minus_baseline"]
    joint_relative = metric_effects[PRIMARY_JOINT_METRIC][
        "arm_relative_to_baseline"
    ]
    survival_relative = metric_effects["survival_s"][
        "arm_relative_to_baseline"
    ]
    completion_floor_passed = completion_delta >= COMPLETION_FLOOR_DELTA
    improvements = {
        "completion_at_least_5pp": (
            completion_delta >= COMPLETION_IMPROVEMENT
        ),
        "survival_at_least_5pct": (
            survival_relative is not None
            and survival_relative >= RELATIVE_IMPROVEMENT
        ),
        "joint_rmse_at_least_5pct_lower": (
            joint_relative is not None
            and joint_relative <= -RELATIVE_IMPROVEMENT
        ),
    }
    return {
        "completion_floor_passed": completion_floor_passed,
        "improvement_checks": improvements,
        "eligible_for_replication": (
            completion_floor_passed and any(improvements.values())
        ),
    }


def analyze(
    baseline_report_path: pathlib.Path,
    baseline_checkpoint_path: pathlib.Path,
    reference_path: pathlib.Path,
    augmented_path: pathlib.Path,
    evaluation_rows: list[EvaluationRow],
) -> dict[str, Any]:
    errors = []
    reference_summary, reference_errors = _validate_augmented_reference(
        reference_path, augmented_path
    )
    errors.extend(reference_errors)

    if (
        not baseline_checkpoint_path.is_file()
        or baseline_checkpoint_path.name != "model_07999.pt"
    ):
        errors.append(
            f"invalid baseline checkpoint {baseline_checkpoint_path}"
        )
    try:
        baseline_report = json.loads(baseline_report_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        baseline_report = {}
        errors.append(f"cannot load baseline report: {exc}")
    baseline_name = "horizon_gmr_walk1_seed0_to8000"
    errors.extend(
        f"baseline: {error}"
        for error in _validate_rollout_report(
            baseline_report,
            expected_training_name=baseline_name,
            expected_motion_file=reference_path,
        )
    )

    actual_arms = [row.arm for row in evaluation_rows]
    if (
        len(actual_arms) != len(set(actual_arms))
        or set(actual_arms) != set(ARMS)
    ):
        errors.append("evaluation map must contain exactly the three frozen arms")

    reports = {}
    rows_by_arm = {}
    for row in evaluation_rows:
        rows_by_arm[row.arm] = row
        expected = ARMS.get(row.arm)
        if expected is None:
            errors.append(f"unexpected arm {row.arm!r}")
            continue
        for field in ("training_name", "actor_func", "critic_func"):
            if getattr(row, field) != expected[field]:
                errors.append(
                    f"{row.arm}: {field}={getattr(row, field)!r}, "
                    f"expected {expected[field]!r}"
                )
        if (
            not row.checkpoint_path.is_file()
            or row.checkpoint_path.name != "model_07999.pt"
        ):
            errors.append(
                f"{row.arm}: invalid checkpoint {row.checkpoint_path}"
            )
        elif _sha256(row.checkpoint_path) != row.checkpoint_sha256:
            errors.append(f"{row.arm}: checkpoint SHA-256 mismatch")
        errors.extend(
            f"{row.arm}: {error}"
            for error in _validate_training_config(row, augmented_path)
        )
        try:
            report = json.loads(row.report_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{row.arm}: cannot load report: {exc}")
            continue
        errors.extend(
            f"{row.arm}: {error}"
            for error in _validate_rollout_report(
                report,
                expected_training_name=row.training_name,
                expected_motion_file=augmented_path,
            )
        )
        reports[row.arm] = report

    baseline_starts = [
        rollout.get("start_step")
        for rollout in baseline_report.get("rollouts", [])
    ]
    for arm, report in reports.items():
        arm_starts = [
            rollout.get("start_step")
            for rollout in report.get("rollouts", [])
        ]
        if arm_starts != baseline_starts:
            errors.append(f"{arm}: paired start steps differ from baseline")
        if report.get("motion_steps") != baseline_report.get("motion_steps"):
            errors.append(f"{arm}: motion length differs from baseline")

    if errors:
        return {
            "passed": False,
            "verdict": "invalid_latent_screen_artifacts",
            "interpretation": (
                "invalid one-clip, one-seed screen; no tracking claim"
            ),
            "protocol_errors": errors,
        }

    baseline_arrays = _report_arrays(baseline_report)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0, ROLLOUTS, size=(BOOTSTRAP_REPLICATES, ROLLOUTS)
    )
    arm_summaries = {}
    for arm in ARMS:
        arm_arrays = _report_arrays(reports[arm])
        metric_effects = {
            metric: _effect_summary(
                baseline_arrays[metric],
                arm_arrays[metric],
                bootstrap_indices,
            )
            for metric in ("completion", "survival_s", *METRICS)
        }
        arm_summaries[arm] = {
            "training_name": rows_by_arm[arm].training_name,
            "actor_func": rows_by_arm[arm].actor_func,
            "critic_func": rows_by_arm[arm].critic_func,
            "metric_effects": metric_effects,
            "promotion": _promotion_decision(metric_effects),
        }

    promoted = [
        arm
        for arm, summary in arm_summaries.items()
        if summary["promotion"]["eligible_for_replication"]
    ]
    return {
        "passed": True,
        "verdict": (
            "promote_for_multiseed_replication"
            if promoted
            else "no_arm_meets_frozen_promotion_rule"
        ),
        "interpretation": (
            "one-clip, one-training-seed development screen; bootstrap "
            "intervals resample paired phase windows and do not establish "
            "cross-motion or cross-training-seed benefit"
        ),
        "protocol_errors": [],
        "protocol": {
            "training_seed": TRAINING_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "training_iterations": TRAINING_ITERATIONS,
            "num_envs": NUM_ENVS,
            "rollouts": ROLLOUTS,
            "horizon_s": HORIZON_S,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "completion_floor_delta": COMPLETION_FLOOR_DELTA,
            "completion_improvement": COMPLETION_IMPROVEMENT,
            "relative_improvement": RELATIVE_IMPROVEMENT,
        },
        "reference_integrity": reference_summary,
        "baseline": {
            "training_name": baseline_name,
            "report_path": str(_resolve(baseline_report_path)),
            "checkpoint_path": str(_resolve(baseline_checkpoint_path)),
            "checkpoint_sha256": _sha256(baseline_checkpoint_path),
        },
        "promoted_arms": promoted,
        "arms": arm_summaries,
    }


def _print_report(report: dict[str, Any]) -> None:
    for error in report.get("protocol_errors", []):
        print(f"ERROR: {error}")
    for arm, summary in report.get("arms", {}).items():
        metrics = summary["metric_effects"]
        print(
            f"{arm}: completion={metrics['completion']['arm_mean']:.1%} "
            f"(delta {metrics['completion']['arm_minus_baseline']:+.1%}), "
            f"survival={metrics['survival_s']['arm_mean']:.4f}s, "
            f"joint_rmse={metrics[PRIMARY_JOINT_METRIC]['arm_mean']:.6f}rad, "
            f"promote={summary['promotion']['eligible_for_replication']}"
        )
    print(f"verdict: {report['verdict']}")
    print(report["interpretation"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", required=True, type=pathlib.Path)
    parser.add_argument(
        "--baseline-checkpoint", required=True, type=pathlib.Path
    )
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument(
        "--augmented-reference", required=True, type=pathlib.Path
    )
    parser.add_argument("--evaluation-map", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    report = analyze(
        args.baseline_report,
        args.baseline_checkpoint,
        args.reference,
        args.augmented_reference,
        read_evaluation_map(args.evaluation_map),
    )
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
