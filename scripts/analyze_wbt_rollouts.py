#!/usr/bin/env python3
"""Validate and analyze paired independent WBT policy rollouts."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any, NamedTuple

import numpy as np


CLIPS = ("walk1", "dance2", "fight1")
SOURCES = ("gmr", "snmr")
TRAINING_SEEDS = (0, 1, 2)
EVALUATION_SEEDS = (101, 202, 303)
ROLLOUTS_PER_EVALUATION = 100
HORIZON_STEPS = 500
HORIZON_S = 10.0
COMPLETION_MARGIN = -0.05
JOINT_ERROR_MARGIN = 0.10
GMR_COMPLETION_FLOOR = 0.50
BOOTSTRAP_SEED = 20260714
BOOTSTRAP_REPLICATES = 10_000
PRIMARY_JOINT_METRIC = "joint_position_rmse_rad"
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
NAME_PATTERN = re.compile(
    r"^pilot_(gmr|snmr)_(walk1|dance2|fight1)_seed([0-2])_eval(101|202|303)$"
)


class EvaluationRow(NamedTuple):
    name: str
    train_name: str
    source: str
    clip: str
    training_seed: int
    evaluation_seed: int
    report_path: pathlib.Path
    checkpoint_path: pathlib.Path


def read_eval_map(path: pathlib.Path) -> list[EvaluationRow]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number}: expected 8 tab-separated fields, found {len(fields)}"
                )
            (
                name,
                train_name,
                source,
                clip,
                train_seed,
                eval_seed,
                report,
                checkpoint,
            ) = fields
            rows.append(
                EvaluationRow(
                    name=name,
                    train_name=train_name,
                    source=source,
                    clip=clip,
                    training_seed=int(train_seed),
                    evaluation_seed=int(eval_seed),
                    report_path=pathlib.Path(report),
                    checkpoint_path=pathlib.Path(checkpoint),
                )
            )
    return rows


def _expected_name(
    source: str, clip: str, training_seed: int, evaluation_seed: int
) -> str:
    return f"pilot_{source}_{clip}_seed{training_seed}_eval{evaluation_seed}"


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _validate_row(row: EvaluationRow, report: dict[str, Any]) -> list[str]:
    errors = []
    expected_name = _expected_name(
        row.source, row.clip, row.training_seed, row.evaluation_seed
    )
    expected_train_name = f"pilot_{row.source}_{row.clip}_seed{row.training_seed}"
    match = NAME_PATTERN.fullmatch(row.name)
    checks = {
        "name": (row.name, expected_name),
        "train_name": (row.train_name, expected_train_name),
        "source": (row.source, match.group(1) if match else None),
        "clip": (row.clip, match.group(2) if match else None),
        "training_seed": (row.training_seed, int(match.group(3)) if match else None),
        "evaluation_seed": (
            row.evaluation_seed,
            int(match.group(4)) if match else None,
        ),
        "report.schema_version": (report.get("schema_version"), 1),
        "report.passed": (report.get("passed"), True),
        "report.seed": (report.get("seed"), row.evaluation_seed),
        "report.training_name": (report.get("training_name"), row.train_name),
        "report.num_rollouts": (report.get("num_rollouts"), ROLLOUTS_PER_EVALUATION),
        "report.horizon_steps": (report.get("horizon_steps"), HORIZON_STEPS),
        "report.horizon_s": (report.get("horizon_s"), HORIZON_S),
        "report.policy_dt": (report.get("policy_dt"), HORIZON_S / HORIZON_STEPS),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{field}={actual!r}, expected {expected!r}")

    motion_file = report.get("motion_file")
    motion_path = pathlib.Path(motion_file) if isinstance(motion_file, str) else None
    if (
        motion_path is None
        or motion_path.parent.name != row.source
        or not motion_path.name.startswith(row.clip + "_")
    ):
        errors.append(f"unexpected motion_file={motion_file!r}")

    if (
        row.checkpoint_path.name != "model_00999.pt"
        or not row.checkpoint_path.is_file()
    ):
        errors.append(
            f"checkpoint is not an existing model_00999.pt: {row.checkpoint_path}"
        )

    rollouts = report.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != ROLLOUTS_PER_EVALUATION:
        errors.append(
            f"rollouts has length {len(rollouts) if isinstance(rollouts, list) else None}, "
            f"expected {ROLLOUTS_PER_EVALUATION}"
        )
        return errors

    env_ids = []
    start_steps = []
    completion_values = []
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
                f"rollout {index}: completed/failed must be complementary booleans"
            )
        if (
            not isinstance(survival_steps, int)
            or not 1 <= survival_steps <= HORIZON_STEPS
        ):
            errors.append(f"rollout {index}: invalid survival_steps={survival_steps!r}")
        elif completed and survival_steps != HORIZON_STEPS:
            errors.append(
                f"rollout {index}: completed rollout is inconsistent with "
                f"survival_steps={survival_steps}"
            )
        elif not _finite_number(survival_s) or not np.isclose(
            survival_s, survival_steps * HORIZON_S / HORIZON_STEPS, atol=1e-12
        ):
            errors.append(f"rollout {index}: survival_s does not match survival_steps")
        metrics = rollout.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
            errors.append(f"rollout {index}: metric set mismatch")
        else:
            metric_values = [metrics[name] for name in METRICS]
            if not all(
                _finite_number(value) and value >= 0.0 for value in metric_values
            ):
                errors.append(
                    f"rollout {index}: metrics must be finite and nonnegative"
                )
        if isinstance(completed, bool):
            completion_values.append(float(completed))
        if isinstance(survival_steps, int):
            survival_values.append(survival_steps * HORIZON_S / HORIZON_STEPS)

    if env_ids != list(range(ROLLOUTS_PER_EVALUATION)):
        errors.append("env_ids must be exactly 0..99 in order")
    valid_start_steps = all(isinstance(value, int) for value in start_steps)
    if not valid_start_steps or len(set(start_steps)) != ROLLOUTS_PER_EVALUATION:
        errors.append("start_steps must contain 100 distinct phase-stratified frames")
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
            errors.append("start_steps do not match the frozen phase-stratified grid")
    else:
        errors.append(f"invalid motion_steps={motion_steps!r}")
    if completion_values:
        observed = float(np.mean(completion_values))
        completion_rate = report.get("completion_rate")
        if not _finite_number(completion_rate) or not np.isclose(
            observed, completion_rate, atol=1e-12
        ):
            errors.append("completion_rate does not match rollout rows")
    if survival_values:
        observed = float(np.mean(survival_values))
        mean_survival_s = report.get("mean_survival_s")
        if not _finite_number(mean_survival_s) or not np.isclose(
            observed, mean_survival_s, atol=1e-9
        ):
            errors.append("mean_survival_s does not match rollout rows")
    return errors


def _load_and_validate(
    rows: list[EvaluationRow],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    reports = {}
    errors = []
    expected_names = {
        _expected_name(source, clip, train_seed, eval_seed)
        for source in SOURCES
        for clip in CLIPS
        for train_seed in TRAINING_SEEDS
        for eval_seed in EVALUATION_SEEDS
    }
    actual_names = [row.name for row in rows]
    if len(actual_names) != len(set(actual_names)):
        errors.append("evaluation map contains duplicate names")
    if set(actual_names) != expected_names:
        errors.append(
            f"evaluation set mismatch: missing={sorted(expected_names - set(actual_names))}, "
            f"unexpected={sorted(set(actual_names) - expected_names)}"
        )

    for row in rows:
        if not row.report_path.is_file():
            errors.append(f"{row.name}: missing report {row.report_path}")
            continue
        try:
            report = json.loads(row.report_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{row.name}: cannot read report: {exc}")
            continue
        row_errors = _validate_row(row, report)
        errors.extend(f"{row.name}: {message}" for message in row_errors)
        reports[row.name] = report

    for clip in CLIPS:
        for train_seed in TRAINING_SEEDS:
            for eval_seed in EVALUATION_SEEDS:
                gmr = reports.get(_expected_name("gmr", clip, train_seed, eval_seed))
                snmr = reports.get(_expected_name("snmr", clip, train_seed, eval_seed))
                if gmr is None or snmr is None:
                    continue
                gmr_starts = [
                    rollout["start_step"] for rollout in gmr.get("rollouts", [])
                ]
                snmr_starts = [
                    rollout["start_step"] for rollout in snmr.get("rollouts", [])
                ]
                if gmr_starts != snmr_starts:
                    errors.append(
                        f"{clip} train={train_seed} eval={eval_seed}: paired start steps differ"
                    )
                if gmr.get("motion_steps") != snmr.get("motion_steps"):
                    errors.append(
                        f"{clip} train={train_seed} eval={eval_seed}: paired motion lengths differ"
                    )
    return reports, errors


def _paired_arrays(
    reports: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    shape = (
        len(CLIPS),
        len(TRAINING_SEEDS),
        len(EVALUATION_SEEDS),
        ROLLOUTS_PER_EVALUATION,
    )
    arrays = {
        source: {
            "completion": np.empty(shape, dtype=np.float64),
            "survival_s": np.empty(shape, dtype=np.float64),
            **{metric: np.empty(shape, dtype=np.float64) for metric in METRICS},
        }
        for source in SOURCES
    }
    for source in SOURCES:
        for clip_index, clip in enumerate(CLIPS):
            for train_index, train_seed in enumerate(TRAINING_SEEDS):
                for eval_index, eval_seed in enumerate(EVALUATION_SEEDS):
                    report = reports[
                        _expected_name(source, clip, train_seed, eval_seed)
                    ]
                    for rollout_index, rollout in enumerate(report["rollouts"]):
                        arrays[source]["completion"][
                            clip_index, train_index, eval_index, rollout_index
                        ] = float(rollout["completed"])
                        arrays[source]["survival_s"][
                            clip_index, train_index, eval_index, rollout_index
                        ] = rollout["survival_s"]
                        for metric in METRICS:
                            arrays[source][metric][
                                clip_index, train_index, eval_index, rollout_index
                            ] = rollout["metrics"][metric]
    return arrays


def _bootstrap_primary(
    gmr_completion: np.ndarray,
    snmr_completion: np.ndarray,
    gmr_joint: np.ndarray,
    snmr_joint: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    completion_effects = np.empty(replicates, dtype=np.float64)
    joint_relative_effects = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        gmr_completion_samples = []
        snmr_completion_samples = []
        gmr_joint_samples = []
        snmr_joint_samples = []
        for clip_index in rng.integers(0, len(CLIPS), size=len(CLIPS)):
            for train_index in rng.integers(
                0, len(TRAINING_SEEDS), size=len(TRAINING_SEEDS)
            ):
                for eval_index in rng.integers(
                    0, len(EVALUATION_SEEDS), size=len(EVALUATION_SEEDS)
                ):
                    rollout_indices = rng.integers(
                        0,
                        ROLLOUTS_PER_EVALUATION,
                        size=ROLLOUTS_PER_EVALUATION,
                    )
                    index = (clip_index, train_index, eval_index, rollout_indices)
                    gmr_completion_samples.append(gmr_completion[index])
                    snmr_completion_samples.append(snmr_completion[index])
                    gmr_joint_samples.append(gmr_joint[index])
                    snmr_joint_samples.append(snmr_joint[index])
        gmr_completion_mean = float(np.mean(gmr_completion_samples))
        snmr_completion_mean = float(np.mean(snmr_completion_samples))
        gmr_joint_mean = float(np.mean(gmr_joint_samples))
        snmr_joint_mean = float(np.mean(snmr_joint_samples))
        completion_effects[replicate] = snmr_completion_mean - gmr_completion_mean
        joint_relative_effects[replicate] = snmr_joint_mean / gmr_joint_mean - 1.0
    return completion_effects, joint_relative_effects


def _confidence_interval(samples: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _classify_verdict(
    *,
    assay_valid: bool,
    completion_noninferior: bool,
    joint_noninferior: bool,
    completion_inferior: bool,
    joint_inferior: bool,
) -> str:
    if completion_inferior or joint_inferior:
        return "inferior_on_three_clip_pilot"
    if not assay_valid:
        return "undertrained_gmr_control_on_three_clip_pilot"
    if completion_noninferior and joint_noninferior:
        return "noninferior_on_three_clip_pilot"
    return "inconclusive_on_three_clip_pilot"


def analyze(rows: list[EvaluationRow]) -> dict[str, Any]:
    reports, errors = _load_and_validate(rows)
    if errors:
        return {
            "passed": False,
            "protocol_errors": errors,
            "interpretation": "invalid independent-rollout artifact set; no tracking claim",
        }

    arrays = _paired_arrays(reports)
    completion_bootstrap, joint_bootstrap = _bootstrap_primary(
        arrays["gmr"]["completion"],
        arrays["snmr"]["completion"],
        arrays["gmr"][PRIMARY_JOINT_METRIC],
        arrays["snmr"][PRIMARY_JOINT_METRIC],
    )
    completion_ci = _confidence_interval(completion_bootstrap)
    joint_ci = _confidence_interval(joint_bootstrap)
    completion_effect = float(
        np.mean(arrays["snmr"]["completion"]) - np.mean(arrays["gmr"]["completion"])
    )
    joint_relative_effect = float(
        np.mean(arrays["snmr"][PRIMARY_JOINT_METRIC])
        / np.mean(arrays["gmr"][PRIMARY_JOINT_METRIC])
        - 1.0
    )
    completion_noninferior = completion_ci[0] >= COMPLETION_MARGIN
    joint_noninferior = joint_ci[1] <= JOINT_ERROR_MARGIN
    gmr_completion_mean = float(np.mean(arrays["gmr"]["completion"]))
    assay_valid = gmr_completion_mean >= GMR_COMPLETION_FLOOR
    completion_inferior = completion_ci[1] < COMPLETION_MARGIN
    joint_inferior = joint_ci[0] > JOINT_ERROR_MARGIN
    verdict = _classify_verdict(
        assay_valid=assay_valid,
        completion_noninferior=completion_noninferior,
        joint_noninferior=joint_noninferior,
        completion_inferior=completion_inferior,
        joint_inferior=joint_inferior,
    )

    metric_effects = {}
    for metric in ("completion", "survival_s", *METRICS):
        gmr_mean = float(np.mean(arrays["gmr"][metric]))
        snmr_mean = float(np.mean(arrays["snmr"][metric]))
        metric_effects[metric] = {
            "gmr_mean": gmr_mean,
            "snmr_mean": snmr_mean,
            "snmr_minus_gmr": snmr_mean - gmr_mean,
            "snmr_relative_to_gmr": snmr_mean / gmr_mean - 1.0
            if gmr_mean > 0.0
            else None,
        }

    pair_summaries = {}
    for clip_index, clip in enumerate(CLIPS):
        pair_summaries[clip] = {}
        for train_index, train_seed in enumerate(TRAINING_SEEDS):
            gmr_completion = arrays["gmr"]["completion"][clip_index, train_index]
            snmr_completion = arrays["snmr"]["completion"][clip_index, train_index]
            gmr_joint = arrays["gmr"][PRIMARY_JOINT_METRIC][clip_index, train_index]
            snmr_joint = arrays["snmr"][PRIMARY_JOINT_METRIC][clip_index, train_index]
            pair_summaries[clip][str(train_seed)] = {
                "gmr_completion": float(np.mean(gmr_completion)),
                "snmr_completion": float(np.mean(snmr_completion)),
                "completion_effect": float(
                    np.mean(snmr_completion) - np.mean(gmr_completion)
                ),
                "gmr_joint_rmse_rad": float(np.mean(gmr_joint)),
                "snmr_joint_rmse_rad": float(np.mean(snmr_joint)),
                "joint_relative_effect": float(
                    np.mean(snmr_joint) / np.mean(gmr_joint) - 1.0
                ),
            }

    return {
        "passed": True,
        "verdict": verdict,
        "interpretation": (
            "independent fixed-policy rollouts on the three-clip pilot; "
            "not a broad Stage C tracking claim"
        ),
        "protocol_errors": [],
        "protocol": {
            "clips": list(CLIPS),
            "training_seeds": list(TRAINING_SEEDS),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "rollouts_per_policy_seed": ROLLOUTS_PER_EVALUATION,
            "horizon_s": HORIZON_S,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "completion_noninferiority_margin": COMPLETION_MARGIN,
            "joint_error_relative_noninferiority_margin": JOINT_ERROR_MARGIN,
            "gmr_completion_assay_floor": GMR_COMPLETION_FLOOR,
        },
        "assay_validity": {
            "gmr_completion_mean": gmr_completion_mean,
            "minimum_required": GMR_COMPLETION_FLOOR,
            "passed": assay_valid,
        },
        "primary": {
            "completion": {
                "effect_snmr_minus_gmr": completion_effect,
                "confidence_interval_95": completion_ci,
                "noninferior": completion_noninferior,
            },
            "joint_position_rmse": {
                "relative_effect_snmr_over_gmr_minus_one": joint_relative_effect,
                "confidence_interval_95": joint_ci,
                "noninferior": joint_noninferior,
            },
        },
        "metric_effects": metric_effects,
        "clip_training_seed_pairs": pair_summaries,
    }


def _print_report(report: dict[str, Any]) -> None:
    for error in report.get("protocol_errors", []):
        print(f"ERROR: {error}")
    if report.get("passed"):
        completion = report["primary"]["completion"]
        joint = report["primary"]["joint_position_rmse"]
        print(
            "completion SNMR-GMR: "
            f"{completion['effect_snmr_minus_gmr']:.6f} "
            f"CI={completion['confidence_interval_95']}"
        )
        print(
            "joint RMSE relative effect: "
            f"{joint['relative_effect_snmr_over_gmr_minus_one']:.6f} "
            f"CI={joint['confidence_interval_95']}"
        )
        print(f"verdict: {report['verdict']}")
    print(report["interpretation"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_map", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    report = analyze(read_eval_map(args.eval_map))
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
