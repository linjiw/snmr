#!/usr/bin/env python3
"""Validate the calibrated SNMR-reference WBT development policy."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

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


TRAINING_NAME = "reference_dev_snmr_walk1_seed0_to8000"
GMR_TRAINING_NAME = "horizon_gmr_walk1_seed0_to8000"
TRAINING_SEED = 0
EVALUATION_SEED = 404
TRAINING_ITERATIONS = 8000
SAVE_INTERVAL = 2000
NUM_ENVS = 1024
COMPLETION_FLOOR = 0.70


def development_decision(completion_rate: float) -> dict[str, Any]:
    passed = completion_rate >= COMPLETION_FLOOR
    return {
        "completion_floor": COMPLETION_FLOOR,
        "completion_rate": completion_rate,
        "completion_floor_passed": passed,
        "promote_confirmatory_matrix": passed,
    }


def _validate_training(
    run_dir: pathlib.Path,
    checkpoint_path: pathlib.Path,
    reference_path: pathlib.Path,
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    try:
        config = _load_config(run_dir)
        scalars = _load_scalars(run_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return {}, [str(exc)]

    checks = {
        "training.name": (_get(config, "training", "name"), TRAINING_NAME),
        "training.num_envs": (_get(config, "training", "num_envs"), NUM_ENVS),
        "training.seed": (_get(config, "training", "seed"), TRAINING_SEED),
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
        errors.append(
            f"training motion_file={motion_file!r}, expected {str(reference_path)!r}"
        )

    expected_steps = list(range(TRAINING_ITERATIONS))
    scalar_summary = {}
    for tag in REQUIRED_TAGS:
        series = scalars.get(tag)
        if series is None:
            errors.append(f"missing scalar tag {tag}")
            continue
        steps = [step for step, _ in series]
        values = np.asarray([value for _, value in series], dtype=np.float64)
        if steps != expected_steps:
            errors.append(
                f"{tag}: expected steps 0..7999, found "
                f"{steps[0] if steps else None}.."
                f"{steps[-1] if steps else None} ({len(steps)} events)"
            )
        if values.size != TRAINING_ITERATIONS or not np.isfinite(values).all():
            errors.append(f"{tag}: values are incomplete or nonfinite")
        if values.size:
            scalar_summary[tag] = {
                "events": int(values.size),
                "final_100_mean": float(values[-100:].mean()),
            }

    if (
        checkpoint_path.parent.resolve() != run_dir.resolve()
        or checkpoint_path.name != "model_07999.pt"
        or not checkpoint_path.is_file()
    ):
        errors.append(f"invalid final checkpoint {checkpoint_path}")
        return scalar_summary, errors
    try:
        checkpoint = _load_checkpoint(checkpoint_path)
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(str(exc))
        return scalar_summary, errors
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
    return scalar_summary, errors


def _metric_summary(report: dict[str, Any]) -> dict[str, float]:
    arrays = _report_arrays(report)
    return {
        metric: float(np.mean(values))
        for metric, values in arrays.items()
    }


def analyze(
    *,
    run_dir: pathlib.Path,
    checkpoint_path: pathlib.Path,
    report_path: pathlib.Path,
    snmr_reference: pathlib.Path,
    gmr_report_path: pathlib.Path,
    gmr_reference: pathlib.Path,
) -> dict[str, Any]:
    errors = []
    scalar_summary, training_errors = _validate_training(
        run_dir, checkpoint_path, snmr_reference
    )
    errors.extend(f"training: {error}" for error in training_errors)

    try:
        report = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        report = {}
        errors.append(f"cannot load SNMR rollout report: {exc}")
    errors.extend(
        f"SNMR report: {error}"
        for error in _validate_rollout_report(
            report,
            expected_training_name=TRAINING_NAME,
            expected_motion_file=snmr_reference,
            expected_seed=EVALUATION_SEED,
        )
    )

    try:
        gmr_report = json.loads(gmr_report_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        gmr_report = {}
        errors.append(f"cannot load GMR context report: {exc}")
    errors.extend(
        f"GMR context: {error}"
        for error in _validate_rollout_report(
            gmr_report,
            expected_training_name=GMR_TRAINING_NAME,
            expected_motion_file=gmr_reference,
            expected_seed=EVALUATION_SEED,
        )
    )
    snmr_starts = [row.get("start_step") for row in report.get("rollouts", [])]
    gmr_starts = [row.get("start_step") for row in gmr_report.get("rollouts", [])]
    if snmr_starts != gmr_starts:
        errors.append("SNMR and GMR development reports use different phase starts")
    if report.get("motion_steps") != gmr_report.get("motion_steps"):
        errors.append("SNMR and GMR development references have different lengths")

    if errors:
        return {
            "passed": False,
            "verdict": "invalid_development_artifacts",
            "protocol_errors": errors,
        }

    snmr_summary = _metric_summary(report)
    gmr_summary = _metric_summary(gmr_report)
    decision = development_decision(snmr_summary["completion"])
    verdict = (
        "promote_confirmatory_matrix"
        if decision["promote_confirmatory_matrix"]
        else "stop_reference_comparison"
    )
    return {
        "passed": True,
        "verdict": verdict,
        "protocol_errors": [],
        "interpretation": (
            "one-clip, one-training-seed development gate; the GMR policy is a continued "
            "calibration run and is context only, not a matched source comparison"
        ),
        "protocol": {
            "clip": "walk1_subject5",
            "training_seed": TRAINING_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "training_iterations": TRAINING_ITERATIONS,
            "num_envs": NUM_ENVS,
            "completion_floor": COMPLETION_FLOOR,
        },
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
            },
            "snmr_reference": {
                "path": str(snmr_reference.resolve()),
                "sha256": _sha256(snmr_reference),
            },
            "gmr_context_report": {
                "path": str(gmr_report_path.resolve()),
                "sha256": _sha256(gmr_report_path),
            },
            "gmr_reference": {
                "path": str(gmr_reference.resolve()),
                "sha256": _sha256(gmr_reference),
            },
        },
        "training_scalars": scalar_summary,
        "snmr": snmr_summary,
        "gmr_context": gmr_summary,
        "snmr_minus_gmr_context": {
            metric: snmr_summary[metric] - gmr_summary[metric]
            for metric in snmr_summary
        },
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--snmr-reference", type=pathlib.Path, required=True)
    parser.add_argument("--gmr-report", type=pathlib.Path, required=True)
    parser.add_argument("--gmr-reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    result = analyze(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        report_path=args.report,
        snmr_reference=args.snmr_reference,
        gmr_report_path=args.gmr_report,
        gmr_reference=args.gmr_reference,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for error in result.get("protocol_errors", []):
        print(f"ERROR: {error}")
    if result["passed"]:
        print(f"SNMR completion: {result['snmr']['completion']:.1%}")
        print(f"SNMR survival: {result['snmr']['survival_s']:.4f}s")
        print(
            "SNMR joint RMSE: "
            f"{result['snmr']['joint_position_rmse_rad']:.6f}rad"
        )
    print(f"verdict: {result['verdict']}")
    print(result.get("interpretation", "invalid artifacts"))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
