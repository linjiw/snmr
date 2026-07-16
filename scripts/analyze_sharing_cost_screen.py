#!/usr/bin/env python3
"""Validate and decide the matched-exposure sharing-cost screen."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
from statistics import fmean
from typing import Any

import torch

ROBOTS = (
    "unitree_g1",
    "booster_t1_29dof",
    "fourier_n1",
    "engineai_pm01",
    "stanford_toddy",
)
SHARED_ARMS = ("shared_base_seed0", "shared_wide_seed0", "shared_adapter_seed0")
CANDIDATE_ARMS = ("shared_wide_seed0", "shared_adapter_seed0")
SPECIALIST_PREFIX = "specialist_"
SEED = 0
EXPOSURES_PER_ROBOT = 50_000
SPECIALIST_STEPS = 50_000
SHARED_STEPS = 125_000
BASE_PARAMETERS = 1_540_810
WIDE_PARAMETERS = 2_751_178
ADAPTER_PARAMETERS = 1_561_290
ROBOT_SPECIFIC_PARAMETERS = 20_480
GAP_CLOSURE_FRACTION = 0.50
MAX_TOTAL_PARAMETER_RATIO = 2.0
MAX_ROBOT_SPECIFIC_FRACTION = 0.20


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_resume_checks(manifest: dict[str, Any]) -> list[str]:
    errors = []
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        return ["manifest invocations are missing"]
    if not isinstance(invocations[0], dict) or invocations[0].get("resume") is not False:
        errors.append("the first invocation must be a fresh launch")
    resume_count = sum(
        isinstance(invocation, dict) and invocation.get("resume") is True
        for invocation in invocations
    )
    checks = manifest.get("resume_checks", [])
    if not isinstance(checks, list):
        return ["resume_checks must be a list"]
    if len(checks) != resume_count:
        errors.append("resume invocation and integrity-check counts differ")
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            errors.append(f"resume check {index} is malformed")
            continue
        if check.get("config_matches_original_except_resume") is not True:
            errors.append(f"resume check {index} changed the original config")
        if check.get("config_differences") != {}:
            errors.append(f"resume check {index} has config differences")
        if check.get("dataset_matches_original") is not True:
            errors.append(f"resume check {index} changed the original dataset")
        if not _valid_sha256(check.get("dataset_hash")):
            errors.append(f"resume check {index} has an invalid dataset hash")
    return errors


def _dataset_file_sets(
    dataset: dict[str, Any],
) -> dict[str, set[tuple[str, int, str]]]:
    if not isinstance(dataset, dict) or not _valid_sha256(dataset.get("sha256")):
        raise ValueError("dataset fingerprint is missing or invalid")
    splits = dataset.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("dataset split fingerprints are missing")
    records = {}
    for split, split_record in splits.items():
        if not isinstance(split, str) or not isinstance(split_record, dict):
            raise ValueError("dataset split fingerprint is malformed")
        files = split_record.get("files")
        if (
            not isinstance(files, list)
            or split_record.get("file_count") != len(files)
            or not _valid_sha256(split_record.get("sha256"))
        ):
            raise ValueError(f"dataset split {split!r} is incomplete")
        file_records = set()
        for file_record in files:
            if not isinstance(file_record, dict):
                raise ValueError(f"dataset split {split!r} has a malformed file")
            path = file_record.get("path")
            size = file_record.get("size_bytes")
            sha256 = file_record.get("sha256")
            if (
                not isinstance(path, str)
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not _valid_sha256(sha256)
            ):
                raise ValueError(f"dataset split {split!r} has an invalid file")
            file_records.add((path, size, sha256))
        if len(file_records) != len(files):
            raise ValueError(f"dataset split {split!r} has duplicate files")
        records[split] = file_records
    return records


def _validate_dataset_partition(
    datasets: dict[str, dict[str, Any]],
) -> list[str]:
    errors = []
    shared_hashes = {
        datasets[arm].get("sha256")
        for arm in SHARED_ARMS
    }
    if len(shared_hashes) != 1:
        errors.append(
            f"shared arms do not use one dataset fingerprint: {sorted(shared_hashes)}"
        )
        return errors

    try:
        shared_files = _dataset_file_sets(datasets["shared_base_seed0"])
        for arm in SHARED_ARMS[1:]:
            if _dataset_file_sets(datasets[arm]) != shared_files:
                errors.append(f"{arm} dataset files differ from shared base")
        specialist_files = {
            robot: _dataset_file_sets(
                datasets[f"{SPECIALIST_PREFIX}{robot}_seed0"]
            )
            for robot in ROBOTS
        }
    except ValueError as exc:
        return [str(exc)]

    if any(set(files) != set(shared_files) for files in specialist_files.values()):
        errors.append("specialist and shared datasets have different split names")
        return errors
    for split, expected in shared_files.items():
        observed: set[tuple[str, int, str]] = set()
        total_records = 0
        for robot in ROBOTS:
            robot_records = specialist_files[robot][split]
            total_records += len(robot_records)
            observed.update(robot_records)
        if len(observed) != total_records:
            errors.append(f"specialist dataset split {split!r} overlaps across robots")
        if observed != expected:
            errors.append(
                f"specialist datasets do not partition shared split {split!r}"
            )
    return errors


def _arm_specs() -> dict[str, dict[str, Any]]:
    specs = {
        f"{SPECIALIST_PREFIX}{robot}_seed0": {
            "robots": [robot],
            "steps": SPECIALIST_STEPS,
            "robots_per_step": 1,
            "enc_hidden": 256,
            "dec_hidden": 256,
            "decoder_adapter_rank": 0,
            "eval_every": 5000,
            "ckpt_every": 25000,
            "total_parameters": BASE_PARAMETERS,
            "robot_specific_parameters": 0,
        }
        for robot in ROBOTS
    }
    specs.update({
        "shared_base_seed0": {
            "robots": list(ROBOTS),
            "steps": SHARED_STEPS,
            "robots_per_step": 2,
            "enc_hidden": 256,
            "dec_hidden": 256,
            "decoder_adapter_rank": 0,
            "eval_every": 12500,
            "ckpt_every": 25000,
            "total_parameters": BASE_PARAMETERS,
            "robot_specific_parameters": 0,
        },
        "shared_wide_seed0": {
            "robots": list(ROBOTS),
            "steps": SHARED_STEPS,
            "robots_per_step": 2,
            "enc_hidden": 384,
            "dec_hidden": 384,
            "decoder_adapter_rank": 0,
            "eval_every": 12500,
            "ckpt_every": 25000,
            "total_parameters": WIDE_PARAMETERS,
            "robot_specific_parameters": 0,
        },
        "shared_adapter_seed0": {
            "robots": list(ROBOTS),
            "steps": SHARED_STEPS,
            "robots_per_step": 2,
            "enc_hidden": 256,
            "dec_hidden": 256,
            "decoder_adapter_rank": 8,
            "eval_every": 12500,
            "ckpt_every": 25000,
            "total_parameters": ADAPTER_PARAMETERS,
            "robot_specific_parameters": ROBOT_SPECIFIC_PARAMETERS,
        },
    })
    return specs


def _expected_adapter_keys() -> set[str]:
    return {
        f"decoder.adapters.{robot}.{layer}.weight"
        for robot in ROBOTS
        for layer in ("down", "up")
    }


def _validate_schedule(
    manifest: dict[str, Any],
    spec: dict[str, Any],
) -> list[str]:
    errors = []
    sampling = manifest.get("training", {}).get("robot_sampling", {})
    if sampling.get("strategy") != "balanced_combinations":
        return ["robot sampling strategy is not balanced_combinations"]
    cycle = sampling.get("cycle")
    expected_groups = {
        tuple(group)
        for group in itertools.combinations(spec["robots"], spec["robots_per_step"])
    }
    if not isinstance(cycle, list):
        return ["balanced robot cycle is missing"]
    try:
        actual_groups = [tuple(group) for group in cycle]
    except TypeError:
        return ["balanced robot cycle is malformed"]
    if len(actual_groups) != len(set(actual_groups)):
        errors.append("balanced robot cycle contains duplicate groups")
    if set(actual_groups) != expected_groups:
        errors.append("balanced robot cycle does not cover every robot combination")
    counts = {
        robot: sum(robot in group for group in actual_groups)
        for robot in spec["robots"]
    }
    if len(set(counts.values())) != 1:
        errors.append(f"balanced robot cycle has unequal counts: {counts}")
    return errors


def _validate_arm(
    arm_dir: pathlib.Path,
    spec: dict[str, Any],
    launch_revision: str,
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    errors = []
    required = {
        "manifest": arm_dir / "manifest.json",
        "final_eval": arm_dir / "final_eval.json",
        "checkpoint": arm_dir / "ckpt.pt",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return [f"missing artifacts: {missing}"], {}, {}
    manifest = _read_json(required["manifest"])
    final_eval = _read_json(required["final_eval"])

    if manifest.get("trainer") != "scripts/train_phase2.py":
        errors.append(f"unexpected trainer {manifest.get('trainer')!r}")
    if manifest.get("status") != "completed":
        errors.append(f"manifest status is {manifest.get('status')!r}")
    errors.extend(_validate_resume_checks(manifest))
    git = manifest.get("git", {})
    if git.get("sha") != launch_revision or git.get("dirty") is not False:
        errors.append("manifest does not use the clean frozen launch revision")
    source_sha256 = manifest.get("source", {}).get("sha256")
    if not _valid_sha256(source_sha256):
        errors.append("manifest source fingerprint is missing or invalid")
    dataset = manifest.get("dataset", {})
    try:
        _dataset_file_sets(dataset)
    except ValueError as exc:
        errors.append(str(exc))
    progress = manifest.get("progress", {})
    if (
        progress.get("completion_state") != "completed"
        or progress.get("step") != spec["steps"]
    ):
        errors.append("manifest completion step is incorrect")
    expected_exposures = {
        robot: EXPOSURES_PER_ROBOT for robot in spec["robots"]
    }
    if progress.get("robot_exposures") != expected_exposures:
        errors.append(
            f"observed exposures={progress.get('robot_exposures')!r}, "
            f"expected={expected_exposures!r}"
        )
    training = manifest.get("training", {})
    if training.get("planned_effective_robot_exposures") != expected_exposures:
        errors.append("planned exposure counts are incorrect")
    schedule = training.get("lr_schedule", {})
    if (
        schedule.get("name") != "CosineAnnealingLR"
        or schedule.get("t_max_steps") != spec["steps"]
        or schedule.get("eta_min") != 0.00001
    ):
        errors.append("learning-rate schedule is incorrect")

    config = manifest.get("config", {})
    expected_config = {
        "robots": spec["robots"],
        "holdout_robot": None,
        "steps": spec["steps"],
        "window": 64,
        "robots_per_step": spec["robots_per_step"],
        "robot_sampling": "balanced_combinations",
        "lr": 0.0003,
        "min_lr": 0.00001,
        "latent_weight": 1.0,
        "contact_weight": 0.0,
        "contact_bce_weight": 0.0,
        "contact_mask": "teacher_height",
        "edge_velocity_weight": 0.0,
        "stance_velocity_weight": 0.0,
        "penetration_weight": 0.0,
        "teacher_velocity_weight": 0.0,
        "phase_balanced_velocity": False,
        "zr_decode_prob": 0.0,
        "latent_dim": 128,
        "enc_hidden": spec["enc_hidden"],
        "dec_hidden": spec["dec_hidden"],
        "decoder_adapter_rank": spec["decoder_adapter_rank"],
        "eval_every": spec["eval_every"],
        "ckpt_every": spec["ckpt_every"],
        "diag_every": 0,
        "seed": SEED,
        "device": "cuda",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(f"config {key}={config.get(key)!r}, expected {expected!r}")
    errors.extend(_validate_schedule(manifest, spec))

    parameter_record = manifest.get("training", {}).get("model_parameters", {})
    expected_fraction = (
        spec["robot_specific_parameters"] / spec["total_parameters"]
    )
    if parameter_record.get("total") != spec["total_parameters"]:
        errors.append("manifest total parameter count is incorrect")
    if parameter_record.get("robot_specific") != spec["robot_specific_parameters"]:
        errors.append("manifest robot-specific parameter count is incorrect")
    actual_fraction = parameter_record.get("robot_specific_fraction")
    if not _finite(actual_fraction) or not math.isclose(
        float(actual_fraction), expected_fraction, rel_tol=0.0, abs_tol=1e-12
    ):
        errors.append("manifest robot-specific parameter fraction is incorrect")

    checkpoints = manifest.get("checkpoints", [])
    if len(checkpoints) != 1:
        errors.append(f"manifest has {len(checkpoints)} retained checkpoint records")
    else:
        checkpoint_record = checkpoints[0]
        from snmr.experiment import sha256_file

        if checkpoint_record.get("step") != spec["steps"]:
            errors.append("retained checkpoint step is incorrect")
        if checkpoint_record.get("sha256") != sha256_file(required["checkpoint"]):
            errors.append("retained checkpoint hash mismatch")

    checkpoint = torch.load(
        required["checkpoint"], map_location="cpu", weights_only=False
    )
    if checkpoint.get("step") != spec["steps"]:
        errors.append("checkpoint step is incorrect")
    if checkpoint.get("robot_exposures") != expected_exposures:
        errors.append("checkpoint exposure counts are incorrect")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        errors.append("checkpoint config is missing")
    else:
        config_keys = (set(checkpoint_config) | set(config)) - {"resume"}
        if any(checkpoint_config.get(key) != config.get(key) for key in config_keys):
            errors.append("checkpoint config differs from the manifest config")
    for key in ("opt", "sched", "rng_state"):
        if not isinstance(checkpoint.get(key), dict) or not checkpoint[key]:
            errors.append(f"checkpoint is missing {key}")
    state = checkpoint.get("model", {})
    if not isinstance(state, dict) or not state:
        errors.append("checkpoint model state is missing")
        state = {}
    state_parameters = sum(
        tensor.numel() for tensor in state.values() if isinstance(tensor, torch.Tensor)
    )
    if state_parameters != spec["total_parameters"]:
        errors.append(
            f"checkpoint parameters={state_parameters}, "
            f"expected={spec['total_parameters']}"
        )
    adapter_keys = {
        key for key in state if key.startswith("decoder.adapters.")
    }
    if spec["decoder_adapter_rank"] > 0:
        if adapter_keys != _expected_adapter_keys():
            errors.append("adapter checkpoint keys do not match the five registered adapters")
        for robot in ROBOTS:
            up = state.get(f"decoder.adapters.{robot}.up.weight")
            if (
                not isinstance(up, torch.Tensor)
                or not torch.isfinite(up).all()
                or torch.count_nonzero(up).item() == 0
            ):
                errors.append(f"{robot} adapter up-projection did not train")
    elif adapter_keys:
        errors.append("non-adapter arm contains adapter checkpoint keys")

    expected_eval_robots = set(spec["robots"])
    if set(final_eval) != expected_eval_robots:
        errors.append(
            f"final evaluation robots={sorted(final_eval)}, "
            f"expected={sorted(expected_eval_robots)}"
        )
    metrics = {}
    for robot in spec["robots"]:
        row = final_eval.get(robot, {})
        mpjpe = row.get("mpjpe_m")
        dof = row.get("dof_err_rad")
        if not _finite(mpjpe) or float(mpjpe) <= 0:
            errors.append(f"{robot} MPJPE is missing or invalid")
        else:
            metrics[robot] = float(mpjpe)
        if not _finite(dof) or float(dof) <= 0:
            errors.append(f"{robot} DOF error is missing or invalid")
    provenance = {
        "source_sha256": source_sha256,
        "dataset_sha256": dataset.get("sha256"),
        "checkpoint_sha256": (
            checkpoints[0].get("sha256") if len(checkpoints) == 1 else None
        ),
        "parameters": parameter_record,
    }
    return errors, metrics, provenance


def sharing_decision(
    specialist: dict[str, float],
    shared: dict[str, dict[str, float]],
    parameter_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = shared["shared_base_seed0"]
    base_gaps = {robot: base[robot] - specialist[robot] for robot in ROBOTS}
    base_mean_gap = fmean(base_gaps.values())
    base_worst_gap = max(base_gaps.values())
    assay_passed = base_mean_gap > 0 and base_worst_gap > 0
    candidates = {}
    for arm in CANDIDATE_ARMS:
        values = shared[arm]
        gaps = {robot: values[robot] - specialist[robot] for robot in ROBOTS}
        mean_gap = fmean(gaps.values())
        worst_gap = max(gaps.values())
        mean_closure = (
            1.0 - mean_gap / base_mean_gap if base_mean_gap > 0 else float("nan")
        )
        worst_closure = (
            1.0 - worst_gap / base_worst_gap if base_worst_gap > 0 else float("nan")
        )
        parameters = parameter_records[arm]
        checks = {
            "sharing_cost_assay_positive": assay_passed,
            "mean_gap_closure_ge_0.5": (
                assay_passed and mean_closure >= GAP_CLOSURE_FRACTION
            ),
            "worst_gap_closure_ge_0.5": (
                assay_passed and worst_closure >= GAP_CLOSURE_FRACTION
            ),
            "total_parameters_le_2x": (
                parameters["total"] <= MAX_TOTAL_PARAMETER_RATIO * BASE_PARAMETERS
            ),
            "robot_specific_fraction_le_0.2": (
                parameters["robot_specific_fraction"]
                <= MAX_ROBOT_SPECIFIC_FRACTION
            ),
        }
        candidates[arm] = {
            "passed": all(checks.values()),
            "checks": checks,
            "mpjpe_m": values,
            "paired_gaps_m": gaps,
            "mean_gap_m": mean_gap,
            "worst_gap_m": worst_gap,
            "mean_gap_closure_fraction": mean_closure,
            "worst_gap_closure_fraction": worst_closure,
            "parameters": parameters,
        }
    promoted = [arm for arm in CANDIDATE_ARMS if candidates[arm]["passed"]]
    promoted.sort(key=lambda arm: (
        candidates[arm]["mean_gap_m"],
        candidates[arm]["parameters"]["total"],
        arm,
    ))
    return {
        "assay_passed": assay_passed,
        "specialist_mpjpe_m": specialist,
        "base_mpjpe_m": base,
        "base_paired_gaps_m": base_gaps,
        "base_mean_gap_m": base_mean_gap,
        "base_worst_gap_m": base_worst_gap,
        "candidates": candidates,
        "promoted_arms": promoted,
        "winner": promoted[0] if promoted else None,
        "replicate_winner_seeds": [1, 2] if promoted else [],
    }


def analyze(root: pathlib.Path) -> dict[str, Any]:
    launch_path = root / "launch_revision.txt"
    if not launch_path.is_file():
        return {
            "passed": False,
            "verdict": "invalid",
            "protocol_errors": ["missing launch_revision.txt"],
        }
    launch_revision = launch_path.read_text().strip()
    specs = _arm_specs()
    protocol_errors = []
    arm_directories = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    if arm_directories != set(specs):
        protocol_errors.append(
            f"arm directories mismatch: found={sorted(arm_directories)}, "
            f"expected={sorted(specs)}"
        )
    arms = {}
    source_hashes = set()
    datasets = {}
    for arm, spec in specs.items():
        errors, metrics, provenance = _validate_arm(
            root / arm, spec, launch_revision
        )
        manifest_path = root / arm / "manifest.json"
        datasets[arm] = (
            _read_json(manifest_path).get("dataset", {})
            if manifest_path.is_file()
            else {}
        )
        arms[arm] = {
            "passed": not errors,
            "errors": errors,
            "mpjpe_m": metrics,
            "provenance": provenance,
        }
        protocol_errors.extend(f"{arm}: {error}" for error in errors)
        if provenance.get("source_sha256"):
            source_hashes.add(provenance["source_sha256"])
    if len(source_hashes) != 1:
        protocol_errors.append(
            f"arms do not share one source fingerprint: {sorted(source_hashes)}"
        )
    if all(arm in datasets for arm in specs):
        protocol_errors.extend(_validate_dataset_partition(datasets))
    if protocol_errors:
        return {
            "passed": False,
            "verdict": "invalid",
            "protocol_errors": protocol_errors,
            "launch_revision": launch_revision,
            "arms": arms,
        }

    specialist = {
        robot: arms[f"{SPECIALIST_PREFIX}{robot}_seed0"]["mpjpe_m"][robot]
        for robot in ROBOTS
    }
    shared = {arm: arms[arm]["mpjpe_m"] for arm in SHARED_ARMS}
    parameter_records = {
        arm: arms[arm]["provenance"]["parameters"] for arm in SHARED_ARMS
    }
    decision = sharing_decision(specialist, shared, parameter_records)
    winner = decision["winner"]
    verdict = f"promote_{winner}" if winner else "no_intervention_promotes"
    return {
        "passed": True,
        "verdict": verdict,
        "protocol_errors": [],
        "launch_revision": launch_revision,
        "protocol": {
            "seed": SEED,
            "robots": list(ROBOTS),
            "exposures_per_robot": EXPOSURES_PER_ROBOT,
            "specialist_steps": SPECIALIST_STEPS,
            "shared_steps": SHARED_STEPS,
            "gap_closure_fraction": GAP_CLOSURE_FRACTION,
            "max_total_parameter_ratio": MAX_TOTAL_PARAMETER_RATIO,
            "max_robot_specific_fraction": MAX_ROBOT_SPECIFIC_FRACTION,
        },
        "arms": arms,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"verdict: {result['verdict']}")
    if result["passed"]:
        decision = result["decision"]
        print(
            f"base mean/worst paired gap: {decision['base_mean_gap_m']:.6f} / "
            f"{decision['base_worst_gap_m']:.6f} m"
        )
        for arm, candidate in decision["candidates"].items():
            print(
                f"{arm}: mean/worst closure "
                f"{candidate['mean_gap_closure_fraction']:.1%} / "
                f"{candidate['worst_gap_closure_fraction']:.1%}; "
                f"passed={candidate['passed']}"
            )
    else:
        for error in result["protocol_errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
