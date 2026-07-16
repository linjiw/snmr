#!/usr/bin/env python3
"""Validate and decide the registered Gate-1b M3 contact-head arm."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
from typing import Any

import torch

from snmr.experiment import sha256_file


EXPECTED_TRAINABLE = (
    "decoder.contact_head.0.weight",
    "decoder.contact_head.0.bias",
    "decoder.contact_head.2.weight",
    "decoder.contact_head.2.bias",
)
EXPECTED_CONFIG = {
    "robot": "unitree_g1",
    "window": 64,
    "lr": 0.0003,
    "min_lr": 0.00001,
    "latent_dim": 128,
    "enc_hidden": 256,
    "dec_hidden": 256,
    "contact_mask": "teacher_height",
    "contact_weight": 0.0,
    "foot_vel_weight": 0.0,
    "contact_bce_weight": 0.25,
    "edge_contact": False,
    "edge_velocity_weight": 0.0,
    "stance_velocity_weight": 0.0,
    "penetration_weight": 0.0,
    "teacher_velocity_weight": 0.0,
    "phase_balanced_velocity": False,
    "no_temporal": False,
    "temporal_positional": False,
    "train_contact_head_only": True,
    "steps": 50000,
    "eval_every": 5000,
    "ckpt_every": 5000,
    "diag_every": 5000,
    "seed": 0,
}
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCIENTIFIC_PATHS = (
    "scripts/train_phase1.py",
    "scripts/audit_contact_masks.py",
    "scripts/eval_footlock.py",
    "snmr",
)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def compare_checkpoint_backbone(
    source_state: dict[str, torch.Tensor],
    trained_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Verify that M3 changed only its newly added contact-head tensors."""
    source_keys = set(source_state)
    trained_keys = set(trained_state)
    changed_inherited = sorted(
        name
        for name in source_keys & trained_keys
        if not torch.equal(source_state[name], trained_state[name])
    )
    missing_inherited = sorted(source_keys - trained_keys)
    new_keys = sorted(trained_keys - source_keys)
    expected_new = sorted(EXPECTED_TRAINABLE)
    return {
        "passed": (
            not changed_inherited
            and not missing_inherited
            and new_keys == expected_new
        ),
        "source_keys": len(source_keys),
        "trained_keys": len(trained_keys),
        "changed_inherited_keys": changed_inherited,
        "missing_inherited_keys": missing_inherited,
        "new_keys": new_keys,
        "expected_new_keys": expected_new,
    }


def validate_artifact_revision(
    launch_revision: str,
    artifact_git: dict[str, Any],
) -> list[str]:
    """Allow clean descendant commits only when all scientific source paths are unchanged."""
    errors = []
    artifact_revision = artifact_git.get("sha")
    if artifact_git.get("dirty") is not False:
        errors.append("artifact revision is dirty")
    if not isinstance(artifact_revision, str):
        return [*errors, "artifact revision is missing"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", launch_revision, artifact_revision],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        errors.append("artifact revision is not a descendant of the launch revision")
        return errors
    scientific_diff = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            launch_revision,
            artifact_revision,
            "--",
            *SCIENTIFIC_PATHS,
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if scientific_diff.returncode != 0:
        errors.append("scientific source paths changed after launch")
    return errors


def analyze(root: pathlib.Path) -> dict[str, Any]:
    errors = []
    required = {
        "launch_revision": root / "launch_revision.txt",
        "manifest": root / "manifest.json",
        "checkpoint": root / "ckpt.pt",
        "mask_audit": root / "mask_audit.json",
        "projection": root / "windowed_projection.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "passed": False,
            "verdict": "invalid",
            "errors": [f"missing required artifacts: {missing}"],
        }

    launch_revision = required["launch_revision"].read_text().strip()
    manifest = _read_json(required["manifest"])
    mask_audit = _read_json(required["mask_audit"])
    projection = _read_json(required["projection"])

    if manifest.get("status") != "completed":
        errors.append(f"manifest status is {manifest.get('status')!r}")
    progress = manifest.get("progress", {})
    if progress.get("completion_state") != "completed" or progress.get("step") != 50000:
        errors.append("manifest is not completed at step 50000")
    git = manifest.get("git", {})
    if git.get("sha") != launch_revision or git.get("dirty") is not False:
        errors.append("training did not use the clean launch revision")

    config = manifest.get("config", {})
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            errors.append(f"config {key}={actual!r}, expected {expected!r}")
    initialization = manifest.get("training", {}).get("initialization", {})
    init_path_value = initialization.get("path")
    init_path = pathlib.Path(init_path_value) if isinstance(init_path_value, str) else None
    if init_path is None or not init_path.is_file():
        errors.append("initialization checkpoint path is missing")
    else:
        actual_init_sha = sha256_file(init_path)
        if initialization.get("sha256") != actual_init_sha:
            errors.append("initialization checkpoint hash mismatch")
    if initialization.get("source_step") != 50000:
        errors.append("initialization source step is not 50000")
    if initialization.get("new_parameter_keys") != sorted(EXPECTED_TRAINABLE):
        errors.append("initialization did not add exactly the contact-head tensors")
    trainable = manifest.get("training", {}).get("trainable_parameter_names")
    if trainable != list(EXPECTED_TRAINABLE):
        errors.append("manifest trainable parameter list is not the four contact-head tensors")
    if manifest.get("training", {}).get("frozen_backbone_eval_mode") is not True:
        errors.append("frozen backbone was not recorded in eval mode")

    trained_checkpoint = torch.load(
        required["checkpoint"], map_location="cpu", weights_only=False
    )
    if init_path is not None and init_path.is_file():
        source_checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        backbone = compare_checkpoint_backbone(
            source_checkpoint["model"], trained_checkpoint["model"]
        )
    else:
        backbone = {"passed": False}
    if not backbone["passed"]:
        errors.append("trained checkpoint changed inherited backbone tensors")
    checkpoint_sha = sha256_file(required["checkpoint"])

    predicted = mask_audit.get("aggregate", {}).get("predicted_t0.5", {})
    for metric in ("precision", "recall", "f1", "candidate_prevalence"):
        if not _finite(predicted.get(metric)):
            errors.append(f"mask audit predicted_t0.5 {metric} is missing or nonfinite")
    mask_artifact = mask_audit.get("artifacts", {}).get("mask_checkpoint", {})
    if mask_artifact.get("sha256") != checkpoint_sha:
        errors.append("mask audit did not use the trained M3 checkpoint")
    mask_git = mask_audit.get("git", {})
    errors.extend(
        f"mask audit: {error}"
        for error in validate_artifact_revision(launch_revision, mask_git)
    )

    protocol = projection.get("protocol", {})
    if protocol.get("num_windows") != 42:
        errors.append(f"projection num_windows={protocol.get('num_windows')!r}, expected 42")
    if protocol.get("method") != "windowed":
        errors.append("projection method is not windowed")
    if protocol.get("lock_mask") != "predicted_contact":
        errors.append("projection mask is not predicted_contact")
    if protocol.get("contact_probability_threshold") != 0.5:
        errors.append("projection contact threshold is not 0.5")
    projection_mask = (
        projection.get("provenance", {})
        .get("artifacts", {})
        .get("mask_checkpoint", {})
    )
    if projection_mask.get("sha256") != checkpoint_sha:
        errors.append("projection did not use the trained M3 checkpoint")
    projection_git = projection.get("provenance", {}).get("git", {})
    errors.extend(
        f"projection: {error}"
        for error in validate_artifact_revision(launch_revision, projection_git)
    )

    raw = projection.get("summary", {}).get("raw", {})
    projected = projection.get("summary", {}).get("windowed", {})
    metrics = (
        "teacher_height_stance_speed_ms",
        "source_contact_stance_speed_ms",
        "mpjpe_m",
        "dof_jerk",
        "limit_violation_fraction",
        "penetration_mean_m",
        "penetration_fraction",
    )
    for metric in metrics:
        if not _finite(raw.get(metric)) or not _finite(projected.get(metric)):
            errors.append(f"projection metric {metric} is missing or nonfinite")
    decision = projection.get("decisions", {}).get("windowed", {})
    required_decisions = (
        "teacher_height_speed_le_0.08",
        "source_contact_speed_le_0.10",
        "mpjpe_delta_le_0.005",
        "dof_jerk_le_1.2x",
        "zero_limit_violations",
        "penetration_mean_guard",
        "penetration_fraction_guard",
        "all_relative_guards_pass",
    )
    for name in required_decisions:
        if not isinstance(decision.get(name), bool):
            errors.append(f"projection decision {name} is missing")

    protocol_passed = not errors
    endpoint_passed = (
        protocol_passed and decision.get("all_relative_guards_pass") is True
    )
    verdict = (
        "pass_contact_closed"
        if endpoint_passed
        else "fail_close_mask_iteration"
        if protocol_passed
        else "invalid"
    )
    return {
        "passed": protocol_passed,
        "verdict": verdict,
        "errors": errors,
        "launch_revision": launch_revision,
        "artifact_revisions": {
            "training": manifest.get("git", {}).get("sha"),
            "mask_audit": mask_git.get("sha"),
            "projection": projection_git.get("sha"),
        },
        "checkpoint_sha256": checkpoint_sha,
        "backbone_integrity": backbone,
        "mask_quality": {
            metric: predicted.get(metric)
            for metric in (
                "precision",
                "recall",
                "f1",
                "candidate_prevalence",
                "oracle_prevalence",
                "samples",
                "windows",
            )
        },
        "raw": {metric: raw.get(metric) for metric in metrics},
        "projected": {metric: projected.get(metric) for metric in metrics},
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()

    report = analyze(args.root)
    output = args.root / "analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    text = (
        f"verdict: {report['verdict']}\n"
        f"protocol_passed: {report['passed']}\n"
        f"errors: {len(report['errors'])}\n"
    )
    if "mask_quality" in report:
        text += (
            f"mask_precision: {report['mask_quality']['precision']}\n"
            f"mask_recall: {report['mask_quality']['recall']}\n"
            f"mask_prevalence: {report['mask_quality']['candidate_prevalence']}\n"
            f"teacher_height_speed_ms: "
            f"{report['projected']['teacher_height_stance_speed_ms']}\n"
            f"source_contact_speed_ms: "
            f"{report['projected']['source_contact_stance_speed_ms']}\n"
            f"mpjpe_m: {report['projected']['mpjpe_m']}\n"
        )
    (args.root / "analysis.txt").write_text(text)
    print(text, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
