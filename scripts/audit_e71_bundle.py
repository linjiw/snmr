#!/usr/bin/env python
"""Fail-closed integrity audit for the E71 command-swap evidence bundle.

The auditor is deliberately independent of the simulator.  It replays file hashes,
validates every rollout report, requires an identical four-cell grid across policies,
and recomputes both the explicit feasibility gate and the confirmatory analysis.  Its
own output is written with create-if-absent semantics so an earlier certificate cannot
be silently replaced.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_e71_command_swap import (
    EXPECTED_EVALUATION_CONDITIONS,
    MIN_ELIGIBLE_PAIRS,
    MIN_ELIGIBLE_TEMPORAL_COMPONENTS,
    PROTOCOL as REPORT_PROTOCOL,
    analyze,
    eligible_pairs,
    validate_report as validate_analysis_report,
)
from snmr.integration.counterfactual_eval import E71_RUNTIME_CONTRACT


MANIFEST_PROTOCOL = "E71 command-swap freeze manifest v1"
GATE_PROTOCOL = "E71 explicit feasibility gate v1"
BUNDLE_PROTOCOL = "E71 command-swap integrity bundle v1"
SMOKE_REPORT_PROTOCOL = "E71 same-state valid-command swap smoke v1"
SMOKE_AUDIT_PROTOCOL = "E71 command-swap smoke audit v1"
E70_POSTPROCESS_GATE = pathlib.Path(
    "/data/robotixx/snmr-research/e70/POSTPROCESS_COMPLETE"
)
EXPECTED_ARMS = ("explicit", "snmr")
EXPECTED_STUDENT_ARMS = {
    "explicit": "c_prior_explicit",
    "snmr": "a_prior_snmr",
}
EXPECTED_TRAINING_SEEDS = (0, 1, 2)
EXPECTED_EVALUATION_SEED = 404
EXPECTED_PAIRS = 69
EXPECTED_CELLS = EXPECTED_PAIRS * 4

GRID_FIELDS = (
    "pair_ids",
    "state_sides",
    "command_sides",
    "state_start_steps",
    "command_start_steps",
    "state_motion_ids",
    "command_motion_ids",
)
REQUIRED_FROZEN_FILES = (
    "ambiguity_precheck",
    "teacher_manifest",
    "teacher_checkpoint_0",
    "teacher_checkpoint_1",
    "motion_0",
    "motion_1",
    "evaluator",
    "reset_layer",
    "bodyfix_layer",
    "latent_runtime",
    "distillation_runtime",
    "analyzer",
    "launcher",
    "bundle_auditor",
    "protocol_document",
)
IMPORTED_RUNTIME_KEYS = (
    "evaluator",
    "reset_layer",
    "bodyfix_layer",
    "latent_runtime",
    "distillation_runtime",
    "holosoma_base_task",
    "holosoma_action_manager",
    "holosoma_joint_action_term",
    "holosoma_mujoco_simulator",
    "holosoma_wbt_manager",
    "holosoma_warp_backend",
    "holosoma_command_runtime",
    "holosoma_observation_manager",
    "holosoma_observation_terms",
    "holosoma_termination_manager",
    "holosoma_termination_terms",
    "mujoco_warp_forward",
    "mujoco_warp_io",
    "mujoco_warp_sleep",
    "mujoco_warp_types",
)
LOAD_BEARING_RUNTIME_KEYS = (
    "evaluator",
    "reset_layer",
    "bodyfix_layer",
    "latent_runtime",
    "distillation_runtime",
)
ENVIRONMENT_KEYS = (
    "python",
    "torch",
    "cuda",
    "mujoco",
    "mujoco_warp",
    "warp",
    "numpy",
    "gpu",
    "cuda_visible_devices",
    "cuda_logical_device",
    "gpu_total_memory_mb",
    "gpu_uuid",
    "mujoco_warp_types",
    "nvidia_driver",
)
ARTIFACT_PATH_KEYS = {
    "e71_root",
    "draft_manifest",
    "preregistered_manifest",
    "smoke_report",
    "smoke_audit",
    "explicit_reports",
    "snmr_reports",
    "explicit_gate",
    "analysis",
    "integrity_bundle",
    "logs_root",
    "holosoma_logs",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{7,64}")

REQUIRED_CONDITION_VALUES = {
    **EXPECTED_EVALUATION_CONDITIONS,
    "passed": True,
    "all_observation_noise": False,
    "future_only_branch_samples": True,
    "warmup_callback_order_override": True,
    "torch_deterministic": True,
    "adaptive_timestep_sampler": False,
    "terrain_spawn_randomization": False,
    "default_dof_pose_equal": True,
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _resolve_file(value: Any, *, base_dir: pathlib.Path, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a nonempty string")
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def validate_hash_record(
    record: Mapping[str, Any], *, base_dir: pathlib.Path, label: str
) -> dict[str, str]:
    """Resolve one ``path``/``sha256`` record and replay its digest."""

    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a path/hash record")
    path = _resolve_file(record.get("path"), base_dir=base_dir, label=label)
    expected = _require_sha256(record.get("sha256"), label=f"{label} sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch")
    return {"path": str(path), "sha256": observed}


def _validate_named_files(
    records: Any, *, base_dir: pathlib.Path
) -> dict[str, dict[str, str]]:
    if not isinstance(records, Mapping):
        raise ValueError("manifest frozen_files must be a mapping")
    missing = [name for name in REQUIRED_FROZEN_FILES if name not in records]
    if missing:
        raise ValueError("manifest is missing frozen files: " + ", ".join(missing))
    return {
        str(name): validate_hash_record(record, base_dir=base_dir, label=f"frozen file {name}")
        for name, record in records.items()
    }


def _validate_commits(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"snmr", "holosoma"}:
        raise ValueError(f"{label} must record exactly the SNMR and Holosoma revisions")
    result: dict[str, str] = {}
    for name in ("snmr", "holosoma"):
        revision = value.get(name)
        if not isinstance(revision, str) or _COMMIT_RE.fullmatch(revision) is None:
            raise ValueError(f"{label} {name} commit must be a hexadecimal revision")
        result[name] = revision
    return result


def _validate_working_trees(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"snmr", "holosoma"}:
        raise ValueError(f"{label} must record exactly the SNMR and Holosoma trees")
    result: dict[str, dict[str, Any]] = {}
    for name in ("snmr", "holosoma"):
        record = value.get(name)
        if not isinstance(record, Mapping) or set(record) != {
            "root",
            "tracked_changes",
            "status_sha256",
        }:
            raise ValueError(f"{label} {name} tree record has the wrong schema")
        root = record.get("root")
        changed = record.get("tracked_changes")
        status_digest = record.get("status_sha256")
        if not isinstance(root, str) or not root or not pathlib.Path(root).is_absolute():
            raise ValueError(f"{label} {name} root must be an absolute path")
        if not isinstance(changed, bool):
            raise ValueError(f"{label} {name} tracked_changes must be boolean")
        result[name] = {
            "root": str(pathlib.Path(root).resolve()),
            "tracked_changes": changed,
            "status_sha256": _require_sha256(
                status_digest, label=f"{label} {name} status sha256"
            ),
        }
    return result


def _validate_environment(
    value: Any, *, label: str, require_gpu: bool
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(ENVIRONMENT_KEYS):
        raise ValueError(f"{label} must record the complete frozen runtime environment")
    result: dict[str, str] = {}
    for name in ENVIRONMENT_KEYS:
        item = value.get(name)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} field {name} must be a nonempty string")
        result[name] = item
    if not pathlib.Path(result["mujoco_warp_types"]).is_absolute():
        raise ValueError(f"{label} mujoco_warp_types must be an absolute path")
    if require_gpu and (
        result["cuda"] in {"None", "unavailable"}
        or result["gpu"] == "unavailable"
        or result["cuda_visible_devices"] != "0"
        or result["cuda_logical_device"] != "0"
        or result["gpu_total_memory_mb"] == "unavailable"
        or result["gpu_uuid"] == "unavailable"
        or result["nvidia_driver"] == "unavailable"
    ):
        raise ValueError(f"{label} does not identify an available frozen CUDA runtime")
    return result


def _validate_confirmatory_argv(value: Any) -> dict[str, list[str]]:
    expected_keys = {
        f"{arm}_seed{seed}"
        for arm in EXPECTED_ARMS
        for seed in EXPECTED_TRAINING_SEEDS
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("manifest confirmatory argv grid must contain all six arm/seed runs")
    result: dict[str, list[str]] = {}
    for key in sorted(expected_keys):
        argv = value.get(key)
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) for item in argv)
            or not argv[0]
        ):
            raise ValueError(f"manifest confirmatory argv {key} must be a string list")
        if not pathlib.Path(argv[0]).is_absolute():
            raise ValueError(f"manifest confirmatory argv {key} must use an absolute evaluator")
        result[key] = list(argv)
    return result


def _validate_smoke_argv(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) for item in value)
        or not value[0]
    ):
        raise ValueError("manifest smoke argv must be a nonempty string list")
    if not pathlib.Path(value[0]).is_absolute():
        raise ValueError("manifest smoke argv must use an absolute evaluator")
    return list(value)


def _validate_artifact_paths(value: Any, *, smoke_pair_id: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != ARTIFACT_PATH_KEYS:
        raise ValueError("manifest artifact_paths has the wrong schema")
    result: dict[str, Any] = {}
    for name in ARTIFACT_PATH_KEYS - {"explicit_reports", "snmr_reports"}:
        path = value.get(name)
        if not isinstance(path, str) or not path or not pathlib.Path(path).is_absolute():
            raise ValueError(f"manifest artifact path {name} must be absolute")
        result[name] = str(pathlib.Path(path).resolve())
    for arm in ("explicit", "snmr"):
        key = f"{arm}_reports"
        paths = value.get(key)
        if (
            not isinstance(paths, list)
            or len(paths) != 3
            or any(not isinstance(path, str) or not pathlib.Path(path).is_absolute() for path in paths)
        ):
            raise ValueError(f"manifest artifact paths {key} must contain three absolute paths")
        result[key] = [str(pathlib.Path(path).resolve()) for path in paths]
        expected_names = [f"{arm}_seed{seed}.json" for seed in EXPECTED_TRAINING_SEEDS]
        if [pathlib.Path(path).name for path in result[key]] != expected_names:
            raise ValueError(f"manifest artifact paths {key} changed seed order or names")
    if pathlib.Path(result["smoke_report"]).name != (
        f"smoke_explicit_seed0_pair{smoke_pair_id}.json"
    ):
        raise ValueError("manifest smoke artifact name differs from the frozen pair")
    leaf_paths = [
        result["draft_manifest"],
        result["preregistered_manifest"],
        result["smoke_report"],
        result["smoke_audit"],
        *result["explicit_reports"],
        *result["snmr_reports"],
        result["explicit_gate"],
        result["analysis"],
        result["integrity_bundle"],
    ]
    if len(set(leaf_paths)) != len(leaf_paths):
        raise ValueError("manifest artifact destinations must be unique")
    e71_root = pathlib.Path(result["e71_root"])
    for name in (
        "smoke_report",
        "smoke_audit",
        "explicit_gate",
        "analysis",
        "integrity_bundle",
        "logs_root",
        "holosoma_logs",
    ):
        if not pathlib.Path(result[name]).is_relative_to(e71_root):
            raise ValueError(f"manifest artifact path {name} escapes the E71 root")
    for key in ("explicit_reports", "snmr_reports"):
        if any(not pathlib.Path(path).is_relative_to(e71_root) for path in result[key]):
            raise ValueError(f"manifest artifact paths {key} escape the E71 root")
    return result


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    base_dir: pathlib.Path,
    require_preregistered: bool = False,
) -> dict[str, Any]:
    """Validate and replay a machine-readable freeze manifest.

    ``DRAFT`` is accepted for preflight tooling, but a final bundle audit passes
    ``require_preregistered=True`` and therefore cannot certify a pre-registration run.
    """

    if manifest.get("protocol") != MANIFEST_PROTOCOL:
        raise ValueError(f"manifest protocol must be {MANIFEST_PROTOCOL!r}")
    status = manifest.get("status")
    if status not in ("DRAFT", "PREREGISTERED"):
        raise ValueError("manifest status must be DRAFT or PREREGISTERED")
    if require_preregistered and status != "PREREGISTERED":
        raise ValueError("final bundle requires a PREREGISTERED manifest")
    if manifest.get("report_protocol") != REPORT_PROTOCOL:
        raise ValueError("manifest report protocol is not the frozen E71 report protocol")
    if manifest.get("evaluation_seed") != EXPECTED_EVALUATION_SEED:
        raise ValueError("manifest evaluation seed must be 404")
    if manifest.get("training_seeds") != list(EXPECTED_TRAINING_SEEDS):
        raise ValueError("manifest training seeds must be [0, 1, 2]")
    if manifest.get("num_pairs") != EXPECTED_PAIRS:
        raise ValueError(f"manifest must freeze exactly {EXPECTED_PAIRS} pairs")
    if manifest.get("cells_per_report") != EXPECTED_CELLS:
        raise ValueError(f"manifest must freeze exactly {EXPECTED_CELLS} cells per report")
    runtime_ready = manifest.get("runtime_ready")
    if not isinstance(runtime_ready, bool):
        raise ValueError("manifest runtime_ready must be boolean")
    smoke_pair_id = manifest.get("smoke_pair_id")
    if (
        not isinstance(smoke_pair_id, int)
        or isinstance(smoke_pair_id, bool)
        or smoke_pair_id not in range(EXPECTED_PAIRS)
    ):
        raise ValueError("manifest smoke_pair_id must be an integer from 0 through 68")
    artifact_paths = _validate_artifact_paths(
        manifest.get("artifact_paths"), smoke_pair_id=smoke_pair_id
    )
    smoke_argv = _validate_smoke_argv(manifest.get("smoke_argv"))

    commits = _validate_commits(manifest.get("commits"), label="manifest")
    working_trees = _validate_working_trees(
        manifest.get("working_trees"), label="manifest"
    )
    environment = _validate_environment(
        manifest.get("environment"),
        label="manifest environment",
        require_gpu=status == "PREREGISTERED",
    )
    derived_runtime_ready = bool(
        environment["cuda"] not in {"None", "unavailable"}
        and environment["gpu"] != "unavailable"
        and environment["cuda_visible_devices"] == "0"
        and environment["cuda_logical_device"] == "0"
        and environment["gpu_total_memory_mb"] != "unavailable"
        and environment["gpu_uuid"] != "unavailable"
        and environment["nvidia_driver"] != "unavailable"
    )
    if runtime_ready is not derived_runtime_ready:
        raise ValueError("manifest runtime_ready disagrees with the frozen environment")
    if status == "PREREGISTERED" and runtime_ready is not True:
        raise ValueError("PREREGISTERED manifest must bind an available CUDA runtime")
    if runtime_ready and any(
        record["tracked_changes"] for record in working_trees.values()
    ):
        raise ValueError("runtime-ready E71 manifests require clean tracked working trees")
    confirmatory_argv = _validate_confirmatory_argv(
        manifest.get("confirmatory_argv_by_arm_seed")
    )

    evaluation_conditions = manifest.get("evaluation_conditions")
    if not isinstance(evaluation_conditions, Mapping):
        raise ValueError("manifest must freeze evaluation_conditions")
    for name, expected in REQUIRED_CONDITION_VALUES.items():
        observed = evaluation_conditions.get(name)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, int):
            matches = (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed == expected
            )
        else:
            matches = (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and float(observed) == expected
            )
        if not matches:
            raise ValueError(f"manifest evaluation condition {name} must be {expected!r}")
    full_state_contract = manifest.get("full_state_contract")
    if not isinstance(full_state_contract, Mapping):
        raise ValueError("manifest must freeze a full_state_contract")
    tensor_names = full_state_contract.get("tensor_names")
    expanded_fields = full_state_contract.get("expanded_mujoco_model_fields")
    tolerance = full_state_contract.get("tolerance")
    if (
        not isinstance(tensor_names, list)
        or not tensor_names
        or any(not isinstance(name, str) or not name for name in tensor_names)
        or len(set(tensor_names)) != len(tensor_names)
    ):
        raise ValueError("full-state tensor names must be a unique nonempty string list")
    if not isinstance(expanded_fields, list) or any(
        not isinstance(name, str) or not name for name in expanded_fields
    ) or len(set(expanded_fields)) != len(expanded_fields):
        raise ValueError("expanded MuJoCo model fields must be a string list")
    if tolerance != 1.0e-6:
        raise ValueError("full-state comparison tolerance must be 1e-6")
    if evaluation_conditions.get("expanded_mujoco_model_fields") != expanded_fields:
        raise ValueError("evaluation and full-state contracts disagree on expanded fields")
    runtime_contract = evaluation_conditions.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise ValueError("evaluation conditions must freeze the runtime contract")
    if dict(runtime_contract) != E71_RUNTIME_CONTRACT:
        raise ValueError("runtime contract differs from the registered E71 contract")
    if not str(runtime_contract.get("simulator_backend_class", "")).endswith(
        ".WarpBackend"
    ) or not str(runtime_contract.get("device", "")).startswith("cuda"):
        raise ValueError("runtime contract must identify the CUDA Warp backend")
    termination_terms = runtime_contract.get("termination_terms")
    if not isinstance(termination_terms, Mapping) or set(termination_terms) != {
        "timeout",
        "bad_tracking",
    }:
        raise ValueError("runtime contract termination terms changed")
    if (
        not isinstance(termination_terms["timeout"], Mapping)
        or termination_terms["timeout"].get("is_timeout") is not True
        or not isinstance(termination_terms["bad_tracking"], Mapping)
        or termination_terms["bad_tracking"].get("is_timeout") is not False
    ):
        raise ValueError("runtime termination timeout roles changed")

    frozen_files = _validate_named_files(manifest.get("frozen_files"), base_dir=base_dir)
    missing_runtime = [name for name in IMPORTED_RUNTIME_KEYS if name not in frozen_files]
    if missing_runtime:
        raise ValueError(
            "manifest is missing imported runtime files: " + ", ".join(missing_runtime)
        )
    if pathlib.Path(environment["mujoco_warp_types"]).resolve() != pathlib.Path(
        frozen_files["mujoco_warp_types"]["path"]
    ):
        raise ValueError("manifest environment and frozen MJWarp types path disagree")
    evaluator_path = frozen_files["evaluator"]["path"]
    if str(pathlib.Path(smoke_argv[0]).resolve()) != evaluator_path:
        raise ValueError("manifest smoke argv uses the wrong evaluator")
    try:
        smoke_logger = smoke_argv[smoke_argv.index("--logger.base-dir") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("manifest smoke argv has no logger destination") from exc
    if str(pathlib.Path(smoke_logger).resolve()) != artifact_paths["holosoma_logs"]:
        raise ValueError("manifest smoke argv and artifact logger path disagree")
    for key, argv in confirmatory_argv.items():
        if str(pathlib.Path(argv[0]).resolve()) != evaluator_path:
            raise ValueError(f"manifest confirmatory argv {key} uses the wrong evaluator")
        try:
            logger = argv[argv.index("--logger.base-dir") + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"manifest confirmatory argv {key} has no logger destination") from exc
        if str(pathlib.Path(logger).resolve()) != artifact_paths["holosoma_logs"]:
            raise ValueError(f"manifest confirmatory argv {key} and artifact path disagree")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 6:
        raise ValueError("manifest must contain exactly six student checkpoints")
    resolved_checkpoints: dict[tuple[str, int], dict[str, str]] = {}
    for index, record in enumerate(checkpoints):
        if not isinstance(record, Mapping):
            raise ValueError(f"checkpoint record {index} must be a mapping")
        arm = record.get("arm")
        seed = record.get("training_seed")
        if (
            arm not in EXPECTED_ARMS
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed not in EXPECTED_TRAINING_SEEDS
        ):
            raise ValueError(f"checkpoint record {index} has an invalid arm or seed")
        key = (str(arm), int(seed))
        if key in resolved_checkpoints:
            raise ValueError(f"manifest duplicates checkpoint {arm} seed {seed}")
        resolved_checkpoints[key] = validate_hash_record(
            record, base_dir=base_dir, label=f"{arm} seed {seed} checkpoint"
        )
    expected_keys = {
        (arm, seed) for arm in EXPECTED_ARMS for seed in EXPECTED_TRAINING_SEEDS
    }
    if set(resolved_checkpoints) != expected_keys:
        raise ValueError("manifest checkpoint arm/seed grid is incomplete")
    preregistration_state: dict[str, Any] | None = None
    preregistration = manifest.get("preregistration")
    if status == "DRAFT":
        if preregistration is not None:
            raise ValueError("DRAFT manifest cannot contain preregistration lineage")
    else:
        expected_lineage_fields = {
            "owner",
            "created_at_utc",
            "parent_draft",
            "smoke_report",
            "smoke_audit",
            "postprocess_gate",
            "capacity_gate",
        }
        if not isinstance(preregistration, Mapping) or set(preregistration) != expected_lineage_fields:
            raise ValueError("PREREGISTERED manifest has incomplete transition lineage")
        owner = preregistration.get("owner")
        created_at = preregistration.get("created_at_utc")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("preregistration owner must be nonempty")
        if not isinstance(created_at, str):
            raise ValueError("preregistration timestamp must be an ISO-8601 string")
        try:
            timestamp = dt.datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise ValueError("preregistration timestamp is not valid ISO-8601") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("preregistration timestamp must be timezone-aware")

        parent_record = validate_hash_record(
            preregistration["parent_draft"], base_dir=base_dir, label="parent DRAFT manifest"
        )
        smoke_record = validate_hash_record(
            preregistration["smoke_report"], base_dir=base_dir, label="smoke report"
        )
        audit_record = validate_hash_record(
            preregistration["smoke_audit"], base_dir=base_dir, label="smoke audit"
        )
        postprocess_record = validate_hash_record(
            preregistration["postprocess_gate"],
            base_dir=base_dir,
            label="cross-project postprocess gate",
        )
        if pathlib.Path(postprocess_record["path"]) != E70_POSTPROCESS_GATE.resolve():
            raise ValueError(
                "preregistration must bind the canonical E70 postprocess gate"
            )
        capacity = preregistration.get("capacity_gate")
        if (
            not isinstance(capacity, Mapping)
            or set(capacity) != {"required_free_mb", "observed_free_mb"}
            or not isinstance(capacity.get("required_free_mb"), int)
            or isinstance(capacity.get("required_free_mb"), bool)
            or capacity["required_free_mb"] < 26_000
            or not isinstance(capacity.get("observed_free_mb"), int)
            or isinstance(capacity.get("observed_free_mb"), bool)
            or capacity["observed_free_mb"] < capacity["required_free_mb"]
        ):
            raise ValueError("preregistration capacity gate must prove at least 26,000 MiB")

        parent_path = pathlib.Path(parent_record["path"])
        smoke_path = pathlib.Path(smoke_record["path"])
        audit_path = pathlib.Path(audit_record["path"])
        expected_lineage_paths = {
            "parent DRAFT manifest": artifact_paths["draft_manifest"],
            "smoke report": artifact_paths["smoke_report"],
            "smoke audit": artifact_paths["smoke_audit"],
        }
        observed_lineage_paths = {
            "parent DRAFT manifest": str(parent_path.resolve()),
            "smoke report": str(smoke_path.resolve()),
            "smoke audit": str(audit_path.resolve()),
        }
        for label, expected_path in expected_lineage_paths.items():
            if observed_lineage_paths[label] != expected_path:
                raise ValueError(f"preregistration {label} is outside its registered path")
        parent_manifest = json.loads(parent_path.read_text())
        parent_state = validate_manifest(
            parent_manifest, base_dir=parent_path.parent, require_preregistered=False
        )
        if parent_manifest.get("status") != "DRAFT" or parent_state["runtime_ready"] is not True:
            raise ValueError("preregistration parent must be a runtime-ready DRAFT manifest")
        expected_smoke_audit = validate_smoke_report(parent_path, smoke_path)
        observed_smoke_audit = json.loads(audit_path.read_text())
        if observed_smoke_audit != expected_smoke_audit:
            raise ValueError("preregistration smoke audit differs from independent recomputation")

        immutable_fields = (
            "report_protocol",
            "evaluation_seed",
            "training_seeds",
            "num_pairs",
            "cells_per_report",
            "runtime_ready",
            "artifact_paths",
            "smoke_pair_id",
            "smoke_argv",
            "commits",
            "working_trees",
            "environment",
            "evaluation_conditions",
            "full_state_contract",
            "frozen_files",
            "checkpoints",
            "confirmatory_argv_by_arm_seed",
        )
        changed = [
            name for name in immutable_fields if manifest.get(name) != parent_manifest.get(name)
        ]
        if changed:
            raise ValueError(
                "PREREGISTERED manifest changed DRAFT content: " + ", ".join(changed)
            )
        preregistration_state = {
            "owner": owner.strip(),
            "created_at_utc": created_at,
            "parent_draft": parent_record,
            "smoke_report": smoke_record,
            "smoke_audit": audit_record,
            "postprocess_gate": postprocess_record,
            "capacity_gate": dict(capacity),
        }
    return {
        "frozen_files": frozen_files,
        "checkpoints": resolved_checkpoints,
        "commits": commits,
        "working_trees": working_trees,
        "environment": environment,
        "runtime_ready": runtime_ready,
        "artifact_paths": artifact_paths,
        "smoke_pair_id": smoke_pair_id,
        "smoke_argv": smoke_argv,
        "confirmatory_argv_by_arm_seed": confirmatory_argv,
        "evaluation_conditions": dict(evaluation_conditions),
        "full_state_contract": dict(full_state_contract),
        "preregistration": preregistration_state,
    }


def _integer_report_vector(report: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in report:
        raise ValueError(f"report is missing required field {key}")
    raw = report.get(key)
    if not isinstance(raw, list) or len(raw) != EXPECTED_CELLS:
        raise ValueError(f"report field {key} must be a {EXPECTED_CELLS}-cell vector")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in raw):
        raise ValueError(f"report field {key} must contain only integers")
    return np.asarray(raw, dtype=np.int64)


def _float_report_vector(report: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in report:
        raise ValueError(f"report is missing required field {key}")
    raw = report.get(key)
    if not isinstance(raw, list) or len(raw) != EXPECTED_CELLS:
        raise ValueError(f"report field {key} must be a {EXPECTED_CELLS}-cell vector")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool) for value in raw
    ):
        raise ValueError(f"report field {key} must contain only numbers")
    return np.asarray(raw, dtype=np.float64)


def _path_hash_pair(
    report: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    base_dir: pathlib.Path,
    label: str,
) -> dict[str, str]:
    return validate_hash_record(
        {"path": report.get(path_key), "sha256": report.get(hash_key)},
        base_dir=base_dir,
        label=label,
    )


def _path_hash_list(
    report: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    expected_length: int,
    base_dir: pathlib.Path,
    label: str,
) -> list[dict[str, str]]:
    paths = report.get(path_key)
    hashes = report.get(hash_key)
    if (
        not isinstance(paths, list)
        or not isinstance(hashes, list)
        or len(paths) != expected_length
        or len(hashes) != expected_length
    ):
        raise ValueError(f"report {label} paths and hashes must have length {expected_length}")
    return [
        validate_hash_record(
            {"path": path, "sha256": digest},
            base_dir=base_dir,
            label=f"{label} {index}",
        )
        for index, (path, digest) in enumerate(zip(paths, hashes, strict=True))
    ]


def validate_rollout_report(
    report: Mapping[str, Any],
    *,
    expected_arm: str,
    expected_seed: int,
    base_dir: pathlib.Path,
) -> dict[str, Any]:
    """Validate one complete 276-cell report and replay all of its input hashes."""

    if not isinstance(report, dict):
        raise ValueError("rollout report must be a JSON object")
    validate_analysis_report(report)
    if report.get("status") != "complete":
        raise ValueError("rollout report status must be complete")
    if report.get("arm") != expected_arm:
        raise ValueError(f"report arm must be {expected_arm!r}")
    if report.get("student_arm") != EXPECTED_STUDENT_ARMS[expected_arm]:
        raise ValueError(
            f"report student arm must be {EXPECTED_STUDENT_ARMS[expected_arm]!r}"
        )
    if report.get("training_seed") != expected_seed:
        raise ValueError(f"{expected_arm} report training seed must be {expected_seed}")
    if report.get("evaluation_seed") != EXPECTED_EVALUATION_SEED:
        raise ValueError("report evaluation seed must be 404")
    if report.get("num_rollouts") != EXPECTED_CELLS:
        raise ValueError(f"report num_rollouts must be {EXPECTED_CELLS}")
    if report.get("smoke_pair_id") is not None:
        raise ValueError("a smoke report cannot enter the confirmatory bundle")
    if report.get("branch_samples_0p5_s") != 25 or report.get("branch_samples_1p0_s") != 50:
        raise ValueError("branch endpoints must use 25 and 50 future-only samples")

    integer_vectors = {
        key: _integer_report_vector(report, key)
        for key in (
            "pair_ids",
            "state_sides",
            "command_sides",
            "state_start_steps",
            "command_start_steps",
            "state_motion_ids",
            "command_motion_ids",
        )
    }
    float_vectors = {
        key: _float_report_vector(report, key)
        for key in (
            "q_a_0p5_s",
            "q_b_0p5_s",
            "q_ab_0p5_s",
            "branch_coordinate_0p5_s",
            "d_a_0p5_s",
            "d_b_0p5_s",
            "q_a_1p0_s",
            "q_b_1p0_s",
            "q_ab_1p0_s",
            "branch_coordinate_1p0_s",
            "d_a_1p0_s",
            "d_b_1p0_s",
            "survival_s",
        )
    }
    goal_scale = report.get("goal_scale")
    if (
        not isinstance(goal_scale, list)
        or len(goal_scale) != 58
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0.0
            for value in goal_scale
        )
    ):
        raise ValueError("goal_scale must contain 58 finite positive values")
    raw_completed = report.get("completed")
    if not isinstance(raw_completed, list) or len(raw_completed) != EXPECTED_CELLS:
        raise ValueError(f"completed must be a {EXPECTED_CELLS}-cell vector")
    if any(
        not isinstance(value, (bool, int))
        or (isinstance(value, int) and value not in (0, 1))
        for value in raw_completed
    ):
        raise ValueError("completed must contain only booleans or 0/1 values")
    if set(np.unique(integer_vectors["pair_ids"])) != set(range(EXPECTED_PAIRS)):
        raise ValueError("report pair IDs must be exactly 0 through 68")
    if np.any(integer_vectors["state_start_steps"] < 0) or np.any(
        integer_vectors["command_start_steps"] < 0
    ):
        raise ValueError("report cursors must be nonnegative")
    if not np.array_equal(
        integer_vectors["state_motion_ids"], integer_vectors["state_sides"]
    ):
        raise ValueError("state motion IDs do not match state sides")
    if not np.array_equal(
        integer_vectors["command_motion_ids"], integer_vectors["command_sides"]
    ):
        raise ValueError("command motion IDs do not match command sides")

    for horizon in ("0p5_s", "1p0_s"):
        q_a = float_vectors[f"q_a_{horizon}"]
        q_b = float_vectors[f"q_b_{horizon}"]
        q_ab = float_vectors[f"q_ab_{horizon}"]
        coordinate = float_vectors[f"branch_coordinate_{horizon}"]
        d_a = float_vectors[f"d_a_{horizon}"]
        d_b = float_vectors[f"d_b_{horizon}"]
        if any(
            not np.all(np.isfinite(values))
            for values in (q_a, q_b, q_ab, coordinate, d_a, d_b)
        ):
            raise ValueError(f"branch metrics at {horizon} must be finite")
        if np.any(q_a < 0.0) or np.any(q_b < 0.0) or np.any(q_ab <= 0.0):
            raise ValueError(f"branch squared errors at {horizon} must be nonnegative with q_ab > 0")
        expected_coordinate = (q_a - q_b) / (q_ab + 1.0e-8)
        if not np.allclose(coordinate, expected_coordinate, rtol=1.0e-7, atol=1.0e-9):
            raise ValueError(
                f"branch coordinate at {horizon} is inconsistent with (q_a-q_b)/(q_ab+1e-8)"
            )
        if not np.allclose(d_a, np.sqrt(q_a), rtol=1.0e-7, atol=1.0e-9) or not np.allclose(
            d_b, np.sqrt(q_b), rtol=1.0e-7, atol=1.0e-9
        ):
            raise ValueError(f"branch distances at {horizon} are inconsistent with sqrt(q)")
        legacy_key = f"branch_score_{horizon}"
        if legacy_key in report:
            legacy_score = _float_report_vector(report, legacy_key)
            if not np.allclose(legacy_score, d_a - d_b, rtol=1.0e-7, atol=1.0e-9):
                raise ValueError(
                    f"legacy branch score at {horizon} is inconsistent with d_a - d_b"
                )
        for pair_id in range(EXPECTED_PAIRS):
            pair_q_ab = q_ab[integer_vectors["pair_ids"] == pair_id]
            if not np.allclose(pair_q_ab, pair_q_ab[0], rtol=0.0, atol=1.0e-12):
                raise ValueError(f"pair {pair_id} q_ab at {horizon} changed across cells")
    survival = float_vectors["survival_s"]
    if not np.all(np.isfinite(survival)) or np.any(survival < 0.0) or np.any(survival > 10.0):
        raise ValueError("survival_s must be finite and within the registered 10-second horizon")

    pair_ids = integer_vectors["pair_ids"]
    states = integer_vectors["state_sides"]
    commands = integer_vectors["command_sides"]
    state_steps = integer_vectors["state_start_steps"]
    command_steps = integer_vectors["command_start_steps"]
    for pair_id in range(EXPECTED_PAIRS):
        side_steps: dict[int, int] = {}
        command_side_steps: dict[int, int] = {}
        for side in (0, 1):
            state_values = np.unique(state_steps[(pair_ids == pair_id) & (states == side)])
            command_values = np.unique(
                command_steps[(pair_ids == pair_id) & (commands == side)]
            )
            if len(state_values) != 1 or len(command_values) != 1:
                raise ValueError(f"pair {pair_id} does not have fixed cursors by side")
            side_steps[side] = int(state_values[0])
            command_side_steps[side] = int(command_values[0])
        if side_steps != command_side_steps:
            raise ValueError(f"pair {pair_id} state and command branch cursors disagree")

    warmup = report.get("warmup_audit")
    if not isinstance(warmup, Mapping) or warmup.get("passed") is not True:
        raise ValueError("warm-up reset audit did not pass")
    expected_warmup = {
        "reset_count": 0,
        "command_dependent_reset": False,
        "episode_length_min": 1,
        "episode_length_max": 1,
        "callback_order_before_warmup": False,
        "callback_order_during_warmup": True,
        "callback_order_after_warmup": False,
    }
    for name, expected in expected_warmup.items():
        if warmup.get(name) != expected:
            raise ValueError(f"warm-up audit field {name} must be {expected!r}")
    if warmup.get("command_dependent_reset") not in (None, False):
        raise ValueError("warm-up audit records a command-dependent reset")
    raw_proprio = report.get("raw_proprio_audit")
    if not isinstance(raw_proprio, Mapping) or raw_proprio.get("passed") is not True:
        raise ValueError("raw-proprioception same-state audit did not pass")
    for audit_name, audit in (
        ("normalized proprioception", report.get("proprio_audit")),
        ("raw proprioception", raw_proprio),
    ):
        if not isinstance(audit, Mapping):
            raise ValueError(f"{audit_name} audit is missing")
        if audit.get("num_state_comparisons") != EXPECTED_PAIRS * 2:
            raise ValueError(f"{audit_name} audit must contain 138 state comparisons")
        maximum = audit.get("max_abs_difference")
        tolerance = audit.get("tolerance")
        if (
            not isinstance(maximum, (int, float))
            or not np.isfinite(maximum)
            or tolerance != 1.0e-6
            or maximum > tolerance
            or audit.get("passed") is not True
        ):
            raise ValueError(f"{audit_name} audit is internally inconsistent")

    full_state_contract = report.get("full_state_contract")
    if not isinstance(full_state_contract, Mapping):
        raise ValueError("report is missing its frozen full-state contract")
    full_state = report.get("full_state_audit")
    if not isinstance(full_state, Mapping) or full_state.get("passed") is not True:
        raise ValueError("full simulator/policy state audit did not pass")
    tensor_names = full_state_contract.get("tensor_names")
    tolerance = full_state_contract.get("tolerance")
    per_tensor = full_state.get("per_tensor")
    if not isinstance(tensor_names, list) or not isinstance(per_tensor, Mapping):
        raise ValueError("full-state audit has no tensor contract")
    if set(per_tensor) != set(tensor_names):
        raise ValueError("full-state audit tensors differ from its frozen contract")
    if full_state.get("num_tensors") != len(tensor_names):
        raise ValueError("full-state audit tensor count is inconsistent")
    if full_state.get("num_tensor_state_comparisons") != len(tensor_names) * EXPECTED_PAIRS * 2:
        raise ValueError("full-state audit comparison count is inconsistent")
    maxima: list[float] = []
    for name in tensor_names:
        item = per_tensor[name]
        if not isinstance(item, Mapping) or item.get("num_state_comparisons") != EXPECTED_PAIRS * 2:
            raise ValueError(f"full-state tensor {name} has an invalid comparison count")
        maximum = item.get("max_abs_difference")
        if (
            not isinstance(maximum, (int, float))
            or not np.isfinite(maximum)
            or maximum > tolerance
            or item.get("passed") is not True
        ):
            raise ValueError(f"full-state tensor {name} did not pass")
        maxima.append(float(maximum))
    overall = full_state.get("max_abs_difference")
    if (
        full_state.get("tolerance") != tolerance
        or not isinstance(overall, (int, float))
        or not np.isfinite(overall)
        or not np.isclose(overall, max(maxima), rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("full-state aggregate audit is internally inconsistent")
    overflow = report.get("overflow_audit")
    if (
        not isinstance(overflow, Mapping)
        or overflow.get("passed") is not True
        or overflow.get("checked_transitions") != 501
        or overflow.get("nonzero_entries") != 0
    ):
        raise ValueError("MJWarp overflow audit did not cover warm-up and all rollout steps")
    conditions = report.get("evaluation_conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("report is missing registered evaluation conditions")
    termination = report.get("termination_audit")
    if not isinstance(termination, Mapping) or termination.get("passed") is not True:
        raise ValueError("command-independent termination audit did not pass")
    if (
        termination.get("primary_horizon_done_count") != 0
        or termination.get("suppressed_steps") != 50
        or termination.get("reference_termination_restored_after_primary_horizon") is not True
    ):
        raise ValueError("termination audit did not prove an uncensored primary horizon")
    expected_terms = [
        "actions",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
        "motion_command",
        "motion_ref_ori_b",
    ]
    layout = report.get("observation_layout_audit")
    if (
        not isinstance(layout, Mapping)
        or layout.get("passed") is not True
        or layout.get("configured_terms_sorted") != expected_terms
        or layout.get("expected_terms_sorted") != expected_terms
        or layout.get("observed_width") != 154
        or layout.get("expected_width") != 154
        or layout.get("tolerance") != 1.0e-6
        or layout.get("configuration")
        != E71_RUNTIME_CONTRACT["actor_observation_contract"]
        or not isinstance(layout.get("max_abs_difference"), (int, float))
        or not np.isfinite(layout["max_abs_difference"])
        or layout["max_abs_difference"] > layout["tolerance"]
    ):
        raise ValueError("semantic actor-observation layout audit did not pass")
    latent_route = report.get("latent_route_audit")
    if not isinstance(latent_route, Mapping) or latent_route.get("passed") is not True:
        raise ValueError("latent command-route audit did not pass")
    if latent_route.get("tolerance") != 1.0e-6:
        raise ValueError("latent command-route tolerance changed")
    for name in (
        "direct_lookup_max_abs_difference",
        "same_command_across_state_max_abs_difference",
    ):
        value = latent_route.get(name)
        if (
            not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
            or value > latent_route["tolerance"]
        ):
            raise ValueError(f"latent command-route field {name} failed")

    provenance = {
        "student_checkpoint": _path_hash_pair(
            report,
            path_key="student_checkpoint",
            hash_key="student_checkpoint_sha256",
            base_dir=base_dir,
            label=f"{expected_arm} seed {expected_seed} student checkpoint",
        ),
        "teacher_manifest": _path_hash_pair(
            report,
            path_key="teacher_manifest",
            hash_key="teacher_manifest_sha256",
            base_dir=base_dir,
            label="teacher manifest",
        ),
        "teacher_checkpoints": _path_hash_list(
            report,
            path_key="teacher_ckpts",
            hash_key="teacher_checkpoint_sha256",
            expected_length=2,
            base_dir=base_dir,
            label="teacher checkpoint",
        ),
        "motions": _path_hash_list(
            report,
            path_key="motion_files",
            hash_key="motion_sha256",
            expected_length=2,
            base_dir=base_dir,
            label="motion",
        ),
        "ambiguity_precheck": _path_hash_pair(
            report,
            path_key="ambiguity_precheck",
            hash_key="ambiguity_precheck_sha256",
            base_dir=base_dir,
            label="ambiguity precheck",
        ),
        "runtime": _path_hash_pair(
            report,
            path_key="runtime",
            hash_key="runtime_sha256",
            base_dir=base_dir,
            label="evaluator runtime",
        ),
        "reset_runtime": _path_hash_pair(
            report,
            path_key="reset_runtime",
            hash_key="reset_runtime_sha256",
            base_dir=base_dir,
            label="reset runtime",
        ),
        "freeze_manifest": _path_hash_pair(
            report,
            path_key="freeze_manifest",
            hash_key="freeze_manifest_sha256",
            base_dir=base_dir,
            label="freeze manifest",
        ),
    }
    invoked_argv = report.get("invoked_argv")
    if (
        not isinstance(invoked_argv, list)
        or not invoked_argv
        or any(not isinstance(item, str) for item in invoked_argv)
        or not invoked_argv[0]
    ):
        raise ValueError("report invoked_argv must be a string list")
    imported_paths = report.get("imported_runtime_paths")
    imported_hashes = report.get("imported_runtime_sha256")
    if (
        not isinstance(imported_paths, Mapping)
        or not isinstance(imported_hashes, Mapping)
        or set(imported_paths) != set(IMPORTED_RUNTIME_KEYS)
        or set(imported_hashes) != set(IMPORTED_RUNTIME_KEYS)
    ):
        raise ValueError("report must bind the exact imported-runtime path/hash set")
    imported_runtime: dict[str, dict[str, str]] = {}
    for name in IMPORTED_RUNTIME_KEYS:
        imported_runtime[name] = validate_hash_record(
            {"path": imported_paths[name], "sha256": imported_hashes[name]},
            base_dir=base_dir,
            label=f"imported runtime {name}",
        )

    repository_commits = _validate_commits(
        report.get("repository_commits"), label="report"
    )
    working_trees = _validate_working_trees(
        report.get("working_trees"), label="report"
    )
    runtime_environment = _validate_environment(
        report.get("runtime_environment"),
        label="report runtime environment",
        require_gpu=True,
    )

    toctou = report.get("runtime_toctou_audit")
    if not isinstance(toctou, Mapping) or toctou.get("passed") is not True:
        raise ValueError("runtime TOCTOU audit did not pass")

    def digest_map(name: str) -> dict[str, str]:
        value = toctou.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"runtime TOCTOU audit is missing {name}")
        return {
            str(key): _require_sha256(digest, label=f"TOCTOU {name}.{key}")
            for key, digest in value.items()
        }

    runtime_start = digest_map("startup_sha256")
    runtime_end = digest_map("completion_sha256")
    all_start = digest_map("all_frozen_startup_sha256")
    all_end = digest_map("all_frozen_completion_sha256")
    if runtime_start != runtime_end or all_start != all_end:
        raise ValueError("runtime or frozen input changed during evaluation")
    expected_runtime = {
        name: imported_runtime[name]["sha256"] for name in LOAD_BEARING_RUNTIME_KEYS
    }
    if runtime_start != expected_runtime:
        raise ValueError("runtime TOCTOU map differs from the imported load-bearing runtimes")
    expected_all = {
        "freeze_manifest": provenance["freeze_manifest"]["sha256"],
        "student_checkpoint": provenance["student_checkpoint"]["sha256"],
        "teacher_manifest": provenance["teacher_manifest"]["sha256"],
        "teacher_checkpoint_0": provenance["teacher_checkpoints"][0]["sha256"],
        "teacher_checkpoint_1": provenance["teacher_checkpoints"][1]["sha256"],
        "motion_0": provenance["motions"][0]["sha256"],
        "motion_1": provenance["motions"][1]["sha256"],
        "ambiguity_precheck": provenance["ambiguity_precheck"]["sha256"],
        **{
            f"runtime.{name}": imported_runtime[name]["sha256"]
            for name in IMPORTED_RUNTIME_KEYS
        },
    }
    if all_start != expected_all:
        raise ValueError("all-frozen TOCTOU map differs from report provenance")

    return {
        "grid": {key: integer_vectors[key].tolist() for key in GRID_FIELDS},
        "provenance": provenance,
        "evaluation_conditions": dict(conditions),
        "full_state_contract": dict(full_state_contract),
        "reference_metrics": {
            "goal_scale": list(goal_scale),
            "q_ab_0p5_s": float_vectors["q_ab_0p5_s"].tolist(),
            "q_ab_1p0_s": float_vectors["q_ab_1p0_s"].tolist(),
        },
        "runtime_contracts": {
            "observation_layout_audit": dict(layout),
            "latent_route_audit": dict(latent_route),
        },
        "execution_provenance": {
            "invoked_argv": list(invoked_argv),
            "imported_runtime": imported_runtime,
            "repository_commits": repository_commits,
            "working_trees": working_trees,
            "runtime_environment": runtime_environment,
        },
    }


def _assert_same_record(
    observed: Mapping[str, str], expected: Mapping[str, str], *, label: str
) -> None:
    if dict(observed) != dict(expected):
        raise ValueError(f"{label} does not match the freeze manifest")


def _validate_report_groups(
    groups: Sequence[
        tuple[str, Sequence[tuple[pathlib.Path, dict[str, Any]]]]
    ],
    *,
    manifest_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate one or more complete arm groups against one frozen manifest."""

    if not groups:
        raise ValueError("at least one report arm is required")
    ordered: list[dict[str, Any]] = []
    reference_grid: dict[str, list[int]] | None = None
    shared_provenance: dict[str, Any] | None = None
    shared_reference_metrics: dict[str, Any] | None = None
    shared_runtime_contracts: dict[str, Any] | None = None
    frozen = manifest_state["frozen_files"]
    checkpoints = manifest_state["checkpoints"]
    for arm, entries in groups:
        if arm not in EXPECTED_ARMS or len(entries) != 3:
            raise ValueError(f"{arm} requires exactly three frozen reports")
        observed_seeds: list[int] = []
        for path, report in entries:
            seed = report.get("training_seed")
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError(f"{arm} report training seed must be an integer")
            if seed not in EXPECTED_TRAINING_SEEDS:
                raise ValueError(f"{arm} report training seed must be 0, 1, or 2")
            expected_path = manifest_state["artifact_paths"][f"{arm}_reports"][seed]
            if str(path.resolve()) != expected_path:
                raise ValueError(f"{arm} seed {seed} report path differs from the manifest")
            observed_seeds.append(seed)
            validated = validate_rollout_report(
                report, expected_arm=arm, expected_seed=seed, base_dir=path.parent
            )
            if reference_grid is None:
                reference_grid = validated["grid"]
            elif validated["grid"] != reference_grid:
                raise ValueError("explicit and SNMR reports do not share the exact frozen grid")
            if validated["evaluation_conditions"] != manifest_state["evaluation_conditions"]:
                raise ValueError("report evaluation conditions differ from the freeze manifest")
            if validated["full_state_contract"] != manifest_state["full_state_contract"]:
                raise ValueError("report full-state contract differs from the freeze manifest")
            if shared_reference_metrics is None:
                shared_reference_metrics = validated["reference_metrics"]
            elif validated["reference_metrics"] != shared_reference_metrics:
                raise ValueError("reports do not share identical reference-only metrics")
            if shared_runtime_contracts is None:
                shared_runtime_contracts = validated["runtime_contracts"]
            elif validated["runtime_contracts"] != shared_runtime_contracts:
                raise ValueError("reports do not share identical runtime contracts")

            execution = validated["execution_provenance"]
            argv_key = f"{arm}_seed{seed}"
            if (
                execution["invoked_argv"]
                != manifest_state["confirmatory_argv_by_arm_seed"][argv_key]
            ):
                raise ValueError(f"{arm} seed {seed} argv differs from the freeze manifest")
            if execution["repository_commits"] != manifest_state["commits"]:
                raise ValueError("report repository revisions differ from the freeze manifest")
            if execution["working_trees"] != manifest_state["working_trees"]:
                raise ValueError("report working-tree state differs from the freeze manifest")
            if execution["runtime_environment"] != manifest_state["environment"]:
                raise ValueError("report runtime environment differs from the freeze manifest")
            for name in IMPORTED_RUNTIME_KEYS:
                _assert_same_record(
                    execution["imported_runtime"][name],
                    frozen[name],
                    label=f"imported runtime {name}",
                )

            provenance = validated["provenance"]
            _assert_same_record(
                provenance["student_checkpoint"],
                checkpoints[(arm, seed)],
                label=f"{arm} seed {seed} checkpoint",
            )
            common = {
                "teacher_manifest": provenance["teacher_manifest"],
                "teacher_checkpoints": provenance["teacher_checkpoints"],
                "motions": provenance["motions"],
                "ambiguity_precheck": provenance["ambiguity_precheck"],
                "runtime": provenance["runtime"],
                "reset_runtime": provenance["reset_runtime"],
                "freeze_manifest": provenance["freeze_manifest"],
            }
            if shared_provenance is None:
                shared_provenance = common
            elif common != shared_provenance:
                raise ValueError("rollout reports do not share identical frozen provenance")
            ordered.append({"path": path.resolve(), "report": report})
        if observed_seeds != list(EXPECTED_TRAINING_SEEDS):
            raise ValueError(f"{arm} reports must be ordered as seeds 0, 1, and 2")

    assert shared_provenance is not None
    expected_common = {
        "teacher_manifest": frozen["teacher_manifest"],
        "teacher_checkpoints": [frozen["teacher_checkpoint_0"], frozen["teacher_checkpoint_1"]],
        "motions": [frozen["motion_0"], frozen["motion_1"]],
        "ambiguity_precheck": frozen["ambiguity_precheck"],
        "runtime": frozen["evaluator"],
        "reset_runtime": frozen["reset_layer"],
    }
    if "freeze_manifest" in manifest_state:
        expected_common["freeze_manifest"] = manifest_state["freeze_manifest"]
    else:
        expected_common["freeze_manifest"] = shared_provenance["freeze_manifest"]
    if shared_provenance != expected_common:
        raise ValueError("rollout provenance does not match the freeze manifest")
    return ordered


def validate_report_set(
    explicit_reports: Sequence[tuple[pathlib.Path, dict[str, Any]]],
    snmr_reports: Sequence[tuple[pathlib.Path, dict[str, Any]]],
    *,
    manifest_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate six reports, their provenance, and their shared intervention grid."""

    if len(explicit_reports) != 3 or len(snmr_reports) != 3:
        raise ValueError("bundle requires exactly three explicit and three SNMR reports")
    return _validate_report_groups(
        (("explicit", explicit_reports), ("snmr", snmr_reports)),
        manifest_state=manifest_state,
    )


def validate_explicit_report_set(
    manifest_path: pathlib.Path,
    explicit_paths: Sequence[pathlib.Path],
) -> list[dict[str, Any]]:
    """Certify all explicit reports before the gate or any SNMR rollout.

    This is intentionally independent of gate analysis.  It requires the final
    ``PREREGISTERED`` manifest, replays all hashes, and verifies the seed-ordered
    explicit reports against the exact frozen grid, conditions, state contract,
    provenance, and arm-specific checkpoints.
    """

    manifest_path = pathlib.Path(manifest_path)
    explicit_paths = [pathlib.Path(path) for path in explicit_paths]
    if len(explicit_paths) != 3:
        raise ValueError("explicit preflight requires exactly three reports")
    all_paths = [manifest_path, *explicit_paths]
    if len({path.resolve() for path in all_paths}) != len(all_paths):
        raise ValueError("explicit preflight artifact paths must be unique")
    for path in all_paths:
        if not path.is_file():
            raise ValueError(f"required explicit-preflight artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text())
    manifest_state = validate_manifest(
        manifest, base_dir=manifest_path.parent, require_preregistered=True
    )
    if str(manifest_path.resolve()) != manifest_state["artifact_paths"][
        "preregistered_manifest"
    ]:
        raise ValueError("PREREGISTERED manifest path differs from its artifact contract")
    manifest_state["freeze_manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
    }
    entries = [(path, json.loads(path.read_text())) for path in explicit_paths]
    _validate_report_groups(
        (("explicit", entries),), manifest_state=manifest_state
    )
    reports = [report for _, report in entries]
    # Reproduce the temporal partition here so malformed precheck/cursor alignment
    # cannot survive preflight and fail only after the gate artifact is attempted.
    eligible_pairs(reports)
    return reports


def _smoke_integer_vector(report: Mapping[str, Any], key: str) -> np.ndarray:
    raw = report.get(key)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"smoke field {key} must be an exact four-cell vector")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in raw):
        raise ValueError(f"smoke field {key} must contain only integers")
    return np.asarray(raw, dtype=np.int64)


def _smoke_float_vector(report: Mapping[str, Any], key: str) -> np.ndarray:
    raw = report.get(key)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"smoke field {key} must be an exact four-cell vector")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        for value in raw
    ):
        raise ValueError(f"smoke field {key} must contain only finite numbers")
    return np.asarray(raw, dtype=np.float64)


def _smoke_finite_matrix(
    report: Mapping[str, Any], key: str, *, rows: int, columns: int
) -> None:
    raw = report.get(key)
    if (
        not isinstance(raw, list)
        or len(raw) != rows
        or any(not isinstance(row, list) or len(row) != columns for row in raw)
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            for row in raw
            for value in row
        )
    ):
        raise ValueError(f"smoke field {key} must be a finite {rows}x{columns} matrix")


def _registered_smoke_branch_steps(
    manifest_state: Mapping[str, Any], *, pair_id: int, fps: float
) -> list[int]:
    """Rebuild the two global branch cursors from the frozen E70 inputs."""

    frozen = manifest_state["frozen_files"]
    precheck_path = pathlib.Path(frozen["ambiguity_precheck"]["path"])
    try:
        precheck = json.loads(precheck_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen ambiguity precheck is not valid JSON") from exc
    if precheck.get("protocol") != "E70 reference-only ambiguity precheck v1":
        raise ValueError("frozen ambiguity precheck has the wrong protocol")
    preferred_pair = precheck.get("preferred_pair")
    pairs = precheck.get("pairs")
    pair = pairs.get(preferred_pair) if isinstance(pairs, Mapping) else None
    windows = pair.get("windows") if isinstance(pair, Mapping) else None
    if not isinstance(windows, list) or len(windows) != EXPECTED_PAIRS:
        raise ValueError("frozen ambiguity precheck must contain exactly 69 windows")
    window = windows[pair_id]
    if not isinstance(window, Mapping):
        raise ValueError("registered smoke window must be a mapping")
    local_steps: list[int] = []
    for name in ("time_seconds_first", "time_seconds_second"):
        value = window.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"registered smoke window {name} is invalid")
        local_steps.append(int(round(float(value) * fps)))

    frame_counts: list[int] = []
    for side in (0, 1):
        motion_path = pathlib.Path(frozen[f"motion_{side}"]["path"])
        try:
            with np.load(motion_path, allow_pickle=False) as motion:
                if "joint_pos" not in motion or motion["joint_pos"].ndim < 1:
                    raise ValueError("motion has no frame-aligned joint_pos")
                frame_count = int(motion["joint_pos"].shape[0])
                if "fps" not in motion:
                    raise ValueError("motion has no sampling rate")
                motion_fps = np.asarray(motion["fps"], dtype=np.float64).reshape(-1)
        except (OSError, ValueError) as exc:
            raise ValueError(f"frozen smoke motion {side} is not a valid NPZ motion") from exc
        if (
            frame_count <= 0
            or motion_fps.size != 1
            or not np.isfinite(motion_fps[0])
            or float(motion_fps[0]) != fps
        ):
            raise ValueError(f"frozen smoke motion {side} changed its 50 Hz contract")
        frame_counts.append(frame_count)
    if any(
        step < 1 or step + 500 >= frame_count
        for step, frame_count in zip(local_steps, frame_counts, strict=True)
    ):
        raise ValueError("registered smoke branch cannot support the full rollout horizon")
    return [local_steps[0], frame_counts[0] + local_steps[1]]


def _validate_smoke_audits(
    report: Mapping[str, Any], *, manifest_state: Mapping[str, Any]
) -> dict[str, Any]:
    warmup = report.get("warmup_audit")
    expected_warmup = {
        "passed": True,
        "reset_count": 0,
        "command_dependent_reset": False,
        "episode_length_min": 1,
        "episode_length_max": 1,
        "callback_order_before_warmup": False,
        "callback_order_during_warmup": True,
        "callback_order_after_warmup": False,
    }
    if not isinstance(warmup, Mapping) or any(
        warmup.get(name) != expected for name, expected in expected_warmup.items()
    ):
        raise ValueError("smoke warm-up audit did not prove a command-independent reset")

    for audit_name, audit in (
        ("normalized proprioception", report.get("proprio_audit")),
        ("raw proprioception", report.get("raw_proprio_audit")),
    ):
        if not isinstance(audit, Mapping):
            raise ValueError(f"smoke {audit_name} audit is missing")
        maximum = audit.get("max_abs_difference")
        if (
            audit.get("passed") is not True
            or audit.get("num_state_comparisons") != 2
            or audit.get("tolerance") != 1.0e-6
            or not isinstance(maximum, (int, float))
            or isinstance(maximum, bool)
            or not np.isfinite(maximum)
            or maximum < 0.0
            or maximum > 1.0e-6
        ):
            raise ValueError(
                f"smoke {audit_name} audit must pass exactly two physical-state comparisons"
            )

    full_state_contract = report.get("full_state_contract")
    if full_state_contract != manifest_state["full_state_contract"]:
        raise ValueError("smoke full-state contract differs from the DRAFT manifest")
    tensor_names = manifest_state["full_state_contract"]["tensor_names"]
    tolerance = manifest_state["full_state_contract"]["tolerance"]
    full_state = report.get("full_state_audit")
    if not isinstance(full_state, Mapping) or full_state.get("passed") is not True:
        raise ValueError("smoke full-state audit did not pass")
    per_tensor = full_state.get("per_tensor")
    if not isinstance(per_tensor, Mapping) or set(per_tensor) != set(tensor_names):
        raise ValueError("smoke full-state audit tensors differ from the frozen contract")
    if (
        full_state.get("num_tensors") != len(tensor_names)
        or full_state.get("num_tensor_state_comparisons") != 2 * len(tensor_names)
        or full_state.get("tolerance") != tolerance
    ):
        raise ValueError("smoke full-state audit has inconsistent two-state counts")
    tensor_maxima: list[float] = []
    for name in tensor_names:
        item = per_tensor[name]
        maximum = item.get("max_abs_difference") if isinstance(item, Mapping) else None
        if (
            not isinstance(item, Mapping)
            or item.get("passed") is not True
            or item.get("num_state_comparisons") != 2
            or not isinstance(maximum, (int, float))
            or isinstance(maximum, bool)
            or not np.isfinite(maximum)
            or maximum < 0.0
            or maximum > tolerance
        ):
            raise ValueError(f"smoke full-state tensor {name} failed its two-state audit")
        tensor_maxima.append(float(maximum))
    full_maximum = full_state.get("max_abs_difference")
    if (
        not isinstance(full_maximum, (int, float))
        or isinstance(full_maximum, bool)
        or not np.isfinite(full_maximum)
        or not np.isclose(
            full_maximum, max(tensor_maxima), rtol=0.0, atol=1.0e-12
        )
    ):
        raise ValueError("smoke full-state aggregate is internally inconsistent")

    overflow = report.get("overflow_audit")
    if (
        not isinstance(overflow, Mapping)
        or overflow.get("passed") is not True
        or overflow.get("checked_transitions") != 501
        or overflow.get("nonzero_entries") != 0
    ):
        raise ValueError(
            "smoke MJWarp overflow audit did not cover warm-up and all rollout steps"
        )

    conditions = report.get("evaluation_conditions")
    if conditions != manifest_state["evaluation_conditions"]:
        raise ValueError("smoke evaluation conditions differ from the DRAFT manifest")

    termination = report.get("termination_audit")
    if (
        not isinstance(termination, Mapping)
        or termination.get("passed") is not True
        or termination.get("primary_horizon_done_count") != 0
        or termination.get("suppressed_steps") != 50
        or termination.get("reference_termination_restored_after_primary_horizon")
        is not True
    ):
        raise ValueError("smoke termination audit did not prove an uncensored horizon")

    expected_terms = [
        "actions",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
        "motion_command",
        "motion_ref_ori_b",
    ]
    layout = report.get("observation_layout_audit")
    layout_maximum = (
        layout.get("max_abs_difference") if isinstance(layout, Mapping) else None
    )
    if (
        not isinstance(layout, Mapping)
        or layout.get("passed") is not True
        or layout.get("configured_terms_sorted") != expected_terms
        or layout.get("expected_terms_sorted") != expected_terms
        or layout.get("observed_width") != 154
        or layout.get("expected_width") != 154
        or layout.get("tolerance") != 1.0e-6
        or layout.get("configuration")
        != E71_RUNTIME_CONTRACT["actor_observation_contract"]
        or not isinstance(layout_maximum, (int, float))
        or isinstance(layout_maximum, bool)
        or not np.isfinite(layout_maximum)
        or layout_maximum < 0.0
        or layout_maximum > 1.0e-6
    ):
        raise ValueError("smoke semantic actor-observation audit did not pass")

    latent = report.get("latent_route_audit")
    if (
        not isinstance(latent, Mapping)
        or latent.get("passed") is not True
        or latent.get("tolerance") != 1.0e-6
    ):
        raise ValueError("smoke latent command-route audit did not pass")
    for name in (
        "direct_lookup_max_abs_difference",
        "same_command_across_state_max_abs_difference",
    ):
        value = latent.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0.0
            or value > 1.0e-6
        ):
            raise ValueError(f"smoke latent command-route field {name} failed")
    return {
        "warmup_audit": dict(warmup),
        "proprio_audit": dict(report["proprio_audit"]),
        "raw_proprio_audit": dict(report["raw_proprio_audit"]),
        "full_state_audit": dict(full_state),
        "overflow_audit": dict(overflow),
        "observation_layout_audit": dict(layout),
        "latent_route_audit": dict(latent),
        "termination_audit": dict(termination),
    }


def _validate_smoke_provenance(
    report: Mapping[str, Any],
    *,
    base_dir: pathlib.Path,
    manifest_state: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = {
        "student_checkpoint": _path_hash_pair(
            report,
            path_key="student_checkpoint",
            hash_key="student_checkpoint_sha256",
            base_dir=base_dir,
            label="smoke explicit seed 0 student checkpoint",
        ),
        "teacher_manifest": _path_hash_pair(
            report,
            path_key="teacher_manifest",
            hash_key="teacher_manifest_sha256",
            base_dir=base_dir,
            label="smoke teacher manifest",
        ),
        "teacher_checkpoints": _path_hash_list(
            report,
            path_key="teacher_ckpts",
            hash_key="teacher_checkpoint_sha256",
            expected_length=2,
            base_dir=base_dir,
            label="smoke teacher checkpoint",
        ),
        "motions": _path_hash_list(
            report,
            path_key="motion_files",
            hash_key="motion_sha256",
            expected_length=2,
            base_dir=base_dir,
            label="smoke motion",
        ),
        "ambiguity_precheck": _path_hash_pair(
            report,
            path_key="ambiguity_precheck",
            hash_key="ambiguity_precheck_sha256",
            base_dir=base_dir,
            label="smoke ambiguity precheck",
        ),
        "runtime": _path_hash_pair(
            report,
            path_key="runtime",
            hash_key="runtime_sha256",
            base_dir=base_dir,
            label="smoke evaluator runtime",
        ),
        "reset_runtime": _path_hash_pair(
            report,
            path_key="reset_runtime",
            hash_key="reset_runtime_sha256",
            base_dir=base_dir,
            label="smoke reset runtime",
        ),
        "freeze_manifest": _path_hash_pair(
            report,
            path_key="freeze_manifest",
            hash_key="freeze_manifest_sha256",
            base_dir=base_dir,
            label="smoke freeze manifest",
        ),
    }
    frozen = manifest_state["frozen_files"]
    expected_records = {
        "student_checkpoint": manifest_state["checkpoints"][("explicit", 0)],
        "teacher_manifest": frozen["teacher_manifest"],
        "teacher_checkpoints": [
            frozen["teacher_checkpoint_0"],
            frozen["teacher_checkpoint_1"],
        ],
        "motions": [frozen["motion_0"], frozen["motion_1"]],
        "ambiguity_precheck": frozen["ambiguity_precheck"],
        "runtime": frozen["evaluator"],
        "reset_runtime": frozen["reset_layer"],
        "freeze_manifest": manifest_state["freeze_manifest"],
    }
    if provenance != expected_records:
        raise ValueError("smoke provenance differs from the DRAFT manifest")

    invoked_argv = report.get("invoked_argv")
    if invoked_argv != manifest_state["smoke_argv"]:
        raise ValueError("smoke evaluator argv differs from the DRAFT manifest")

    imported_paths = report.get("imported_runtime_paths")
    imported_hashes = report.get("imported_runtime_sha256")
    if (
        not isinstance(imported_paths, Mapping)
        or not isinstance(imported_hashes, Mapping)
        or set(imported_paths) != set(IMPORTED_RUNTIME_KEYS)
        or set(imported_hashes) != set(IMPORTED_RUNTIME_KEYS)
    ):
        raise ValueError("smoke report must bind the exact imported-runtime set")
    imported_runtime: dict[str, dict[str, str]] = {}
    for name in IMPORTED_RUNTIME_KEYS:
        imported_runtime[name] = validate_hash_record(
            {"path": imported_paths[name], "sha256": imported_hashes[name]},
            base_dir=base_dir,
            label=f"smoke imported runtime {name}",
        )
        _assert_same_record(
            imported_runtime[name], frozen[name], label=f"smoke imported runtime {name}"
        )

    commits = _validate_commits(report.get("repository_commits"), label="smoke report")
    if commits != manifest_state["commits"]:
        raise ValueError("smoke repository revisions differ from the DRAFT manifest")
    working_trees = _validate_working_trees(
        report.get("working_trees"), label="smoke report"
    )
    if working_trees != manifest_state["working_trees"]:
        raise ValueError("smoke working-tree state differs from the DRAFT manifest")
    environment = _validate_environment(
        report.get("runtime_environment"),
        label="smoke report runtime environment",
        require_gpu=True,
    )
    if environment != manifest_state["environment"]:
        raise ValueError("smoke runtime environment differs from the DRAFT manifest")

    toctou = report.get("runtime_toctou_audit")
    if not isinstance(toctou, Mapping) or toctou.get("passed") is not True:
        raise ValueError("smoke runtime TOCTOU audit did not pass")

    def digest_map(name: str) -> dict[str, str]:
        value = toctou.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"smoke runtime TOCTOU audit is missing {name}")
        return {
            str(key): _require_sha256(
                digest, label=f"smoke TOCTOU {name}.{key}"
            )
            for key, digest in value.items()
        }

    runtime_start = digest_map("startup_sha256")
    runtime_end = digest_map("completion_sha256")
    frozen_start = digest_map("all_frozen_startup_sha256")
    frozen_end = digest_map("all_frozen_completion_sha256")
    if runtime_start != runtime_end or frozen_start != frozen_end:
        raise ValueError("smoke runtime or frozen input changed during evaluation")
    expected_runtime = {
        name: imported_runtime[name]["sha256"] for name in LOAD_BEARING_RUNTIME_KEYS
    }
    if runtime_start != expected_runtime:
        raise ValueError("smoke TOCTOU runtime map differs from imported runtimes")
    expected_frozen = {
        "freeze_manifest": provenance["freeze_manifest"]["sha256"],
        "student_checkpoint": provenance["student_checkpoint"]["sha256"],
        "teacher_manifest": provenance["teacher_manifest"]["sha256"],
        "teacher_checkpoint_0": provenance["teacher_checkpoints"][0]["sha256"],
        "teacher_checkpoint_1": provenance["teacher_checkpoints"][1]["sha256"],
        "motion_0": provenance["motions"][0]["sha256"],
        "motion_1": provenance["motions"][1]["sha256"],
        "ambiguity_precheck": provenance["ambiguity_precheck"]["sha256"],
        **{
            f"runtime.{name}": imported_runtime[name]["sha256"]
            for name in IMPORTED_RUNTIME_KEYS
        },
    }
    if frozen_start != expected_frozen:
        raise ValueError("smoke all-frozen TOCTOU map differs from its provenance")
    return {
        "provenance": provenance,
        "imported_runtime": imported_runtime,
        "repository_commits": commits,
        "working_trees": working_trees,
        "runtime_environment": environment,
        "invoked_argv": list(invoked_argv),
    }


def validate_smoke_report(
    manifest_path: pathlib.Path, smoke_path: pathlib.Path
) -> dict[str, Any]:
    """Validate one DRAFT-bound four-cell smoke run and return its certificate.

    The returned payload is deterministic and side-effect free so freeze tooling can
    recompute it byte-for-byte before changing the manifest to ``PREREGISTERED``.
    """

    manifest_path = pathlib.Path(manifest_path)
    smoke_path = pathlib.Path(smoke_path)
    if manifest_path.resolve() == smoke_path.resolve():
        raise ValueError("smoke manifest and report paths must be unique")
    for path in (manifest_path, smoke_path):
        if not path.is_file():
            raise ValueError(f"required smoke-audit artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "DRAFT":
        raise ValueError("smoke audit requires a DRAFT manifest")
    manifest_state = validate_manifest(manifest, base_dir=manifest_path.parent)
    artifacts = manifest_state["artifact_paths"]
    if str(manifest_path.resolve()) != artifacts["draft_manifest"]:
        raise ValueError("DRAFT manifest path differs from its artifact contract")
    if str(smoke_path.resolve()) != artifacts["smoke_report"]:
        raise ValueError("smoke report path differs from the DRAFT artifact contract")
    manifest_record = {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
    }
    manifest_state["freeze_manifest"] = manifest_record

    report = json.loads(smoke_path.read_text())
    if not isinstance(report, dict):
        raise ValueError("smoke report must be a JSON object")
    if report.get("protocol") != SMOKE_REPORT_PROTOCOL:
        raise ValueError(f"smoke report protocol must be {SMOKE_REPORT_PROTOCOL!r}")
    if report.get("status") != "complete":
        raise ValueError("smoke report status must be complete")
    if report.get("arm") != "explicit" or report.get("student_arm") != "c_prior_explicit":
        raise ValueError("smoke report must use the explicit seed-0 student arm")
    if (
        not isinstance(report.get("training_seed"), int)
        or isinstance(report.get("training_seed"), bool)
        or report.get("training_seed") != 0
    ):
        raise ValueError("smoke report training seed must be integer 0")
    if (
        not isinstance(report.get("evaluation_seed"), int)
        or isinstance(report.get("evaluation_seed"), bool)
        or report.get("evaluation_seed") != EXPECTED_EVALUATION_SEED
    ):
        raise ValueError("smoke report evaluation seed must be integer 404")
    smoke_pair_id = manifest_state["smoke_pair_id"]
    if report.get("smoke_pair_id") != smoke_pair_id:
        raise ValueError("smoke report pair ID differs from the DRAFT manifest")
    if report.get("num_rollouts") != 4 or report.get("simulator_num_envs") != 4:
        raise ValueError("smoke report must contain exactly four simulator rollouts")
    if report.get("branch_samples_0p5_s") != 25 or report.get("branch_samples_1p0_s") != 50:
        raise ValueError("smoke branch endpoints must use 25 and 50 future-only samples")
    if report.get("policy_dt_s") != 0.02 or report.get("motion_fps") != 50.0:
        raise ValueError("smoke report changed the registered 50 Hz timing contract")

    integer_vectors = {
        key: _smoke_integer_vector(report, key) for key in GRID_FIELDS
    }
    expected_grid = {
        "pair_ids": [smoke_pair_id] * 4,
        "state_sides": [0, 0, 1, 1],
        "command_sides": [0, 1, 0, 1],
        "state_motion_ids": [0, 0, 1, 1],
        "command_motion_ids": [0, 1, 0, 1],
    }
    for key, expected in expected_grid.items():
        if integer_vectors[key].tolist() != expected:
            raise ValueError(f"smoke {key} does not match the exact four-cell grid")
    state_steps = integer_vectors["state_start_steps"].tolist()
    command_steps = integer_vectors["command_start_steps"].tolist()
    if any(step < 0 for step in (*state_steps, *command_steps)):
        raise ValueError("smoke grid cursors must be nonnegative")
    if (
        state_steps != [state_steps[0], state_steps[0], state_steps[2], state_steps[2]]
        or command_steps
        != [state_steps[0], state_steps[2], state_steps[0], state_steps[2]]
    ):
        raise ValueError("smoke state and command cursors do not realize the four-cell grid")
    registered_steps = _registered_smoke_branch_steps(
        manifest_state, pair_id=smoke_pair_id, fps=float(report["motion_fps"])
    )
    if state_steps != [
        registered_steps[0],
        registered_steps[0],
        registered_steps[1],
        registered_steps[1],
    ]:
        raise ValueError("smoke grid cursors differ from the frozen ambiguity pair")

    metric_keys = (
        "q_a_0p5_s",
        "q_b_0p5_s",
        "q_ab_0p5_s",
        "d_a_0p5_s",
        "d_b_0p5_s",
        "branch_coordinate_0p5_s",
        "q_a_1p0_s",
        "q_b_1p0_s",
        "q_ab_1p0_s",
        "d_a_1p0_s",
        "d_b_1p0_s",
        "branch_coordinate_1p0_s",
        "survival_s",
    )
    metrics = {key: _smoke_float_vector(report, key) for key in metric_keys}
    for horizon in ("0p5_s", "1p0_s"):
        q_a = metrics[f"q_a_{horizon}"]
        q_b = metrics[f"q_b_{horizon}"]
        q_ab = metrics[f"q_ab_{horizon}"]
        if np.any(q_a < 0.0) or np.any(q_b < 0.0) or np.any(q_ab <= 0.0):
            raise ValueError(f"smoke branch squared errors at {horizon} are invalid")
        if not np.allclose(q_ab, q_ab[0], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"smoke reference-only q_ab at {horizon} changed across cells")
        expected_coordinate = (q_a - q_b) / (q_ab + 1.0e-8)
        if not np.allclose(
            metrics[f"branch_coordinate_{horizon}"],
            expected_coordinate,
            rtol=1.0e-7,
            atol=1.0e-9,
        ):
            raise ValueError(f"smoke branch coordinate at {horizon} failed q/C algebra")
        if not np.allclose(
            metrics[f"d_a_{horizon}"], np.sqrt(q_a), rtol=1.0e-7, atol=1.0e-9
        ) or not np.allclose(
            metrics[f"d_b_{horizon}"], np.sqrt(q_b), rtol=1.0e-7, atol=1.0e-9
        ):
            raise ValueError(f"smoke branch distance at {horizon} failed sqrt(q) algebra")
        legacy_key = f"branch_score_{horizon}"
        if legacy_key in report:
            legacy = _smoke_float_vector(report, legacy_key)
            if not np.allclose(
                legacy,
                metrics[f"d_a_{horizon}"] - metrics[f"d_b_{horizon}"],
                rtol=1.0e-7,
                atol=1.0e-9,
            ):
                raise ValueError(f"smoke legacy branch score at {horizon} is inconsistent")

    goal_scale = report.get("goal_scale")
    if (
        not isinstance(goal_scale, list)
        or len(goal_scale) != 58
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value <= 0.0
            for value in goal_scale
        )
    ):
        raise ValueError("smoke goal_scale must contain 58 finite positive values")
    completed = report.get("completed")
    if (
        not isinstance(completed, list)
        or len(completed) != 4
        or any(
            not isinstance(value, (bool, int))
            or (isinstance(value, int) and not isinstance(value, bool) and value not in (0, 1))
            for value in completed
        )
    ):
        raise ValueError("smoke completed must contain exactly four boolean outcomes")
    completion_rate = report.get("completion_rate")
    if (
        not isinstance(completion_rate, (int, float))
        or isinstance(completion_rate, bool)
        or not np.isclose(completion_rate, np.mean(completed), rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("smoke completion_rate disagrees with completed")
    survival = metrics["survival_s"]
    if np.any(survival < 0.0) or np.any(survival > 10.0):
        raise ValueError("smoke survival must remain within the registered 10-second horizon")
    mean_survival = report.get("mean_survival_s")
    if (
        not isinstance(mean_survival, (int, float))
        or isinstance(mean_survival, bool)
        or not np.isclose(mean_survival, np.mean(survival), rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("smoke mean_survival_s disagrees with survival_s")
    _smoke_finite_matrix(report, "first_z_cmd", rows=4, columns=64)
    _smoke_finite_matrix(report, "first_student_action", rows=4, columns=29)
    _smoke_finite_matrix(report, "first_teacher_action", rows=4, columns=29)

    _validate_smoke_audits(report, manifest_state=manifest_state)
    _validate_smoke_provenance(
        report, base_dir=smoke_path.parent, manifest_state=manifest_state
    )

    smoke_record = {
        "path": str(smoke_path.resolve()),
        "sha256": sha256_file(smoke_path),
    }
    return {
        "protocol": SMOKE_AUDIT_PROTOCOL,
        "passed": True,
        "manifest_protocol": MANIFEST_PROTOCOL,
        "report_protocol": SMOKE_REPORT_PROTOCOL,
        "manifest_status": "DRAFT",
        "arm": "explicit",
        "student_arm": "c_prior_explicit",
        "training_seed": 0,
        "evaluation_seed": EXPECTED_EVALUATION_SEED,
        "smoke_pair_id": smoke_pair_id,
        "num_rollouts": 4,
        "state_command_cells": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "artifacts": {
            "manifest": manifest_record,
            "smoke_report": smoke_record,
        },
    }


def _validate_input_records(
    records: Any,
    *,
    expected_paths: Sequence[pathlib.Path],
    base_dir: pathlib.Path,
    label: str,
) -> list[dict[str, str]]:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        raise ValueError(f"{label} inputs must contain exactly {len(expected_paths)} reports")
    observed = [
        validate_hash_record(record, base_dir=base_dir, label=f"{label} input {index}")
        for index, record in enumerate(records)
    ]
    expected = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in expected_paths
    ]
    if observed != expected:
        raise ValueError(f"{label} inputs do not match the registered report order and hashes")
    return observed


def validate_gate_artifact(
    gate_artifact: Mapping[str, Any],
    *,
    explicit_entries: Sequence[tuple[pathlib.Path, dict[str, Any]]],
    base_dir: pathlib.Path,
) -> dict[str, Any]:
    """Recompute the explicit feasibility gate and bind its immutable input list."""

    if gate_artifact.get("protocol") != GATE_PROTOCOL:
        raise ValueError(f"gate protocol must be {GATE_PROTOCOL!r}")
    if gate_artifact.get("analysis_protocol") != REPORT_PROTOCOL:
        raise ValueError("gate artifact has the wrong analysis protocol")
    if gate_artifact.get("minimum_eligible_pairs") != MIN_ELIGIBLE_PAIRS:
        raise ValueError("gate artifact changed the registered minimum eligible pairs")
    if (
        gate_artifact.get("minimum_eligible_temporal_components")
        != MIN_ELIGIBLE_TEMPORAL_COMPONENTS
    ):
        raise ValueError("gate artifact changed the registered temporal-component minimum")
    _validate_input_records(
        gate_artifact.get("inputs"),
        expected_paths=[path for path, _ in explicit_entries],
        base_dir=base_dir,
        label="gate",
    )
    recomputed = eligible_pairs([report for _, report in explicit_entries])
    if gate_artifact.get("explicit_feasibility_gate") != recomputed:
        raise ValueError("gate artifact does not equal the recomputed explicit gate")
    expected_valid = (
        recomputed["num_eligible_pairs"] >= MIN_ELIGIBLE_PAIRS
        and recomputed["num_eligible_temporal_components"]
        >= MIN_ELIGIBLE_TEMPORAL_COMPONENTS
    )
    if gate_artifact.get("valid_explicit_gate") is not expected_valid:
        raise ValueError("gate artifact valid flag is inconsistent with the eligible set")
    return recomputed


def validate_analysis_artifact(
    analysis_artifact: Mapping[str, Any],
    *,
    explicit_entries: Sequence[tuple[pathlib.Path, dict[str, Any]]],
    snmr_entries: Sequence[tuple[pathlib.Path, dict[str, Any]]],
    gate: Mapping[str, Any],
    base_dir: pathlib.Path,
) -> dict[str, Any]:
    """Recompute the confirmatory result and require its exact registered fields."""

    ordered_entries = [*explicit_entries, *snmr_entries]
    _validate_input_records(
        analysis_artifact.get("inputs"),
        expected_paths=[path for path, _ in ordered_entries],
        base_dir=base_dir,
        label="analysis",
    )
    recomputed = analyze(
        [report for _, report in explicit_entries],
        [report for _, report in snmr_entries],
    )
    for key, expected in recomputed.items():
        if analysis_artifact.get(key) != expected:
            raise ValueError(f"analysis artifact field {key} differs from recomputation")
    if analysis_artifact.get("explicit_feasibility_gate") != gate:
        raise ValueError("analysis eligible set differs from the immutable gate artifact")
    return recomputed


def audit_bundle(
    *,
    manifest_path: pathlib.Path,
    explicit_paths: Sequence[pathlib.Path],
    snmr_paths: Sequence[pathlib.Path],
    gate_path: pathlib.Path,
    analysis_path: pathlib.Path,
) -> dict[str, Any]:
    """Load and validate every E71 evidence artifact."""

    all_paths = [manifest_path, *explicit_paths, *snmr_paths, gate_path, analysis_path]
    resolved = [path.resolve() for path in all_paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("bundle artifact paths must be unique")
    for path in all_paths:
        if not path.is_file():
            raise ValueError(f"required bundle artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text())
    manifest_state = validate_manifest(
        manifest, base_dir=manifest_path.parent, require_preregistered=True
    )
    if str(manifest_path.resolve()) != manifest_state["artifact_paths"][
        "preregistered_manifest"
    ]:
        raise ValueError("PREREGISTERED manifest path differs from its artifact contract")
    artifact_paths = manifest_state["artifact_paths"]
    if [str(path.resolve()) for path in explicit_paths] != artifact_paths[
        "explicit_reports"
    ] or [str(path.resolve()) for path in snmr_paths] != artifact_paths["snmr_reports"]:
        raise ValueError("bundle report paths differ from the manifest artifact contract")
    if str(gate_path.resolve()) != artifact_paths["explicit_gate"]:
        raise ValueError("explicit gate path differs from the manifest artifact contract")
    if str(analysis_path.resolve()) != artifact_paths["analysis"]:
        raise ValueError("analysis path differs from the manifest artifact contract")
    manifest_state["freeze_manifest"] = {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
    }
    explicit_entries = [
        (path, json.loads(path.read_text())) for path in explicit_paths
    ]
    snmr_entries = [(path, json.loads(path.read_text())) for path in snmr_paths]
    validate_report_set(
        explicit_entries, snmr_entries, manifest_state=manifest_state
    )
    gate_artifact = json.loads(gate_path.read_text())
    gate = validate_gate_artifact(
        gate_artifact,
        explicit_entries=explicit_entries,
        base_dir=gate_path.parent,
    )
    analysis_artifact = json.loads(analysis_path.read_text())
    analysis = validate_analysis_artifact(
        analysis_artifact,
        explicit_entries=explicit_entries,
        snmr_entries=snmr_entries,
        gate=gate,
        base_dir=analysis_path.parent,
    )
    artifacts = {
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "gate": {"path": str(gate_path.resolve()), "sha256": sha256_file(gate_path)},
        "analysis": {"path": str(analysis_path.resolve()), "sha256": sha256_file(analysis_path)},
        "explicit_reports": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in explicit_paths
        ],
        "snmr_reports": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in snmr_paths
        ],
    }
    return {
        "protocol": BUNDLE_PROTOCOL,
        "manifest_protocol": MANIFEST_PROTOCOL,
        "report_protocol": REPORT_PROTOCOL,
        "gate_protocol": GATE_PROTOCOL,
        "valid_explicit_gate": analysis["valid_explicit_gate"],
        "positive_target_specific_gate": analysis["positive_target_specific_gate"],
        "eligible_pair_ids": gate["eligible_pair_ids"],
        "artifacts": artifacts,
        "replayed_frozen_files": manifest_state["frozen_files"],
    }


def write_json_once(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a JSON artifact and refuse any existing destination.

    A fully flushed temporary inode is hard-linked into place.  ``os.link`` fails when
    the destination exists, giving the write an atomic create-if-absent contract.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite write-once artifact {path}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite write-once artifact {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--explicit-reports", type=pathlib.Path, nargs=3)
    parser.add_argument("--explicit-preflight", action="store_true")
    parser.add_argument("--smoke-report", type=pathlib.Path)
    parser.add_argument("--smoke-out", type=pathlib.Path)
    parser.add_argument("--snmr-reports", type=pathlib.Path, nargs=3)
    parser.add_argument("--gate", type=pathlib.Path)
    parser.add_argument("--analysis", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    if args.smoke_report is not None or args.smoke_out is not None:
        if args.smoke_report is None or args.smoke_out is None:
            parser.error("smoke audit requires both --smoke-report and --smoke-out")
        if (
            args.explicit_reports is not None
            or args.explicit_preflight
            or any(
                value is not None
                for value in (args.snmr_reports, args.gate, args.analysis, args.out)
            )
        ):
            parser.error("smoke audit cannot be combined with preflight or final-bundle inputs")
        certificate = validate_smoke_report(args.manifest, args.smoke_report)
        registered_smoke_out = json.loads(args.manifest.read_text())["artifact_paths"][
            "smoke_audit"
        ]
        if str(args.smoke_out.resolve()) != str(pathlib.Path(registered_smoke_out).resolve()):
            parser.error("--smoke-out differs from the manifest artifact contract")
        write_json_once(args.smoke_out, certificate)
        print(json.dumps(certificate, indent=2, sort_keys=True))
        return
    if args.explicit_preflight:
        if args.explicit_reports is None:
            parser.error("--explicit-preflight requires --explicit-reports")
        if any(value is not None for value in (args.snmr_reports, args.gate, args.analysis, args.out)):
            parser.error("--explicit-preflight cannot be combined with final-bundle inputs")
        reports = validate_explicit_report_set(args.manifest, args.explicit_reports)
        print(
            json.dumps(
                {
                    "protocol": "E71 explicit report-set preflight v1",
                    "passed": True,
                    "training_seeds": [report["training_seed"] for report in reports],
                    "report_sha256": [sha256_file(path) for path in args.explicit_reports],
                },
                indent=2,
            )
        )
        return
    if args.explicit_reports is None or any(
        value is None for value in (args.snmr_reports, args.gate, args.analysis, args.out)
    ):
        parser.error(
            "final audit requires --explicit-reports, --snmr-reports, --gate, --analysis, and --out"
        )
    assert args.snmr_reports is not None
    assert args.gate is not None
    assert args.analysis is not None
    assert args.out is not None
    bundle = audit_bundle(
        manifest_path=args.manifest,
        explicit_paths=args.explicit_reports,
        snmr_paths=args.snmr_reports,
        gate_path=args.gate,
        analysis_path=args.analysis,
    )
    registered_bundle_out = json.loads(args.manifest.read_text())["artifact_paths"][
        "integrity_bundle"
    ]
    if str(args.out.resolve()) != str(pathlib.Path(registered_bundle_out).resolve()):
        parser.error("--out differs from the manifest integrity-bundle path")
    write_json_once(args.out, bundle)
    print(f"certified E71 command-swap bundle -> {args.out}")


if __name__ == "__main__":
    main()
