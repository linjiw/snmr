from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import scripts.audit_e71_bundle as audit_module
from scripts.analyze_e71_command_swap import analyze, eligible_pairs
from scripts.analyze_e71_command_swap import EXPECTED_EVALUATION_CONDITIONS
from scripts.audit_e71_bundle import (
    BUNDLE_PROTOCOL,
    GATE_PROTOCOL,
    IMPORTED_RUNTIME_KEYS,
    LOAD_BEARING_RUNTIME_KEYS,
    MANIFEST_PROTOCOL,
    REPORT_PROTOCOL,
    SMOKE_AUDIT_PROTOCOL,
    SMOKE_REPORT_PROTOCOL,
    audit_bundle,
    sha256_file,
    validate_manifest,
    validate_explicit_report_set,
    validate_report_set,
    validate_rollout_report,
    validate_smoke_report,
    write_json_once,
)
from snmr.integration.counterfactual_eval import E71_RUNTIME_CONTRACT


@pytest.fixture(autouse=True)
def _test_postprocess_gate(monkeypatch, tmp_path: pathlib.Path) -> None:
    """Keep the production gate canonical while giving unit tests an isolated root."""

    monkeypatch.setattr(
        audit_module,
        "E70_POSTPROCESS_GATE",
        (tmp_path / "POSTPROCESS_COMPLETE").resolve(),
    )


def _record(path: pathlib.Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _make_file(root: pathlib.Path, name: str) -> pathlib.Path:
    path = root / name
    path.write_bytes(f"frozen:{name}".encode())
    return path


def _frozen_inputs(tmp_path: pathlib.Path) -> tuple[dict, dict]:
    frozen_names = (
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
    files = {name: _make_file(tmp_path, name) for name in frozen_names}
    for name in ("motion_0", "motion_1"):
        with files[name].open("wb") as handle:
            np.savez(
                handle,
                fps=np.asarray([50.0], dtype=np.float32),
                joint_pos=np.empty((10_000, 0), dtype=np.float32),
            )
    ambiguity_windows = []
    for pair_id in range(69):
        component = pair_id % 12
        within_component = pair_id // 12
        ambiguity_windows.append(
            {
                "time_seconds_first": component * 10.0
                + (10 + within_component) / 50.0,
                "time_seconds_second": component * 10.0
                + (10 + within_component) / 50.0,
            }
        )
    files["ambiguity_precheck"].write_text(
        json.dumps(
            {
                "protocol": "E70 reference-only ambiguity precheck v1",
                "preferred_pair": "walk1_subject1,walk1_subject5",
                "thresholds": {"rollout_seconds": 10.0},
                "loaded_motion_order": ["walk1_subject1", "walk1_subject5"],
                "pairs": {
                    "walk1_subject1,walk1_subject5": {"windows": ambiguity_windows}
                },
            }
        )
    )
    checkpoints = {}
    checkpoint_records = []
    for arm in ("explicit", "snmr"):
        for seed in range(3):
            path = _make_file(tmp_path, f"{arm}-seed{seed}.pt")
            checkpoints[(arm, seed)] = path
            checkpoint_records.append(
                {"arm": arm, "training_seed": seed, **_record(path)}
            )
    evaluation_conditions = {
        **EXPECTED_EVALUATION_CONDITIONS,
        "passed": True,
        "all_observation_noise": False,
        "future_only_branch_samples": True,
        "warmup_callback_order_override": True,
        "torch_deterministic": True,
        "adaptive_timestep_sampler": False,
        "terrain_spawn_randomization": False,
        "expanded_mujoco_model_fields": [],
        "default_dof_pose_equal": True,
        "runtime_contract": copy.deepcopy(E71_RUNTIME_CONTRACT),
    }
    full_state_contract = {
        "tensor_names": ["dof_pos", "dof_vel"],
        "expanded_mujoco_model_fields": [],
        "tolerance": 1.0e-6,
    }
    commits = {"snmr": "a" * 40, "holosoma": "b" * 40}
    working_trees = {
        "snmr": {
            "root": str((tmp_path / "snmr-root").resolve()),
            "tracked_changes": False,
            "status_sha256": "c" * 64,
        },
        "holosoma": {
            "root": str((tmp_path / "holosoma-root").resolve()),
            "tracked_changes": False,
            "status_sha256": "d" * 64,
        },
    }
    environment = {
        "python": "3.10.0",
        "torch": "2.7.0",
        "cuda": "12.8",
        "mujoco": "3.3.5",
        "mujoco_warp": "0.0.1",
        "warp": "1.8.0",
        "numpy": "2.2.0",
        "gpu": "fixture-gpu",
        "cuda_visible_devices": "0",
        "cuda_logical_device": "0",
        "gpu_total_memory_mb": "49152",
        "gpu_uuid": "GPU-fixture",
        "mujoco_warp_types": str(files["mujoco_warp_types"].resolve()),
        "nvidia_driver": "570.0",
    }
    confirmatory_argv = {
        f"{arm}_seed{seed}": [
            str(files["evaluator"].resolve()),
            f"--fixture-arm={arm}",
            f"--fixture-seed={seed}",
        ]
        for arm in ("explicit", "snmr")
        for seed in range(3)
    }
    smoke_argv = [
        str(files["evaluator"].resolve()),
        "--fixture-smoke",
        "--training.num-envs",
        "4",
    ]
    artifact_paths = {
        "e71_root": str(tmp_path.resolve()),
        "draft_manifest": str((tmp_path / "freeze_manifest.json").resolve()),
        "preregistered_manifest": str(
            (tmp_path / "freeze_manifest_preregistered.json").resolve()
        ),
        "smoke_report": str(
            (tmp_path / "smoke_explicit_seed0_pair0.json").resolve()
        ),
        "smoke_audit": str((tmp_path / "smoke-audit.json").resolve()),
        "explicit_reports": [
            str((tmp_path / f"explicit_seed{seed}.json").resolve())
            for seed in range(3)
        ],
        "snmr_reports": [
            str((tmp_path / f"snmr_seed{seed}.json").resolve())
            for seed in range(3)
        ],
        "explicit_gate": str((tmp_path / "explicit_gate.json").resolve()),
        "analysis": str((tmp_path / "analysis.json").resolve()),
        "integrity_bundle": str((tmp_path / "integrity_bundle.json").resolve()),
        "logs_root": str((tmp_path / "logs").resolve()),
        "holosoma_logs": str((tmp_path / "holosoma_logs").resolve()),
    }
    for argv in [smoke_argv, *confirmatory_argv.values()]:
        argv.extend(["--logger.base-dir", artifact_paths["holosoma_logs"]])
    manifest = {
        "protocol": MANIFEST_PROTOCOL,
        "status": "DRAFT",
        "report_protocol": REPORT_PROTOCOL,
        "evaluation_seed": 404,
        "training_seeds": [0, 1, 2],
        "num_pairs": 69,
        "cells_per_report": 276,
        "runtime_ready": True,
        "artifact_paths": artifact_paths,
        "smoke_pair_id": 0,
        "smoke_argv": smoke_argv,
        "commits": commits,
        "working_trees": working_trees,
        "environment": environment,
        "evaluation_conditions": evaluation_conditions,
        "full_state_contract": full_state_contract,
        "frozen_files": {name: _record(path) for name, path in files.items()},
        "checkpoints": checkpoint_records,
        "confirmatory_argv_by_arm_seed": confirmatory_argv,
    }
    manifest_path = tmp_path / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest, {
        "files": files,
        "checkpoints": checkpoints,
        "evaluation_conditions": evaluation_conditions,
        "full_state_contract": full_state_contract,
        "freeze_manifest": manifest_path,
        "commits": commits,
        "working_trees": working_trees,
        "environment": environment,
        "confirmatory_argv_by_arm_seed": confirmatory_argv,
        "smoke_pair_id": 0,
        "smoke_argv": smoke_argv,
        "artifact_paths": artifact_paths,
    }


def _report(arm: str, seed: int, frozen: dict) -> dict:
    pair_ids = []
    states = []
    commands = []
    state_steps = []
    command_steps = []
    coordinates = []
    legacy_scores = []
    q_a = []
    q_b = []
    q_ab = []
    d_a = []
    d_b = []
    for pair_id in range(69):
        component = pair_id % 12
        within_component = pair_id // 12
        starts = (
            component * 500 + 10 + within_component,
            (20 + component) * 500 + 10 + within_component,
        )
        for state in (0, 1):
            for command in (0, 1):
                pair_ids.append(pair_id)
                states.append(state)
                commands.append(command)
                state_steps.append(starts[state])
                command_steps.append(starts[command])
                coordinate = -0.5 if command == 0 else 0.5
                q_a_value = 0.5 if command == 0 else 1.0
                q_b_value = 1.0 if command == 0 else 0.5
                coordinates.append(coordinate / (1.0 + 1.0e-8))
                q_a.append(q_a_value)
                q_b.append(q_b_value)
                q_ab.append(1.0)
                d_a.append(q_a_value**0.5)
                d_b.append(q_b_value**0.5)
                legacy_scores.append(q_a_value**0.5 - q_b_value**0.5)
    files = frozen["files"]
    checkpoint = frozen["checkpoints"][(arm, seed)]
    imported_paths = {
        name: str(files[name].resolve()) for name in IMPORTED_RUNTIME_KEYS
    }
    imported_hashes = {
        name: sha256_file(files[name]) for name in IMPORTED_RUNTIME_KEYS
    }
    runtime_hashes = {
        name: imported_hashes[name] for name in LOAD_BEARING_RUNTIME_KEYS
    }
    all_frozen_hashes = {
        "freeze_manifest": sha256_file(frozen["freeze_manifest"]),
        "student_checkpoint": sha256_file(checkpoint),
        "teacher_manifest": sha256_file(files["teacher_manifest"]),
        "teacher_checkpoint_0": sha256_file(files["teacher_checkpoint_0"]),
        "teacher_checkpoint_1": sha256_file(files["teacher_checkpoint_1"]),
        "motion_0": sha256_file(files["motion_0"]),
        "motion_1": sha256_file(files["motion_1"]),
        "ambiguity_precheck": sha256_file(files["ambiguity_precheck"]),
        **{f"runtime.{name}": digest for name, digest in imported_hashes.items()},
    }
    return {
        "protocol": REPORT_PROTOCOL,
        "status": "complete",
        "arm": arm,
        "student_arm": {
            "explicit": "c_prior_explicit",
            "snmr": "a_prior_snmr",
        }[arm],
        "training_seed": seed,
        "evaluation_seed": 404,
        "smoke_pair_id": None,
        "num_rollouts": 276,
        "pair_ids": pair_ids,
        "state_sides": states,
        "command_sides": commands,
        "state_start_steps": state_steps,
        "command_start_steps": command_steps,
        "state_motion_ids": states.copy(),
        "command_motion_ids": commands.copy(),
        "q_a_0p5_s": q_a,
        "q_b_0p5_s": q_b,
        "q_ab_0p5_s": q_ab,
        "branch_coordinate_0p5_s": coordinates,
        "d_a_0p5_s": d_a,
        "d_b_0p5_s": d_b,
        "branch_score_0p5_s": legacy_scores,
        "branch_samples_0p5_s": 25,
        "q_a_1p0_s": q_a,
        "q_b_1p0_s": q_b,
        "q_ab_1p0_s": q_ab,
        "branch_coordinate_1p0_s": coordinates,
        "d_a_1p0_s": d_a,
        "d_b_1p0_s": d_b,
        "branch_score_1p0_s": legacy_scores,
        "branch_samples_1p0_s": 50,
        "goal_scale": [1.0] * 58,
        "completed": [True] * 276,
        "survival_s": [2.0] * 276,
        "first_z_cmd": [[0.0] * 64 for _ in range(276)],
        "first_student_action": [
            [(-1.0 if command == 0 else 1.0)] * 29 for command in commands
        ],
        "first_teacher_action": [
            [(-1.0 if command == 0 else 1.0)] * 29 for command in commands
        ],
        "proprio_audit": {
            "passed": True,
            "num_state_comparisons": 138,
            "max_abs_difference": 0.0,
            "tolerance": 1.0e-6,
        },
        "raw_proprio_audit": {
            "passed": True,
            "num_state_comparisons": 138,
            "max_abs_difference": 0.0,
            "tolerance": 1.0e-6,
        },
        "warmup_audit": {
            "passed": True,
            "reset_count": 0,
            "command_dependent_reset": False,
            "episode_length_min": 1,
            "episode_length_max": 1,
            "callback_order_before_warmup": False,
            "callback_order_during_warmup": True,
            "callback_order_after_warmup": False,
        },
        "full_state_contract": copy.deepcopy(frozen["full_state_contract"]),
        "full_state_audit": {
            "passed": True,
            "num_tensors": 2,
            "num_tensor_state_comparisons": 276,
            "max_abs_difference": 0.0,
            "tolerance": 1.0e-6,
            "per_tensor": {
                name: {
                    "passed": True,
                    "num_state_comparisons": 138,
                    "max_abs_difference": 0.0,
                }
                for name in ("dof_pos", "dof_vel")
            },
        },
        "overflow_audit": {
            "checked_transitions": 501,
            "nonzero_entries": 0,
            "passed": True,
        },
        "evaluation_conditions": copy.deepcopy(frozen["evaluation_conditions"]),
        "termination_audit": {
            "passed": True,
            "primary_horizon_done_count": 0,
            "suppressed_steps": 50,
            "reference_termination_restored_after_primary_horizon": True,
        },
        "observation_layout_audit": {
            "passed": True,
            "configured_terms_sorted": [
                "actions",
                "base_ang_vel",
                "dof_pos",
                "dof_vel",
                "motion_command",
                "motion_ref_ori_b",
            ],
            "expected_terms_sorted": [
                "actions",
                "base_ang_vel",
                "dof_pos",
                "dof_vel",
                "motion_command",
                "motion_ref_ori_b",
            ],
            "observed_width": 154,
            "expected_width": 154,
            "max_abs_difference": 0.0,
            "tolerance": 1.0e-6,
            "configuration": copy.deepcopy(
                E71_RUNTIME_CONTRACT["actor_observation_contract"]
            ),
        },
        "latent_route_audit": {
            "passed": True,
            "direct_lookup_max_abs_difference": 0.0,
            "same_command_across_state_max_abs_difference": 0.0,
            "tolerance": 1.0e-6,
        },
        "policy_dt_s": 0.02,
        "motion_fps": 50.0,
        "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": sha256_file(checkpoint),
        "teacher_manifest": str(files["teacher_manifest"]),
        "teacher_manifest_sha256": sha256_file(files["teacher_manifest"]),
        "teacher_ckpts": [
            str(files["teacher_checkpoint_0"]),
            str(files["teacher_checkpoint_1"]),
        ],
        "teacher_checkpoint_sha256": [
            sha256_file(files["teacher_checkpoint_0"]),
            sha256_file(files["teacher_checkpoint_1"]),
        ],
        "motion_files": [str(files["motion_0"]), str(files["motion_1"])],
        "motion_sha256": [
            sha256_file(files["motion_0"]),
            sha256_file(files["motion_1"]),
        ],
        "ambiguity_precheck": str(files["ambiguity_precheck"]),
        "ambiguity_precheck_sha256": sha256_file(files["ambiguity_precheck"]),
        "runtime": str(files["evaluator"]),
        "runtime_sha256": sha256_file(files["evaluator"]),
        "reset_runtime": str(files["reset_layer"]),
        "reset_runtime_sha256": sha256_file(files["reset_layer"]),
        "freeze_manifest": str(frozen["freeze_manifest"]),
        "freeze_manifest_sha256": sha256_file(frozen["freeze_manifest"]),
        "invoked_argv": copy.deepcopy(
            frozen["confirmatory_argv_by_arm_seed"][f"{arm}_seed{seed}"]
        ),
        "runtime_toctou_audit": {
            "startup_sha256": copy.deepcopy(runtime_hashes),
            "completion_sha256": copy.deepcopy(runtime_hashes),
            "all_frozen_startup_sha256": copy.deepcopy(all_frozen_hashes),
            "all_frozen_completion_sha256": copy.deepcopy(all_frozen_hashes),
            "passed": True,
        },
        "imported_runtime_paths": imported_paths,
        "imported_runtime_sha256": imported_hashes,
        "repository_commits": copy.deepcopy(frozen["commits"]),
        "working_trees": copy.deepcopy(frozen["working_trees"]),
        "runtime_environment": copy.deepcopy(frozen["environment"]),
    }


def _write_smoke_inputs(tmp_path: pathlib.Path) -> dict:
    manifest, frozen = _frozen_inputs(tmp_path)
    manifest["status"] = "DRAFT"
    manifest_path = frozen["freeze_manifest"]
    manifest_path.write_text(json.dumps(manifest))
    report = _report("explicit", 0, frozen)
    vector_keys = (
        "pair_ids",
        "state_sides",
        "command_sides",
        "state_start_steps",
        "command_start_steps",
        "state_motion_ids",
        "command_motion_ids",
        "q_a_0p5_s",
        "q_b_0p5_s",
        "q_ab_0p5_s",
        "branch_coordinate_0p5_s",
        "d_a_0p5_s",
        "d_b_0p5_s",
        "branch_score_0p5_s",
        "q_a_1p0_s",
        "q_b_1p0_s",
        "q_ab_1p0_s",
        "branch_coordinate_1p0_s",
        "d_a_1p0_s",
        "d_b_1p0_s",
        "branch_score_1p0_s",
        "completed",
        "survival_s",
    )
    for key in vector_keys:
        report[key] = report[key][:4]
    report.update(
        {
            "protocol": SMOKE_REPORT_PROTOCOL,
            "smoke_pair_id": frozen["smoke_pair_id"],
            "num_rollouts": 4,
            "simulator_num_envs": 4,
            "completion_rate": 1.0,
            "mean_survival_s": 2.0,
            "first_z_cmd": [[0.0] * 64 for _ in range(4)],
            "first_student_action": [[0.0] * 29 for _ in range(4)],
            "first_teacher_action": [[0.0] * 29 for _ in range(4)],
            "invoked_argv": copy.deepcopy(frozen["smoke_argv"]),
        }
    )
    for key in ("proprio_audit", "raw_proprio_audit"):
        report[key]["num_state_comparisons"] = 2
    report["full_state_audit"]["num_tensor_state_comparisons"] = 4
    for item in report["full_state_audit"]["per_tensor"].values():
        item["num_state_comparisons"] = 2
    smoke_path = tmp_path / "smoke_explicit_seed0_pair0.json"
    smoke_path.write_text(json.dumps(report))
    return {
        "manifest": manifest_path,
        "smoke": smoke_path,
        "manifest_payload": manifest,
        "report_payload": report,
        "frozen": frozen,
    }


def _shift_smoke_cursors(report: dict) -> None:
    report["state_start_steps"] = [value + 5 for value in report["state_start_steps"]]
    report["command_start_steps"] = [
        value + 5 for value in report["command_start_steps"]
    ]


def _write_bundle_inputs(tmp_path: pathlib.Path) -> dict:
    smoke_inputs = _write_smoke_inputs(tmp_path)
    parent_path = smoke_inputs["manifest"]
    smoke_path = smoke_inputs["smoke"]
    smoke_audit = validate_smoke_report(parent_path, smoke_path)
    smoke_audit_path = tmp_path / "smoke-audit.json"
    smoke_audit_path.write_text(json.dumps(smoke_audit))
    postprocess_path = _make_file(tmp_path, "POSTPROCESS_COMPLETE")
    manifest = copy.deepcopy(smoke_inputs["manifest_payload"])
    manifest["status"] = "PREREGISTERED"
    manifest["preregistration"] = {
        "owner": "fixture-owner",
        "created_at_utc": "2026-08-14T00:00:00+00:00",
        "parent_draft": _record(parent_path),
        "smoke_report": _record(smoke_path),
        "smoke_audit": _record(smoke_audit_path),
        "postprocess_gate": _record(postprocess_path),
        "capacity_gate": {
            "required_free_mb": 26_000,
            "observed_free_mb": 30_000,
        },
    }
    manifest_path = tmp_path / "freeze_manifest_preregistered.json"
    manifest_path.write_text(json.dumps(manifest))
    frozen = smoke_inputs["frozen"]
    frozen["freeze_manifest"] = manifest_path

    explicit = []
    snmr = []
    for arm, destination in (("explicit", explicit), ("snmr", snmr)):
        for seed in range(3):
            path = tmp_path / f"{arm}_seed{seed}.json"
            report = _report(arm, seed, frozen)
            path.write_text(json.dumps(report))
            destination.append((path, report))

    gate_result = eligible_pairs([report for _, report in explicit])
    gate = {
        "protocol": GATE_PROTOCOL,
        "analysis_protocol": REPORT_PROTOCOL,
        "minimum_eligible_pairs": 20,
        "minimum_eligible_temporal_components": 6,
        "valid_explicit_gate": True,
        "explicit_feasibility_gate": gate_result,
        "inputs": [_record(path) for path, _ in explicit],
    }
    gate_path = tmp_path / "explicit_gate.json"
    gate_path.write_text(json.dumps(gate))

    analysis = analyze(
        [report for _, report in explicit], [report for _, report in snmr]
    )
    analysis["inputs"] = [_record(path) for path, _ in (*explicit, *snmr)]
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis))
    return {
        "manifest": manifest_path,
        "explicit": [path for path, _ in explicit],
        "snmr": [path for path, _ in snmr],
        "gate": gate_path,
        "analysis": analysis_path,
        "manifest_payload": manifest,
        "frozen": frozen,
    }


def test_final_bundle_replays_hashes_grids_gate_and_analysis(tmp_path) -> None:
    paths = _write_bundle_inputs(tmp_path)
    result = audit_bundle(
        manifest_path=paths["manifest"],
        explicit_paths=paths["explicit"],
        snmr_paths=paths["snmr"],
        gate_path=paths["gate"],
        analysis_path=paths["analysis"],
    )
    assert result["protocol"] == BUNDLE_PROTOCOL
    assert result["valid_explicit_gate"] is True
    assert result["positive_target_specific_gate"] is True
    assert result["eligible_pair_ids"] == list(range(69))


def test_explicit_preflight_certifies_three_reports_without_snmr(tmp_path) -> None:
    paths = _write_bundle_inputs(tmp_path)
    reports = validate_explicit_report_set(paths["manifest"], paths["explicit"])
    assert [report["training_seed"] for report in reports] == [0, 1, 2]
    assert all(report["arm"] == "explicit" for report in reports)


def test_explicit_preflight_cli_is_output_free_and_machine_readable(
    tmp_path, monkeypatch, capsys
) -> None:
    paths = _write_bundle_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_e71_bundle.py",
            "--manifest",
            str(paths["manifest"]),
            "--explicit-reports",
            *(str(path) for path in paths["explicit"]),
            "--explicit-preflight",
        ],
    )
    audit_module.main()
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True
    assert result["training_seeds"] == [0, 1, 2]
    assert not list(tmp_path.glob("*preflight*.json"))


def test_explicit_preflight_rejects_draft_manifest(tmp_path) -> None:
    paths = _write_bundle_inputs(tmp_path)
    manifest = json.loads(paths["manifest"].read_text())
    manifest["status"] = "DRAFT"
    paths["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="PREREGISTERED"):
        validate_explicit_report_set(paths["manifest"], paths["explicit"])


def test_manifest_allows_draft_preflight_but_final_audit_rejects_it(tmp_path) -> None:
    paths = _write_bundle_inputs(tmp_path)
    manifest = paths["manifest_payload"]
    manifest["status"] = "DRAFT"
    manifest.pop("preregistration")
    validate_manifest(manifest, base_dir=tmp_path)
    paths["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="PREREGISTERED"):
        audit_bundle(
            manifest_path=paths["manifest"],
            explicit_paths=paths["explicit"],
            snmr_paths=paths["snmr"],
            gate_path=paths["gate"],
            analysis_path=paths["analysis"],
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda report: report.pop("d_a_1p0_s"), "vector of numbers"),
        (
            lambda report: report["branch_coordinate_1p0_s"].__setitem__(0, 9.0),
            r"inconsistent with \(q_a-q_b\)",
        ),
        (
            lambda report: report["warmup_audit"].update(passed=False),
            "warmup_audit did not pass",
        ),
        (
            lambda report: report["state_motion_ids"].__setitem__(0, 1),
            "state motion IDs",
        ),
    ],
)
def test_report_schema_fails_closed(tmp_path, mutation, match: str) -> None:
    manifest, frozen = _frozen_inputs(tmp_path)
    report = _report("explicit", 0, frozen)
    mutation(report)
    with pytest.raises(ValueError, match=match):
        validate_rollout_report(
            report, expected_arm="explicit", expected_seed=0, base_dir=tmp_path
        )


def test_report_set_rejects_one_misaligned_snmr_grid(tmp_path) -> None:
    manifest, frozen = _frozen_inputs(tmp_path)
    state = validate_manifest(manifest, base_dir=tmp_path)
    explicit = []
    snmr = []
    for arm, destination in (("explicit", explicit), ("snmr", snmr)):
        for seed in range(3):
            report = _report(arm, seed, frozen)
            path = pathlib.Path(frozen["artifact_paths"][f"{arm}_reports"][seed])
            if arm == "snmr" and seed == 2:
                for index in (0, 1):
                    report["state_start_steps"][index] += 1
                for index in (0, 2):
                    report["command_start_steps"][index] += 1
            destination.append((path, report))
    with pytest.raises(ValueError, match="exact frozen grid"):
        validate_report_set(explicit, snmr, manifest_state=state)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda report: report["goal_scale"].__setitem__(0, 1.25),
            "reference-only metrics",
        ),
        (
            lambda report: report["evaluation_conditions"].update(
                default_dof_pose_equal=False
            ),
            "evaluation conditions",
        ),
    ],
)
def test_report_set_rejects_seed_specific_reference_or_condition_drift(
    tmp_path, mutation, match: str
) -> None:
    manifest, frozen = _frozen_inputs(tmp_path)
    state = validate_manifest(manifest, base_dir=tmp_path)
    explicit = []
    snmr = []
    for arm, destination in (("explicit", explicit), ("snmr", snmr)):
        for seed in range(3):
            report = _report(arm, seed, frozen)
            if arm == "snmr" and seed == 2:
                mutation(report)
            destination.append(
                (
                    pathlib.Path(frozen["artifact_paths"][f"{arm}_reports"][seed]),
                    report,
                )
            )
    with pytest.raises(ValueError, match=match):
        validate_report_set(explicit, snmr, manifest_state=state)


def test_manifest_rejects_unsafe_registered_conditions(tmp_path) -> None:
    manifest, _ = _frozen_inputs(tmp_path)
    manifest["evaluation_conditions"]["future_only_branch_samples"] = False
    with pytest.raises(ValueError, match="future_only_branch_samples"):
        validate_manifest(manifest, base_dir=tmp_path)


def test_preregistered_manifest_requires_audited_smoke_lineage(tmp_path) -> None:
    manifest, _ = _frozen_inputs(tmp_path)
    manifest["status"] = "PREREGISTERED"
    with pytest.raises(ValueError, match="transition lineage"):
        validate_manifest(manifest, base_dir=tmp_path, require_preregistered=True)


def test_preregistered_manifest_replays_parent_smoke_and_certificate(tmp_path) -> None:
    paths = _write_bundle_inputs(tmp_path)
    state = validate_manifest(
        paths["manifest_payload"],
        base_dir=paths["manifest"].parent,
        require_preregistered=True,
    )
    assert state["runtime_ready"] is True
    assert state["preregistration"]["owner"] == "fixture-owner"

    parent = pathlib.Path(
        paths["manifest_payload"]["preregistration"]["parent_draft"]["path"]
    )
    parent.write_text(parent.read_text() + "\n")
    with pytest.raises(ValueError, match="parent DRAFT manifest hash mismatch"):
        validate_manifest(
            paths["manifest_payload"],
            base_dir=paths["manifest"].parent,
            require_preregistered=True,
        )


def test_preregistered_manifest_rejects_substitute_postprocess_gate(tmp_path) -> None:
    paths = _write_bundle_inputs(tmp_path)
    substitute = _make_file(tmp_path, "SUBSTITUTE_POSTPROCESS_COMPLETE")
    paths["manifest_payload"]["preregistration"]["postprocess_gate"] = _record(
        substitute
    )
    with pytest.raises(ValueError, match="canonical E70 postprocess gate"):
        validate_manifest(
            paths["manifest_payload"],
            base_dir=paths["manifest"].parent,
            require_preregistered=True,
        )


def test_preregistered_manifest_rejects_copied_smoke_certificate(tmp_path) -> None:
    paths = _write_bundle_inputs(tmp_path)
    registered = pathlib.Path(paths["frozen"]["artifact_paths"]["smoke_audit"])
    copied = tmp_path / "copied-smoke-audit.json"
    copied.write_bytes(registered.read_bytes())
    paths["manifest_payload"]["preregistration"]["smoke_audit"] = _record(copied)
    with pytest.raises(ValueError, match="smoke audit is outside its registered path"):
        validate_manifest(
            paths["manifest_payload"],
            base_dir=paths["manifest"].parent,
            require_preregistered=True,
        )


def test_report_hash_replay_rejects_tampered_checkpoint(tmp_path) -> None:
    manifest, frozen = _frozen_inputs(tmp_path)
    report = _report("explicit", 0, frozen)
    frozen["checkpoints"][("explicit", 0)].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        validate_rollout_report(
            report, expected_arm="explicit", expected_seed=0, base_dir=tmp_path
        )


@pytest.mark.parametrize("artifact", ["gate", "analysis"])
def test_bundle_rejects_stale_derived_artifact(tmp_path, artifact: str) -> None:
    paths = _write_bundle_inputs(tmp_path)
    payload = json.loads(paths[artifact].read_text())
    if artifact == "gate":
        payload["explicit_feasibility_gate"]["eligible_pair_ids"].pop()
    else:
        payload["positive_target_specific_gate"] = False
    paths[artifact].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="recomputed|recomputation"):
        audit_bundle(
            manifest_path=paths["manifest"],
            explicit_paths=paths["explicit"],
            snmr_paths=paths["snmr"],
            gate_path=paths["gate"],
            analysis_path=paths["analysis"],
        )


def test_write_json_once_cannot_replace_gate_or_analysis(tmp_path) -> None:
    output = tmp_path / "analysis.json"
    write_json_once(output, {"version": 1})
    first_bytes = output.read_bytes()
    with pytest.raises(FileExistsError, match="write-once"):
        write_json_once(output, {"version": 2})
    assert output.read_bytes() == first_bytes
    assert json.loads(output.read_text()) == {"version": 1}


def test_smoke_audit_returns_canonical_draft_bound_certificate(tmp_path) -> None:
    paths = _write_smoke_inputs(tmp_path)
    certificate = validate_smoke_report(paths["manifest"], paths["smoke"])
    assert certificate == {
        "protocol": SMOKE_AUDIT_PROTOCOL,
        "passed": True,
        "manifest_protocol": MANIFEST_PROTOCOL,
        "report_protocol": SMOKE_REPORT_PROTOCOL,
        "manifest_status": "DRAFT",
        "arm": "explicit",
        "student_arm": "c_prior_explicit",
        "training_seed": 0,
        "evaluation_seed": 404,
        "smoke_pair_id": 0,
        "num_rollouts": 4,
        "state_command_cells": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "artifacts": {
            "manifest": _record(paths["manifest"]),
            "smoke_report": _record(paths["smoke"]),
        },
    }


def test_smoke_audit_rejects_preregistered_manifest(tmp_path) -> None:
    paths = _write_smoke_inputs(tmp_path)
    manifest = json.loads(paths["manifest"].read_text())
    manifest["status"] = "PREREGISTERED"
    paths["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires a DRAFT manifest"):
        validate_smoke_report(paths["manifest"], paths["smoke"])


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda report: report["state_sides"].__setitem__(1, 1),
            "exact four-cell grid",
        ),
        (_shift_smoke_cursors, "frozen ambiguity pair"),
        (
            lambda report: report["branch_coordinate_1p0_s"].__setitem__(0, 9.0),
            "failed q/C algebra",
        ),
        (
            lambda report: report["proprio_audit"].update(num_state_comparisons=138),
            "exactly two physical-state comparisons",
        ),
        (
            lambda report: report["full_state_audit"].update(
                num_tensor_state_comparisons=276
            ),
            "two-state counts",
        ),
        (
            lambda report: report["overflow_audit"].update(nonzero_entries=1),
            "overflow audit",
        ),
        (
            lambda report: report["observation_layout_audit"].update(passed=False),
            "semantic actor-observation audit",
        ),
        (
            lambda report: report["latent_route_audit"].update(passed=False),
            "latent command-route audit",
        ),
        (
            lambda report: report["termination_audit"].update(
                primary_horizon_done_count=1
            ),
            "uncensored horizon",
        ),
        (
            lambda report: report["evaluation_conditions"].update(
                future_only_branch_samples=False
            ),
            "evaluation conditions",
        ),
        (
            lambda report: report["invoked_argv"].append("--unregistered"),
            "argv differs",
        ),
        (
            lambda report: report["runtime_toctou_audit"][
                "completion_sha256"
            ].update(evaluator="f" * 64),
            "changed during evaluation",
        ),
    ],
)
def test_smoke_audit_fails_closed_on_schema_audit_or_provenance_drift(
    tmp_path, mutation, match: str
) -> None:
    paths = _write_smoke_inputs(tmp_path)
    report = json.loads(paths["smoke"].read_text())
    mutation(report)
    paths["smoke"].write_text(json.dumps(report))
    with pytest.raises(ValueError, match=match):
        validate_smoke_report(paths["manifest"], paths["smoke"])


@pytest.mark.parametrize("field", ["smoke_pair_id", "smoke_argv"])
def test_manifest_requires_frozen_smoke_registration(tmp_path, field: str) -> None:
    manifest, _ = _frozen_inputs(tmp_path)
    manifest.pop(field)
    with pytest.raises(ValueError, match="smoke"):
        validate_manifest(manifest, base_dir=tmp_path)


def test_smoke_cli_writes_once_and_matches_library_certificate(tmp_path) -> None:
    paths = _write_smoke_inputs(tmp_path)
    expected = validate_smoke_report(paths["manifest"], paths["smoke"])
    certificate_path = pathlib.Path(
        paths["manifest_payload"]["artifact_paths"]["smoke_audit"]
    )
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/audit_e71_bundle.py"
    argv = [
        sys.executable,
        str(script),
        "--manifest",
        str(paths["manifest"]),
        "--smoke-report",
        str(paths["smoke"]),
        "--smoke-out",
        str(certificate_path),
    ]
    completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == expected
    assert json.loads(certificate_path.read_text()) == expected
    original = certificate_path.read_bytes()

    repeated = subprocess.run(argv, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "write-once" in repeated.stderr
    assert certificate_path.read_bytes() == original
