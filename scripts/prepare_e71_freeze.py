#!/usr/bin/env python
"""Create a write-once machine-readable E71 freeze manifest.

Draft manifests support simulator smoke testing.  A PREREGISTERED manifest additionally
requires an available NVIDIA driver and is the only status accepted by confirmatory runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_e71_bundle import (  # noqa: E402
    MANIFEST_PROTOCOL,
    validate_manifest,
    validate_smoke_report,
    write_json_once,
)
from scripts.analyze_e71_command_swap import PROTOCOL as REPORT_PROTOCOL  # noqa: E402
from snmr.integration.counterfactual_eval import (  # noqa: E402
    E71_EVALUATION_CONDITIONS,
    E71_FULL_STATE_TENSOR_NAMES,
)


E70_ROOT = pathlib.Path("/data/robotixx/snmr-research/e70")
E70_POSTPROCESS_GATE = E70_ROOT / "POSTPROCESS_COMPLETE"
PRECHECK = ROOT / "autoresearch/iterate-260808-0338/e70_ambiguity_precheck.json"
HOLOSOMA_ROOT = ROOT.parent / "holosoma"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: pathlib.Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def git_state(root: pathlib.Path) -> tuple[str, dict[str, Any]]:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, {
        "root": str(root.resolve()),
        "tracked_changes": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def runtime_environment(wbt_python: pathlib.Path) -> dict[str, str]:
    program = """
import importlib.metadata, inspect, json, pathlib, platform
import mujoco, mujoco_warp, mujoco_warp._src.types as mjw_types, numpy, torch, warp
print(json.dumps({
    'python': platform.python_version(),
    'torch': torch.__version__,
    'cuda': str(torch.version.cuda),
    'mujoco': mujoco.__version__,
    'mujoco_warp': importlib.metadata.version('mujoco-warp'),
    'warp': warp.__version__,
    'numpy': numpy.__version__,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable',
    'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES', ''),
    'cuda_logical_device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'unavailable',
    'gpu_total_memory_mb': str(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)) if torch.cuda.is_available() else 'unavailable',
    'mujoco_warp_types': str(pathlib.Path(inspect.getsourcefile(mjw_types)).resolve()),
}))
"""
    completed = subprocess.run(
        [str(wbt_python), "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    return json.loads(completed.stdout)


def gpu_driver() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unavailable"
    value = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    return value or "unavailable"


def gpu_uuid() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return "unavailable"
    return completed.stdout.splitlines()[0].strip() or "unavailable"


def gpu_free_mb(wbt_python: pathlib.Path) -> int | None:
    program = "import torch; print(torch.cuda.mem_get_info(0)[0] // (1024 * 1024))"
    completed = subprocess.run(
        [str(wbt_python), "-c", program],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    last = completed.stdout.splitlines()[-1].strip()
    return int(last) if last.isdigit() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", choices=("DRAFT", "PREREGISTERED"), required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--wbt-python",
        type=pathlib.Path,
        default=HOLOSOMA_ROOT / ".venv/hsmujoco/bin/python",
    )
    parser.add_argument("--parent-draft", type=pathlib.Path)
    parser.add_argument("--smoke-report", type=pathlib.Path)
    parser.add_argument("--smoke-audit", type=pathlib.Path)
    parser.add_argument("--owner")
    parser.add_argument("--min-free-mb", type=int, default=26_000)
    args = parser.parse_args()

    transition_values = (
        args.parent_draft,
        args.smoke_report,
        args.smoke_audit,
        args.owner,
    )
    if args.status == "DRAFT" and any(value is not None for value in transition_values):
        raise ValueError("DRAFT creation cannot claim preregistration lineage")
    if args.status == "PREREGISTERED" and any(
        value is None for value in transition_values
    ):
        raise ValueError(
            "PREREGISTERED creation requires --parent-draft, --smoke-report, "
            "--smoke-audit, and --owner"
        )
    observed_free_mb: int | None = None
    if args.status == "PREREGISTERED":
        assert args.parent_draft is not None
        assert args.smoke_report is not None
        assert args.smoke_audit is not None
        assert args.owner is not None
        if not args.owner.strip():
            raise ValueError("preregistration owner must be nonempty")
        lineage_paths = (
            args.parent_draft.absolute(),
            args.smoke_report.absolute(),
            args.smoke_audit.absolute(),
        )
        if any(not path.is_file() for path in lineage_paths):
            missing = [str(path) for path in lineage_paths if not path.is_file()]
            raise FileNotFoundError("missing preregistration lineage: " + ", ".join(missing))
        if args.out.absolute() in lineage_paths:
            raise ValueError("final manifest path must be distinct from its immutable lineage")
    if args.min_free_mb != 26_000:
        raise ValueError("E71 freezes the GPU capacity gate at exactly 26,000 MiB")

    # Preserve the virtual-environment launcher itself.  ``resolve()`` follows the
    # interpreter symlink into the base uv installation and drops the hsmujoco venv.
    wbt_python = args.wbt_python.expanduser().absolute()
    if not wbt_python.is_file():
        raise FileNotFoundError(wbt_python)
    environment = runtime_environment(wbt_python)
    environment["nvidia_driver"] = gpu_driver()
    environment["gpu_uuid"] = gpu_uuid()
    runtime_ready = bool(
        environment["gpu"] != "unavailable"
        and environment["nvidia_driver"] != "unavailable"
        and environment["gpu_uuid"] != "unavailable"
        and environment["cuda"] not in {"None", "unavailable"}
        and environment["cuda_visible_devices"] == "0"
        and environment["cuda_logical_device"] == "0"
    )
    if args.status == "PREREGISTERED":
        if not runtime_ready:
            raise RuntimeError("refusing preregistration while the CUDA runtime is unavailable")
        if not E70_POSTPROCESS_GATE.is_file():
            raise RuntimeError(
                "refusing preregistration before canonical cross-project gate "
                f"{E70_POSTPROCESS_GATE}"
            )
        observed_free_mb = gpu_free_mb(wbt_python)
        if observed_free_mb is None or observed_free_mb < args.min_free_mb:
            raise RuntimeError(
                "refusing preregistration without at least "
                f"{args.min_free_mb} MiB free GPU memory; observed {observed_free_mb}"
            )

    preregistration: dict[str, Any] | None = None
    parent_manifest: dict[str, Any] | None = None
    if args.status == "PREREGISTERED":
        assert args.parent_draft is not None
        assert args.smoke_report is not None
        assert args.smoke_audit is not None
        assert args.owner is not None
        parent_path = args.parent_draft.absolute()
        smoke_path = args.smoke_report.absolute()
        audit_path = args.smoke_audit.absolute()
        expected_smoke_audit = validate_smoke_report(parent_path, smoke_path)
        observed_smoke_audit = json.loads(audit_path.read_text())
        if observed_smoke_audit != expected_smoke_audit:
            raise ValueError("smoke certificate does not equal the independent recomputation")
        parent_manifest = json.loads(parent_path.read_text())
        validate_manifest(
            parent_manifest, base_dir=parent_path.parent, require_preregistered=False
        )
        if parent_manifest.get("status") != "DRAFT" or parent_manifest.get(
            "runtime_ready"
        ) is not True:
            raise ValueError("preregistration requires a runtime-ready DRAFT parent")
        preregistration = {
            "owner": args.owner.strip(),
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "parent_draft": record(parent_path),
            "smoke_report": record(smoke_path),
            "smoke_audit": record(audit_path),
            "postprocess_gate": record(E70_POSTPROCESS_GATE),
            "capacity_gate": {
                "required_free_mb": args.min_free_mb,
                "observed_free_mb": observed_free_mb,
            },
        }

    teacher_manifest_path = E70_ROOT / "teacher_manifest.json"
    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    teacher_entries = teacher_manifest.get("motions", [])
    if len(teacher_entries) != 2:
        raise ValueError("E70 teacher manifest must contain exactly two specialists")
    teacher_checkpoints = [
        pathlib.Path(str(entry["checkpoint"])) for entry in teacher_entries
    ]
    motion_paths = [
        E70_ROOT / "motions/walk1_subject1_mj_z.npz",
        E70_ROOT / "motions/walk1_subject5_mj_z.npz",
    ]

    holosoma_files = {
        "holosoma_base_task": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/envs/base_task/base_task.py",
        "holosoma_action_manager": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/managers/action/manager.py",
        "holosoma_joint_action_term": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/managers/action/terms/joint_control.py",
        "holosoma_mujoco_simulator": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/simulator/mujoco/mujoco.py",
        "holosoma_wbt_manager": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/envs/wbt/wbt_manager.py",
        "holosoma_warp_backend": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/simulator/mujoco/backends/warp_backend.py",
        "holosoma_command_runtime": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/managers/command/terms/wbt.py",
        "holosoma_observation_manager": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/managers/observation/manager.py",
        "holosoma_observation_terms": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/managers/observation/terms/wbt.py",
        "holosoma_termination_manager": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/managers/termination/manager.py",
        "holosoma_termination_terms": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/managers/termination/terms/wbt.py",
        "holosoma_randomization_config": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/config_values/wbt/g1/randomization.py",
        "holosoma_observation_config": HOLOSOMA_ROOT
        / "src/holosoma/holosoma/config_values/wbt/g1/observation.py",
        "mujoco_warp_types": pathlib.Path(environment["mujoco_warp_types"]),
    }
    mujoco_warp_src = pathlib.Path(environment["mujoco_warp_types"]).parent
    holosoma_files.update(
        {
            "mujoco_warp_forward": mujoco_warp_src / "forward.py",
            "mujoco_warp_io": mujoco_warp_src / "io.py",
            "mujoco_warp_sleep": mujoco_warp_src / "sleep.py",
        }
    )
    frozen_paths = {
        "ambiguity_precheck": PRECHECK,
        "teacher_manifest": teacher_manifest_path,
        "teacher_checkpoint_0": teacher_checkpoints[0],
        "teacher_checkpoint_1": teacher_checkpoints[1],
        "motion_0": motion_paths[0],
        "motion_1": motion_paths[1],
        "evaluator": ROOT / "scripts/eval_e71_command_swap.py",
        "reset_layer": ROOT / "snmr/integration/counterfactual_eval.py",
        "bodyfix_layer": ROOT / "snmr/integration/wbt_bodyfix.py",
        "latent_runtime": ROOT / "snmr/integration/wbt_latent.py",
        "distillation_runtime": ROOT / "snmr/integration/distillation.py",
        "analyzer": ROOT / "scripts/analyze_e71_command_swap.py",
        "launcher": ROOT / "scripts/run_e71_command_swap.sh",
        "bundle_auditor": ROOT / "scripts/audit_e71_bundle.py",
        "protocol_document": ROOT / "docs/E71_COMMAND_SWAP_PROTOCOL.md",
        "freeze_manifest_generator": pathlib.Path(__file__),
        **holosoma_files,
    }
    frozen_files = {name: record(path) for name, path in frozen_paths.items()}

    checkpoints = []
    for arm, tag, filename in (
        ("explicit", "explicit", "c_prior_explicit_student.pt"),
        ("snmr", "snmr", "a_prior_snmr_student.pt"),
    ):
        for seed in (0, 1, 2):
            item = record(E70_ROOT / f"students/seed{seed}_{tag}/{filename}")
            checkpoints.append({"arm": arm, "training_seed": seed, **item})

    snmr_commit, snmr_state = git_state(ROOT)
    holosoma_commit, holosoma_state = git_state(HOLOSOMA_ROOT)
    e71_root = E70_ROOT.parent / "e71"
    if parent_manifest is None:
        draft_manifest_path = args.out.absolute()
        final_name = (
            "e71_freeze_manifest.json"
            if draft_manifest_path.name == "e71_freeze_draft.json"
            else f"{draft_manifest_path.stem}_preregistered.json"
        )
        artifact_paths = {
            "e71_root": str(e71_root.resolve()),
            "draft_manifest": str(draft_manifest_path),
            "preregistered_manifest": str(
                draft_manifest_path.with_name(final_name)
            ),
            "smoke_report": str(
                (e71_root / "reports/smoke_explicit_seed0_pair0.json").resolve()
            ),
            "smoke_audit": str((e71_root / "smoke_audit.json").resolve()),
            "explicit_reports": [
                str((e71_root / f"reports/explicit_seed{seed}.json").resolve())
                for seed in (0, 1, 2)
            ],
            "snmr_reports": [
                str((e71_root / f"reports/snmr_seed{seed}.json").resolve())
                for seed in (0, 1, 2)
            ],
            "explicit_gate": str((e71_root / "explicit_gate.json").resolve()),
            "analysis": str((e71_root / "analysis.json").resolve()),
            "integrity_bundle": str((e71_root / "integrity_bundle.json").resolve()),
            "logs_root": str((e71_root / "logs").resolve()),
            "holosoma_logs": str((e71_root / "holosoma_logs").resolve()),
        }
    else:
        artifact_paths = json.loads(json.dumps(parent_manifest["artifact_paths"]))
        if args.out.absolute() != pathlib.Path(
            artifact_paths["preregistered_manifest"]
        ):
            raise ValueError(
                "PREREGISTERED output path differs from the DRAFT artifact contract"
            )

    def evaluator_argv(arm: str, seed: int, num_envs: int) -> list[str]:
        return [
            str((ROOT / "scripts/eval_e71_command_swap.py").resolve()),
            "exp:g1-29dof-wbt",
            "simulator:mjwarp",
            "logger:disabled",
            "--training.num-envs",
            str(num_envs),
            "--training.seed",
            "404",
            "--training.headless",
            "True",
            "--training.name",
            f"e71_{arm}_seed{seed}",
            "--logger.base-dir",
            artifact_paths["holosoma_logs"],
            "--simulator.config.sim.max-episode-length-s",
            "20.0",
            "--command.setup-terms.motion-command.params.motion-config.motion-file",
            "",
            "--command.setup-terms.motion-command.params.motion-config.motion-dir",
            str((E70_ROOT / "motions").resolve()),
        ]

    payload = {
        "protocol": MANIFEST_PROTOCOL,
        "status": args.status,
        "report_protocol": REPORT_PROTOCOL,
        "evaluation_seed": 404,
        "training_seeds": [0, 1, 2],
        "num_pairs": 69,
        "cells_per_report": 276,
        "runtime_ready": runtime_ready,
        "artifact_paths": artifact_paths,
        "smoke_pair_id": 0,
        "smoke_argv": evaluator_argv("explicit", 0, 4),
        "commits": {"snmr": snmr_commit, "holosoma": holosoma_commit},
        "working_trees": {"snmr": snmr_state, "holosoma": holosoma_state},
        "environment": environment,
        "evaluation_conditions": json.loads(json.dumps(E71_EVALUATION_CONDITIONS)),
        "full_state_contract": {
            "tensor_names": list(E71_FULL_STATE_TENSOR_NAMES),
            "expanded_mujoco_model_fields": [],
            "tolerance": 1.0e-6,
        },
        "frozen_files": frozen_files,
        "checkpoints": checkpoints,
        "confirmatory_argv_by_arm_seed": {
            f"{arm}_seed{seed}": evaluator_argv(arm, seed, 276)
            for arm in ("explicit", "snmr")
            for seed in (0, 1, 2)
        },
    }
    if preregistration is not None:
        payload["preregistration"] = preregistration
    if parent_manifest is not None:
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
            name for name in immutable_fields if parent_manifest.get(name) != payload.get(name)
        ]
        if changed:
            raise ValueError(
                "DRAFT inputs changed before preregistration: " + ", ".join(changed)
            )
    validate_manifest(
        payload,
        base_dir=args.out.parent.resolve(),
        require_preregistered=args.status == "PREREGISTERED",
    )
    write_json_once(args.out, payload)
    print(f"wrote {args.status} E71 freeze manifest -> {args.out}")


if __name__ == "__main__":
    main()
