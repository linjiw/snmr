"""Reproducible run manifests and deterministic artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import torch


MANIFEST_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(
    split_paths: Mapping[str, Sequence[str | pathlib.Path]],
    *,
    root: str | pathlib.Path | None = None,
) -> dict:
    """Hash every dataset file and derive deterministic per-split and whole-dataset hashes."""
    root_path = pathlib.Path(root).resolve() if root is not None else None
    split_records = {}
    dataset_digest = hashlib.sha256()
    for split_name in sorted(split_paths):
        files = []
        split_digest = hashlib.sha256()
        total_bytes = 0
        for input_path in sorted((pathlib.Path(p).resolve() for p in split_paths[split_name]),
                                 key=str):
            stat = input_path.stat()
            file_hash = sha256_file(input_path)
            try:
                display_path = str(input_path.relative_to(root_path)) if root_path else str(input_path)
            except ValueError:
                display_path = str(input_path)
            record = {
                "path": display_path,
                "size_bytes": stat.st_size,
                "sha256": file_hash,
            }
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            split_digest.update(encoded)
            files.append(record)
            total_bytes += stat.st_size
        split_hash = split_digest.hexdigest()
        split_record = {
            "file_count": len(files),
            "size_bytes": total_bytes,
            "sha256": split_hash,
            "files": files,
        }
        split_records[split_name] = split_record
        dataset_digest.update(split_name.encode())
        dataset_digest.update(split_hash.encode())
    return {
        "algorithm": "sha256",
        "root": str(root_path) if root_path else None,
        "sha256": dataset_digest.hexdigest(),
        "splits": split_records,
    }


def git_state(repo_root: str | pathlib.Path) -> dict:
    root = pathlib.Path(repo_root)

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    if revision.returncode != 0:
        return {"available": False, "sha": None, "dirty": None}
    status_lines = [line for line in status.stdout.splitlines() if line]
    tracked_diff = run("diff", "--binary", "HEAD", "--")
    return {
        "available": True,
        "sha": revision.stdout.strip(),
        "dirty": bool(status_lines),
        "tracked_dirty": any(not line.startswith("??") for line in status_lines),
        "tracked_diff_sha256": (
            hashlib.sha256(tracked_diff.stdout.encode()).hexdigest()
            if tracked_diff.returncode == 0 and tracked_diff.stdout
            else None
        ),
        "untracked_paths": sum(line.startswith("??") for line in status_lines),
    }


def runtime_state(device: str) -> dict:
    cuda_device = None
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        cuda_device = torch.cuda.get_device_name(torch.device(device))
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_device,
    }


def capture_rng_state() -> dict:
    """Capture Python, NumPy, CPU Torch, and available CUDA generator states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([generator_state.cpu() for generator_state in state["cuda"]])


def source_fingerprint(
    repo_root: str | pathlib.Path,
    trainer: str | pathlib.Path,
) -> dict:
    """Fingerprint package code, the active trainer, and dependency metadata."""
    root = pathlib.Path(repo_root).resolve()
    paths = list((root / "snmr").rglob("*.py"))
    trainer_path = root / trainer
    if trainer_path.is_file():
        paths.append(trainer_path)
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        paths.append(pyproject)
    return dataset_fingerprint({"source": sorted(set(paths), key=str)}, root=root)


class RunManifest:
    """Atomic JSON manifest updated as checkpoints are retained."""

    def __init__(self, path: str | pathlib.Path, data: dict):
        self.path = pathlib.Path(path)
        self.data = data

    @classmethod
    def start(
        cls,
        path: str | pathlib.Path,
        *,
        trainer: str,
        repo_root: str | pathlib.Path,
        argv: Sequence[str],
        config: Mapping[str, Any],
        dataset: Mapping[str, Any],
        training: Mapping[str, Any],
        objectives: Mapping[str, Any],
        resume: bool,
    ) -> "RunManifest":
        path = pathlib.Path(path)
        invocation = {
            "started_utc": utc_now(),
            "argv": list(argv),
            "command": shlex.join(argv),
            "cwd": os.getcwd(),
            "resume": bool(resume),
        }
        if resume and path.exists():
            data = json.loads(path.read_text())
            data.setdefault("invocations", []).append(invocation)
            current_config = _json_safe(dict(config))
            current_dataset = _json_safe(dict(dataset))
            original_config = data.get("config", {})
            compared_keys = (set(original_config) | set(current_config)) - {"resume"}
            config_differences = {
                key: {
                    "original": original_config.get(key),
                    "current": current_config.get(key),
                }
                for key in sorted(compared_keys)
                if original_config.get(key) != current_config.get(key)
            }
            data.setdefault("resume_checks", []).append({
                "checked_utc": utc_now(),
                "config_matches_original_except_resume": not config_differences,
                "config_differences": config_differences,
                "dataset_hash": current_dataset.get("sha256"),
                "dataset_matches_original": (
                    current_dataset.get("sha256") == data.get("dataset", {}).get("sha256")
                ),
                "config": current_config,
            })
            data["status"] = "running"
            data["updated_utc"] = utc_now()
            manifest = cls(path, data)
            manifest.write()
            return manifest

        now = utc_now()
        data = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "trainer": trainer,
            "status": "running",
            "created_utc": now,
            "updated_utc": now,
            "git": git_state(repo_root),
            "source": source_fingerprint(repo_root, trainer),
            "runtime": runtime_state(str(config.get("device", "unknown"))),
            "invocations": [invocation],
            "config": _json_safe(dict(config)),
            "dataset": _json_safe(dict(dataset)),
            "training": _json_safe(dict(training)),
            "objectives": _json_safe(dict(objectives)),
            "progress": {
                "step": 0,
                "completion_state": "running",
                "robot_exposures": {},
            },
            "checkpoints": [],
        }
        manifest = cls(path, data)
        manifest.write()
        return manifest

    def update_progress(
        self,
        *,
        step: int,
        robot_exposures: Mapping[str, int],
        checkpoint_path: str | pathlib.Path | None = None,
        complete: bool = False,
    ) -> None:
        self.data["updated_utc"] = utc_now()
        self.data["status"] = "completed" if complete else "running"
        self.data["progress"] = {
            "step": int(step),
            "completion_state": "completed" if complete else "checkpointed",
            "robot_exposures": {key: int(value) for key, value in robot_exposures.items()},
        }
        if checkpoint_path is not None:
            checkpoint = pathlib.Path(checkpoint_path)
            record = {
                "step": int(step),
                "path": str(checkpoint.resolve()),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "retained_utc": utc_now(),
                "complete": bool(complete),
            }
            existing = [
                item for item in self.data.setdefault("checkpoints", [])
                if item.get("path") != record["path"]
            ]
            self.data["checkpoints"] = existing + [record]
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
