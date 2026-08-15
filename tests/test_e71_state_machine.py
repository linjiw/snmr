from __future__ import annotations

import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_e71_command_swap.sh"
FREEZER = ROOT / "scripts/prepare_e71_freeze.py"


def test_launcher_is_executable_and_has_valid_shell_syntax() -> None:
    assert os.access(LAUNCHER, os.X_OK)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_status_is_read_only_and_reports_both_manifest_stages() -> None:
    completed = subprocess.run(
        [str(LAUNCHER), "status"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "E71 draft manifest:" in completed.stdout
    assert "E71 final manifest:" in completed.stdout
    assert "GPU free MiB:" in completed.stdout


def test_direct_preregistration_without_lineage_is_rejected_before_write(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "final.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(FREEZER),
            "--status",
            "PREREGISTERED",
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode != 0
    assert "requires --parent-draft" in completed.stderr
    assert not output.exists()


def test_freezer_rejects_capacity_threshold_drift_before_write(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "draft.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(FREEZER),
            "--status",
            "DRAFT",
            "--out",
            str(output),
            "--min-free-mb",
            "30000",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode != 0
    assert "exactly 26,000 MiB" in completed.stderr
    assert not output.exists()
