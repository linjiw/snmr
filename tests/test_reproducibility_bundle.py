from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_reproducibility_bundle import (
    anonymity_findings,
    audit_bundle,
    validate_manifest,
    validate_report_inputs,
)
from scripts.build_reproducibility_report import anonymize_analysis


REPO = Path(__file__).resolve().parents[1]


def test_anonymized_report_rewrites_repository_and_artifact_paths() -> None:
    analysis = {
        "protocol": "E70 preregistered analysis v1",
        "seeds": [0],
        "inputs": [
            {"path": str(REPO / "paper" / "main.tex"), "sha256": "a" * 64},
            {
                "path": "/volume/snmr-research/e70/students/report.json",
                "sha256": "b" * 64,
            },
        ],
    }
    result = anonymize_analysis(analysis, source_sha256="c" * 64, repo_root=REPO)
    assert result["inputs"][0]["path"] == "repo/paper/main.tex"
    assert result["inputs"][1]["path"] == "artifacts/e70/students/report.json"
    assert result["reproducibility_copy"]["anonymous_paths"] is True


def test_anonymity_scan_detects_email_and_home_path(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text("contact person@example.org in /home/account/project")
    findings = anonymity_findings(tmp_path)
    assert any("email" in item for item in findings)
    assert any("home path" in item for item in findings)


def test_manifest_rejects_hash_drift(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("first")
    manifest = {
        "protocol": "SNMR ICRA 2027 anonymous reproducibility bundle v1",
        "entries": [
            {"path": "file.txt", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        ],
    }
    changed = copy.deepcopy(manifest)
    target.write_text("second")
    with pytest.raises(ValueError, match="hash verification failed"):
        validate_manifest(changed, repo_root=tmp_path)


def test_anonymous_report_inputs_resolve_under_declared_roots(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    report_dir = repo / "reproducibility" / "reports"
    target = artifacts / "e70" / "report.json"
    report_dir.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    target.write_text("payload")
    report = {
        "inputs": [
            {
                "path": "artifacts/e70/report.json",
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        ]
    }
    report_path = report_dir / "analysis.json"
    report_path.write_text(json.dumps(report))
    assert validate_report_inputs(
        report_path, repo_root=repo, artifact_root=artifacts
    ) == ["artifacts/e70/report.json"]


def test_tracked_reproducibility_bundle_passes() -> None:
    result = audit_bundle(REPO / "reproducibility", REPO)
    assert result["hashes_verified"] >= 10
    assert result["anonymity_findings"] == 0
    assert result["broken_links"] == 0
    assert result["behavioral_hashes_verified"] == 0
