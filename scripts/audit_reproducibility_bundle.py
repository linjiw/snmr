#!/usr/bin/env python
"""Verify hashes, links, and anonymity of the tracked reproducibility index."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import pathlib
import re
import socket
from typing import Any


PROTOCOL = "SNMR ICRA 2027 anonymous reproducibility bundle v1"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HOME_PATH_RE = re.compile(r"/(?:home|Users)/[^/\s]+")
DATA_USER_PATH_RE = re.compile(r"/data/[^/\s]+/")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
TEXT_SUFFIXES = {".md", ".json", ".txt", ".tsv"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: dict[str, Any], *, repo_root: pathlib.Path) -> list[str]:
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("unexpected reproducibility-manifest protocol")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("reproducibility manifest has no entries")
    repo_root = repo_root.resolve()
    seen: set[str] = set()
    validated = []
    for entry in entries:
        try:
            relative = str(entry["path"])
            expected = str(entry["sha256"])
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed reproducibility-manifest entry") from exc
        if relative in seen or len(expected) != 64:
            raise ValueError(f"duplicate path or malformed hash: {relative}")
        seen.add(relative)
        path = (repo_root / relative).resolve()
        if not path.is_relative_to(repo_root):
            raise ValueError(f"manifest path escapes the repository: {relative}")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"hash verification failed: {relative}")
        validated.append(relative)
    return validated


def anonymity_findings(bundle_root: pathlib.Path) -> list[str]:
    identifiers = {
        value.casefold()
        for value in (
            getpass.getuser(),
            os.environ.get("USER", ""),
            os.environ.get("LOGNAME", ""),
            socket.gethostname(),
        )
        if len(value) >= 3
    }
    findings = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="replace")
        lowered = text.casefold()
        if EMAIL_RE.search(text):
            findings.append(f"{path}: email address")
        if HOME_PATH_RE.search(text):
            findings.append(f"{path}: user home path")
        if DATA_USER_PATH_RE.search(text):
            findings.append(f"{path}: host-local data path")
        for identifier in sorted(identifiers):
            if identifier in lowered:
                findings.append(f"{path}: host/user identifier")
    return findings


def broken_links(bundle_root: pathlib.Path) -> list[str]:
    broken = []
    for path in sorted(bundle_root.rglob("*.md")):
        for target in MARKDOWN_LINK_RE.findall(path.read_text()):
            target = target.split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{path}: {target}")
    return broken


def validate_report_inputs(
    report_path: pathlib.Path, *, repo_root: pathlib.Path, artifact_root: pathlib.Path
) -> list[str]:
    report = json.loads(report_path.read_text())
    records = report.get("inputs")
    if not isinstance(records, list) or not records:
        raise ValueError("anonymous analyzer report has no inputs")
    roots = {"repo": repo_root.resolve(), "artifacts": artifact_root.resolve()}
    validated = []
    for record in records:
        logical = pathlib.PurePosixPath(str(record["path"]))
        if not logical.parts or logical.parts[0] not in roots:
            raise ValueError(f"unsupported anonymous input path: {logical}")
        root = roots[logical.parts[0]]
        path = root.joinpath(*logical.parts[1:]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"anonymous input path escapes its root: {logical}")
        if not path.is_file() or sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"behavioral input hash verification failed: {logical}")
        validated.append(str(logical))
    return validated


def audit_bundle(
    bundle_root: pathlib.Path,
    repo_root: pathlib.Path,
    artifact_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    bundle_root, repo_root = bundle_root.resolve(), repo_root.resolve()
    if not bundle_root.is_relative_to(repo_root):
        raise ValueError("bundle root must be inside the repository")
    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = validate_manifest(manifest, repo_root=repo_root)
    anonymity = anonymity_findings(bundle_root)
    links = broken_links(bundle_root)
    if anonymity:
        raise ValueError("anonymity audit failed: " + "; ".join(anonymity))
    if links:
        raise ValueError("bundle links do not resolve: " + "; ".join(links))
    artifact_hashes = (
        validate_report_inputs(
            bundle_root / "reports" / "e70_seed0_analysis.json",
            repo_root=repo_root,
            artifact_root=artifact_root,
        )
        if artifact_root is not None
        else []
    )
    return {
        "protocol": PROTOCOL,
        "hashes_verified": len(hashes),
        "text_files_scanned": sum(
            path.is_file() and path.suffix in TEXT_SUFFIXES for path in bundle_root.rglob("*")
        ),
        "links_checked": sum(
            len(MARKDOWN_LINK_RE.findall(path.read_text()))
            for path in bundle_root.rglob("*.md")
        ),
        "anonymity_findings": 0,
        "broken_links": 0,
        "behavioral_hashes_verified": len(artifact_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=pathlib.Path, required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--artifact-root", type=pathlib.Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_bundle(args.bundle_root, args.repo_root, args.artifact_root), indent=2
        )
    )


if __name__ == "__main__":
    main()
