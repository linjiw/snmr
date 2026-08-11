#!/usr/bin/env python
"""Create an anonymous, deterministic copy of an E70 analyzer report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
from typing import Any


PROTOCOL = "E70 preregistered analysis v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def anonymous_path(path: pathlib.Path, *, repo_root: pathlib.Path) -> str:
    path = path.resolve()
    repo_root = repo_root.resolve()
    try:
        return "repo/" + path.relative_to(repo_root).as_posix()
    except ValueError:
        pass
    parts = path.parts
    if "snmr-research" in parts:
        index = parts.index("snmr-research")
        return "artifacts/" + pathlib.PurePosixPath(*parts[index + 1 :]).as_posix()
    return "external/" + path.name


def anonymize_analysis(
    analysis: dict[str, Any], *, source_sha256: str, repo_root: pathlib.Path
) -> dict[str, Any]:
    if analysis.get("protocol") != PROTOCOL:
        raise ValueError("reproducibility report requires the frozen E70 protocol")
    if analysis.get("seeds") not in ([0], [0, 1, 2]):
        raise ValueError("reproducibility report accepts only seed 0 or the final seed set")
    records = analysis.get("inputs")
    if not isinstance(records, list) or not records:
        raise ValueError("analysis has no hash-bound inputs")
    result = copy.deepcopy(analysis)
    anonymized = []
    for record in records:
        try:
            path = pathlib.Path(record["path"])
            digest = str(record["sha256"])
        except (KeyError, TypeError) as exc:
            raise ValueError("analysis contains a malformed input record") from exc
        if len(digest) != 64:
            raise ValueError("analysis contains a malformed SHA-256")
        anonymized.append(
            {"path": anonymous_path(path, repo_root=repo_root), "sha256": digest}
        )
    result["inputs"] = anonymized
    result["reproducibility_copy"] = {
        "anonymous_paths": True,
        "source_analysis_sha256": source_sha256,
        "note": "Numbers are unchanged; only host-local input path prefixes were replaced.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=pathlib.Path, required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.analysis.read_text())
    result = anonymize_analysis(
        source,
        source_sha256=sha256_file(args.analysis),
        repo_root=args.repo_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"wrote anonymous reproducibility report -> {args.out}")


if __name__ == "__main__":
    main()
