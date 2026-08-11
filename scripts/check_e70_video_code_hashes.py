#!/usr/bin/env python
"""Fail closed if the frozen E70 paper/video implementation has drifted."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib


PROTOCOL = "E70 paper-video code freeze v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_hash_manifest(manifest: dict, root: pathlib.Path) -> list[str]:
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("unexpected E70 paper-video code-freeze protocol")
    expected = manifest.get("sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("E70 paper-video code-freeze manifest has no hashes")

    root = root.resolve()
    validated = []
    for relative, wanted in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"hash target escapes repository root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != wanted:
            raise ValueError(
                f"E70 paper-video code drift for {relative}: "
                f"expected {wanted}, observed {observed}"
            )
        validated.append(relative)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    validated = validate_hash_manifest(json.loads(args.manifest.read_text()), args.root)
    print(f"validated {len(validated)} frozen E70 paper/video files")


if __name__ == "__main__":
    main()
