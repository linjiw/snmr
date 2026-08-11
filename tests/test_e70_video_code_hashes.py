from __future__ import annotations

import hashlib

import pytest

from scripts.check_e70_video_code_hashes import PROTOCOL, validate_hash_manifest


def test_validate_hash_manifest_accepts_exact_files_and_rejects_drift(tmp_path) -> None:
    target = tmp_path / "script.py"
    target.write_bytes(b"frozen")
    manifest = {
        "protocol": PROTOCOL,
        "sha256": {"script.py": hashlib.sha256(target.read_bytes()).hexdigest()},
    }
    assert validate_hash_manifest(manifest, tmp_path) == ["script.py"]

    target.write_bytes(b"drifted")
    with pytest.raises(ValueError, match="code drift"):
        validate_hash_manifest(manifest, tmp_path)


def test_validate_hash_manifest_rejects_path_escape(tmp_path) -> None:
    manifest = {"protocol": PROTOCOL, "sha256": {"../escape": "0" * 64}}
    with pytest.raises(ValueError, match="escapes repository root"):
        validate_hash_manifest(manifest, tmp_path)
