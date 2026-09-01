import hashlib
from pathlib import Path
import subprocess

import pytest

from snmr.provenance import (
    ArtifactMismatchError,
    ArtifactSnapshot,
    MJCF_BUNDLE_HASH_DOMAIN,
    MJCF_BUNDLE_HASH_SCHEMA,
    MjcfBundleSnapshot,
    git_revision,
    source_revision_manifest,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _committed_repo(path: Path) -> str:
    path.mkdir()
    _git(path, "init", "-q")
    tracked = path / "tracked.txt"
    tracked.write_text("frozen\n")
    _git(path, "add", "tracked.txt")
    _git(
        path,
        "-c",
        "user.name=SNMR Test",
        "-c",
        "user.email=snmr-test@example.invalid",
        "commit",
        "-q",
        "-m",
        "freeze",
    )
    return _git(path, "rev-parse", "HEAD")


def _bundle_digest(snapshot: MjcfBundleSnapshot, domain: bytes) -> str:
    """Independent spelling of the versioned bundle-hash wire format."""

    digest = hashlib.sha256()
    digest.update(domain)
    entrypoint = snapshot.entrypoint.encode("utf-8")
    digest.update(len(entrypoint).to_bytes(8, "big"))
    digest.update(entrypoint)
    for relative, member in snapshot.members:
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(member.size_bytes.to_bytes(8, "big"))
        digest.update(member.data)
    return digest.hexdigest()


def test_artifact_snapshot_hashes_consumed_buffer_and_detects_mutation(tmp_path: Path):
    source = tmp_path / "motion.npz"
    original = b"motion-buffer-v1\x00\xff"
    source.write_bytes(original)

    snapshot = ArtifactSnapshot.capture(source)
    assert snapshot.data == original
    assert snapshot.sha256 == hashlib.sha256(original).hexdigest()
    assert snapshot.manifest()["size_bytes"] == len(original)

    source.write_bytes(b"motion-buffer-v2")
    with pytest.raises(ArtifactMismatchError, match="changed after capture"):
        snapshot.assert_source_unchanged()

    materialized = snapshot.materialize(tmp_path / "worker" / "motion.npz")
    assert materialized.read_bytes() == original


def test_mjcf_bundle_hash_covers_only_referenced_files_and_materializes_layout(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    mesh = assets / "link.obj"
    mesh.write_bytes(b"v 0 0 0\n")
    (assets / "unrelated.obj").write_bytes(b"not referenced")
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text(
        '<mujoco><compiler meshdir="assets"/><asset>'
        '<mesh name="link" file="link.obj"/></asset><worldbody/></mujoco>'
    )

    first = MjcfBundleSnapshot.capture(mjcf)
    assert first.entrypoint_snapshot.sha256 == hashlib.sha256(mjcf.read_bytes()).hexdigest()
    assert {name for name, _ in first.members} == {"robot.xml", "assets/link.obj"}
    manifest = first.manifest(include_paths=False)
    assert manifest["hash_schema"] == MJCF_BUNDLE_HASH_SCHEMA == "snmr.mjcf-bundle.v0.2"
    assert MJCF_BUNDLE_HASH_DOMAIN == b"snmr.mjcf-bundle.v0.2\0"
    assert first.sha256 == _bundle_digest(first, MJCF_BUNDLE_HASH_DOMAIN)
    assert first.sha256 != _bundle_digest(first, b"snmr.mjcf-bundle.v0.1\0")

    (assets / "unrelated.obj").write_bytes(b"changed but still not referenced")
    assert MjcfBundleSnapshot.capture(mjcf).sha256 == first.sha256
    mesh.write_bytes(b"v 1 2 3\n")
    second = MjcfBundleSnapshot.capture(mjcf)
    assert second.sha256 != first.sha256

    runtime_mjcf = first.materialize(tmp_path / "worker-asset")
    assert runtime_mjcf.relative_to(tmp_path / "worker-asset").as_posix() == "robot.xml"
    assert (runtime_mjcf.parent / "assets" / "link.obj").read_bytes() == b"v 0 0 0\n"


def test_mjcf_bundle_hash_binds_entrypoint_identity(tmp_path: Path):
    first_xml = tmp_path / "first.xml"
    second_xml = tmp_path / "second.xml"
    first_xml.write_text('<mujoco><include file="second.xml"/></mujoco>')
    second_xml.write_text('<mujoco><include file="first.xml"/></mujoco>')

    first = MjcfBundleSnapshot.capture(first_xml)
    second = MjcfBundleSnapshot.capture(second_xml)
    assert {name for name, _ in first.members} == {name for name, _ in second.members}
    assert first.entrypoint != second.entrypoint
    assert first.sha256 != second.sha256


def test_git_revision_records_exact_clean_and_dirty_states(tmp_path: Path):
    repo = tmp_path / "repo"
    commit = _committed_repo(repo)

    clean = git_revision(repo / "tracked.txt")
    assert clean.status == "available"
    assert clean.commit == commit
    assert clean.dirty is False
    assert clean.repo_root == str(repo.resolve())

    (repo / "tracked.txt").write_text("mutated\n")
    dirty = git_revision(repo)
    assert dirty.commit == commit
    assert dirty.dirty is True


def test_source_manifest_uses_explicit_null_for_absent_repositories(tmp_path: Path):
    repo = tmp_path / "snmr"
    commit = _committed_repo(repo)
    manifest = source_revision_manifest(
        snmr_path=repo,
        newton_path=tmp_path / "absent-newton",
        isaac_lab_path=tmp_path / "absent-isaac-lab",
    )

    assert manifest["snmr_commit"] == commit
    assert manifest["snmr_dirty"] is False
    assert manifest["newton_commit"] is None
    assert manifest["newton_dirty"] is None
    assert manifest["newton_repo_status"] == "unavailable"
    assert manifest["isaac_lab_commit"] is None
    assert manifest["isaac_lab_dirty"] is None
    assert manifest["isaac_lab_repo_status"] == "unavailable"
