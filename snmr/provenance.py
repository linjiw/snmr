"""Fail-closed input and source provenance for simulator workers.

The simulator adapters run in different Python environments, so provenance must be
established by each worker rather than copied from a launcher manifest.  This module
provides two small building blocks for that contract:

* immutable in-memory snapshots whose digest is computed from the same bytes consumed by
  the worker; and
* explicit Git states that distinguish a clean checkout from an unavailable repository.

Path-only simulator APIs can consume a snapshot after it is materialized in a private
temporary directory.  This removes the time-of-check/time-of-use race against the original
artifact while retaining relative MJCF asset paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


PROVENANCE_SCHEMA_VERSION = "snmr.provenance.v0.1"
MJCF_BUNDLE_HASH_SCHEMA = "snmr.mjcf-bundle.v0.2"
MJCF_BUNDLE_HASH_DOMAIN = (MJCF_BUNDLE_HASH_SCHEMA + "\0").encode("ascii")


class ArtifactMismatchError(ValueError):
    """Raised when an input no longer matches its bound digest."""


def sha256_bytes(data: bytes) -> str:
    """Hash an immutable buffer that will also be handed to the consumer."""

    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ArtifactSnapshot:
    """One file captured atomically enough for deterministic worker consumption."""

    source_path: Path
    data: bytes = field(repr=False)
    sha256: str
    size_bytes: int

    @classmethod
    def capture(cls, path: str | Path) -> "ArtifactSnapshot":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        before = source.stat()
        with source.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            data = stream.read()
        after = source.stat()
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_opened = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_opened or identity_opened != identity_after:
            raise ArtifactMismatchError(f"artifact changed while being captured: {source}")
        if len(data) != opened.st_size:
            raise ArtifactMismatchError(
                f"short read while capturing {source}: read {len(data)} of {opened.st_size} bytes"
            )
        return cls(
            source_path=source,
            data=data,
            sha256=sha256_bytes(data),
            size_bytes=len(data),
        )

    def manifest(self, *, include_path: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if include_path:
            result["source_path"] = str(self.source_path)
        return result

    def assert_source_unchanged(self) -> None:
        """Fail if the source path no longer contains the captured bytes."""

        current = ArtifactSnapshot.capture(self.source_path)
        if current.sha256 != self.sha256 or current.size_bytes != self.size_bytes:
            raise ArtifactMismatchError(
                f"artifact changed after capture: {self.source_path} "
                f"({self.sha256} -> {current.sha256})"
            )

    def materialize(self, path: str | Path) -> Path:
        """Write the captured buffer to a new path for a path-only consumer."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(self.data)
        return destination


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _resolve_reference(base: Path, reference: str) -> Path:
    candidate = Path(reference).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _reject_absolute_reference(reference: str, *, label: str) -> None:
    if Path(reference).expanduser().is_absolute():
        raise ValueError(
            f"absolute MJCF {label} reference cannot be safely materialized: {reference!r}"
        )


def _compiler_directories(trees: Sequence[ET.Element]) -> dict[str, str]:
    values: dict[str, str] = {}
    for tree in trees:
        for element in tree.iter():
            if _tag(element) != "compiler":
                continue
            for name in ("assetdir", "meshdir", "texturedir"):
                value = element.get(name)
                if value is None:
                    continue
                _reject_absolute_reference(value, label=name)
                previous = values.get(name)
                if previous is not None and previous != value:
                    raise ValueError(f"conflicting MJCF compiler {name}: {previous!r} vs {value!r}")
                values[name] = value
    return values


def _asset_base(entrypoint: Path, tag: str, compiler: Mapping[str, str]) -> Path:
    assetdir = compiler.get("assetdir", "")
    if tag == "mesh":
        directory = compiler.get("meshdir", assetdir)
    elif tag == "texture":
        directory = compiler.get("texturedir", assetdir)
    else:
        directory = assetdir
    return _resolve_reference(entrypoint.parent, directory) if directory else entrypoint.parent


@dataclass(frozen=True)
class MjcfBundleSnapshot:
    """An MJCF entrypoint plus every file reference required by that XML bundle."""

    source_root: Path
    entrypoint: str
    members: tuple[tuple[str, ArtifactSnapshot], ...]
    sha256: str
    size_bytes: int

    @property
    def entrypoint_snapshot(self) -> ArtifactSnapshot:
        by_name = dict(self.members)
        return by_name[self.entrypoint]

    @classmethod
    def capture(cls, path: str | Path) -> "MjcfBundleSnapshot":
        entrypoint = Path(path).expanduser().resolve()
        if not entrypoint.is_file():
            raise FileNotFoundError(entrypoint)

        snapshots: dict[Path, ArtifactSnapshot] = {}
        roots: dict[Path, ET.Element] = {}
        pending = [entrypoint]
        while pending:
            xml_path = pending.pop()
            if xml_path in roots:
                continue
            snapshot = ArtifactSnapshot.capture(xml_path)
            try:
                root = ET.fromstring(snapshot.data)
            except ET.ParseError as exc:
                raise ValueError(f"invalid MJCF XML {xml_path}: {exc}") from exc
            snapshots[xml_path] = snapshot
            roots[xml_path] = root
            for element in root.iter():
                if _tag(element) == "include" and element.get("file"):
                    reference = str(element.get("file"))
                    _reject_absolute_reference(reference, label="include")
                    pending.append(_resolve_reference(xml_path.parent, reference))

        compiler = _compiler_directories(tuple(roots.values()))
        for xml_path, root in roots.items():
            for element in root.iter():
                tag = _tag(element)
                reference = element.get("file")
                if reference is None or tag in {"include", "compiler"}:
                    continue
                _reject_absolute_reference(reference, label=tag)
                if tag not in {"mesh", "texture", "hfield", "skin"}:
                    # Unknown file-bearing elements are still inputs.  Resolve them from the
                    # XML containing the element instead of silently omitting provenance.
                    base = xml_path.parent
                else:
                    base = _asset_base(entrypoint, tag, compiler)
                asset_path = _resolve_reference(base, reference)
                if asset_path not in snapshots:
                    snapshots[asset_path] = ArtifactSnapshot.capture(asset_path)

        common_root = Path(os.path.commonpath([str(path.parent) for path in snapshots])).resolve()
        members = tuple(sorted(
            ((path.relative_to(common_root).as_posix(), snapshot)
             for path, snapshot in snapshots.items()),
            key=lambda item: item[0],
        ))
        entrypoint_name = entrypoint.relative_to(common_root).as_posix()
        digest = hashlib.sha256()
        # v0.2 binds both the captured member bytes/layout and which XML member is
        # interpreted as the model entrypoint.  The explicit domain prevents the
        # entrypoint-bound format from being confused with legacy v0.1 digests.
        digest.update(MJCF_BUNDLE_HASH_DOMAIN)
        encoded_entrypoint = entrypoint_name.encode("utf-8")
        digest.update(len(encoded_entrypoint).to_bytes(8, "big"))
        digest.update(encoded_entrypoint)
        total_bytes = 0
        for relative, snapshot in members:
            encoded_path = relative.encode("utf-8")
            digest.update(len(encoded_path).to_bytes(8, "big"))
            digest.update(encoded_path)
            digest.update(snapshot.size_bytes.to_bytes(8, "big"))
            digest.update(snapshot.data)
            total_bytes += snapshot.size_bytes
        return cls(
            source_root=common_root,
            entrypoint=entrypoint_name,
            members=members,
            sha256=digest.hexdigest(),
            size_bytes=total_bytes,
        )

    def manifest(self, *, include_paths: bool = True) -> dict[str, Any]:
        files = [
            {
                "relative_path": relative,
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            }
            for relative, snapshot in self.members
        ]
        result: dict[str, Any] = {
            "hash_schema": MJCF_BUNDLE_HASH_SCHEMA,
            "sha256": self.sha256,
            "entrypoint": self.entrypoint,
            "entrypoint_sha256": self.entrypoint_snapshot.sha256,
            "file_count": len(files),
            "size_bytes": self.size_bytes,
            "files": files,
        }
        if include_paths:
            result["source_root"] = str(self.source_root)
        return result

    def materialize(self, directory: str | Path) -> Path:
        destination = Path(directory)
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True)
        for relative, snapshot in self.members:
            snapshot.materialize(destination / relative)
        return destination / self.entrypoint


@dataclass(frozen=True)
class GitRevision:
    """A Git checkout state with explicit absence/error semantics."""

    status: str
    commit: str | None
    dirty: bool | None
    repo_root: str | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "available"


def git_revision(path: str | Path | None) -> GitRevision:
    if path is None:
        return GitRevision("unavailable", None, None, None, "no repository path detected")
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return GitRevision("unavailable", None, None, None, f"path does not exist: {candidate}")
    start = candidate.parent if candidate.is_file() else candidate

    def invoke(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(start), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    try:
        root_result = invoke("rev-parse", "--show-toplevel")
    except OSError as exc:
        return GitRevision("error", None, None, None, str(exc))
    if root_result.returncode != 0:
        message = root_result.stderr.strip() or "not a Git checkout"
        return GitRevision("unavailable", None, None, None, message)
    root = Path(root_result.stdout.strip()).resolve()
    commit_result = invoke("rev-parse", "HEAD")
    status_result = invoke("status", "--porcelain", "--untracked-files=normal")
    if commit_result.returncode != 0 or status_result.returncode != 0:
        error = commit_result.stderr.strip() or status_result.stderr.strip() or "Git query failed"
        return GitRevision("error", None, None, str(root), error)
    commit = commit_result.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        return GitRevision("error", None, None, str(root), f"invalid Git revision {commit!r}")
    return GitRevision(
        status="available",
        commit=commit,
        dirty=bool(status_result.stdout.strip()),
        repo_root=str(root),
    )


def module_checkout_hint(module_name: str) -> Path | None:
    """Find a package location without importing its simulator stack."""

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        return Path(spec.origin)
    locations = spec.submodule_search_locations
    if locations:
        first = next(iter(locations), None)
        return Path(first) if first is not None else None
    return None


def source_revision_manifest(
    *,
    snmr_path: str | Path,
    newton_path: str | Path | None = None,
    isaac_lab_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return flat, stable revision fields for every generated report.

    Missing optional simulator checkouts are represented by null commit/dirty values and an
    ``unavailable`` status.  They are never confused with a clean checkout.
    """

    resolved_newton = newton_path if newton_path is not None else module_checkout_hint("newton")
    resolved_isaac = (
        isaac_lab_path if isaac_lab_path is not None else module_checkout_hint("isaaclab")
    )
    revisions = {
        "snmr": git_revision(snmr_path),
        "newton": git_revision(resolved_newton),
        "isaac_lab": git_revision(resolved_isaac),
    }
    result: dict[str, Any] = {"provenance_schema_version": PROVENANCE_SCHEMA_VERSION}
    for name, revision in revisions.items():
        result[f"{name}_commit"] = revision.commit
        result[f"{name}_dirty"] = revision.dirty
        result[f"{name}_repo_status"] = revision.status
        result[f"{name}_repo_root"] = revision.repo_root
        if revision.error is not None:
            result[f"{name}_repo_error"] = revision.error
    return result
