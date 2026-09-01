"""Deterministic contracts shared by the two G0 forward-kinematics workers.

The MuJoCo and PhysX workers intentionally run in different Python environments.  This
module contains only NumPy/XML helpers so both workers can validate the exact sample
buffer and apply the same numerical gate without importing either simulator.

Quaternion arrays in this protocol use ``xyzw``.  SNMR's internal convention remains
``wxyz``; each backend adapter converts explicitly at its boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np


FK_PARITY_SCHEMA_VERSION = "snmr.g0_fk_parity.v0.1"
URDF_BUNDLE_HASH_SCHEMA_VERSION = "snmr.urdf-bundle.v0.2"
DIRECTORY_BUNDLE_HASH_SCHEMA_VERSION = "snmr.directory-bundle.v0.2"
POSITION_THRESHOLD_M = 1.0e-3
ORIENTATION_THRESHOLD_RAD = 1.0e-3

G1_KEY_LINKS: tuple[tuple[str, str], ...] = (
    ("pelvis", "pelvis"),
    ("torso", "torso_link"),
    ("head", "head_link"),
    ("left_hand", "left_rubber_hand"),
    ("right_hand", "right_rubber_hand"),
    ("left_foot", "left_foot_contact_point"),
    ("right_foot", "right_foot_contact_point"),
)


def _require_float64_c(array: np.ndarray, *, label: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.dtype("<f8") and value.dtype != np.dtype("=f8"):
        raise TypeError(f"{label} must be a float64 buffer, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{label} must be C contiguous")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value


def float64_buffer_sha256(array: np.ndarray) -> str:
    """Hash the exact bytes of a finite, C-contiguous float64 memory buffer.

    Shape and field meaning are recorded separately in the manifest.  The function does
    not silently cast or copy: a worker must hash the buffer it actually consumes.
    """

    value = _require_float64_c(array, label="array")
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _framed_bytes(payload: bytes) -> bytes:
    return len(payload).to_bytes(8, "big") + payload


def joint_sample_payload_sha256(
    joint_names: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
    samples: np.ndarray,
) -> str:
    """Hash names, bounds, shape, and the exact canonical sample buffers."""

    lower_array = _require_float64_c(np.asarray(lower), label="lower")
    upper_array = _require_float64_c(np.asarray(upper), label="upper")
    sample_array = _require_float64_c(np.asarray(samples), label="samples")
    names = tuple(str(name) for name in joint_names)
    if lower_array.shape != (len(names),) or upper_array.shape != (len(names),):
        raise ValueError("joint bounds must have one entry per joint name")
    if sample_array.ndim != 2 or sample_array.shape[1] != len(names):
        raise ValueError("samples must have shape (N, len(joint_names))")

    digest = hashlib.sha256()
    digest.update(b"snmr.g0-joint-samples.v0.1\0")
    for name in names:
        digest.update(_framed_bytes(name.encode("utf-8")))
    for value in (lower_array, upper_array, sample_array):
        shape = json.dumps(value.shape, separators=(",", ":")).encode("ascii")
        digest.update(_framed_bytes(shape))
        digest.update(_framed_bytes(value.dtype.str.encode("ascii")))
        digest.update(_framed_bytes(memoryview(value).cast("B")))
    return digest.hexdigest()


def sample_uniform_joint_positions(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    num_samples: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """Draw deterministic ``q ~ Uniform(q_min, q_max)`` as canonical float64."""

    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.ndim != 1 or hi.shape != lo.shape or lo.size == 0:
        raise ValueError("lower and upper must be non-empty one-dimensional arrays of equal shape")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("all joint limits must be finite")
    if np.any(hi <= lo):
        raise ValueError("every upper joint limit must be greater than its lower limit")
    if isinstance(num_samples, bool) or int(num_samples) != num_samples or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    result = rng.uniform(lo, hi, size=(int(num_samples), lo.size))
    return np.ascontiguousarray(result, dtype=np.float64)


def xyzw_to_wxyz(quaternions: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternions)
    if value.shape[-1] != 4:
        raise ValueError("quaternion arrays must end in dimension 4")
    return np.ascontiguousarray(value[..., [3, 0, 1, 2]])


def wxyz_to_xyzw(quaternions: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternions)
    if value.shape[-1] != 4:
        raise ValueError("quaternion arrays must end in dimension 4")
    return np.ascontiguousarray(value[..., [1, 2, 3, 0]])


def quaternion_geodesic_xyzw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return sign-invariant SO(3) geodesic distance in radians."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError("quaternion arrays must have equal shape ending in 4")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("quaternion arrays contain non-finite values")
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    if np.any(norm_a < 1.0e-12) or np.any(norm_b < 1.0e-12):
        raise ValueError("zero-norm quaternion")
    a = a / norm_a[..., None]
    b = b / norm_b[..., None]
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


@dataclass(frozen=True)
class FKParityResult:
    num_samples: int
    num_links: int
    position_threshold_m: float
    orientation_threshold_rad: float
    max_position_error_m: float
    max_orientation_error_rad: float
    worst_position_sample: int
    worst_position_link: int
    worst_orientation_sample: int
    worst_orientation_link: int
    position_pass: bool
    orientation_pass: bool
    g0_pass: bool

    def to_manifest(self, key_links: Sequence[str]) -> dict[str, object]:
        result = asdict(self)
        result["worst_position_link_name"] = str(key_links[self.worst_position_link])
        result["worst_orientation_link_name"] = str(key_links[self.worst_orientation_link])
        result["threshold_semantics"] = "strict_less_than"
        return result


def compare_fk_poses(
    reference_positions: np.ndarray,
    reference_quaternions_xyzw: np.ndarray,
    candidate_positions: np.ndarray,
    candidate_quaternions_xyzw: np.ndarray,
    *,
    position_threshold_m: float = POSITION_THRESHOLD_M,
    orientation_threshold_rad: float = ORIENTATION_THRESHOLD_RAD,
) -> FKParityResult:
    """Apply the preregistered maximum-error G0 criterion to two FK tensors."""

    ref_pos = np.asarray(reference_positions, dtype=np.float64)
    cand_pos = np.asarray(candidate_positions, dtype=np.float64)
    ref_quat = np.asarray(reference_quaternions_xyzw, dtype=np.float64)
    cand_quat = np.asarray(candidate_quaternions_xyzw, dtype=np.float64)
    if ref_pos.shape != cand_pos.shape or ref_pos.ndim != 3 or ref_pos.shape[-1] != 3:
        raise ValueError("position arrays must have equal shape (N, K, 3)")
    if ref_quat.shape != cand_quat.shape or ref_quat.shape != ref_pos.shape[:-1] + (4,):
        raise ValueError("quaternion arrays must have equal shape (N, K, 4)")
    if ref_pos.shape[0] == 0 or ref_pos.shape[1] == 0:
        raise ValueError("FK tensors must contain at least one sample and link")
    if not np.isfinite(ref_pos).all() or not np.isfinite(cand_pos).all():
        raise ValueError("position arrays contain non-finite values")
    if not np.isfinite(position_threshold_m) or position_threshold_m <= 0.0:
        raise ValueError("position threshold must be finite and positive")
    if not np.isfinite(orientation_threshold_rad) or orientation_threshold_rad <= 0.0:
        raise ValueError("orientation threshold must be finite and positive")

    position_error = np.linalg.norm(ref_pos - cand_pos, axis=-1)
    orientation_error = quaternion_geodesic_xyzw(ref_quat, cand_quat)
    worst_pos_flat = int(np.argmax(position_error))
    worst_quat_flat = int(np.argmax(orientation_error))
    worst_pos = np.unravel_index(worst_pos_flat, position_error.shape)
    worst_quat = np.unravel_index(worst_quat_flat, orientation_error.shape)
    max_pos = float(position_error[worst_pos])
    max_quat = float(orientation_error[worst_quat])
    pos_pass = bool(max_pos < position_threshold_m)
    quat_pass = bool(max_quat < orientation_threshold_rad)
    return FKParityResult(
        num_samples=ref_pos.shape[0],
        num_links=ref_pos.shape[1],
        position_threshold_m=float(position_threshold_m),
        orientation_threshold_rad=float(orientation_threshold_rad),
        max_position_error_m=max_pos,
        max_orientation_error_rad=max_quat,
        worst_position_sample=int(worst_pos[0]),
        worst_position_link=int(worst_pos[1]),
        worst_orientation_sample=int(worst_quat[0]),
        worst_orientation_link=int(worst_quat[1]),
        position_pass=pos_pass,
        orientation_pass=quat_pass,
        g0_pass=bool(pos_pass and quat_pass),
    )


def _quat_multiply_xyzw(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(first, -1, 0)
    bx, by, bz, bw = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def _rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q_xyz = quaternion[..., :3]
    uv = np.cross(q_xyz, vector)
    uuv = np.cross(q_xyz, uv)
    return vector + 2.0 * (quaternion[..., 3:4] * uv + uuv)


def _rpy_to_xyzw(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy * 0.5
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class FixedLinkFrame:
    semantic_name: str
    target_link: str
    runtime_body: str
    local_position: tuple[float, float, float]
    local_quaternion_xyzw: tuple[float, float, float, float]


def resolve_fixed_link_frames(
    urdf_bytes: bytes,
    key_links: Sequence[tuple[str, str]],
    runtime_body_names: Iterable[str],
) -> tuple[FixedLinkFrame, ...]:
    """Resolve key links to PhysX/MuJoCo bodies through URDF fixed-joint chains.

    Isaac's generated USD merges fixed joints.  A target such as ``head_link`` is
    therefore represented as a deterministic offset from its nearest surviving body.
    Crossing a revolute joint while searching is an error, preventing a plausible but
    physically wrong proxy mapping.
    """

    try:
        root = ET.fromstring(urdf_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"invalid URDF XML: {exc}") from exc
    bodies = set(str(name) for name in runtime_body_names)
    if not bodies:
        raise ValueError("runtime body set is empty")
    incoming: dict[str, tuple[str, str, np.ndarray, np.ndarray]] = {}
    links = {element.get("name") for element in root.findall("link")}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            continue
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        if xyz.shape != (3,) or rpy.shape != (3,):
            raise ValueError(f"invalid URDF origin for joint {joint.get('name')!r}")
        incoming[child_name] = (
            parent_name,
            str(joint.get("type", "")),
            xyz.astype(np.float64),
            _rpy_to_xyzw(rpy.astype(np.float64)),
        )

    resolved: list[FixedLinkFrame] = []
    for semantic_name, target_link in key_links:
        if target_link not in links:
            raise ValueError(f"key link {target_link!r} is absent from URDF")
        current = target_link
        local_position = np.zeros(3, dtype=np.float64)
        local_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        visited: set[str] = set()
        while current not in bodies:
            if current in visited:
                raise ValueError(f"cycle while resolving URDF link {target_link!r}")
            visited.add(current)
            if current not in incoming:
                raise ValueError(
                    f"cannot resolve key link {target_link!r} to a runtime body; stopped at {current!r}"
                )
            parent, joint_type, parent_to_child_pos, parent_to_child_quat = incoming[current]
            if joint_type != "fixed":
                raise ValueError(
                    f"key link {target_link!r} would cross non-fixed joint into runtime body "
                    f"({current!r}, type={joint_type!r})"
                )
            local_position = parent_to_child_pos + _rotate_xyzw(
                parent_to_child_quat, local_position
            )
            local_quaternion = _quat_multiply_xyzw(parent_to_child_quat, local_quaternion)
            current = parent
        resolved.append(
            FixedLinkFrame(
                semantic_name=str(semantic_name),
                target_link=str(target_link),
                runtime_body=current,
                local_position=tuple(float(value) for value in local_position),
                local_quaternion_xyzw=tuple(float(value) for value in local_quaternion),
            )
        )
    return tuple(resolved)


def evaluate_fixed_link_frames(
    body_positions: np.ndarray,
    body_quaternions_xyzw: np.ndarray,
    body_names: Sequence[str],
    frames: Sequence[FixedLinkFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate resolved key frames from batched runtime-body poses."""

    positions = np.asarray(body_positions, dtype=np.float64)
    quaternions = np.asarray(body_quaternions_xyzw, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("body positions must have shape (N, B, 3)")
    if quaternions.shape != positions.shape[:-1] + (4,):
        raise ValueError("body quaternions must have shape (N, B, 4)")
    if positions.shape[1] != len(body_names):
        raise ValueError("body name count does not match pose tensors")
    by_name = {str(name): index for index, name in enumerate(body_names)}
    if len(by_name) != len(body_names):
        raise ValueError("runtime body names are not unique")
    output_positions = np.empty((positions.shape[0], len(frames), 3), dtype=np.float64)
    output_quaternions = np.empty((positions.shape[0], len(frames), 4), dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        if frame.runtime_body not in by_name:
            raise ValueError(f"runtime body disappeared: {frame.runtime_body!r}")
        body_index = by_name[frame.runtime_body]
        body_quaternion = quaternions[:, body_index]
        local_position = np.asarray(frame.local_position, dtype=np.float64)
        local_quaternion = np.asarray(frame.local_quaternion_xyzw, dtype=np.float64)
        output_positions[:, frame_index] = positions[:, body_index] + _rotate_xyzw(
            body_quaternion, np.broadcast_to(local_position, (positions.shape[0], 3))
        )
        output_quaternions[:, frame_index] = _quat_multiply_xyzw(
            body_quaternion, np.broadcast_to(local_quaternion, (positions.shape[0], 4))
        )
    return np.ascontiguousarray(output_positions), np.ascontiguousarray(output_quaternions)


def capture_urdf_bundle(urdf_path: str | Path) -> tuple[dict[str, object], bytes]:
    """Read a URDF and all local mesh/texture references into a deterministic bundle hash."""

    entrypoint = Path(urdf_path).expanduser().resolve()
    urdf_bytes = entrypoint.read_bytes()
    try:
        root = ET.fromstring(urdf_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"invalid URDF XML {entrypoint}: {exc}") from exc
    members: dict[str, bytes] = {entrypoint.name: urdf_bytes}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] not in {"mesh", "texture"}:
            continue
        reference = element.get("filename")
        if not reference:
            continue
        if reference.startswith("package://") or Path(reference).expanduser().is_absolute():
            raise ValueError(f"non-local URDF asset reference is not supported: {reference!r}")
        source = (entrypoint.parent / reference).resolve()
        try:
            relative = source.relative_to(entrypoint.parent).as_posix()
        except ValueError as exc:
            raise ValueError(f"URDF asset escapes bundle root: {reference!r}") from exc
        members[relative] = source.read_bytes()
    digest = hashlib.sha256()
    digest.update(URDF_BUNDLE_HASH_SCHEMA_VERSION.encode("ascii") + b"\0")
    digest.update(_framed_bytes(entrypoint.name.encode("utf-8")))
    files: list[dict[str, object]] = []
    for relative in sorted(members):
        data = members[relative]
        digest.update(_framed_bytes(relative.encode("utf-8")))
        digest.update(_framed_bytes(data))
        files.append(
            {
                "relative_path": relative,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return (
        {
            "source_path": str(entrypoint),
            "entrypoint": entrypoint.name,
            "hash_schema_version": URDF_BUNDLE_HASH_SCHEMA_VERSION,
            "hash_algorithm": "sha256",
            "entrypoint_bound": True,
            "sha256": digest.hexdigest(),
            "size_bytes": sum(len(data) for data in members.values()),
            "file_count": len(files),
            "files": files,
        },
        urdf_bytes,
    )


def capture_directory_bundle(
    root_path: str | Path,
    *,
    entrypoint: str,
) -> tuple[dict[str, object], tuple[tuple[str, bytes], ...]]:
    """Read a local USD directory into memory for hashing and private materialization."""

    root = Path(root_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    members: list[tuple[str, bytes]] = []
    for source in sorted(root.rglob("*")):
        if source.is_symlink():
            raise ValueError(f"symlinks are not permitted in an asset bundle: {source}")
        if source.is_file():
            members.append((source.relative_to(root).as_posix(), source.read_bytes()))
    if entrypoint not in {relative for relative, _ in members}:
        raise FileNotFoundError(root / entrypoint)
    digest = hashlib.sha256()
    digest.update(DIRECTORY_BUNDLE_HASH_SCHEMA_VERSION.encode("ascii") + b"\0")
    digest.update(_framed_bytes(entrypoint.encode("utf-8")))
    files: list[dict[str, object]] = []
    for relative, data in members:
        digest.update(_framed_bytes(relative.encode("utf-8")))
        digest.update(_framed_bytes(data))
        files.append(
            {
                "relative_path": relative,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "source_root": str(root),
        "entrypoint": entrypoint,
        "hash_schema_version": DIRECTORY_BUNDLE_HASH_SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "entrypoint_bound": True,
        "sha256": digest.hexdigest(),
        "size_bytes": sum(len(data) for _, data in members),
        "file_count": len(files),
        "files": files,
    }
    return manifest, tuple(members)


def materialize_memory_bundle(
    members: Sequence[tuple[str, bytes]],
    destination: str | Path,
) -> Path:
    """Materialize already-hashed bytes, rejecting path traversal and overwrites."""

    root = Path(destination)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for relative, data in members:
        target = (root / relative).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"bundle member escapes destination: {relative!r}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.write_bytes(data)
    return root
