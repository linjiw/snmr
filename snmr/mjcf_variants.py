"""Materialized, constant-density same-topology MJCF morphology variants.

Unlike :mod:`snmr.robot_variants`, this module emits an actual MJCF bundle that can be
loaded by MuJoCo and re-hashed by :class:`snmr.provenance.MjcfBundleSnapshot`.  The input
is an immutable in-memory bundle snapshot; the mutable source path is never reopened.

The first qualified dialect is deliberately narrow and matches the frozen Holosoma G1
asset: one XML entrypoint, no includes/default inheritance, explicit inertials on every
robot body, one free root, and explicit named hinge joints.  Unsupported MJCF constructs
fail closed.  A body/link scale ``s`` owns that body's internal geometry and its
outgoing child anchors.  It is applied coherently as

``local lengths -> s``, ``direct child body positions -> s``, ``mass -> s**3``,
and ``inertia -> s**5``.  In particular, a child's incoming ``<body pos>`` is
scaled by the *parent* link's scale, never by an independently drawn child scale.

Mesh bytes are not rewritten.  Their MJCF ``<mesh scale>`` is changed (and mesh assets are
duplicated when one source mesh is used at multiple scales), so visual and collision mesh
geometry follows the same rule.  The free-root spawn transform and ``qpos0`` are invariant.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .provenance import (
    MJCF_BUNDLE_HASH_DOMAIN,
    MJCF_BUNDLE_HASH_SCHEMA,
    MjcfBundleSnapshot,
    sha256_bytes,
)
from .robot_spec import RobotSpec, SemanticManifest

MATERIALIZED_VARIANT_SCHEMA = "snmr.materialized-mjcf-variant.v0.1"
GENERATION_BASIS = "semantic-limb-path-depth-parent-anchor.v0.2"
_FLOAT_TOL = 2.0e-10
# MuJoCo recenters mesh assets while compiling them. Uniformly scaling a nearly
# symmetric mesh can perturb the compiler's eigensystem even though the XML mesh
# scale and physical shape are correct. These explicit gates remain substantially
# tighter than G0's 1 mm / 1e-3 rad FK tolerances.
_COMPILED_MESH_RECENTER_ROTATION_TOL_RAD = 2.5e-4
_COMPILED_MESH_AABB_TOL_METERS = 1.0e-5


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(payload: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(payload))


def _bundle_hash(entrypoint: str, members: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    digest.update(MJCF_BUNDLE_HASH_DOMAIN)
    encoded_entrypoint = entrypoint.encode("utf-8")
    digest.update(len(encoded_entrypoint).to_bytes(8, "big"))
    digest.update(encoded_entrypoint)
    for relative, data in sorted(members):
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a finite real number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a finite real number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _number_list(value: str, label: str) -> list[float]:
    values = [_finite(item, label) for item in value.split()]
    if not values:
        raise ValueError(f"{label} must not be empty")
    return values


def _format_numbers(values: Iterable[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _scaled_attribute(
    element: ET.Element,
    name: str,
    scale: float,
    *,
    expected_lengths: tuple[int, ...] | None = None,
) -> None:
    raw = element.get(name)
    if raw is None:
        return
    values = _number_list(raw, f"<{_tag(element)}> {name}")
    if expected_lengths is not None and len(values) not in expected_lengths:
        raise ValueError(
            f"<{_tag(element)}> {name} must have length in {expected_lengths}, "
            f"got {len(values)}"
        )
    element.set(name, _format_numbers(value * scale for value in values))


def _named_elements(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    result: dict[str, ET.Element] = {}
    for element in root.iter():
        if _tag(element) != tag:
            continue
        name = element.get("name")
        if not name:
            continue
        if name in result:
            raise ValueError(
                f"qualified MJCF requires unique <{tag}> names; duplicate {name!r}"
            )
        result[name] = element
    return result


def _checked_relative(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe bundle member path {relative!r}")
    return path


@dataclass(frozen=True)
class MjcfVariantPlan:
    """A complete, immutable perturbation plan bound to one captured source bundle."""

    source_bundle_sha256: str
    root_body: str
    semantic_manifest: SemanticManifest
    body_scale_groups: tuple[tuple[str, tuple[str, ...]], ...]
    body_scales: tuple[tuple[str, float], ...]
    joint_limit_center_shifts_rad: tuple[tuple[str, float], ...]
    max_link_length_fraction: float
    max_joint_limit_shift_rad: float
    seed: int | None = None
    generation_basis: str = GENERATION_BASIS

    def validate(self) -> None:
        if len(self.source_bundle_sha256) != 64:
            raise ValueError("source_bundle_sha256 must be a 64-character digest")
        try:
            int(self.source_bundle_sha256, 16)
        except ValueError as exc:
            raise ValueError("source_bundle_sha256 must be hexadecimal") from exc
        if not isinstance(self.root_body, str) or not self.root_body.strip():
            raise ValueError("root_body must be non-empty")
        if not isinstance(self.semantic_manifest, SemanticManifest):
            raise TypeError("semantic_manifest must be a SemanticManifest")
        if self.semantic_manifest.root_link != self.root_body:
            raise ValueError("semantic_manifest root_link must match root_body")
        link_fraction = _finite(
            self.max_link_length_fraction, "max_link_length_fraction"
        )
        limit_shift = _finite(
            self.max_joint_limit_shift_rad, "max_joint_limit_shift_rad"
        )
        if not 0.0 <= link_fraction < 1.0:
            raise ValueError("max_link_length_fraction must lie in [0, 1)")
        if not 0.0 <= limit_shift < math.pi:
            raise ValueError("max_joint_limit_shift_rad must lie in [0, pi)")
        if limit_shift != 0.0:
            raise ValueError(
                "nonzero joint-limit variants are not qualified: mirrored physical "
                "coordinate signs must be derived and validated first"
            )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer))
        ):
            raise TypeError("seed must be an integer or None")
        if self.generation_basis != GENERATION_BASIS:
            raise ValueError(f"unsupported generation_basis {self.generation_basis!r}")

        body_names = [name for name, _ in self.body_scales]
        joint_names = [name for name, _ in self.joint_limit_center_shifts_rad]
        if len(set(body_names)) != len(body_names):
            raise ValueError("body_scales contains duplicate body names")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(
                "joint_limit_center_shifts_rad contains duplicate joint names"
            )
        if self.root_body not in set(body_names):
            raise ValueError("body_scales must include root_body")
        scale_by_body = dict(self.body_scales)
        group_names: set[str] = set()
        grouped_bodies: set[str] = set()
        for group_name, members in self.body_scale_groups:
            if (
                not isinstance(group_name, str)
                or not group_name.strip()
                or group_name in group_names
            ):
                raise ValueError("body scale group names must be non-empty and unique")
            group_names.add(group_name)
            if not isinstance(members, tuple) or not members:
                raise ValueError(f"body scale group {group_name!r} must be non-empty")
            if len(set(members)) != len(members):
                raise ValueError(
                    f"body scale group {group_name!r} has duplicate bodies"
                )
            for member in members:
                if member not in scale_by_body:
                    raise ValueError(
                        f"body scale group {group_name!r} references unknown body {member!r}"
                    )
                if member in grouped_bodies:
                    raise ValueError(
                        f"body {member!r} belongs to multiple scale groups"
                    )
                grouped_bodies.add(member)
            realized = {scale_by_body[member] for member in members}
            if len(realized) != 1:
                raise ValueError(
                    f"body scale group {group_name!r} does not share one realized scale"
                )
        for name, scale in self.body_scales:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("body scale names must be non-empty")
            parsed = _finite(scale, f"body scale {name!r}")
            if link_fraction == 0.0 and parsed != 1.0:
                raise ValueError(
                    f"body scale {name!r} must be exactly 1 for a zero bound"
                )
            if abs(parsed - 1.0) > link_fraction + 1.0e-12:
                raise ValueError(f"body scale {name!r} exceeds registered bound")
            if parsed <= 0.0:
                raise ValueError(f"body scale {name!r} must be positive")
        if dict(self.body_scales)[self.root_body] != 1.0:
            raise ValueError("the free-root body scale must be exactly 1.0")
        if (
            any(scale != 1.0 for _, scale in self.body_scales)
            and not self.body_scale_groups
        ):
            raise ValueError("a nonzero body-scale plan must register its scale groups")
        for name, shift in self.joint_limit_center_shifts_rad:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("joint shift names must be non-empty")
            parsed_shift = _finite(shift, f"joint shift {name!r}")
            if parsed_shift != 0.0:
                raise ValueError(
                    f"joint shift {name!r} must be exactly zero until mirrored physical "
                    "coordinate signs are qualified"
                )
            if abs(parsed_shift) > limit_shift + 1.0e-12:
                raise ValueError(f"joint shift {name!r} exceeds registered bound")

    @property
    def is_zero(self) -> bool:
        return all(scale == 1.0 for _, scale in self.body_scales) and all(
            shift == 0.0 for _, shift in self.joint_limit_center_shifts_rad
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "source_bundle_sha256": self.source_bundle_sha256,
            "root_body": self.root_body,
            "semantic_manifest": asdict(self.semantic_manifest),
            "semantic_manifest_sha256": self.semantic_manifest_sha256,
            "body_scale_groups": self.body_scale_groups_payload,
            "body_scale_groups_sha256": self.body_scale_groups_sha256,
            "body_scales": [list(item) for item in sorted(self.body_scales)],
            "joint_limit_center_shifts_rad": [
                list(item) for item in sorted(self.joint_limit_center_shifts_rad)
            ],
            "max_link_length_fraction": self.max_link_length_fraction,
            "max_joint_limit_shift_rad": self.max_joint_limit_shift_rad,
            "seed": self.seed,
            "generation_basis": self.generation_basis,
        }

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.to_dict())

    @property
    def semantic_manifest_sha256(self) -> str:
        return _canonical_hash(asdict(self.semantic_manifest))

    @property
    def body_scale_groups_payload(self) -> list[dict[str, Any]]:
        return [
            {"group": name, "bodies": list(members)}
            for name, members in sorted(self.body_scale_groups)
        ]

    @property
    def body_scale_groups_sha256(self) -> str:
        return _canonical_hash(self.body_scale_groups_payload)


@dataclass(frozen=True)
class MjcfVariantValidation:
    body_count: int
    hinge_joint_count: int
    geom_count: int
    output_robot_spec_hash: str
    output_entrypoint_sha256: str
    qpos0_max_abs_error: float
    root_spawn_max_abs_error: float
    body_position_rule_max_abs_error: float
    mass_rule_max_abs_error: float
    inertia_rule_max_abs_error: float
    geom_rule_max_abs_error: float
    compiled_mesh_recenter_rotation_max_rad: float
    compiled_mesh_aabb_max_abs_error: float
    joint_rule_max_abs_error: float
    nominal_fk_finite: bool
    qpos0_inside_shifted_limits: bool

    @property
    def passed(self) -> bool:
        errors = (
            self.qpos0_max_abs_error,
            self.root_spawn_max_abs_error,
            self.body_position_rule_max_abs_error,
            self.mass_rule_max_abs_error,
            self.inertia_rule_max_abs_error,
            self.geom_rule_max_abs_error,
            self.joint_rule_max_abs_error,
        )
        return (
            max(errors, default=0.0) <= _FLOAT_TOL
            and self.compiled_mesh_recenter_rotation_max_rad
            <= _COMPILED_MESH_RECENTER_ROTATION_TOL_RAD
            and self.compiled_mesh_aabb_max_abs_error <= _COMPILED_MESH_AABB_TOL_METERS
            and self.nominal_fk_finite
            and self.qpos0_inside_shifted_limits
        )


@dataclass(frozen=True)
class GeneratorProvenance:
    """Exact software/command provenance for an archival variant publication."""

    command: tuple[str, ...]
    python_version: str
    mujoco_version: str
    revisions: tuple[tuple[str, str | bool | None], ...]

    def to_dict(self) -> dict[str, Any]:
        if not self.command or any(
            not isinstance(item, str) or not item for item in self.command
        ):
            raise ValueError("generator command must contain non-empty argv strings")
        if not self.python_version or not self.mujoco_version:
            raise ValueError("Python and MuJoCo versions must be recorded")
        revision_dict = dict(self.revisions)
        if len(revision_dict) != len(self.revisions):
            raise ValueError("generator revision fields must be unique")
        required = {
            f"{name}_{field}"
            for name in ("snmr", "newton", "isaac_lab")
            for field in ("commit", "dirty", "repo_status")
        }
        if not required.issubset(revision_dict):
            raise ValueError(
                f"generator revisions are missing {sorted(required - set(revision_dict))}"
            )
        for name in ("snmr", "newton", "isaac_lab"):
            commit = revision_dict[f"{name}_commit"]
            if commit is not None and (
                not isinstance(commit, str)
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None
            ):
                raise ValueError(
                    f"{name}_commit must be a lowercase 40- or 64-character Git SHA"
                )
            dirty = revision_dict[f"{name}_dirty"]
            if dirty is not None and not isinstance(dirty, bool):
                raise TypeError(f"{name}_dirty must be bool or None")
            if not isinstance(revision_dict[f"{name}_repo_status"], str):
                raise TypeError(f"{name}_repo_status must be a string")
        return {
            "command": list(self.command),
            "python_version": self.python_version,
            "mujoco_version": self.mujoco_version,
            "revisions": revision_dict,
        }

    @property
    def archival_ready(self) -> bool:
        revisions = dict(self.revisions)
        return all(
            revisions.get(f"{name}_repo_status") == "available"
            and isinstance(revisions.get(f"{name}_commit"), str)
            and revisions.get(f"{name}_dirty") is False
            for name in ("snmr", "newton", "isaac_lab")
        )


@dataclass(frozen=True)
class MaterializedMjcfVariant:
    """Validated output bytes.  Binary members are exact copies of captured buffers."""

    entrypoint: str
    members: tuple[tuple[str, bytes], ...] = field(repr=False)
    source_bundle_sha256: str
    output_bundle_sha256: str
    plan: MjcfVariantPlan
    validation: MjcfVariantValidation
    generator_provenance: GeneratorProvenance | None = None

    def manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": MATERIALIZED_VARIANT_SCHEMA,
            "bundle_hash_schema": MJCF_BUNDLE_HASH_SCHEMA,
            "source_bundle_sha256": self.source_bundle_sha256,
            "output_bundle_sha256": self.output_bundle_sha256,
            "entrypoint": self.entrypoint,
            "zero_perturbation_exact_copy": self.plan.is_zero,
            "plan": self.plan.to_dict(),
            "plan_sha256": self.plan.sha256,
            "validation": {**asdict(self.validation), "passed": self.validation.passed},
            "validation_tolerances": {
                "exact_rule_max_abs_error": _FLOAT_TOL,
                "compiled_mesh_recenter_rotation_rad": (
                    _COMPILED_MESH_RECENTER_ROTATION_TOL_RAD
                ),
                "compiled_mesh_aabb_meters": _COMPILED_MESH_AABB_TOL_METERS,
            },
            "generator_provenance": (
                None
                if self.generator_provenance is None
                else self.generator_provenance.to_dict()
            ),
            "archival_ready": bool(
                self.generator_provenance is not None
                and self.generator_provenance.archival_ready
            ),
            "members": [
                {
                    "relative_path": relative,
                    "sha256": sha256_bytes(data),
                    "size_bytes": len(data),
                }
                for relative, data in self.members
            ],
            "scope_limits": [
                "same topology only",
                "single XML entrypoint; no includes or MJCF defaults",
                "explicit inertials and named hinge joints required",
                "uniform per-body geometry scaling only",
                "child body anchors are owned by the physical parent link",
                "joint-limit perturbations disabled until physical mirror signs are qualified",
            ],
        }
        payload["manifest_sha256"] = _canonical_hash(payload)
        return payload


def _snapshot_members(snapshot: MjcfBundleSnapshot) -> tuple[tuple[str, bytes], ...]:
    members = tuple((relative, member.data) for relative, member in snapshot.members)
    for relative, _ in members:
        _checked_relative(relative)
    if snapshot.entrypoint not in dict(members):
        raise ValueError("captured bundle does not contain its declared entrypoint")
    if _bundle_hash(snapshot.entrypoint, members) != snapshot.sha256:
        raise ValueError("captured bundle fields do not replay their declared SHA-256")
    xml_members = [
        relative
        for relative, _ in members
        if PurePosixPath(relative).suffix.lower() == ".xml"
    ]
    if xml_members != [snapshot.entrypoint]:
        raise ValueError(
            "qualified materialized variants require exactly one XML member, the entrypoint"
        )
    return members


def _write_raw_members(
    members: Sequence[tuple[str, bytes]], entrypoint: str, directory: Path
) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    return _write_members_into_existing_directory(members, entrypoint, directory)


def _write_members_into_existing_directory(
    members: Sequence[tuple[str, bytes]], entrypoint: str, directory: Path
) -> Path:
    resolved_root = directory.resolve()
    for relative, data in members:
        path = directory.joinpath(*_checked_relative(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve()
        if resolved_root not in resolved.parents:
            raise ValueError(f"bundle member escapes destination: {relative!r}")
        path.write_bytes(data)
    return directory.joinpath(*_checked_relative(entrypoint).parts)


def _load_snapshot_model(snapshot: MjcfBundleSnapshot):
    import mujoco

    temporary = tempfile.TemporaryDirectory(prefix="snmr-mjcf-plan-")
    root = Path(temporary.name) / "bundle"
    entrypoint = _write_raw_members(
        _snapshot_members(snapshot), snapshot.entrypoint, root
    )
    try:
        model = mujoco.MjModel.from_xml_path(str(entrypoint))
    except Exception:
        temporary.cleanup()
        raise
    return temporary, entrypoint, model


def _path_to_root(spec: RobotSpec, endpoint: str) -> tuple[str, ...]:
    parent = {link.name: link.parent for link in spec.links}
    if endpoint not in parent:
        raise ValueError(f"semantic endpoint {endpoint!r} is absent from RobotSpec")
    reverse: list[str] = []
    current: str | None = endpoint
    while current is not None:
        if current in reverse:
            raise ValueError("RobotSpec topology contains a cycle")
        reverse.append(current)
        current = parent[current]
    path = tuple(reversed(reverse))
    if path[0] != spec.semantics.root_link:
        raise ValueError(f"semantic endpoint {endpoint!r} is outside the robot root")
    return path


def build_semantic_limb_variant_plan(
    snapshot: MjcfBundleSnapshot,
    semantics: SemanticManifest,
    *,
    seed: int,
    link_length_fraction: float = 0.15,
    joint_limit_shift_rad: float = 0.0,
) -> MjcfVariantPlan:
    """Draw a serialization-independent bilateral plan from semantic path depth.

    Random draws are ordered by semantic role (feet, then hands) and path depth, never by
    XML serialization or opaque link names.  Names remain necessary only as mutation
    addresses in the returned plan.  Descendants off a semantic path inherit the nearest
    scaled ancestor, keeping auxiliary foot/hand collision bodies coherent.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    fraction = _finite(link_length_fraction, "link_length_fraction")
    shift_bound = _finite(joint_limit_shift_rad, "joint_limit_shift_rad")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("link_length_fraction must lie in [0, 1)")
    if not 0.0 <= shift_bound < math.pi:
        raise ValueError("joint_limit_shift_rad must lie in [0, pi)")
    if shift_bound != 0.0:
        raise ValueError(
            "nonzero joint-limit variants are disabled until mirrored physical "
            "coordinate signs are qualified"
        )

    temporary, entrypoint, _ = _load_snapshot_model(snapshot)
    try:
        spec = RobotSpec.from_mjcf(entrypoint, semantics)
    finally:
        temporary.cleanup()

    rng = np.random.default_rng(int(seed))
    explicit_scale: dict[str, float] = {semantics.root_link: 1.0}
    body_scale_groups: list[tuple[str, tuple[str, ...]]] = []
    paired_paths: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    endpoint_pairs = (
        (semantics.left_foot_link, semantics.right_foot_link, "feet"),
        (semantics.left_hand_link, semantics.right_hand_link, "hands"),
    )
    for left_endpoint, right_endpoint, role in endpoint_pairs:
        if left_endpoint is None and right_endpoint is None:
            continue
        if left_endpoint is None or right_endpoint is None:
            raise ValueError(
                f"{role} semantic endpoints must be supplied as a bilateral pair"
            )
        left_path = _path_to_root(spec, left_endpoint)
        right_path = _path_to_root(spec, right_endpoint)
        if len(left_path) != len(right_path):
            raise ValueError(f"{role} semantic paths have unequal topology depth")
        divergence = next(
            (
                index
                for index, pair in enumerate(zip(left_path, right_path))
                if pair[0] != pair[1]
            ),
            len(left_path),
        )
        if divergence == len(left_path):
            raise ValueError(f"{role} semantic endpoints do not define distinct limbs")
        if left_path[:divergence] != right_path[:divergence]:
            raise AssertionError("invalid common semantic path")
        paired_paths.append((left_path[divergence:], right_path[divergence:]))
        for path_depth, (left_name, right_name) in enumerate(
            zip(left_path[divergence:], right_path[divergence:])
        ):
            scale = 1.0 + float(rng.uniform(-fraction, fraction))
            explicit_scale[left_name] = scale
            explicit_scale[right_name] = scale
            body_scale_groups.append(
                (f"{role}.divergent_path_depth_{path_depth}", (left_name, right_name))
            )

    if not paired_paths:
        raise ValueError(
            "at least one bilateral hand or foot semantic pair is required"
        )

    children: dict[str, list[str]] = {link.name: [] for link in spec.links}
    for link in spec.links:
        if link.parent is not None:
            children[link.parent].append(link.name)
    body_scale: dict[str, float] = {}

    def assign(name: str, inherited: float) -> None:
        scale = explicit_scale.get(name, inherited)
        body_scale[name] = scale
        for child in sorted(children[name]):
            assign(child, scale)

    assign(semantics.root_link, 1.0)

    joint_by_child = {joint.child_link: joint for joint in spec.joints}
    joint_shift = {joint.name: 0.0 for joint in spec.joints}
    for left_path, right_path in paired_paths:
        for left_name, right_name in zip(left_path, right_path):
            left_joint = joint_by_child.get(left_name)
            right_joint = joint_by_child.get(right_name)
            if (left_joint is None) != (right_joint is None):
                raise ValueError(
                    "bilateral semantic paths have mismatched joint attachments"
                )
            if left_joint is None:
                continue
            epsilon = 1.0e-9
            lower = max(
                -shift_bound,
                left_joint.nominal_position - left_joint.upper_limit + epsilon,
                right_joint.nominal_position - right_joint.upper_limit + epsilon,
            )
            upper = min(
                shift_bound,
                left_joint.nominal_position - left_joint.lower_limit - epsilon,
                right_joint.nominal_position - right_joint.lower_limit - epsilon,
            )
            if lower > upper:
                raise ValueError(
                    "no bilateral joint-limit shift preserves both nominal positions"
                )
            shift = float(rng.uniform(lower, upper)) if lower < upper else float(lower)
            joint_shift[left_joint.name] = shift
            joint_shift[right_joint.name] = shift

    plan = MjcfVariantPlan(
        source_bundle_sha256=snapshot.sha256,
        root_body=semantics.root_link,
        semantic_manifest=semantics,
        body_scale_groups=tuple(body_scale_groups),
        body_scales=tuple(sorted(body_scale.items())),
        joint_limit_center_shifts_rad=tuple(sorted(joint_shift.items())),
        max_link_length_fraction=fraction,
        max_joint_limit_shift_rad=shift_bound,
        seed=int(seed),
    )
    plan.validate()
    return plan


def zero_perturbation_plan(
    snapshot: MjcfBundleSnapshot,
    semantics: SemanticManifest,
) -> MjcfVariantPlan:
    """Build a complete zero plan whose output must be byte-identical to the source."""

    temporary, entrypoint, _ = _load_snapshot_model(snapshot)
    try:
        spec = RobotSpec.from_mjcf(entrypoint, semantics)
    finally:
        temporary.cleanup()
    plan = MjcfVariantPlan(
        source_bundle_sha256=snapshot.sha256,
        root_body=semantics.root_link,
        semantic_manifest=semantics,
        body_scale_groups=(),
        body_scales=tuple(sorted((link.name, 1.0) for link in spec.links)),
        joint_limit_center_shifts_rad=tuple(
            sorted((joint.name, 0.0) for joint in spec.joints)
        ),
        max_link_length_fraction=0.0,
        max_joint_limit_shift_rad=0.0,
        seed=None,
    )
    plan.validate()
    return plan


def _qualify_and_mutate_xml(entrypoint_bytes: bytes, plan: MjcfVariantPlan) -> bytes:
    try:
        root = ET.fromstring(entrypoint_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"captured MJCF entrypoint is invalid XML: {exc}") from exc
    if _tag(root) != "mujoco":
        raise ValueError("entrypoint root must be <mujoco>")
    if any(_tag(element) == "include" for element in root.iter()):
        raise ValueError("materialized variants do not yet support MJCF <include>")
    if any(_tag(element) == "default" for element in root.iter()):
        raise ValueError(
            "materialized variants do not support inherited MJCF <default> values"
        )
    compilers = [element for element in root if _tag(element) == "compiler"]
    if len(compilers) != 1 or compilers[0].get("angle") != "radian":
        raise ValueError(
            "qualified variant MJCF requires one <compiler angle='radian'>"
        )

    worldbodies = [element for element in root if _tag(element) == "worldbody"]
    if len(worldbodies) != 1:
        raise ValueError("qualified variant MJCF requires exactly one <worldbody>")
    body_by_name = _named_elements(worldbodies[0], "body")
    if plan.root_body not in body_by_name:
        raise ValueError(f"root body {plan.root_body!r} is absent from MJCF")
    root_body = body_by_name[plan.root_body]
    freejoints = [element for element in root_body if _tag(element) == "freejoint"]
    if len(freejoints) != 1:
        raise ValueError(
            "qualified root body must contain exactly one direct <freejoint>"
        )

    subtree_names: set[str] = set()

    def collect_subtree(body: ET.Element) -> None:
        name = body.get("name")
        if not name:
            raise ValueError("every robot <body> must have an explicit name")
        subtree_names.add(name)
        for child in body:
            if _tag(child) == "body":
                collect_subtree(child)

    collect_subtree(root_body)
    scale_by_body = dict(plan.body_scales)
    if set(scale_by_body) != subtree_names:
        missing = sorted(subtree_names - set(scale_by_body))
        extra = sorted(set(scale_by_body) - subtree_names)
        raise ValueError(
            f"body scale plan must exactly cover robot subtree; missing={missing}, extra={extra}"
        )

    joint_elements: dict[str, ET.Element] = {}
    mesh_usages: dict[str, list[tuple[ET.Element, float]]] = {}

    def mutate_body(body: ET.Element, parent_scale: float | None) -> None:
        name = str(body.get("name"))
        scale = scale_by_body[name]
        if name != plan.root_body:
            if parent_scale is None:
                raise AssertionError("non-root body is missing its parent scale")
            # A child body position is the distal anchor expressed in the physical
            # parent link's frame.  Scaling it by the child's independent draw would
            # change the parent segment/anchor ratio and make adjacent variants
            # geometrically incoherent.
            _scaled_attribute(body, "pos", parent_scale, expected_lengths=(3,))
        elif body.get("pos") is not None and scale != 1.0:
            raise AssertionError("root spawn scale escaped plan validation")

        inertials = [child for child in body if _tag(child) == "inertial"]
        if len(inertials) != 1:
            raise ValueError(f"body {name!r} must have exactly one explicit <inertial>")
        inertial = inertials[0]
        if inertial.get("mass") is None:
            raise ValueError(f"body {name!r} inertial must declare mass")
        if (inertial.get("diaginertia") is None) == (
            inertial.get("fullinertia") is None
        ):
            raise ValueError(
                f"body {name!r} inertial must declare exactly one of diaginertia/fullinertia"
            )
        _scaled_attribute(inertial, "pos", scale, expected_lengths=(3,))
        _scaled_attribute(inertial, "mass", scale**3, expected_lengths=(1,))
        _scaled_attribute(inertial, "diaginertia", scale**5, expected_lengths=(3,))
        _scaled_attribute(inertial, "fullinertia", scale**5, expected_lengths=(6,))

        for child in body:
            tag = _tag(child)
            if tag == "body":
                mutate_body(child, scale)
            elif tag == "joint":
                joint_type = child.get("type", "hinge")
                if joint_type != "hinge":
                    raise ValueError(
                        f"body {name!r} has unsupported joint type {joint_type!r}"
                    )
                joint_name = child.get("name")
                if not joint_name:
                    raise ValueError("every hinge joint must have an explicit name")
                if joint_name in joint_elements:
                    raise ValueError(f"duplicate hinge joint name {joint_name!r}")
                if child.get("range") is None:
                    raise ValueError(
                        f"hinge joint {joint_name!r} must declare an explicit range"
                    )
                _scaled_attribute(child, "pos", scale, expected_lengths=(3,))
                joint_elements[joint_name] = child
            elif tag in {"geom", "site"}:
                _scaled_attribute(child, "pos", scale, expected_lengths=(3,))
                _scaled_attribute(child, "fromto", scale, expected_lengths=(6,))
                _scaled_attribute(child, "size", scale)
                if tag == "geom":
                    _scaled_attribute(child, "mass", scale**3, expected_lengths=(1,))
                    mesh_name = child.get("mesh")
                    if mesh_name:
                        mesh_usages.setdefault(mesh_name, []).append((child, scale))
            elif tag in {"camera", "light"}:
                _scaled_attribute(child, "pos", scale, expected_lengths=(3,))
            elif tag in {"inertial", "freejoint"}:
                continue
            else:
                raise ValueError(f"body {name!r} contains unsupported direct <{tag}>")

    mutate_body(root_body, None)

    shift_by_joint = dict(plan.joint_limit_center_shifts_rad)
    if set(shift_by_joint) != set(joint_elements):
        missing = sorted(set(joint_elements) - set(shift_by_joint))
        extra = sorted(set(shift_by_joint) - set(joint_elements))
        raise ValueError(
            f"joint shift plan must exactly cover hinge joints; missing={missing}, extra={extra}"
        )
    for name, element in joint_elements.items():
        values = _number_list(str(element.get("range")), f"joint {name!r} range")
        if len(values) != 2 or not values[0] < values[1]:
            raise ValueError(f"joint {name!r} must have an ordered two-value range")
        shift = shift_by_joint[name]
        element.set("range", _format_numbers((values[0] + shift, values[1] + shift)))

    assets = [element for element in root if _tag(element) == "asset"]
    if mesh_usages and not assets:
        raise ValueError("mesh-bearing qualified MJCF requires a direct <asset>")
    mesh_by_name: dict[str, tuple[ET.Element, ET.Element]] = {}
    for asset in assets:
        for element in asset:
            if _tag(element) != "mesh":
                continue
            name = element.get("name")
            if not name:
                raise ValueError("every qualified <mesh> asset must have a name")
            if name in mesh_by_name:
                raise ValueError(f"duplicate mesh asset name {name!r}")
            mesh_by_name[name] = (element, asset)
    occupied_mesh_names = set(mesh_by_name)
    for mesh_name, usages in sorted(mesh_usages.items()):
        if mesh_name not in mesh_by_name:
            raise ValueError(
                f"mesh {mesh_name!r} is used but has no direct <asset><mesh> definition"
            )
        source_mesh, source_asset = mesh_by_name[mesh_name]
        raw_scale = source_mesh.get("scale")
        base_scale = (
            [1.0, 1.0, 1.0]
            if raw_scale is None
            else _number_list(raw_scale, "mesh scale")
        )
        if len(base_scale) == 1:
            base_scale *= 3
        if len(base_scale) != 3:
            raise ValueError(
                f"mesh {mesh_name!r} scale must contain one or three values"
            )
        distinct_scales = sorted({scale for _, scale in usages})
        primary = 1.0 if 1.0 in distinct_scales else distinct_scales[0]
        source_mesh.set(
            "scale", _format_numbers(value * primary for value in base_scale)
        )
        mesh_for_scale: dict[float, str] = {primary: mesh_name}
        for scale in distinct_scales:
            if scale == primary:
                continue
            suffix = hashlib.sha256(format(scale, ".17g").encode("ascii")).hexdigest()[
                :12
            ]
            generated_name = f"{mesh_name}__snmr_s_{suffix}"
            if generated_name in occupied_mesh_names:
                raise ValueError(f"generated mesh name collision {generated_name!r}")
            occupied_mesh_names.add(generated_name)
            duplicate = deepcopy(source_mesh)
            duplicate.set("name", generated_name)
            duplicate.set(
                "scale", _format_numbers(value * scale for value in base_scale)
            )
            source_asset.append(duplicate)
            mesh_for_scale[scale] = generated_name
        for geom, scale in usages:
            geom.set("mesh", mesh_for_scale[scale])

    return ET.tostring(
        root, encoding="utf-8", xml_declaration=False, short_empty_elements=True
    )


def _object_names(model: Any, object_type: Any, count: int) -> tuple[str | None, ...]:
    import mujoco

    return tuple(mujoco.mj_id2name(model, object_type, index) for index in range(count))


def _max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.size == 0:
        return 0.0
    return float(
        np.max(
            np.abs(
                np.asarray(actual, dtype=np.float64)
                - np.asarray(expected, dtype=np.float64)
            )
        )
    )


def _quat_geodesic_wxyz(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return math.inf
    dot = abs(float(np.dot(left_array / left_norm, right_array / right_norm)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _validate_materialized_paths(
    source_entrypoint: Path,
    output_entrypoint: Path,
    output_entrypoint_bytes: bytes,
    plan: MjcfVariantPlan,
    semantics: SemanticManifest,
) -> MjcfVariantValidation:
    import mujoco

    source = mujoco.MjModel.from_xml_path(str(source_entrypoint))
    output = mujoco.MjModel.from_xml_path(str(output_entrypoint))
    if (source.nbody, source.njnt, source.ngeom, source.nq, source.nv) != (
        output.nbody,
        output.njnt,
        output.ngeom,
        output.nq,
        output.nv,
    ):
        raise ValueError("variant changed compiled topology or state dimensions")
    if _object_names(source, mujoco.mjtObj.mjOBJ_BODY, source.nbody) != _object_names(
        output, mujoco.mjtObj.mjOBJ_BODY, output.nbody
    ):
        raise ValueError("variant changed compiled body ordering or names")
    if _object_names(source, mujoco.mjtObj.mjOBJ_JOINT, source.njnt) != _object_names(
        output, mujoco.mjtObj.mjOBJ_JOINT, output.njnt
    ):
        raise ValueError("variant changed compiled joint ordering or names")
    if not np.array_equal(source.jnt_type, output.jnt_type):
        raise ValueError("variant changed compiled joint types")
    if not np.array_equal(source.geom_type, output.geom_type) or not np.array_equal(
        source.geom_bodyid, output.geom_bodyid
    ):
        raise ValueError("variant changed compiled geom types or body attachment")

    scale_by_name = dict(plan.body_scales)
    body_names = _object_names(source, mujoco.mjtObj.mjOBJ_BODY, source.nbody)
    body_ids = {
        name: index for index, name in enumerate(body_names) if name in scale_by_name
    }
    if set(body_ids) != set(scale_by_name):
        raise ValueError("compiled body set does not match plan")
    root_id = body_ids[plan.root_body]
    root_spawn_error = max(
        _max_abs(output.body_pos[root_id], source.body_pos[root_id]),
        _max_abs(output.body_quat[root_id], source.body_quat[root_id]),
    )

    body_position_error = 0.0
    mass_error = 0.0
    inertia_error = 0.0
    for name, body_id in body_ids.items():
        scale = scale_by_name[name]
        if name == plan.root_body:
            expected_position = source.body_pos[body_id]
        else:
            parent_id = int(source.body_parentid[body_id])
            parent_name = body_names[parent_id]
            if parent_name not in scale_by_name:
                raise ValueError(
                    f"body {name!r} has parent {parent_name!r} outside the scale plan"
                )
            expected_position = source.body_pos[body_id] * scale_by_name[parent_name]
        body_position_error = max(
            body_position_error,
            _max_abs(output.body_pos[body_id], expected_position),
        )
        body_position_error = max(
            body_position_error,
            _max_abs(output.body_ipos[body_id], source.body_ipos[body_id] * scale),
        )
        body_position_error = max(
            body_position_error,
            _max_abs(output.body_quat[body_id], source.body_quat[body_id]),
        )
        mass_error = max(
            mass_error,
            abs(
                float(output.body_mass[body_id])
                - float(source.body_mass[body_id]) * scale**3
            ),
        )
        inertia_error = max(
            inertia_error,
            _max_abs(
                output.body_inertia[body_id], source.body_inertia[body_id] * scale**5
            ),
        )

    joint_names = _object_names(source, mujoco.mjtObj.mjOBJ_JOINT, source.njnt)
    shift_by_name = dict(plan.joint_limit_center_shifts_rad)
    joint_error = 0.0
    qpos_inside = True
    for joint_id, name in enumerate(joint_names):
        if source.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if name not in shift_by_name:
            raise ValueError(f"compiled hinge joint {name!r} is absent from plan")
        body_id = int(source.jnt_bodyid[joint_id])
        body_name = body_names[body_id]
        if body_name is None:
            raise ValueError("hinge joint has unnamed child body")
        scale = scale_by_name[body_name]
        joint_error = max(
            joint_error,
            _max_abs(output.jnt_pos[joint_id], source.jnt_pos[joint_id] * scale),
            _max_abs(output.jnt_axis[joint_id], source.jnt_axis[joint_id]),
            _max_abs(
                output.jnt_range[joint_id],
                source.jnt_range[joint_id] + shift_by_name[name],
            ),
        )
        qpos_address = int(output.jnt_qposadr[joint_id])
        nominal = float(output.qpos0[qpos_address])
        qpos_inside = qpos_inside and bool(
            output.jnt_range[joint_id, 0] <= nominal <= output.jnt_range[joint_id, 1]
        )

    # Primitive geometry must compile to the exact registered scale rule.  Meshes are
    # checked through their explicit mesh_scale plus local translation.  MuJoCo may
    # choose a slightly different internal recentering eigensystem for a uniformly
    # scaled, near-symmetric mesh, so its compiled quaternion/AABB are recorded under
    # separate, physically small tolerances rather than weakening the exact XML rule.
    geom_error = 0.0
    compiled_mesh_rotation_error = 0.0
    compiled_mesh_aabb_error = 0.0
    for geom_id in range(source.ngeom):
        body_id = int(source.geom_bodyid[geom_id])
        body_name = body_names[body_id]
        scale = scale_by_name.get(str(body_name), 1.0)
        geom_position_error = _max_abs(
            output.geom_pos[geom_id], source.geom_pos[geom_id] * scale
        )
        if source.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            source_mesh_id = int(source.geom_dataid[geom_id])
            output_mesh_id = int(output.geom_dataid[geom_id])
            if source_mesh_id < 0 or output_mesh_id < 0:
                raise ValueError("compiled mesh geom has no mesh data id")
            geom_error = max(
                geom_error,
                geom_position_error,
                _max_abs(
                    output.mesh_scale[output_mesh_id],
                    source.mesh_scale[source_mesh_id] * scale,
                ),
            )
            compiled_mesh_rotation_error = max(
                compiled_mesh_rotation_error,
                _quat_geodesic_wxyz(
                    output.geom_quat[geom_id], source.geom_quat[geom_id]
                ),
            )
            compiled_mesh_aabb_error = max(
                compiled_mesh_aabb_error,
                _max_abs(output.geom_aabb[geom_id], source.geom_aabb[geom_id] * scale),
            )
        else:
            geom_error = max(
                geom_error,
                geom_position_error,
                _max_abs(output.geom_quat[geom_id], source.geom_quat[geom_id]),
                _max_abs(output.geom_size[geom_id], source.geom_size[geom_id] * scale),
                _max_abs(output.geom_aabb[geom_id], source.geom_aabb[geom_id] * scale),
            )

    qpos_error = _max_abs(output.qpos0, source.qpos0)
    data = mujoco.MjData(output)
    data.qpos[:] = output.qpos0
    mujoco.mj_forward(output, data)
    nominal_finite = bool(
        np.isfinite(data.xpos).all()
        and np.isfinite(data.xquat).all()
        and np.allclose(np.linalg.norm(data.xquat, axis=1), 1.0, atol=1.0e-10, rtol=0.0)
    )

    spec = RobotSpec.from_mjcf(output_entrypoint, semantics)
    if len(spec.links) != len(scale_by_name) or len(spec.joints) != len(shift_by_name):
        raise ValueError("RobotSpec round-trip changed robot topology")
    validation = MjcfVariantValidation(
        body_count=len(spec.links),
        hinge_joint_count=len(spec.joints),
        geom_count=int(output.ngeom),
        output_robot_spec_hash=spec.spec_hash,
        output_entrypoint_sha256=sha256_bytes(output_entrypoint_bytes),
        qpos0_max_abs_error=qpos_error,
        root_spawn_max_abs_error=root_spawn_error,
        body_position_rule_max_abs_error=body_position_error,
        mass_rule_max_abs_error=mass_error,
        inertia_rule_max_abs_error=inertia_error,
        geom_rule_max_abs_error=geom_error,
        compiled_mesh_recenter_rotation_max_rad=compiled_mesh_rotation_error,
        compiled_mesh_aabb_max_abs_error=compiled_mesh_aabb_error,
        joint_rule_max_abs_error=joint_error,
        nominal_fk_finite=nominal_finite,
        qpos0_inside_shifted_limits=qpos_inside,
    )
    if not validation.passed:
        raise ValueError(
            f"materialized MJCF failed physical validation: {asdict(validation)}"
        )
    return validation


def materialize_mjcf_variant(
    snapshot: MjcfBundleSnapshot,
    plan: MjcfVariantPlan,
    semantics: SemanticManifest,
    *,
    generator_provenance: GeneratorProvenance | None = None,
) -> MaterializedMjcfVariant:
    """Transform captured bytes, load both models, and enforce all registered invariants."""

    plan.validate()
    if plan.source_bundle_sha256 != snapshot.sha256:
        raise ValueError("variant plan is bound to a different source MJCF bundle")
    if semantics != plan.semantic_manifest:
        raise ValueError(
            "materialization SemanticManifest differs from the manifest bound to the plan"
        )
    source_members = _snapshot_members(snapshot)
    source_by_name = dict(source_members)
    entrypoint_bytes = source_by_name[snapshot.entrypoint]
    # Exact identity is a stronger zero-perturbation guarantee than reserializing XML.
    output_entrypoint_bytes = (
        entrypoint_bytes
        if plan.is_zero
        else _qualify_and_mutate_xml(entrypoint_bytes, plan)
    )
    output_members = tuple(
        (relative, output_entrypoint_bytes if relative == snapshot.entrypoint else data)
        for relative, data in source_members
    )
    output_hash = _bundle_hash(snapshot.entrypoint, output_members)

    with tempfile.TemporaryDirectory(prefix="snmr-mjcf-variant-validate-") as temporary:
        root = Path(temporary)
        source_entrypoint = _write_raw_members(
            source_members, snapshot.entrypoint, root / "source"
        )
        output_entrypoint = _write_raw_members(
            output_members, snapshot.entrypoint, root / "output"
        )
        validation = _validate_materialized_paths(
            source_entrypoint,
            output_entrypoint,
            output_entrypoint_bytes,
            plan,
            semantics,
        )
    return MaterializedMjcfVariant(
        entrypoint=snapshot.entrypoint,
        members=output_members,
        source_bundle_sha256=snapshot.sha256,
        output_bundle_sha256=output_hash,
        plan=plan,
        validation=validation,
        generator_provenance=generator_provenance,
    )


def require_archival_ready_variant_manifest(
    manifest: Mapping[str, Any], *, observed_bundle_sha256: str
) -> None:
    """Reject a materialized variant that is not qualified for dataset use.

    Experiment smoke artifacts may intentionally carry ``archival_ready=false`` while
    the source checkout is dirty.  Dataset builders must call this guard instead of
    treating a successful MuJoCo load as sufficient provenance.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("variant manifest must be a mapping")
    if manifest.get("schema_version") != MATERIALIZED_VARIANT_SCHEMA:
        raise ValueError("unsupported materialized variant manifest schema")
    if (
        not isinstance(observed_bundle_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", observed_bundle_sha256) is None
    ):
        raise ValueError("observed_bundle_sha256 must be a lowercase SHA-256 digest")
    recorded_manifest_hash = manifest.get("manifest_sha256")
    if not isinstance(recorded_manifest_hash, str):
        raise TypeError("variant manifest is missing manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if _canonical_hash(unhashed) != recorded_manifest_hash:
        raise ValueError("variant manifest SHA-256 does not replay")
    if manifest.get("archival_ready") is not True:
        raise ValueError("variant manifest is not archival-ready for dataset use")
    if manifest.get("output_bundle_sha256") != observed_bundle_sha256:
        raise ValueError("backend-rehashed variant bundle does not match the manifest")

    validation = manifest.get("validation")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        raise ValueError("variant manifest does not contain a passing validation")
    provenance = manifest.get("generator_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("archival variant is missing generator provenance")
    revisions = provenance.get("revisions")
    if not isinstance(revisions, Mapping):
        raise TypeError("archival variant is missing source revisions")
    for name in ("snmr", "newton", "isaac_lab"):
        commit = revisions.get(f"{name}_commit")
        if (
            not isinstance(commit, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None
            or revisions.get(f"{name}_dirty") is not False
            or revisions.get(f"{name}_repo_status") != "available"
        ):
            raise ValueError(
                f"archival variant is not bound to a clean {name} revision"
            )

    plan = manifest.get("plan")
    if not isinstance(plan, Mapping):
        raise TypeError("archival variant is missing its perturbation plan")
    plan_hash = manifest.get("plan_sha256")
    if not isinstance(plan_hash, str) or _canonical_hash(plan) != plan_hash:
        raise ValueError("archival variant perturbation-plan SHA-256 does not replay")
    if plan.get("generation_basis") != GENERATION_BASIS:
        raise ValueError("archival variant uses an unsupported generation basis")
    semantic_manifest = plan.get("semantic_manifest")
    semantic_hash = plan.get("semantic_manifest_sha256")
    if not isinstance(semantic_manifest, Mapping) or not isinstance(semantic_hash, str):
        raise TypeError("archival variant is missing its SemanticManifest binding")
    if _canonical_hash(semantic_manifest) != semantic_hash:
        raise ValueError("archival variant SemanticManifest SHA-256 does not replay")
    scale_groups = plan.get("body_scale_groups")
    if not isinstance(scale_groups, list):
        raise TypeError("archival variant is missing its body-scale grouping")
    scale_groups_hash = plan.get("body_scale_groups_sha256")
    if (
        not isinstance(scale_groups_hash, str)
        or _canonical_hash(scale_groups) != scale_groups_hash
    ):
        raise ValueError("archival variant body-scale grouping SHA-256 does not replay")
    if manifest.get("zero_perturbation_exact_copy") is not True and not scale_groups:
        raise ValueError("nonzero dataset variant has no registered body-scale groups")

    raw_body_scales = plan.get("body_scales")
    if not isinstance(raw_body_scales, list):
        raise TypeError("archival variant is missing realized body scales")
    try:
        body_scales = {
            str(item[0]): _finite(item[1], "realized body scale")
            for item in raw_body_scales
            if isinstance(item, list) and len(item) == 2
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("archival variant has invalid realized body scales") from exc
    if len(body_scales) != len(raw_body_scales):
        raise ValueError("archival variant has malformed or duplicate body scales")
    group_names: set[str] = set()
    grouped_bodies: set[str] = set()
    for raw_group in scale_groups:
        if not isinstance(raw_group, Mapping):
            raise TypeError("archival variant has a malformed body-scale group")
        group_name = raw_group.get("group")
        members = raw_group.get("bodies")
        if (
            not isinstance(group_name, str)
            or not group_name
            or group_name in group_names
            or not isinstance(members, list)
            or not members
        ):
            raise ValueError("archival variant has an invalid body-scale group")
        group_names.add(group_name)
        if any(
            not isinstance(member, str)
            or member not in body_scales
            or member in grouped_bodies
            for member in members
        ) or len(set(members)) != len(members):
            raise ValueError("archival variant has invalid grouped body membership")
        grouped_bodies.update(members)
        if len({body_scales[member] for member in members}) != 1:
            raise ValueError("archival variant body-scale group is not coherent")

    if _finite(plan.get("max_joint_limit_shift_rad"), "joint-limit bound") != 0.0:
        raise ValueError(
            "dataset variants cannot contain unqualified joint-limit shifts"
        )
    raw_shifts = plan.get("joint_limit_center_shifts_rad")
    if not isinstance(raw_shifts, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or _finite(item[1], "joint-limit shift") != 0.0
        for item in raw_shifts
    ):
        raise ValueError(
            "dataset variants cannot contain unqualified joint-limit shifts"
        )


def write_materialized_mjcf_variant(
    variant: MaterializedMjcfVariant,
    destination: str | Path,
) -> Path:
    """Publish a validated bundle to an exclusively created directory and re-hash it.

    ``destination`` is write-once: an existing path is always rejected.  The audit manifest
    is adjacent to the captured bundle members but is not itself an MJCF dependency, so the
    output bundle digest remains replayable with :class:`MjcfBundleSnapshot`.  The manifest
    is written last and acts as the completion marker; a failed publication removes only the
    directory this call created.
    """

    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(relative == "variant_manifest.json" for relative, _ in variant.members):
        raise ValueError("bundle already contains reserved variant_manifest.json")
    created = False
    try:
        target.mkdir(exist_ok=False)
        created = True
        entrypoint = _write_members_into_existing_directory(
            variant.members, variant.entrypoint, target
        )
        replay = MjcfBundleSnapshot.capture(entrypoint)
        if replay.sha256 != variant.output_bundle_sha256:
            raise ValueError(
                f"published bundle hash mismatch: {replay.sha256} != {variant.output_bundle_sha256}"
            )
        (target / "variant_manifest.json").write_bytes(
            json.dumps(
                variant.manifest(), indent=2, sort_keys=True, allow_nan=False
            ).encode("utf-8")
            + b"\n"
        )
        return entrypoint
    except Exception:
        if created:
            shutil.rmtree(target, ignore_errors=True)
        raise
