"""Deterministic virtual same-topology RobotSpec perturbations.

This module is intentionally limited to architecture counterfactuals.  It emits canonical
JSON bytes describing a virtual RobotSpec and hashes those exact bytes, but it does *not*
claim to emit a simulator-loadable MJCF/URDF/USD asset.  The manifest therefore carries an
explicit ``backend_compatible=false`` field.  P4 training and verifier rollouts must refuse
these virtual variants until a separately tested asset materializer exists.

Geometry changes propagate to centers of mass, collision proxies, masses, and inertias under
a declared uniform constant-density rule::

    length -> s * length
    mass   -> s^3 * mass
    inertia -> s^5 * inertia

The current perturbation is same-topology and requires zero nominal joint angles.  That
restriction is fail-closed: nominal-pose FK for arbitrary authored joint defaults has not
yet been qualified.  The registered kinematic model must still receive
``feature_set="kinematic"`` tokens, so derived mass/inertia values cannot become an
accidental dynamics treatment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np

from .robot_spec import CollisionProxy, JointSpec, LinkSpec, RobotSpec
VARIANT_SCHEMA_VERSION = "snmr.virtual-robot-variant.v0.2"
VIRTUAL_ASSET_FORMAT = "snmr.virtual-robot-spec-json.v0.1"


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scale_vector(values: Sequence[float], scale: float) -> tuple[float, ...]:
    return tuple(float(value) * scale for value in values)


def _quaternion_matrix_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("variant input contains an invalid quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _canonical_source_hash(spec: RobotSpec) -> str:
    """Hash the physical source independent of link/joint serialization order."""

    return _canonical_hash({
        "schema_version": spec.schema_version,
        "asset_sha256": spec.asset_sha256,
        "frames": asdict(spec.frames),
        "semantics": asdict(spec.semantics),
        "control": asdict(spec.control),
        "runtime": asdict(spec.runtime),
        "links": [asdict(link) for link in sorted(spec.links, key=lambda item: item.name)],
        "joints": [asdict(joint) for joint in sorted(spec.joints, key=lambda item: item.name)],
    })


@dataclass(frozen=True)
class CoherentVariantConfig:
    """Registered perturbation bounds for same-topology morphology augmentation."""

    link_length_fraction: float = 0.15
    joint_limit_shift_rad: float = math.radians(10.0)
    preserve_bilateral_symmetry: bool = True
    geometry_rule: str = "uniform_constant_density"

    def validate(self) -> None:
        if not math.isfinite(self.link_length_fraction):
            raise ValueError("link_length_fraction must be finite")
        if not 0.0 <= self.link_length_fraction < 1.0:
            raise ValueError("link_length_fraction must lie in [0, 1)")
        if not math.isfinite(self.joint_limit_shift_rad):
            raise ValueError("joint_limit_shift_rad must be finite")
        if not 0.0 <= self.joint_limit_shift_rad < math.pi:
            raise ValueError("joint_limit_shift_rad must lie in [0, pi)")
        if self.geometry_rule != "uniform_constant_density":
            raise ValueError("only uniform_constant_density geometry scaling is supported")


@dataclass(frozen=True)
class CoherentVariantManifest:
    """Audit metadata kept outside all numeric model inputs."""

    schema_version: str
    seed: int
    parent_family: str
    source_spec_hash: str
    source_canonical_hash: str
    output_spec_hash: str
    output_kinematic_hash: str
    virtual_asset_sha256: str
    virtual_asset_json: str
    backend_compatible: bool
    config: CoherentVariantConfig
    link_scales: tuple[tuple[str, float], ...]
    joint_limit_center_shifts_rad: tuple[tuple[str, float], ...]
    topology_changed: bool
    consistency_checks: tuple[tuple[str, bool], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "parent_family": self.parent_family,
            "source_spec_hash": self.source_spec_hash,
            "source_canonical_hash": self.source_canonical_hash,
            "output_spec_hash": self.output_spec_hash,
            "output_kinematic_hash": self.output_kinematic_hash,
            "virtual_asset_sha256": self.virtual_asset_sha256,
            "virtual_asset_json": self.virtual_asset_json,
            "backend_compatible": self.backend_compatible,
            "config": asdict(self.config),
            "link_scales": [list(value) for value in self.link_scales],
            "joint_limit_center_shifts_rad": [
                list(value) for value in self.joint_limit_center_shifts_rad
            ],
            "topology_changed": self.topology_changed,
            "consistency_checks": [list(value) for value in self.consistency_checks],
        }
        payload["manifest_sha256"] = _canonical_hash(payload)
        return payload

    def require_backend_compatible(self) -> None:
        """Fail closed when a virtual probe is offered to a simulator worker."""

        if not self.backend_compatible:
            raise RuntimeError(
                "virtual RobotSpec variants are tokenizer-only; materialize and parity-test "
                "an exact MJCF/URDF/USD asset before backend use"
            )


def _symmetry_group_by_link(spec: RobotSpec, preserve: bool) -> dict[str, tuple[str, ...]]:
    group_by_link = {link.name: (link.name,) for link in spec.links}
    if not preserve:
        return group_by_link
    for left, right in spec.semantics.symmetry_pairs:
        group = tuple(sorted((left, right)))
        group_by_link[left] = group
        group_by_link[right] = group
    return group_by_link


def _draw_group_values(
    groups: Sequence[tuple[str, ...]],
    *,
    rng: np.random.Generator,
    radius: float,
) -> dict[tuple[str, ...], float]:
    return {
        group: float(rng.uniform(-radius, radius))
        for group in sorted(set(groups))
    }


def _scaled_proxy(proxy: CollisionProxy, scale: float) -> CollisionProxy:
    return replace(
        proxy,
        local_position=_scale_vector(proxy.local_position, scale),
        size=_scale_vector(proxy.size, scale),
    )


def _rest_link_positions(spec: RobotSpec) -> dict[str, np.ndarray]:
    """Resolve rest link origins with the same body-transform convention as MJCF parsing."""

    by_name = {link.name: link for link in spec.links}
    positions: dict[str, np.ndarray] = {}
    rotations: dict[str, np.ndarray] = {}

    def resolve(name: str) -> None:
        if name in positions:
            return
        link = by_name[name]
        local_rotation = _quaternion_matrix_wxyz(link.local_rotation_wxyz)
        if link.parent is None:
            positions[name] = np.asarray(link.local_position, dtype=np.float64)
            rotations[name] = local_rotation
            return
        resolve(link.parent)
        positions[name] = (
            positions[link.parent]
            + rotations[link.parent] @ np.asarray(link.local_position, dtype=np.float64)
        )
        rotations[name] = rotations[link.parent] @ local_rotation

    for link_name in sorted(by_name):
        resolve(link_name)
    return positions


def _rest_scales(spec: RobotSpec) -> tuple[float, float | None]:
    positions = _rest_link_positions(spec)
    stacked = np.stack([positions[link.name] for link in spec.links], axis=0)
    standing_height = float(stacked[:, 2].max() - stacked[:, 2].min())
    if standing_height <= 1e-9:
        root = positions[spec.semantics.root_link]
        standing_height = float(np.linalg.norm(stacked - root[None, :], axis=1).max())
    if standing_height <= 1e-9:
        raise ValueError("variant rest geometry has no usable length scale")
    arm_span: float | None = None
    left_hand = spec.semantics.left_hand_link
    right_hand = spec.semantics.right_hand_link
    if left_hand is not None and right_hand is not None:
        arm_span = float(np.linalg.norm(positions[left_hand] - positions[right_hand]))
        if arm_span <= 1e-9:
            arm_span = None
    return standing_height, arm_span


def generate_virtual_coherent_variant(
    spec: RobotSpec,
    *,
    parent_family: str,
    seed: int,
    config: CoherentVariantConfig | None = None,
) -> tuple[RobotSpec, CoherentVariantManifest]:
    """Generate one deterministic virtual same-topology architecture probe.

    The returned RobotSpec is valid for tokenization and counterfactual model tests.  Its
    ``asset_sha256`` binds the exact ``virtual_asset_json`` bytes in the manifest, not a
    simulator asset.  Callers must check ``backend_compatible`` before any backend use.
    """

    spec.validate()
    if not isinstance(parent_family, str) or not parent_family.strip():
        raise ValueError("parent_family must be non-empty audit metadata")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    parsed_seed = int(seed)
    variant_config = config or CoherentVariantConfig()
    variant_config.validate()
    nonzero_nominals = [
        joint.name for joint in spec.joints if abs(joint.nominal_position) > 1.0e-12
    ]
    if nonzero_nominals:
        raise ValueError(
            "virtual variant generation requires zero nominal joint positions until "
            f"nominal-pose FK is qualified; nonzero joints: {nonzero_nominals}"
        )

    group_by_link = _symmetry_group_by_link(
        spec, variant_config.preserve_bilateral_symmetry
    )
    rng = np.random.default_rng(parsed_seed)
    length_delta_by_group = _draw_group_values(
        tuple(group_by_link.values()),
        rng=rng,
        radius=variant_config.link_length_fraction,
    )
    link_scale = {
        link.name: 1.0 + length_delta_by_group[group_by_link[link.name]]
        for link in spec.links
    }
    # The free-root transform is an asset spawn/default pose, not morphology.  Never
    # perturb it or turn authored world height into a procedural-family cue.
    root_name = spec.semantics.root_link
    link_scale[root_name] = 1.0

    scaled_links = []
    for link in sorted(spec.links, key=lambda item: item.name):
        scale = link_scale[link.name]
        scaled_links.append(replace(
            link,
            local_position=_scale_vector(link.local_position, scale),
            mass=link.mass * scale ** 3,
            center_of_mass=_scale_vector(link.center_of_mass, scale),
            inertia_diagonal=_scale_vector(link.inertia_diagonal, scale ** 5),
            collision_proxies=tuple(
                _scaled_proxy(proxy, scale) for proxy in link.collision_proxies
            ),
        ))

    joint_group = {
        joint.name: group_by_link[joint.child_link]
        for joint in spec.joints
    }
    raw_shift_by_group = _draw_group_values(
        tuple(joint_group.values()),
        rng=rng,
        radius=variant_config.joint_limit_shift_rad,
    )
    applied_shift: dict[str, float] = {}
    scaled_joints: list[JointSpec] = []
    epsilon = 1e-9
    for joint in sorted(spec.joints, key=lambda item: item.name):
        requested = raw_shift_by_group[joint_group[joint.name]]
        minimum = joint.nominal_position - joint.upper_limit + epsilon
        maximum = joint.nominal_position - joint.lower_limit - epsilon
        shift = float(np.clip(requested, minimum, maximum))
        applied_shift[joint.name] = shift
        scale = link_scale[joint.child_link]
        scaled_joints.append(replace(
            joint,
            lower_limit=joint.lower_limit + shift,
            upper_limit=joint.upper_limit + shift,
            local_position=_scale_vector(joint.local_position, scale),
            armature=joint.armature * scale ** 5,
        ))

    source_canonical_hash = _canonical_source_hash(spec)
    virtual_runtime = replace(spec.runtime, source_format=VIRTUAL_ASSET_FORMAT)
    virtual_asset_payload = {
        "format": VIRTUAL_ASSET_FORMAT,
        "frames": asdict(spec.frames),
        "semantics": asdict(spec.semantics),
        "control": asdict(spec.control),
        "runtime": asdict(virtual_runtime),
        "links": [asdict(link) for link in scaled_links],
        "joints": [asdict(joint) for joint in scaled_joints],
    }
    virtual_asset_json = json.dumps(
        virtual_asset_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    virtual_asset_sha256 = hashlib.sha256(virtual_asset_json.encode("utf-8")).hexdigest()
    provisional = replace(
        spec,
        asset_sha256=virtual_asset_sha256,
        runtime=virtual_runtime,
        links=tuple(scaled_links),
        joints=tuple(scaled_joints),
        total_mass=float(sum(link.mass for link in scaled_links)),
    )
    standing_height, arm_span = _rest_scales(provisional)
    variant = replace(
        provisional,
        standing_height=standing_height,
        arm_span=arm_span,
    )
    variant.validate()

    source_links = {link.name: link for link in spec.links}
    source_joints = {joint.name: joint for joint in spec.joints}
    checks = {
        "same_link_names": set(source_links) == {link.name for link in variant.links},
        "same_joint_names": set(source_joints) == {joint.name for joint in variant.joints},
        "same_parent_tree": all(
            source_links[link.name].parent == link.parent for link in variant.links
        ),
        "same_joint_axes": all(
            source_joints[joint.name].axis == joint.axis for joint in variant.joints
        ),
        "nominals_inside_limits": all(
            joint.lower_limit <= joint.nominal_position <= joint.upper_limit
            for joint in variant.joints
        ),
        "mass_matches_geometry_rule": all(math.isclose(
            next(item for item in variant.links if item.name == name).mass,
            source_links[name].mass * scale ** 3,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) for name, scale in link_scale.items()),
    }
    if not all(checks.values()):
        raise AssertionError(f"coherent variant consistency failure: {checks}")

    manifest = CoherentVariantManifest(
        schema_version=VARIANT_SCHEMA_VERSION,
        seed=parsed_seed,
        parent_family=parent_family.strip(),
        source_spec_hash=spec.spec_hash,
        source_canonical_hash=source_canonical_hash,
        output_spec_hash=variant.spec_hash,
        output_kinematic_hash=variant.kinematic_hash,
        virtual_asset_sha256=virtual_asset_sha256,
        virtual_asset_json=virtual_asset_json,
        backend_compatible=False,
        config=variant_config,
        link_scales=tuple(sorted(link_scale.items())),
        joint_limit_center_shifts_rad=tuple(sorted(applied_shift.items())),
        topology_changed=False,
        consistency_checks=tuple(sorted(checks.items())),
    )
    return variant, manifest
