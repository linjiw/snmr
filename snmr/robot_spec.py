"""Canonical, identity-free physical description of a retargeting target.

``RobotKinematics`` deliberately exposes only the eight static features used by the
original SNMR checkpoints.  Those features are sufficient for FK-conditioned decoding,
but they cannot support a claim that a retargeter responds to mass, actuator strength,
latency, or contact geometry.  This module is the versioned contract for that next stage.

The contract has three design rules:

* hashes and source paths are provenance, never model features;
* missing actuator/control values stay missing and receive explicit availability masks;
* kinematics and dynamics have separate hashes, so a dynamics-twin intervention is
  mechanically auditable.

MJCF is the first supported source format because all currently trained SNMR robots have
MuJoCo assets.  URDF/USD adapters can target the same dataclasses without changing the
training or verification records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "snmr.robot.v0.1"
GRAVITY_M_S2 = 9.81
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_QUATERNION_NORM_ATOL = 1e-5
_JOINT_AXIS_NORM_ATOL = 1e-6


def _tuple(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) for x in values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quat_wxyz_to_rot6d(quat: Sequence[float]) -> tuple[float, ...]:
    w, x, y, z = (float(v) for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"invalid quaternion {tuple(quat)}")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # First two columns of the rotation matrix in column-major order, matching
    # ``snmr.rotation.quat_to_rot6d``.
    return (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    )


def _nonempty_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty, whitespace-trimmed string")
    return value


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a real number, not bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _finite_vector(
    values: Any,
    length: int,
    label: str,
    *,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a length-{length} numeric sequence")
    try:
        parsed = tuple(
            _finite_scalar(value, f"{label}[{index}]")
            for index, value in enumerate(values)
        )
    except TypeError as exc:
        raise TypeError(f"{label} must be a length-{length} numeric sequence") from exc
    if len(parsed) != length:
        raise ValueError(f"{label} must have length {length}, got {len(parsed)}")
    if nonnegative and any(value < 0.0 for value in parsed):
        raise ValueError(f"{label} must be nonnegative")
    return parsed


def _unit_quaternion(values: Any, label: str) -> tuple[float, float, float, float]:
    parsed = _finite_vector(values, 4, label)
    norm = math.sqrt(sum(value * value for value in parsed))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_QUATERNION_NORM_ATOL):
        raise ValueError(f"{label} must be a unit wxyz quaternion (norm={norm:g})")
    return parsed  # type: ignore[return-value]


def _quat_wxyz_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = _unit_quaternion(quat, "quaternion")
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_multiply(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = _unit_quaternion(left, "left quaternion")
    rw, rx, ry, rz = _unit_quaternion(right, "right quaternion")
    product = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    norm = math.sqrt(sum(value * value for value in product))
    return tuple(value / norm for value in product)  # type: ignore[return-value]


@dataclass(frozen=True)
class FrameConvention:
    length_unit: str = "meter"
    up_axis: str = "z"
    forward_axis: str = "x"
    quaternion: str = "wxyz"

    def __post_init__(self) -> None:
        expected = {
            "length_unit": "meter",
            "up_axis": "z",
            # The serialized value ``x`` denotes the positive X axis.
            "forward_axis": "x",
            "quaternion": "wxyz",
        }
        for field_name, canonical in expected.items():
            if getattr(self, field_name) != canonical:
                raise ValueError(f"{field_name} must be the canonical value {canonical!r}")


@dataclass(frozen=True)
class SemanticManifest:
    """The bounded manual annotation allowed for a new robot.

    Per-motion scales, correspondence weights, orientation offsets, prompts, and model
    adapters intentionally do not appear here.
    """

    root_link: str
    torso_link: str | None = None
    head_link: str | None = None
    left_hand_link: str | None = None
    right_hand_link: str | None = None
    left_foot_link: str | None = None
    right_foot_link: str | None = None
    allowed_contact_links: tuple[str, ...] = ()
    symmetry_pairs: tuple[tuple[str, str], ...] = ()

    def referenced_links(self) -> tuple[str, ...]:
        scalar = (
            self.root_link,
            self.torso_link,
            self.head_link,
            self.left_hand_link,
            self.right_hand_link,
            self.left_foot_link,
            self.right_foot_link,
        )
        values = [name for name in scalar if name is not None]
        values.extend(self.allowed_contact_links)
        for left, right in self.symmetry_pairs:
            values.extend((left, right))
        return tuple(values)


@dataclass(frozen=True)
class CollisionProxy:
    geometry_type: str
    local_position: tuple[float, float, float]
    local_rotation_wxyz: tuple[float, float, float, float]
    size: tuple[float, float, float]
    friction: tuple[float, float, float]
    contact_type: int
    contact_affinity: int


@dataclass(frozen=True)
class LinkSpec:
    name: str
    parent: str | None
    local_position: tuple[float, float, float]
    local_rotation_wxyz: tuple[float, float, float, float]
    mass: float
    center_of_mass: tuple[float, float, float]
    inertia_diagonal: tuple[float, float, float]
    inertia_rotation_wxyz: tuple[float, float, float, float]
    collision_proxies: tuple[CollisionProxy, ...] = ()


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent_link: str
    child_link: str
    joint_type: str
    axis: tuple[float, float, float]
    lower_limit: float
    upper_limit: float
    velocity_limit: float | None
    torque_limit: float | None
    armature: float
    damping: float
    friction_loss: float
    nominal_position: float
    kp: float | None = None
    kd: float | None = None
    # Pose of the joint frame in the child-link frame.  Defaults preserve legacy
    # serialized RobotSpecs whose hinge was implicitly located at the body origin.
    local_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _parent_to_joint_transform(
    link: LinkSpec,
    joint: JointSpec | None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compose a child-link pose with its optional joint-frame pose.

    Fixed/root links retain their link-frame transform.  For a revolute child, the
    translation and orientation identify the actual hinge frame in parent-link
    coordinates, including nonzero MJCF ``jnt_pos`` offsets.
    """

    if joint is None:
        return link.local_position, link.local_rotation_wxyz
    link_rotation = _quat_wxyz_matrix(link.local_rotation_wxyz)
    joint_offset = np.asarray(joint.local_position, dtype=np.float64)
    position = np.asarray(link.local_position, dtype=np.float64) + link_rotation @ joint_offset
    rotation = _quat_wxyz_multiply(
        link.local_rotation_wxyz,
        joint.local_rotation_wxyz,
    )
    return _tuple(position), rotation


@dataclass(frozen=True)
class ControlSpec:
    mode: str = "position_pd"
    control_dt: float = 0.02
    latency_seconds: float | None = None


@dataclass(frozen=True)
class RuntimeSpec:
    simulation_dt: float
    source_format: str = "mjcf"


@dataclass(frozen=True)
class RobotFeatureBundle:
    """Dense model inputs plus names needed to make their meaning auditable."""

    node_features: np.ndarray
    node_feature_names: tuple[str, ...]
    node_names: tuple[str, ...]
    parent_index: np.ndarray
    dynamics_available: np.ndarray
    dynamics_available_names: tuple[str, ...]


@dataclass(frozen=True)
class RobotSpec:
    schema_version: str
    asset_sha256: str
    frames: FrameConvention
    semantics: SemanticManifest
    control: ControlSpec
    runtime: RuntimeSpec
    total_mass: float
    standing_height: float
    arm_span: float | None
    links: tuple[LinkSpec, ...]
    joints: tuple[JointSpec, ...]

    @classmethod
    def from_mjcf(
        cls,
        path: str | Path,
        semantics: SemanticManifest,
        *,
        control: ControlSpec | None = None,
        velocity_limits: Mapping[str, float] | None = None,
        torque_limits: Mapping[str, float] | None = None,
        kp: Mapping[str, float] | None = None,
        kd: Mapping[str, float] | None = None,
    ) -> "RobotSpec":
        """Resolve an MJCF into a self-contained RobotSpec.

        Optional mappings are keyed by exact joint name.  Absent values remain ``None``;
        they are not silently replaced with a large finite constant.
        """

        import mujoco

        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        model = mujoco.MjModel.from_xml_path(str(source))

        free_joints = [
            j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_joints) != 1:
            raise ValueError(f"{source}: expected one free root joint, found {len(free_joints)}")
        root_body = int(model.jnt_bodyid[free_joints[0]])

        body_ids: list[int] = []

        def walk(body_id: int) -> None:
            body_ids.append(body_id)
            for child in range(model.nbody):
                if child != body_id and int(model.body_parentid[child]) == body_id:
                    walk(child)

        walk(root_body)
        included = set(body_ids)
        body_name = {
            body_id: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in body_ids
        }
        if any(name is None for name in body_name.values()):
            raise ValueError(f"{source}: every robot body must have a name")

        links: list[LinkSpec] = []
        for body_id in body_ids:
            proxies: list[CollisionProxy] = []
            for geom_id in range(model.ngeom):
                if int(model.geom_bodyid[geom_id]) != body_id:
                    continue
                geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id])).name.removeprefix("mjGEOM_").lower()
                proxies.append(CollisionProxy(
                    geometry_type=geom_type,
                    local_position=_tuple(model.geom_pos[geom_id]),
                    local_rotation_wxyz=_tuple(model.geom_quat[geom_id]),
                    size=_tuple(model.geom_size[geom_id]),
                    friction=_tuple(model.geom_friction[geom_id]),
                    contact_type=int(model.geom_contype[geom_id]),
                    contact_affinity=int(model.geom_conaffinity[geom_id]),
                ))
            parent_id = int(model.body_parentid[body_id])
            links.append(LinkSpec(
                name=str(body_name[body_id]),
                parent=str(body_name[parent_id]) if parent_id in included else None,
                local_position=_tuple(model.body_pos[body_id]),
                local_rotation_wxyz=_tuple(model.body_quat[body_id]),
                mass=float(model.body_mass[body_id]),
                center_of_mass=_tuple(model.body_ipos[body_id]),
                inertia_diagonal=_tuple(model.body_inertia[body_id]),
                inertia_rotation_wxyz=_tuple(model.body_iquat[body_id]),
                collision_proxies=tuple(proxies),
            ))

        velocity_limits = velocity_limits or {}
        torque_limits = torque_limits or {}
        kp = kp or {}
        kd = kd or {}
        joints: list[JointSpec] = []
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            child_id = int(model.jnt_bodyid[joint_id])
            if child_id not in included:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None:
                raise ValueError(f"{source}: every hinge joint must have a name")
            parent_id = int(model.body_parentid[child_id])
            if parent_id not in included:
                raise ValueError(f"{source}: hinge {name!r} has a parent outside the robot subtree")
            lo, hi = (float(x) for x in model.jnt_range[joint_id])
            if not bool(model.jnt_limited[joint_id]) or (lo == 0.0 and hi == 0.0):
                lo, hi = -math.pi, math.pi
            dof_id = int(model.jnt_dofadr[joint_id])
            qpos_id = int(model.jnt_qposadr[joint_id])

            parsed_torque: float | None = None
            if hasattr(model, "jnt_actfrclimited") and bool(model.jnt_actfrclimited[joint_id]):
                force_range = model.jnt_actfrcrange[joint_id]
                parsed_torque = float(max(abs(force_range[0]), abs(force_range[1])))
            if name in torque_limits:
                parsed_torque = float(torque_limits[name])

            joints.append(JointSpec(
                name=str(name),
                parent_link=str(body_name[parent_id]),
                child_link=str(body_name[child_id]),
                joint_type="revolute",
                axis=_tuple(model.jnt_axis[joint_id]),
                lower_limit=lo,
                upper_limit=hi,
                velocity_limit=float(velocity_limits[name]) if name in velocity_limits else None,
                torque_limit=parsed_torque,
                armature=float(model.dof_armature[dof_id]),
                damping=float(model.dof_damping[dof_id]),
                friction_loss=float(model.dof_frictionloss[dof_id]),
                nominal_position=float(model.qpos0[qpos_id]),
                kp=float(kp[name]) if name in kp else None,
                kd=float(kd[name]) if name in kd else None,
                local_position=_tuple(model.jnt_pos[joint_id]),
                # MJCF hinge axes and positions are expressed directly in the body frame;
                # unlike URDF, MJCF has no additional joint-frame orientation attribute.
                local_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ))

        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        body_xyz = np.asarray([data.xpos[body_id] for body_id in body_ids], dtype=np.float64)
        standing_height = float(body_xyz[:, 2].max() - body_xyz[:, 2].min())
        if standing_height <= 1e-6:
            # A horizontal nominal pose still needs a stable length scale.
            standing_height = float(np.linalg.norm(body_xyz - body_xyz[0], axis=1).max())

        arm_span: float | None = None
        if semantics.left_hand_link and semantics.right_hand_link:
            by_name = {name: body_id for body_id, name in body_name.items()}
            if semantics.left_hand_link in by_name and semantics.right_hand_link in by_name:
                arm_span = float(np.linalg.norm(
                    data.xpos[by_name[semantics.left_hand_link]]
                    - data.xpos[by_name[semantics.right_hand_link]]
                ))

        spec = cls(
            schema_version=SCHEMA_VERSION,
            asset_sha256=_sha256_file(source),
            frames=FrameConvention(),
            semantics=semantics,
            control=control or ControlSpec(),
            runtime=RuntimeSpec(simulation_dt=float(model.opt.timestep)),
            total_mass=float(sum(link.mass for link in links)),
            standing_height=standing_height,
            arm_span=arm_span,
            links=tuple(links),
            joints=tuple(joints),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported RobotSpec schema {self.schema_version!r}")
        if not isinstance(self.asset_sha256, str) or _SHA256_RE.fullmatch(
            self.asset_sha256.lower()
        ) is None:
            raise ValueError("asset_sha256 must be a 64-character hexadecimal digest")
        if not isinstance(self.frames, FrameConvention) or self.frames != FrameConvention():
            raise ValueError(
                "frames must use the canonical meter, Z-up, forward +X ('x'), wxyz convention"
            )
        if not isinstance(self.control, ControlSpec):
            raise TypeError("control must be a ControlSpec")
        if not isinstance(self.runtime, RuntimeSpec):
            raise TypeError("runtime must be a RuntimeSpec")
        _nonempty_name(self.control.mode, "control.mode")
        control_dt = _finite_scalar(self.control.control_dt, "control.control_dt")
        simulation_dt = _finite_scalar(self.runtime.simulation_dt, "runtime.simulation_dt")
        if control_dt <= 0.0 or simulation_dt <= 0.0:
            raise ValueError("control_dt and simulation_dt must be positive")
        _nonempty_name(self.runtime.source_format, "runtime.source_format")
        if self.control.latency_seconds is not None:
            latency = _finite_scalar(self.control.latency_seconds, "control.latency_seconds")
            if latency < 0.0:
                raise ValueError("latency_seconds must be nonnegative")
        if not self.links:
            raise ValueError("RobotSpec must contain links")
        total_mass = _finite_scalar(self.total_mass, "total_mass")
        standing_height = _finite_scalar(self.standing_height, "standing_height")
        if total_mass <= 0.0:
            raise ValueError(f"invalid total_mass {self.total_mass}")
        if standing_height <= 0.0:
            raise ValueError(f"invalid standing_height {self.standing_height}")
        if self.arm_span is not None and _finite_scalar(self.arm_span, "arm_span") <= 0.0:
            raise ValueError("arm_span must be positive when available")

        for index, link in enumerate(self.links):
            if not isinstance(link, LinkSpec):
                raise TypeError(f"links[{index}] must be a LinkSpec")
            _nonempty_name(link.name, f"links[{index}].name")
        names = [link.name for link in self.links]
        if len(set(names)) != len(names):
            raise ValueError("link names must be unique")
        name_set = set(names)
        roots = [link for link in self.links if link.parent is None]
        if not isinstance(self.semantics, SemanticManifest):
            raise TypeError("semantics must be a SemanticManifest")
        _nonempty_name(self.semantics.root_link, "semantics.root_link")
        if len(roots) != 1 or roots[0].name != self.semantics.root_link:
            raise ValueError("semantic root_link must be the unique kinematic root")

        link_by_name = {link.name: link for link in self.links}
        children: dict[str, list[str]] = {name: [] for name in names}
        summed_mass = 0.0
        for link in self.links:
            if link.parent is not None:
                _nonempty_name(link.parent, f"link {link.name!r} parent")
                if link.parent not in name_set:
                    raise ValueError(f"link {link.name!r} references unknown parent {link.parent!r}")
                if link.parent == link.name:
                    raise ValueError(f"link {link.name!r} cannot be its own parent")
                children[link.parent].append(link.name)
            _finite_vector(link.local_position, 3, f"link {link.name!r} local_position")
            _unit_quaternion(
                link.local_rotation_wxyz, f"link {link.name!r} local_rotation_wxyz"
            )
            mass = _finite_scalar(link.mass, f"link {link.name!r} mass")
            if mass < 0.0:
                raise ValueError(f"link {link.name!r} has negative mass")
            summed_mass += mass
            _finite_vector(link.center_of_mass, 3, f"link {link.name!r} center_of_mass")
            _finite_vector(
                link.inertia_diagonal,
                3,
                f"link {link.name!r} inertia_diagonal",
                nonnegative=True,
            )
            _unit_quaternion(
                link.inertia_rotation_wxyz,
                f"link {link.name!r} inertia_rotation_wxyz",
            )
            for proxy_index, proxy in enumerate(link.collision_proxies):
                if not isinstance(proxy, CollisionProxy):
                    raise TypeError(
                        f"link {link.name!r} collision_proxies[{proxy_index}] "
                        "must be CollisionProxy"
                    )
                prefix = f"link {link.name!r} collision_proxies[{proxy_index}]"
                _nonempty_name(proxy.geometry_type, f"{prefix}.geometry_type")
                _finite_vector(proxy.local_position, 3, f"{prefix}.local_position")
                _unit_quaternion(proxy.local_rotation_wxyz, f"{prefix}.local_rotation_wxyz")
                _finite_vector(proxy.size, 3, f"{prefix}.size", nonnegative=True)
                _finite_vector(proxy.friction, 3, f"{prefix}.friction", nonnegative=True)
                for field_name, value in (
                    ("contact_type", proxy.contact_type),
                    ("contact_affinity", proxy.contact_affinity),
                ):
                    if isinstance(value, (bool, np.bool_)) or not isinstance(
                        value, (int, np.integer)
                    ):
                        raise TypeError(f"{prefix}.{field_name} must be an integer")
                    if int(value) < 0:
                        raise ValueError(f"{prefix}.{field_name} must be nonnegative")

        if not math.isclose(summed_mass, total_mass, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"total_mass {total_mass:g} does not equal the link-mass sum {summed_mass:g}"
            )

        root_name = roots[0].name
        visited: set[str] = set()
        frontier = [root_name]
        while frontier:
            name = frontier.pop()
            if name in visited:
                raise ValueError("link parents must define a rooted connected acyclic tree")
            visited.add(name)
            frontier.extend(children[name])
        if visited != name_set:
            missing = sorted(name_set - visited)
            raise ValueError(
                "link parents must define a rooted connected acyclic tree; "
                f"unreachable links={missing}"
            )

        semantic_scalar_names = (
            "root_link",
            "torso_link",
            "head_link",
            "left_hand_link",
            "right_hand_link",
            "left_foot_link",
            "right_foot_link",
        )
        semantic_references: list[str] = []
        for field_name in semantic_scalar_names:
            value = getattr(self.semantics, field_name)
            if value is not None:
                semantic_references.append(_nonempty_name(value, f"semantics.{field_name}"))
        for left_field, right_field in (
            ("left_hand_link", "right_hand_link"),
            ("left_foot_link", "right_foot_link"),
        ):
            left_value = getattr(self.semantics, left_field)
            right_value = getattr(self.semantics, right_field)
            if left_value is not None and left_value == right_value:
                raise ValueError(f"semantics.{left_field} and {right_field} must be distinct")

        allowed_contacts = tuple(self.semantics.allowed_contact_links)
        for index, name in enumerate(allowed_contacts):
            semantic_references.append(
                _nonempty_name(name, f"semantics.allowed_contact_links[{index}]")
            )
        if len(set(allowed_contacts)) != len(allowed_contacts):
            raise ValueError("semantics.allowed_contact_links must be unique")

        symmetry_members: set[str] = set()
        for index, pair in enumerate(self.semantics.symmetry_pairs):
            if isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError(f"semantics.symmetry_pairs[{index}] must contain two links")
            left = _nonempty_name(pair[0], f"semantics.symmetry_pairs[{index}][0]")
            right = _nonempty_name(pair[1], f"semantics.symmetry_pairs[{index}][1]")
            if left == right:
                raise ValueError("a semantic symmetry pair must contain distinct links")
            if left in symmetry_members or right in symmetry_members:
                raise ValueError("a link may appear in at most one semantic symmetry pair")
            symmetry_members.update((left, right))
            semantic_references.extend((left, right))

        unknown = sorted(set(semantic_references) - name_set)
        if unknown:
            raise ValueError(f"semantic manifest references unknown links: {unknown}")

        for index, joint in enumerate(self.joints):
            if not isinstance(joint, JointSpec):
                raise TypeError(f"joints[{index}] must be a JointSpec")
            _nonempty_name(joint.name, f"joints[{index}].name")
        joint_names = [joint.name for joint in self.joints]
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("joint names must be unique")
        joint_children: set[str] = set()
        for joint in self.joints:
            if joint.parent_link not in name_set or joint.child_link not in name_set:
                raise ValueError(f"joint {joint.name!r} references an unknown link")
            if joint.child_link == root_name:
                raise ValueError(f"joint {joint.name!r} cannot attach the root link")
            expected_parent = link_by_name[joint.child_link].parent
            if joint.parent_link != expected_parent:
                raise ValueError(
                    f"joint {joint.name!r} parent/child is inconsistent with the link tree: "
                    f"expected parent {expected_parent!r}"
                )
            if joint.child_link in joint_children:
                raise ValueError(
                    f"link {joint.child_link!r} has more than one scalar joint attachment"
                )
            joint_children.add(joint.child_link)
            if joint.joint_type != "revolute":
                raise ValueError(
                    f"joint {joint.name!r} has unsupported type {joint.joint_type!r}; "
                    "RobotSpec v0.1 supports one revolute joint per child link"
                )
            axis = _finite_vector(joint.axis, 3, f"joint {joint.name!r} axis")
            axis_norm = math.sqrt(sum(value * value for value in axis))
            if not math.isclose(
                axis_norm, 1.0, rel_tol=0.0, abs_tol=_JOINT_AXIS_NORM_ATOL
            ):
                raise ValueError(f"joint {joint.name!r} axis must be unit length")
            _finite_vector(
                joint.local_position, 3, f"joint {joint.name!r} local_position"
            )
            _unit_quaternion(
                joint.local_rotation_wxyz,
                f"joint {joint.name!r} local_rotation_wxyz",
            )
            lower = _finite_scalar(joint.lower_limit, f"joint {joint.name!r} lower_limit")
            upper = _finite_scalar(joint.upper_limit, f"joint {joint.name!r} upper_limit")
            nominal = _finite_scalar(
                joint.nominal_position, f"joint {joint.name!r} nominal_position"
            )
            if not lower < upper:
                raise ValueError(f"joint {joint.name!r} has an empty range")
            if not lower <= nominal <= upper:
                raise ValueError(f"joint {joint.name!r} nominal_position lies outside its range")
            for label, value in (
                ("armature", joint.armature),
                ("damping", joint.damping),
                ("friction_loss", joint.friction_loss),
            ):
                parsed = _finite_scalar(value, f"joint {joint.name!r} {label}")
                if parsed < 0.0:
                    raise ValueError(f"joint {joint.name!r} has negative {label}")
            for label, value in (
                ("velocity_limit", joint.velocity_limit),
                ("torque_limit", joint.torque_limit),
                ("kp", joint.kp),
                ("kd", joint.kd),
            ):
                if value is not None:
                    parsed = _finite_scalar(value, f"joint {joint.name!r} {label}")
                    if parsed <= 0.0:
                        raise ValueError(f"joint {joint.name!r} has invalid {label}={value}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RobotSpec":
        semantics = payload["semantics"]
        spec = cls(
            schema_version=str(payload["schema_version"]),
            asset_sha256=str(payload["asset_sha256"]),
            frames=FrameConvention(**payload["frames"]),
            semantics=SemanticManifest(
                **{
                    **semantics,
                    "allowed_contact_links": tuple(semantics.get("allowed_contact_links", ())),
                    "symmetry_pairs": tuple(tuple(pair) for pair in semantics.get("symmetry_pairs", ())),
                }
            ),
            control=ControlSpec(**payload["control"]),
            runtime=RuntimeSpec(**payload["runtime"]),
            total_mass=float(payload["total_mass"]),
            standing_height=float(payload["standing_height"]),
            arm_span=None if payload.get("arm_span") is None else float(payload["arm_span"]),
            links=tuple(LinkSpec(
                **{
                    **link,
                    "local_position": tuple(link["local_position"]),
                    "local_rotation_wxyz": tuple(link["local_rotation_wxyz"]),
                    "center_of_mass": tuple(link["center_of_mass"]),
                    "inertia_diagonal": tuple(link["inertia_diagonal"]),
                    "inertia_rotation_wxyz": tuple(link["inertia_rotation_wxyz"]),
                    "collision_proxies": tuple(CollisionProxy(
                        **{
                            **proxy,
                            "local_position": tuple(proxy["local_position"]),
                            "local_rotation_wxyz": tuple(proxy["local_rotation_wxyz"]),
                            "size": tuple(proxy["size"]),
                            "friction": tuple(proxy["friction"]),
                        }
                    ) for proxy in link.get("collision_proxies", ())),
                }
            ) for link in payload["links"]),
            joints=tuple(JointSpec(
                **{
                    **joint,
                    "axis": tuple(joint["axis"]),
                    "local_position": tuple(joint.get("local_position", (0.0, 0.0, 0.0))),
                    "local_rotation_wxyz": tuple(
                        joint.get("local_rotation_wxyz", (1.0, 0.0, 0.0, 0.0))
                    ),
                }
            ) for joint in payload["joints"]),
        )
        spec.validate()
        return spec

    @property
    def kinematic_hash(self) -> str:
        return _canonical_hash({
            "schema_version": self.schema_version,
            "frames": asdict(self.frames),
            "semantics": asdict(self.semantics),
            "links": [
                {
                    "name": link.name,
                    "parent": link.parent,
                    # A free-root body's MJCF pose is an asset spawn/default-world
                    # transform, not morphology.  Canonicalize it out of the
                    # kinematic identity just as model_features() does below.
                    "local_position": (
                        (0.0, 0.0, 0.0) if link.parent is None else link.local_position
                    ),
                    "local_rotation_wxyz": (
                        (1.0, 0.0, 0.0, 0.0)
                        if link.parent is None
                        else link.local_rotation_wxyz
                    ),
                    "collision_proxies": [asdict(proxy) for proxy in link.collision_proxies],
                }
                for link in self.links
            ],
            "joints": [
                {
                    "name": joint.name,
                    "parent_link": joint.parent_link,
                    "child_link": joint.child_link,
                    "joint_type": joint.joint_type,
                    "axis": joint.axis,
                    "local_position": joint.local_position,
                    "local_rotation_wxyz": joint.local_rotation_wxyz,
                    "lower_limit": joint.lower_limit,
                    "upper_limit": joint.upper_limit,
                    "nominal_position": joint.nominal_position,
                }
                for joint in self.joints
            ],
        })

    @property
    def dynamics_hash(self) -> str:
        return _canonical_hash({
            "control": asdict(self.control),
            "runtime": asdict(self.runtime),
            "links": [
                {
                    "name": link.name,
                    "mass": link.mass,
                    "center_of_mass": link.center_of_mass,
                    "inertia_diagonal": link.inertia_diagonal,
                    "inertia_rotation_wxyz": link.inertia_rotation_wxyz,
                }
                for link in self.links
            ],
            "joints": [
                {
                    "name": joint.name,
                    "velocity_limit": joint.velocity_limit,
                    "torque_limit": joint.torque_limit,
                    "armature": joint.armature,
                    "damping": joint.damping,
                    "friction_loss": joint.friction_loss,
                    "kp": joint.kp,
                    "kd": joint.kd,
                }
                for joint in self.joints
            ],
        })

    @property
    def spec_hash(self) -> str:
        return _canonical_hash(self.to_dict())

    def dynamics_twin(
        self,
        *,
        torque_scale: float = 1.0,
        velocity_scale: float = 1.0,
        mass_scale: float = 1.0,
        latency_seconds: float | None = None,
        control_dt: float | None = None,
    ) -> "RobotSpec":
        """Return a same-geometry intervention with coherently scaled dynamics.

        Global mass scaling also scales inertia, corresponding to a density intervention on
        unchanged geometry.  ``None`` actuator values remain unavailable.
        """

        for label, value in (
            ("torque_scale", torque_scale),
            ("velocity_scale", velocity_scale),
            ("mass_scale", mass_scale),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive, got {value}")
        if latency_seconds is not None and latency_seconds < 0.0:
            raise ValueError("latency_seconds must be nonnegative")
        if control_dt is not None and control_dt <= 0.0:
            raise ValueError("control_dt must be positive")

        twin = replace(
            self,
            total_mass=self.total_mass * mass_scale,
            links=tuple(replace(
                link,
                mass=link.mass * mass_scale,
                inertia_diagonal=tuple(x * mass_scale for x in link.inertia_diagonal),
            ) for link in self.links),
            joints=tuple(replace(
                joint,
                torque_limit=(None if joint.torque_limit is None else joint.torque_limit * torque_scale),
                velocity_limit=(
                    None if joint.velocity_limit is None else joint.velocity_limit * velocity_scale
                ),
            ) for joint in self.joints),
            control=replace(
                self.control,
                latency_seconds=(
                    self.control.latency_seconds if latency_seconds is None else latency_seconds
                ),
                control_dt=self.control.control_dt if control_dt is None else control_dt,
            ),
        )
        twin.validate()
        if twin.kinematic_hash != self.kinematic_hash:
            raise AssertionError("a dynamics twin changed the kinematic hash")
        return twin

    def model_features(self) -> RobotFeatureBundle:
        """Create dimensionless, identity-free per-link tokens.

        Missing torque/velocity/gain values are encoded as zero *and* accompanied by four
        availability bits.  Neither names, hashes, asset paths, nor serialization indices
        are present in the numeric feature matrix.
        """

        link_index = {link.name: i for i, link in enumerate(self.links)}
        joint_by_child = {joint.child_link: joint for joint in self.joints}
        semantic_roles = (
            "is_root",
            "is_torso",
            "is_head",
            "is_left_hand",
            "is_right_hand",
            "is_left_foot",
            "is_right_foot",
            "is_allowed_contact",
        )
        names = (
            "parent_to_joint_x_over_height",
            "parent_to_joint_y_over_height",
            "parent_to_joint_z_over_height",
            "rest_rot6d_0", "rest_rot6d_1", "rest_rot6d_2",
            "rest_rot6d_3", "rest_rot6d_4", "rest_rot6d_5",
            "mass_fraction",
            "com_x_over_height", "com_y_over_height", "com_z_over_height",
            "inertia_x_over_mh2", "inertia_y_over_mh2", "inertia_z_over_mh2",
            "collision_radius_over_height",
            "joint_axis_x", "joint_axis_y", "joint_axis_z",
            "joint_lower_over_pi", "joint_upper_over_pi", "has_dof",
            "torque_over_mgh", "velocity_per_control_step",
            "armature_over_mh2", "damping_dt_over_mh2",
            "kp_dt2_over_mh2", "kd_dt_over_mh2",
            *semantic_roles,
        )
        availability_names = ("torque_limit", "velocity_limit", "kp", "kd")
        features: list[list[float]] = []
        availability: list[list[float]] = []
        parent_index: list[int] = []
        height = self.standing_height
        mass = self.total_mass
        inertial_scale = mass * height * height
        torque_scale = mass * GRAVITY_M_S2 * height
        dt = self.control.control_dt
        allowed_contacts = set(self.semantics.allowed_contact_links)

        role_targets = (
            self.semantics.root_link,
            self.semantics.torso_link,
            self.semantics.head_link,
            self.semantics.left_hand_link,
            self.semantics.right_hand_link,
            self.semantics.left_foot_link,
            self.semantics.right_foot_link,
        )
        for link in self.links:
            joint = joint_by_child.get(link.name)
            if link.parent is None:
                # The canonical RobotSpec frame is pelvis-origin.  MJCF commonly stores
                # a standing spawn height/orientation on the free-root body; exposing it
                # here would leak authoring/default-state identity into the model.
                joint_position = (0.0, 0.0, 0.0)
                joint_rotation = (1.0, 0.0, 0.0, 0.0)
            else:
                joint_position, joint_rotation = _parent_to_joint_transform(link, joint)
            collision_radius = max(
                (math.sqrt(sum(value * value for value in proxy.size))
                 for proxy in link.collision_proxies),
                default=0.0,
            )
            if joint is None:
                joint_values = [0.0] * 12
                joint_available = [0.0] * 4
            else:
                torque = 0.0 if joint.torque_limit is None else joint.torque_limit / torque_scale
                velocity = 0.0 if joint.velocity_limit is None else joint.velocity_limit * dt
                kp_value = 0.0 if joint.kp is None else joint.kp * dt * dt / inertial_scale
                kd_value = 0.0 if joint.kd is None else joint.kd * dt / inertial_scale
                joint_values = [
                    *joint.axis,
                    joint.lower_limit / math.pi,
                    joint.upper_limit / math.pi,
                    1.0,
                    torque,
                    velocity,
                    joint.armature / inertial_scale,
                    joint.damping * dt / inertial_scale,
                    kp_value,
                    kd_value,
                ]
                joint_available = [
                    float(joint.torque_limit is not None),
                    float(joint.velocity_limit is not None),
                    float(joint.kp is not None),
                    float(joint.kd is not None),
                ]
            role_values = [float(link.name == target) for target in role_targets]
            role_values.append(float(link.name in allowed_contacts))
            inertia_denom = max(link.mass * height * height, np.finfo(np.float64).eps)
            features.append([
                *(value / height for value in joint_position),
                *_quat_wxyz_to_rot6d(joint_rotation),
                link.mass / mass,
                *(value / height for value in link.center_of_mass),
                *(value / inertia_denom for value in link.inertia_diagonal),
                collision_radius / height,
                *joint_values,
                *role_values,
            ])
            availability.append(joint_available)
            parent_index.append(-1 if link.parent is None else link_index[link.parent])

        node_features = np.asarray(features, dtype=np.float32)
        if node_features.shape[1] != len(names):
            raise AssertionError((node_features.shape, len(names)))
        return RobotFeatureBundle(
            node_features=node_features,
            node_feature_names=tuple(names),
            node_names=tuple(link.name for link in self.links),
            parent_index=np.asarray(parent_index, dtype=np.int64),
            dynamics_available=np.asarray(availability, dtype=np.float32),
            dynamics_available_names=availability_names,
        )
