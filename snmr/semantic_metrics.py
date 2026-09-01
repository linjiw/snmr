"""Teacher-independent semantic evaluation for cross-embodiment retargeting.

The evaluator deliberately has no GMR/teacher trajectory argument.  It compares a canonical
``HumanMotionSpec`` directly with exact robot FK buffers under a versioned correspondence.
Every degree of freedom that could otherwise make the metric movable -- robot scale, root
frame, semantic point offsets, orientation calibration, contacts, and gate thresholds -- is
either obtained from a hash-matched ``RobotSpec`` or included in a canonical digest.

Anchor calibration uses an explicit right-multiplication convention.  For link/body world pose
``(p_WL, q_WL)`` and local calibration ``(p_LS, q_LS)``, the semantic pose is::

    p_WS = p_WL + rotate(q_WL, p_LS)
    q_WS = q_WL ⊗ q_LS

Thus ``q_LS`` maps the semantic frame into the link/body frame.  Both human and robot offsets
are immutable values in ``SemanticCorrespondence.sha256``.  Robot foot contacts are derived
from calibrated sole points using the frozen, hash-bound ``RobotContactProtocol``; callers
cannot supply candidate-friendly contact labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import struct
from typing import Any, Sequence

import numpy as np
import torch

from . import rotation as rot
from .data import heading_quat_inverse
from .motion_spec import CONTACT_LABELS, HumanFrameConvention, HumanMotionSpec
from .robot_spec import RobotSpec


SEMANTIC_METRIC_SCHEMA_VERSION = "snmr.semantic_metrics.v1"
SEMANTIC_ROLE_SET_VERSION = "snmr.semantic_roles.whole_body.v1"
SEMANTIC_CORRESPONDENCE_SCHEMA_VERSION = "snmr.semantic_correspondence.v1"
ROBOT_CONTACT_SCHEMA_VERSION = "snmr.robot_contact.kinematic_sole.v1"

# This exact role set and order is part of the benchmark protocol.  It prevents a run from
# silently omitting a difficult body part or changing aggregation weights after seeing results.
REQUIRED_SEMANTIC_ROLES = (
    "pelvis",
    "torso",
    "head",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
)
ORIENTATION_ROLES = ("left_hand", "right_hand", "left_foot", "right_foot")

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_QUATERNION_NORM_ATOL = 1e-5
_TIME_ATOL = 1e-9
_HEADING_PROJECTION_EPS = 1e-10


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, whitespace-trimmed string")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.lower()) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest")
    return value.lower()


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _numeric_array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _immutable_array(value: Any, name: str) -> np.ndarray:
    contiguous = np.ascontiguousarray(_numeric_array(value, name), dtype="<f8")
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype="<f8")
    return frozen.reshape(contiguous.shape)


def _tuple_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a length-{length} numeric sequence")
    try:
        result = tuple(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be a length-{length} numeric sequence") from exc
    if len(result) != length:
        raise ValueError(f"{name} must have length {length}, got {len(result)}")
    return result


def _unit_quaternion_tuple(value: Any, name: str) -> tuple[float, float, float, float]:
    result = _tuple_vector(value, 4, name)
    norm = math.sqrt(sum(component * component for component in result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_QUATERNION_NORM_ATOL):
        raise ValueError(f"{name} must be a unit wxyz quaternion")
    # q and -q are one rotation.  Canonical sign makes correspondence hashes stable.
    for component in result:
        if abs(component) > np.finfo(np.float64).eps:
            if component < 0.0:
                result = tuple(-item for item in result)
            break
    return result  # type: ignore[return-value]


def _canonical_quaternions(value: Any, name: str) -> np.ndarray:
    result = _numeric_array(value, name)
    if result.ndim < 2 or result.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (..., 4), got {result.shape}")
    norms = np.linalg.norm(result, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_NORM_ATOL):
        raise ValueError(f"{name} must contain unit quaternions")
    canonical = np.array(result, dtype="<f8", order="C", copy=True)
    flat = canonical.reshape(-1, 4)
    nonzero = np.abs(flat) > np.finfo(np.float64).eps
    first = np.argmax(nonzero, axis=1)
    signs = flat[np.arange(flat.shape[0]), first]
    flat[signs < 0.0] *= -1.0
    return _immutable_array(canonical, name)


def _names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of names")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of names") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    for index, value in enumerate(result):
        _nonempty_string(value, f"{name}[{index}]")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _update_exact_buffer_hash(digest: Any, name: str, value: np.ndarray) -> None:
    label = name.encode("utf-8")
    digest.update(struct.pack("<I", len(label)))
    digest.update(label)
    digest.update(struct.pack("<I", value.ndim))
    digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
    # The public contracts intentionally expose read-only buffers.  Hash a private copy so
    # signed-zero canonicalisation cannot alias or mutate the trajectory.
    normalized = np.array(value, dtype="<f8", order="C", copy=True)
    normalized[normalized == 0.0] = 0.0
    digest.update(normalized.tobytes(order="C"))


@dataclass(frozen=True)
class SemanticAnchor:
    """One calibrated human-body/robot-link semantic pose.

    Point offsets are meters in their source body/link frame.  Orientation offsets use the
    module-level right-multiplication convention ``q_semantic = q_source ⊗ q_offset``.
    """

    role: str
    human_body: str
    robot_link: str
    human_point_local_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_point_local_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    human_orientation_offset_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    robot_orientation_offset_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _nonempty_string(self.role, "role"))
        object.__setattr__(self, "human_body", _nonempty_string(self.human_body, "human_body"))
        object.__setattr__(self, "robot_link", _nonempty_string(self.robot_link, "robot_link"))
        object.__setattr__(
            self,
            "human_point_local_m",
            _tuple_vector(self.human_point_local_m, 3, "human_point_local_m"),
        )
        object.__setattr__(
            self,
            "robot_point_local_m",
            _tuple_vector(self.robot_point_local_m, 3, "robot_point_local_m"),
        )
        object.__setattr__(
            self,
            "human_orientation_offset_wxyz",
            _unit_quaternion_tuple(
                self.human_orientation_offset_wxyz, "human_orientation_offset_wxyz"
            ),
        )
        object.__setattr__(
            self,
            "robot_orientation_offset_wxyz",
            _unit_quaternion_tuple(
                self.robot_orientation_offset_wxyz, "robot_orientation_offset_wxyz"
            ),
        )


@dataclass(frozen=True)
class RobotContactProtocol:
    """Frozen differentiable sole-height/velocity contact detector."""

    schema_version: str = ROBOT_CONTACT_SCHEMA_VERSION
    speed_threshold_m_s: float = 0.20
    height_clearance_m: float = 0.05
    speed_softness_m_s: float = 0.05
    height_softness_m: float = 0.01
    ground_height_m: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != ROBOT_CONTACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported robot contact schema {self.schema_version!r}")
        for field in (
            "speed_threshold_m_s",
            "height_clearance_m",
            "speed_softness_m_s",
            "height_softness_m",
        ):
            object.__setattr__(self, field, _finite_float(getattr(self, field), field, positive=True))
        object.__setattr__(
            self, "ground_height_m", _finite_float(self.ground_height_m, "ground_height_m")
        )

    @property
    def sha256(self) -> str:
        return _canonical_hash(asdict(self))


@dataclass(frozen=True)
class SemanticThresholds:
    """Pre-registered two-sided motion-fidelity gate thresholds."""

    contact_probability_threshold: float = 0.5
    amplitude_ratio_min: float = 0.50
    amplitude_ratio_max: float = 1.50
    energy_ratio_min: float = 0.25
    energy_ratio_max: float = 4.00
    jitter_ratio_min: float = 0.25
    jitter_ratio_max: float = 4.00
    motion_floor: float = 1e-12

    def __post_init__(self) -> None:
        values = {}
        for field in self.__dataclass_fields__:
            values[field] = _finite_float(
                getattr(self, field), field, positive=field != "contact_probability_threshold"
            )
            object.__setattr__(self, field, values[field])
        if not 0.0 < self.contact_probability_threshold < 1.0:
            raise ValueError("contact_probability_threshold must lie strictly between 0 and 1")
        for label in ("amplitude", "energy", "jitter"):
            lower = values[f"{label}_ratio_min"]
            upper = values[f"{label}_ratio_max"]
            if not lower <= 1.0 <= upper or lower >= upper:
                raise ValueError(f"{label} ratio bounds must satisfy 0 < min <= 1 <= max")

    @property
    def sha256(self) -> str:
        return _canonical_hash(asdict(self))


@dataclass(frozen=True)
class SemanticCorrespondence:
    """Versioned, complete semantic calibration for one robot/body convention."""

    mapping_version: str
    anchors: tuple[SemanticAnchor, ...]
    contact_protocol: RobotContactProtocol = RobotContactProtocol()
    thresholds: SemanticThresholds = SemanticThresholds()
    schema_version: str = SEMANTIC_CORRESPONDENCE_SCHEMA_VERSION
    role_set_version: str = SEMANTIC_ROLE_SET_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping_version", _nonempty_string(self.mapping_version, "mapping_version"))
        if self.schema_version != SEMANTIC_CORRESPONDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported semantic correspondence schema {self.schema_version!r}")
        if self.role_set_version != SEMANTIC_ROLE_SET_VERSION:
            raise ValueError(f"unsupported semantic role set {self.role_set_version!r}")
        if isinstance(self.anchors, (str, bytes)):
            raise TypeError("anchors must be a sequence of SemanticAnchor values")
        anchors = tuple(self.anchors)
        if any(not isinstance(anchor, SemanticAnchor) for anchor in anchors):
            raise TypeError("anchors must contain only SemanticAnchor values")
        roles = tuple(anchor.role for anchor in anchors)
        if roles != REQUIRED_SEMANTIC_ROLES:
            raise ValueError(
                "anchors must contain the complete canonical role set in order: "
                f"{REQUIRED_SEMANTIC_ROLES}"
            )
        for label, values in (
            ("human bodies", tuple(anchor.human_body for anchor in anchors)),
            ("robot links", tuple(anchor.robot_link for anchor in anchors)),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"anchor {label} must be unique")
        if not isinstance(self.contact_protocol, RobotContactProtocol):
            raise TypeError("contact_protocol must be RobotContactProtocol")
        if not isinstance(self.thresholds, SemanticThresholds):
            raise TypeError("thresholds must be SemanticThresholds")
        object.__setattr__(self, "anchors", anchors)

    @property
    def sha256(self) -> str:
        return _canonical_hash(asdict(self))


@dataclass(frozen=True, eq=False)
class RobotFKTrajectory:
    """Exact robot FK clip, cryptographically bound to a declared ``RobotSpec``.

    Root position/orientation are intentionally absent as independent inputs.  The evaluator
    derives them from ``root_link`` in these same FK buffers, eliminating synthetic-root attacks.
    Candidate contact channels are also intentionally absent.
    """

    timestamps_s: Any
    frames: HumanFrameConvention
    robot_spec_sha256: str
    robot_asset_sha256: str
    root_link: str
    link_names: Sequence[str]
    link_positions_m: Any
    link_orientations_wxyz: Any

    def __post_init__(self) -> None:
        if not isinstance(self.frames, HumanFrameConvention):
            raise TypeError("frames must be HumanFrameConvention")
        timestamps = _immutable_array(self.timestamps_s, "timestamps_s")
        names = _names(self.link_names, "link_names")
        positions = _immutable_array(self.link_positions_m, "link_positions_m")
        orientations = _canonical_quaternions(
            self.link_orientations_wxyz, "link_orientations_wxyz"
        )
        spec_hash = _sha256(self.robot_spec_sha256, "robot_spec_sha256")
        asset_hash = _sha256(self.robot_asset_sha256, "robot_asset_sha256")
        root = _nonempty_string(self.root_link, "root_link")
        if root not in names:
            raise ValueError(f"root_link {root!r} is absent from link_names")
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError("timestamps_s must be a non-empty one-dimensional array")
        if not math.isclose(float(timestamps[0]), 0.0, rel_tol=0.0, abs_tol=_TIME_ATOL):
            raise ValueError("timestamps_s must start at zero")
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("timestamps_s must be strictly increasing")
        expected_position = (timestamps.size, len(names), 3)
        expected_orientation = (timestamps.size, len(names), 4)
        if positions.shape != expected_position:
            raise ValueError(f"link_positions_m must have shape {expected_position}, got {positions.shape}")
        if orientations.shape != expected_orientation:
            raise ValueError(
                f"link_orientations_wxyz must have shape {expected_orientation}, got {orientations.shape}"
            )
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "robot_spec_sha256", spec_hash)
        object.__setattr__(self, "robot_asset_sha256", asset_hash)
        object.__setattr__(self, "root_link", root)
        object.__setattr__(self, "link_names", names)
        object.__setattr__(self, "link_positions_m", positions)
        object.__setattr__(self, "link_orientations_wxyz", orientations)

    @property
    def root_index(self) -> int:
        return self.link_names.index(self.root_link)

    @property
    def root_position_m(self) -> np.ndarray:
        return self.link_positions_m[:, self.root_index]

    @property
    def root_orientation_wxyz(self) -> np.ndarray:
        return self.link_orientations_wxyz[:, self.root_index]

    @property
    def buffer_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"snmr.robot_fk.exact_buffers.v1\0")
        for value in (
            self.robot_spec_sha256,
            self.robot_asset_sha256,
            self.root_link,
            *self.link_names,
        ):
            encoded = value.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
        _update_exact_buffer_hash(digest, "timestamps_s", self.timestamps_s)
        _update_exact_buffer_hash(digest, "link_positions_m", self.link_positions_m)
        _update_exact_buffer_hash(
            digest, "link_orientations_wxyz", self.link_orientations_wxyz
        )
        return digest.hexdigest()


@dataclass(frozen=True)
class SemanticMetricReport:
    """Deterministic metrics plus every hash needed to reproduce their meaning."""

    metric_schema_version: str
    metric_schema_sha256: str
    human_motion_spec_sha256: str
    human_motion_buffer_sha256: str
    robot_spec_sha256: str
    robot_asset_sha256: str
    robot_fk_buffer_sha256: str
    robot_contact_protocol_sha256: str
    robot_contact_buffer_sha256: str
    correspondence_sha256: str
    thresholds_sha256: str
    frame_count: int
    duration_s: float
    robot_normalization_length_m: float
    keypoint_error_m_mean: float
    keypoint_error_m_p95: float
    keypoint_error_m_max: float
    keypoint_error_normalized_mean: float
    keypoint_error_normalized_p95: float
    keypoint_error_normalized_max: float
    per_anchor_keypoint_error_m_mean: tuple[tuple[str, float], ...]
    per_anchor_keypoint_error_normalized_mean: tuple[tuple[str, float], ...]
    end_effector_relative_position_error_m_mean: float
    end_effector_relative_position_error_normalized_mean: float
    end_effector_relative_orientation_error_rad_mean: float
    per_end_effector_relative_orientation_error_rad_mean: tuple[tuple[str, float], ...]
    contact_probability_mae: float
    contact_timing_disagreement_fraction: float
    contact_transition_timing_mae_s: float
    per_contact_probability_mae: tuple[tuple[str, float], ...]
    root_displacement_error_m_mean: float
    root_displacement_error_normalized_mean: float
    root_heading_error_rad_mean: float
    root_height_error_m_mean: float
    root_height_error_normalized_mean: float
    per_anchor_amplitude_ratio: tuple[tuple[str, float | None], ...]
    per_anchor_energy_ratio: tuple[tuple[str, float | None], ...]
    per_anchor_jitter_ratio: tuple[tuple[str, float | None], ...]
    per_anchor_motion_gate_pass: tuple[tuple[str, bool], ...]
    motion_fidelity_gate_pass: bool
    motion_fidelity_violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        for field in (
            "per_anchor_keypoint_error_m_mean",
            "per_anchor_keypoint_error_normalized_mean",
            "per_end_effector_relative_orientation_error_rad_mean",
            "per_contact_probability_mae",
            "per_anchor_amplitude_ratio",
            "per_anchor_energy_ratio",
            "per_anchor_jitter_ratio",
            "per_anchor_motion_gate_pass",
        ):
            result[field] = dict(result[field])
        return result

    @property
    def report_sha256(self) -> str:
        return _canonical_hash(self.to_dict())


_METRIC_SCHEMA_PAYLOAD = {
    "version": SEMANTIC_METRIC_SCHEMA_VERSION,
    "roles": REQUIRED_SEMANTIC_ROLES,
    "orientation_roles": ORIENTATION_ROLES,
    "position_alignment": "per-domain initial/root heading; root_xy removed; human normalized then robot-scaled",
    "semantic_pose_composition": "p_WS=p_WL+R_WL*p_LS;q_WS=q_WL*q_LS",
    "root_source": "declared RobotSpec root link in exact FK buffer",
    "robot_contact_source": ROBOT_CONTACT_SCHEMA_VERSION,
    "motion_gates": "per-anchor two-sided amplitude, velocity-energy, acceleration-energy",
}
SEMANTIC_METRIC_SCHEMA_SHA256 = _canonical_hash(_METRIC_SCHEMA_PAYLOAD)


def _torch(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(value, dtype=np.float64, order="C", copy=True))


def _semantic_poses(
    positions: np.ndarray,
    orientations: np.ndarray,
    point_offsets: Sequence[tuple[float, float, float]],
    orientation_offsets: Sequence[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    q = _torch(orientations)
    point = _torch(np.asarray(point_offsets, dtype=np.float64))[None].expand(q.shape[0], -1, -1)
    semantic_position = _torch(positions) + rot.quat_rotate(q, point)
    offset_q = _torch(np.asarray(orientation_offsets, dtype=np.float64))[None].expand_as(q)
    semantic_orientation = rot.quat_mul(q, offset_q)
    return semantic_position.numpy(), semantic_orientation.numpy()


def _heading_components(quaternions: np.ndarray, name: str) -> tuple[np.ndarray, torch.Tensor]:
    q = _torch(quaternions)
    x_axis = torch.zeros((q.shape[0], 3), dtype=torch.float64)
    x_axis[:, 0] = 1.0
    forward = rot.quat_rotate(q, x_axis)
    norm = torch.linalg.vector_norm(forward[:, :2], dim=-1)
    if torch.any(norm <= _HEADING_PROJECTION_EPS):
        frames = torch.flatnonzero(norm <= _HEADING_PROJECTION_EPS).tolist()
        raise ValueError(f"{name} has undefined +X heading at frames {frames}")
    return torch.atan2(forward[:, 1], forward[:, 0]).numpy(), heading_quat_inverse(q)


def _heading_local_positions_m(
    positions: np.ndarray,
    root_positions: np.ndarray,
    root_orientations: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    _, inverse_heading = _heading_components(root_orientations, name)
    root_xy = _torch(root_positions).clone()
    root_xy[:, 2] = 0.0
    relative = _torch(positions) - root_xy[:, None]
    inverse = inverse_heading[:, None].expand(-1, positions.shape[1], -1)
    return rot.quat_rotate(inverse, relative).numpy()


def _root_relative_positions_m(
    positions: np.ndarray, root_positions: np.ndarray, root_orientations: np.ndarray
) -> np.ndarray:
    inverse = rot.quat_conjugate(_torch(root_orientations))[:, None].expand(
        -1, positions.shape[1], -1
    )
    return rot.quat_rotate(inverse, _torch(positions) - _torch(root_positions)[:, None]).numpy()


def _root_relative_orientations(
    orientations: np.ndarray, root_orientations: np.ndarray
) -> torch.Tensor:
    inverse = rot.quat_conjugate(_torch(root_orientations))[:, None].expand(
        -1, orientations.shape[1], -1
    )
    return rot.quat_mul(inverse, _torch(orientations))


def _error_statistics(values: np.ndarray) -> tuple[float, float, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not flat.size or not np.isfinite(flat).all():
        raise ValueError("semantic errors must be finite and non-empty")
    return float(np.mean(flat)), float(np.quantile(flat, 0.95)), float(np.max(flat))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    output = np.empty_like(value)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _derive_robot_contacts(
    sole_positions_m: np.ndarray,
    timestamps_s: np.ndarray,
    protocol: RobotContactProtocol,
) -> np.ndarray:
    if timestamps_s.size == 1:
        speed = np.zeros((1, len(CONTACT_LABELS)), dtype=np.float64)
    else:
        velocity = np.gradient(sole_positions_m, timestamps_s, axis=0, edge_order=1)
        speed = np.linalg.norm(velocity, axis=-1)
    height = sole_positions_m[..., 2] - protocol.ground_height_m
    speed_score = _sigmoid(
        (protocol.speed_threshold_m_s - speed) / protocol.speed_softness_m_s
    )
    height_score = _sigmoid(
        (protocol.height_clearance_m - height) / protocol.height_softness_m
    )
    return np.clip(speed_score * height_score, 0.0, 1.0)


def _contact_buffer_hash(
    probabilities: np.ndarray, protocol: RobotContactProtocol
) -> str:
    digest = hashlib.sha256()
    digest.update(b"snmr.robot_contact.normalized_buffers.v1\0")
    digest.update(protocol.sha256.encode("ascii"))
    for label in CONTACT_LABELS:
        digest.update(label.encode("utf-8") + b"\0")
    _update_exact_buffer_hash(digest, "contact_probabilities", probabilities)
    return digest.hexdigest()


def _transition_times(
    probability: np.ndarray, timestamps: np.ndarray, threshold: float, entering: bool
) -> np.ndarray:
    active = probability >= threshold
    indices = np.flatnonzero((active[1:] != active[:-1]) & (active[1:] == entering)) + 1
    return 0.5 * (timestamps[indices - 1] + timestamps[indices])


def _transition_error(
    human: np.ndarray, robot: np.ndarray, timestamps: np.ndarray, threshold: float
) -> float:
    penalty = float(timestamps[-1] - timestamps[0])
    values = []
    for entering in (False, True):
        first = _transition_times(human, timestamps, threshold, entering)
        second = _transition_times(robot, timestamps, threshold, entering)
        matched = min(first.size, second.size)
        total = float(np.abs(first[:matched] - second[:matched]).sum())
        total += abs(first.size - second.size) * penalty
        count = max(first.size, second.size)
        values.append(total / count if count else 0.0)
    return float(np.mean(values))


def _per_anchor_amplitude(positions: np.ndarray) -> np.ndarray:
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered * centered, axis=-1), axis=0))


def _per_anchor_energy(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if timestamps.size < 2:
        return np.zeros(positions.shape[1], dtype=np.float64)
    velocity = np.diff(positions, axis=0) / np.diff(timestamps)[:, None, None]
    return np.mean(np.sum(velocity * velocity, axis=-1), axis=0)


def _per_anchor_jitter(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if timestamps.size < 3:
        return np.zeros(positions.shape[1], dtype=np.float64)
    dt = np.diff(timestamps)
    velocity = np.diff(positions, axis=0) / dt[:, None, None]
    midpoint_dt = 0.5 * (dt[1:] + dt[:-1])
    acceleration = np.diff(velocity, axis=0) / midpoint_dt[:, None, None]
    return np.mean(np.sum(acceleration * acceleration, axis=-1), axis=0)


def _ratio_gate(
    candidate: float, reference: float, lower: float, upper: float, floor: float
) -> tuple[float | None, bool]:
    if reference <= floor:
        return None, candidate <= floor
    ratio = candidate / reference
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("motion-fidelity ratio must be finite and nonnegative")
    return float(ratio), bool(lower <= ratio <= upper)


def _initial_heading_local_displacement_m(
    positions: np.ndarray, orientations: np.ndarray, *, name: str
) -> np.ndarray:
    _, inverse_heading = _heading_components(orientations, name)
    displacement = _torch(positions - positions[0])
    displacement[:, 2] = 0.0
    return rot.quat_rotate(
        inverse_heading[0].expand(positions.shape[0], -1), displacement
    ).numpy()


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _validate_robot_binding(
    robot_spec: RobotSpec,
    robot_fk: RobotFKTrajectory,
    correspondence: SemanticCorrespondence,
) -> None:
    if not isinstance(robot_spec, RobotSpec):
        raise TypeError("robot_spec must be RobotSpec")
    robot_spec.validate()
    if robot_fk.robot_spec_sha256 != robot_spec.spec_hash:
        raise ValueError("robot FK robot_spec_sha256 does not match the declared RobotSpec")
    if robot_fk.robot_asset_sha256 != robot_spec.asset_sha256:
        raise ValueError("robot FK robot_asset_sha256 does not match the declared RobotSpec")
    if robot_fk.root_link != robot_spec.semantics.root_link:
        raise ValueError("robot FK root_link must equal RobotSpec.semantics.root_link")
    semantic_links = {
        "pelvis": robot_spec.semantics.root_link,
        "torso": robot_spec.semantics.torso_link,
        "head": robot_spec.semantics.head_link,
        "left_hand": robot_spec.semantics.left_hand_link,
        "right_hand": robot_spec.semantics.right_hand_link,
        "left_foot": robot_spec.semantics.left_foot_link,
        "right_foot": robot_spec.semantics.right_foot_link,
    }
    missing = [role for role, link in semantic_links.items() if link is None]
    if missing:
        raise ValueError(f"RobotSpec is missing required semantic links for roles: {missing}")
    for anchor in correspondence.anchors:
        if anchor.robot_link != semantic_links[anchor.role]:
            raise ValueError(
                f"semantic role {anchor.role!r} must use RobotSpec link "
                f"{semantic_links[anchor.role]!r}, got {anchor.robot_link!r}"
            )
    required_contacts = {
        str(semantic_links["left_foot"]), str(semantic_links["right_foot"])
    }
    if not required_contacts.issubset(set(robot_spec.semantics.allowed_contact_links)):
        raise ValueError("RobotSpec allowed_contact_links must include both semantic foot links")


def evaluate_semantic_retargeting(
    human_motion: HumanMotionSpec,
    robot_spec: RobotSpec,
    robot_fk: RobotFKTrajectory,
    correspondence: SemanticCorrespondence,
) -> SemanticMetricReport:
    """Evaluate a candidate directly against human-side semantics, never a teacher output."""

    if not isinstance(human_motion, HumanMotionSpec):
        raise TypeError("human_motion must be HumanMotionSpec")
    if not isinstance(robot_fk, RobotFKTrajectory):
        raise TypeError("robot_fk must be RobotFKTrajectory")
    if not isinstance(correspondence, SemanticCorrespondence):
        raise TypeError("correspondence must be SemanticCorrespondence")
    human_motion.validate()
    if human_motion.contact_origin != "derived":
        raise ValueError(
            "semantic evaluation requires HumanMotionSpec contacts derived by its "
            "hash-bound FootContactProtocol"
        )
    _validate_robot_binding(robot_spec, robot_fk, correspondence)
    if robot_fk.frames != human_motion.frames:
        raise ValueError("robot and human frame/unit conventions must match exactly")
    timestamps = human_motion.timebase.target_timestamps
    if robot_fk.timestamps_s.shape != timestamps.shape or not np.allclose(
        robot_fk.timestamps_s, timestamps, rtol=0.0, atol=_TIME_ATOL
    ):
        raise ValueError("robot timestamps must match HumanMotionSpec target_timestamps")

    human_by_name = {name: index for index, name in enumerate(human_motion.body_names)}
    robot_by_name = {name: index for index, name in enumerate(robot_fk.link_names)}
    for anchor in correspondence.anchors:
        if anchor.human_body not in human_by_name:
            raise ValueError(
                f"semantic role {anchor.role!r} references unknown human body {anchor.human_body!r}"
            )
        if anchor.robot_link not in robot_by_name:
            raise ValueError(
                f"semantic role {anchor.role!r} references unknown robot link {anchor.robot_link!r}"
            )
    human_indices = [human_by_name[anchor.human_body] for anchor in correspondence.anchors]
    robot_indices = [robot_by_name[anchor.robot_link] for anchor in correspondence.anchors]
    if not np.all(human_motion.validity_mask[:, human_indices]):
        raise ValueError("all required human semantic anchors must be valid for every frame")

    human_positions, human_orientations = _semantic_poses(
        human_motion.body_positions[:, human_indices],
        human_motion.body_orientations_wxyz[:, human_indices],
        [anchor.human_point_local_m for anchor in correspondence.anchors],
        [anchor.human_orientation_offset_wxyz for anchor in correspondence.anchors],
    )
    robot_positions, robot_orientations = _semantic_poses(
        robot_fk.link_positions_m[:, robot_indices],
        robot_fk.link_orientations_wxyz[:, robot_indices],
        [anchor.robot_point_local_m for anchor in correspondence.anchors],
        [anchor.robot_orientation_offset_wxyz for anchor in correspondence.anchors],
    )
    human_scale = human_motion.segment_scales.normalization_length_m
    robot_scale = robot_spec.standing_height

    human_local_norm = _heading_local_positions_m(
        human_positions,
        human_motion.root_position,
        human_motion.root_orientation_wxyz,
        name="human root orientation",
    ) / human_scale
    robot_local_norm = _heading_local_positions_m(
        robot_positions,
        robot_fk.root_position_m,
        robot_fk.root_orientation_wxyz,
        name="robot root orientation",
    ) / robot_scale
    normalized_errors = np.linalg.norm(robot_local_norm - human_local_norm, axis=-1)
    meter_errors = normalized_errors * robot_scale
    norm_mean, norm_p95, norm_max = _error_statistics(normalized_errors)
    meter_mean, meter_p95, meter_max = _error_statistics(meter_errors)
    per_anchor_norm = tuple(
        (role, float(np.mean(normalized_errors[:, index])))
        for index, role in enumerate(REQUIRED_SEMANTIC_ROLES)
    )
    per_anchor_m = tuple(
        (role, float(np.mean(meter_errors[:, index])))
        for index, role in enumerate(REQUIRED_SEMANTIC_ROLES)
    )

    human_relative_norm = _root_relative_positions_m(
        human_positions, human_motion.root_position, human_motion.root_orientation_wxyz
    ) / human_scale
    robot_relative_norm = _root_relative_positions_m(
        robot_positions, robot_fk.root_position_m, robot_fk.root_orientation_wxyz
    ) / robot_scale
    end_indices = [REQUIRED_SEMANTIC_ROLES.index(role) for role in ORIENTATION_ROLES]
    end_position_errors = np.linalg.norm(
        robot_relative_norm[:, end_indices] - human_relative_norm[:, end_indices], axis=-1
    )
    end_position_norm_mean = float(np.mean(end_position_errors))
    end_position_m_mean = end_position_norm_mean * robot_scale

    human_relative_q = _root_relative_orientations(
        human_orientations, human_motion.root_orientation_wxyz
    )
    robot_relative_q = _root_relative_orientations(
        robot_orientations, robot_fk.root_orientation_wxyz
    )
    orientation_errors = rot.quat_geodesic_angle(
        human_relative_q[:, end_indices], robot_relative_q[:, end_indices]
    ).numpy()
    per_orientation = tuple(
        (role, float(np.mean(orientation_errors[:, index])))
        for index, role in enumerate(ORIENTATION_ROLES)
    )

    foot_indices = [
        REQUIRED_SEMANTIC_ROLES.index("left_foot"),
        REQUIRED_SEMANTIC_ROLES.index("right_foot"),
    ]
    robot_contacts = _derive_robot_contacts(
        robot_positions[:, foot_indices], timestamps, correspondence.contact_protocol
    )
    contact_error = np.abs(robot_contacts - human_motion.contacts)
    threshold = correspondence.thresholds.contact_probability_threshold
    contact_disagreement = (robot_contacts >= threshold) != (human_motion.contacts >= threshold)
    per_contact = tuple(
        (label, float(np.mean(contact_error[:, index])))
        for index, label in enumerate(CONTACT_LABELS)
    )
    transition_error = float(
        np.mean(
            [
                _transition_error(
                    human_motion.contacts[:, index], robot_contacts[:, index], timestamps, threshold
                )
                for index in range(len(CONTACT_LABELS))
            ]
        )
    )

    human_root_displacement_norm = _initial_heading_local_displacement_m(
        human_motion.root_position,
        human_motion.root_orientation_wxyz,
        name="human root orientation",
    ) / human_scale
    robot_root_displacement_norm = _initial_heading_local_displacement_m(
        robot_fk.root_position_m,
        robot_fk.root_orientation_wxyz,
        name="robot root orientation",
    ) / robot_scale
    root_displacement_norm = float(
        np.mean(
            np.linalg.norm(
                robot_root_displacement_norm - human_root_displacement_norm, axis=-1
            )
        )
    )
    human_heading, _ = _heading_components(
        human_motion.root_orientation_wxyz, "human root orientation"
    )
    robot_heading, _ = _heading_components(
        robot_fk.root_orientation_wxyz, "robot root orientation"
    )
    human_heading = np.unwrap(human_heading) - np.unwrap(human_heading)[0]
    robot_heading = np.unwrap(robot_heading) - np.unwrap(robot_heading)[0]
    root_heading_error = float(np.mean(np.abs(_wrap_angle(robot_heading - human_heading))))
    root_height_norm = float(
        np.mean(
            np.abs(
                robot_fk.root_position_m[:, 2] / robot_scale
                - human_motion.root_position[:, 2] / human_scale
            )
        )
    )

    human_amplitude = _per_anchor_amplitude(human_local_norm)
    robot_amplitude = _per_anchor_amplitude(robot_local_norm)
    human_energy = _per_anchor_energy(human_local_norm, timestamps)
    robot_energy = _per_anchor_energy(robot_local_norm, timestamps)
    human_jitter = _per_anchor_jitter(human_local_norm, timestamps)
    robot_jitter = _per_anchor_jitter(robot_local_norm, timestamps)
    thresholds = correspondence.thresholds
    amplitude_ratios: list[tuple[str, float | None]] = []
    energy_ratios: list[tuple[str, float | None]] = []
    jitter_ratios: list[tuple[str, float | None]] = []
    gate_results: list[tuple[str, bool]] = []
    violations: list[str] = []
    for index, role in enumerate(REQUIRED_SEMANTIC_ROLES):
        amplitude_ratio, amplitude_pass = _ratio_gate(
            float(robot_amplitude[index]),
            float(human_amplitude[index]),
            thresholds.amplitude_ratio_min,
            thresholds.amplitude_ratio_max,
            thresholds.motion_floor,
        )
        energy_ratio, energy_pass = _ratio_gate(
            float(robot_energy[index]),
            float(human_energy[index]),
            thresholds.energy_ratio_min,
            thresholds.energy_ratio_max,
            thresholds.motion_floor,
        )
        jitter_ratio, jitter_pass = _ratio_gate(
            float(robot_jitter[index]),
            float(human_jitter[index]),
            thresholds.jitter_ratio_min,
            thresholds.jitter_ratio_max,
            thresholds.motion_floor,
        )
        amplitude_ratios.append((role, amplitude_ratio))
        energy_ratios.append((role, energy_ratio))
        jitter_ratios.append((role, jitter_ratio))
        passed = amplitude_pass and energy_pass and jitter_pass
        gate_results.append((role, passed))
        if not amplitude_pass:
            violations.append(f"{role}:amplitude")
        if not energy_pass:
            violations.append(f"{role}:energy")
        if not jitter_pass:
            violations.append(f"{role}:jitter")

    scalar_values = (
        meter_mean,
        meter_p95,
        meter_max,
        norm_mean,
        norm_p95,
        norm_max,
        end_position_m_mean,
        end_position_norm_mean,
        float(np.mean(orientation_errors)),
        float(np.mean(contact_error)),
        float(np.mean(contact_disagreement)),
        transition_error,
        root_displacement_norm,
        root_heading_error,
        root_height_norm,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("semantic evaluation produced nonfinite metrics")

    return SemanticMetricReport(
        metric_schema_version=SEMANTIC_METRIC_SCHEMA_VERSION,
        metric_schema_sha256=SEMANTIC_METRIC_SCHEMA_SHA256,
        human_motion_spec_sha256=human_motion.spec_sha256,
        human_motion_buffer_sha256=human_motion.buffer_sha256,
        robot_spec_sha256=robot_spec.spec_hash,
        robot_asset_sha256=robot_spec.asset_sha256,
        robot_fk_buffer_sha256=robot_fk.buffer_sha256,
        robot_contact_protocol_sha256=correspondence.contact_protocol.sha256,
        robot_contact_buffer_sha256=_contact_buffer_hash(
            robot_contacts, correspondence.contact_protocol
        ),
        correspondence_sha256=correspondence.sha256,
        thresholds_sha256=thresholds.sha256,
        frame_count=int(timestamps.size),
        duration_s=float(timestamps[-1] - timestamps[0]),
        robot_normalization_length_m=robot_scale,
        keypoint_error_m_mean=meter_mean,
        keypoint_error_m_p95=meter_p95,
        keypoint_error_m_max=meter_max,
        keypoint_error_normalized_mean=norm_mean,
        keypoint_error_normalized_p95=norm_p95,
        keypoint_error_normalized_max=norm_max,
        per_anchor_keypoint_error_m_mean=per_anchor_m,
        per_anchor_keypoint_error_normalized_mean=per_anchor_norm,
        end_effector_relative_position_error_m_mean=end_position_m_mean,
        end_effector_relative_position_error_normalized_mean=end_position_norm_mean,
        end_effector_relative_orientation_error_rad_mean=float(np.mean(orientation_errors)),
        per_end_effector_relative_orientation_error_rad_mean=per_orientation,
        contact_probability_mae=float(np.mean(contact_error)),
        contact_timing_disagreement_fraction=float(np.mean(contact_disagreement)),
        contact_transition_timing_mae_s=transition_error,
        per_contact_probability_mae=per_contact,
        root_displacement_error_m_mean=root_displacement_norm * robot_scale,
        root_displacement_error_normalized_mean=root_displacement_norm,
        root_heading_error_rad_mean=root_heading_error,
        root_height_error_m_mean=root_height_norm * robot_scale,
        root_height_error_normalized_mean=root_height_norm,
        per_anchor_amplitude_ratio=tuple(amplitude_ratios),
        per_anchor_energy_ratio=tuple(energy_ratios),
        per_anchor_jitter_ratio=tuple(jitter_ratios),
        per_anchor_motion_gate_pass=tuple(gate_results),
        motion_fidelity_gate_pass=not violations,
        motion_fidelity_violations=tuple(violations),
    )


__all__ = [
    "ORIENTATION_ROLES",
    "REQUIRED_SEMANTIC_ROLES",
    "ROBOT_CONTACT_SCHEMA_VERSION",
    "SEMANTIC_CORRESPONDENCE_SCHEMA_VERSION",
    "SEMANTIC_METRIC_SCHEMA_SHA256",
    "SEMANTIC_METRIC_SCHEMA_VERSION",
    "SEMANTIC_ROLE_SET_VERSION",
    "RobotContactProtocol",
    "RobotFKTrajectory",
    "SemanticAnchor",
    "SemanticCorrespondence",
    "SemanticMetricReport",
    "SemanticThresholds",
    "evaluate_semantic_retargeting",
]
