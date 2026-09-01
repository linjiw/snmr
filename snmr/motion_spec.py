"""Canonical human-motion contract for learning-based retargeting.

The contract is intentionally small and strict.  A :class:`HumanMotionSpec` is a
resampled, self-contained clip in the package-wide convention: Z-up, forward +X,
pelvis-rooted, meters, radians, and scalar-first (``wxyz``) quaternions.  It keeps the
source timebase alongside a strictly uniform 50 Hz target grid, truncated to complete
ticks, so an exported training tensor never relies on an implicit FPS assumption.

All array fields are copied into immutable, C-contiguous buffers.  The integrity digest
is computed over labelled, shape-delimited buffers after normalising floating-point
values to little-endian float32 and masks to uint8.  Consequently, equivalent values
have the same digest regardless of the input NumPy dtype or memory layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
import re
import struct
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


SCHEMA_VERSION = "snmr.human_motion.v0.1"
TARGET_FPS = 50.0
CONTACT_LABELS = ("left_foot", "right_foot")

POSITION_INTERPOLATION = "cubic_spline"
QUATERNION_INTERPOLATION = "slerp"
SHORT_SEQUENCE_POSITION_FALLBACK = "linear"
CONTACT_INTERPOLATION = "linear"
VALIDITY_INTERPOLATION = "fail_closed_all_valid"

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_QUATERNION_NORM_ATOL = 1e-5
_TIME_ATOL = 1e-10


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


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, whitespace-trimmed string")
    return value


def _numeric_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=np.float64, order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _immutable_array(array: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    """Return an array backed by immutable bytes, not merely a write-protected owner."""

    contiguous = np.ascontiguousarray(array, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


def _immutable_float_array(value: Any, name: str) -> np.ndarray:
    return _immutable_array(_numeric_array(value, name), np.dtype(np.float64))


def _immutable_bool_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise TypeError(f"{name} must have boolean dtype")
    return _immutable_array(raw, np.dtype(np.bool_))


def _canonical_quaternions(value: Any, name: str) -> np.ndarray:
    """Validate unit quaternions and choose a deterministic double-cover sign."""

    quaternions = _numeric_array(value, name)
    if quaternions.ndim < 2 or quaternions.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (..., 4), got {quaternions.shape}")
    norms = np.linalg.norm(quaternions, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_NORM_ATOL):
        worst = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(f"{name} must contain unit quaternions (max norm error {worst:g})")

    # q and -q encode the same rotation.  Make the first non-negligible component
    # positive so hashes and JSON are stable across equivalent source conventions.
    result = np.array(quaternions, dtype=np.float64, order="C", copy=True)
    flat = result.reshape(-1, 4)
    nonzero = np.abs(flat) > np.finfo(np.float64).eps
    first = np.argmax(nonzero, axis=1)
    first_value = flat[np.arange(flat.shape[0]), first]
    flat[first_value < 0.0] *= -1.0
    return _immutable_array(result, np.dtype(np.float64))


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"{name} fields do not match the schema ({', '.join(details)})")


@dataclass(frozen=True)
class MotionSource:
    """Stable identity of the unprocessed source clip."""

    dataset: str
    sequence_id: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _nonempty_string(self.dataset, "dataset"))
        object.__setattr__(
            self, "sequence_id", _nonempty_string(self.sequence_id, "sequence_id")
        )
        if not isinstance(self.sha256, str) or _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a 64-character hexadecimal SHA-256 digest")
        object.__setattr__(self, "sha256", self.sha256.lower())


@dataclass(frozen=True)
class MotionProvenance:
    """Auditable identity and preprocessing history for one source clip.

    ``clip_range`` is a half-open source-frame range ``[start, stop)``.  Its
    length must match the source timeline bound to the containing motion spec.
    Transformations are ordered because preprocessing order is semantically
    meaningful.
    """

    motion_id: str
    source_subject: str | None
    clip_range: tuple[int, int]
    split_id: str
    transformations: tuple[str, ...]
    preprocessing_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "motion_id", _nonempty_string(self.motion_id, "motion_id"))
        if self.source_subject is not None:
            object.__setattr__(
                self,
                "source_subject",
                _nonempty_string(self.source_subject, "source_subject"),
            )
        if isinstance(self.clip_range, (str, bytes)):
            raise TypeError("clip_range must be a two-integer sequence")
        try:
            parsed_range = tuple(self.clip_range)
        except TypeError as exc:
            raise TypeError("clip_range must be a two-integer sequence") from exc
        if len(parsed_range) != 2:
            raise ValueError("clip_range must contain exactly (start, stop)")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               for value in parsed_range):
            raise TypeError("clip_range entries must be integers")
        start, stop = (int(value) for value in parsed_range)
        if start < 0 or stop <= start:
            raise ValueError("clip_range must satisfy 0 <= start < stop")
        object.__setattr__(self, "clip_range", (start, stop))
        object.__setattr__(self, "split_id", _nonempty_string(self.split_id, "split_id"))
        if isinstance(self.transformations, (str, bytes)):
            raise TypeError("transformations must be an ordered sequence of strings")
        try:
            transformations = tuple(self.transformations)
        except TypeError as exc:
            raise TypeError("transformations must be an ordered sequence of strings") from exc
        for index, transformation in enumerate(transformations):
            _nonempty_string(transformation, f"transformations[{index}]")
        object.__setattr__(self, "transformations", transformations)
        object.__setattr__(
            self,
            "preprocessing_version",
            _nonempty_string(self.preprocessing_version, "preprocessing_version"),
        )

    @property
    def source_frame_count(self) -> int:
        return self.clip_range[1] - self.clip_range[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_id": self.motion_id,
            "source_subject": self.source_subject,
            "clip_range": list(self.clip_range),
            "split_id": self.split_id,
            "transformations": list(self.transformations),
            "preprocessing_version": self.preprocessing_version,
        }


@dataclass(frozen=True)
class HumanFrameConvention:
    """The only frame/unit convention accepted by this schema version."""

    world_up_axis: str = "z"
    world_forward_axis: str = "+x"
    root_reference: str = "pelvis_origin"
    root_height_convention: str = "absolute_world_z"
    length_unit: str = "meter"
    angle_unit: str = "radian"
    quaternion_convention: str = "wxyz"

    def __post_init__(self) -> None:
        expected = {
            "world_up_axis": "z",
            "world_forward_axis": "+x",
            "root_reference": "pelvis_origin",
            "root_height_convention": "absolute_world_z",
            "length_unit": "meter",
            "angle_unit": "radian",
            "quaternion_convention": "wxyz",
        }
        for name, canonical in expected.items():
            if getattr(self, name) != canonical:
                raise ValueError(f"{name} must be the canonical value {canonical!r}")


@dataclass(frozen=True)
class BodySegmentScales:
    """Dimensionless segment lengths normalised by one declared body-height scale."""

    normalization_length_m: float
    torso: float
    thigh: float
    shin: float
    upper_arm: float
    forearm: float

    def __post_init__(self) -> None:
        for name in (
            "normalization_length_m",
            "torso",
            "thigh",
            "shin",
            "upper_arm",
            "forearm",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, positive=True),
            )

    @classmethod
    def from_lengths(
        cls,
        *,
        normalization_length_m: float,
        torso_m: float,
        thigh_m: float,
        shin_m: float,
        upper_arm_m: float,
        forearm_m: float,
    ) -> "BodySegmentScales":
        """Convert measured segment lengths in meters into dimensionless descriptors."""

        scale = _finite_float(
            normalization_length_m, "normalization_length_m", positive=True
        )
        lengths = {
            "torso": torso_m,
            "thigh": thigh_m,
            "shin": shin_m,
            "upper_arm": upper_arm_m,
            "forearm": forearm_m,
        }
        normalised = {
            name: _finite_float(value, f"{name}_m", positive=True) / scale
            for name, value in lengths.items()
        }
        return cls(normalization_length_m=scale, **normalised)

    def as_buffer(self) -> np.ndarray:
        return np.asarray(
            (self.torso, self.thigh, self.shin, self.upper_arm, self.forearm),
            dtype=np.float64,
        )

    @classmethod
    def from_landmarks(
        cls,
        *,
        body_positions: Any,
        body_names: Sequence[str],
        landmark_pairs: "BilateralSegmentLandmarks",
        normalization_length_m: float,
        validity_mask: Any | None = None,
    ) -> "BodySegmentScales":
        """Extract robust bilateral segment lengths from a motion tensor."""

        return extract_body_segment_scales(
            body_positions=body_positions,
            body_names=body_names,
            landmark_pairs=landmark_pairs,
            normalization_length_m=normalization_length_m,
            validity_mask=validity_mask,
        )


def _bilateral_pairs(value: Any, name: str) -> tuple[tuple[str, str], tuple[str, str]]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must contain exactly two landmark pairs")
    try:
        sides = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain exactly two landmark pairs") from exc
    if len(sides) != 2:
        raise ValueError(f"{name} must contain exactly two landmark pairs")
    parsed: list[tuple[str, str]] = []
    for side_index, pair in enumerate(sides):
        if isinstance(pair, (str, bytes)):
            raise TypeError(f"{name}[{side_index}] must be a two-name landmark pair")
        try:
            endpoints = tuple(pair)
        except TypeError as exc:
            raise TypeError(f"{name}[{side_index}] must be a two-name landmark pair") from exc
        if len(endpoints) != 2:
            raise ValueError(f"{name}[{side_index}] must contain exactly two names")
        first = _nonempty_string(endpoints[0], f"{name}[{side_index}][0]")
        second = _nonempty_string(endpoints[1], f"{name}[{side_index}][1]")
        if first == second:
            raise ValueError(f"{name}[{side_index}] endpoints must be distinct")
        parsed.append((first, second))
    return (parsed[0], parsed[1])


@dataclass(frozen=True)
class BilateralSegmentLandmarks:
    """Left/right landmark endpoint pairs used for each scale descriptor."""

    torso: tuple[tuple[str, str], tuple[str, str]]
    thigh: tuple[tuple[str, str], tuple[str, str]]
    shin: tuple[tuple[str, str], tuple[str, str]]
    upper_arm: tuple[tuple[str, str], tuple[str, str]]
    forearm: tuple[tuple[str, str], tuple[str, str]]

    def __post_init__(self) -> None:
        for name in ("torso", "thigh", "shin", "upper_arm", "forearm"):
            object.__setattr__(self, name, _bilateral_pairs(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            name: [list(pair) for pair in getattr(self, name)]
            for name in ("torso", "thigh", "shin", "upper_arm", "forearm")
        }


def extract_body_segment_scales(
    *,
    body_positions: Any,
    body_names: Sequence[str],
    landmark_pairs: BilateralSegmentLandmarks,
    normalization_length_m: float,
    validity_mask: Any | None = None,
) -> BodySegmentScales:
    """Extract dimensionless median bilateral lengths from declared landmarks.

    The input is ``(T, B, 3)`` in meters.  For each left/right pair, the median
    valid length over time is computed; the descriptor is the mean of the two
    side medians divided by ``normalization_length_m``.  This prevents a side
    with more valid frames from dominating the bilateral descriptor.
    """

    if not isinstance(landmark_pairs, BilateralSegmentLandmarks):
        raise TypeError("landmark_pairs must be BilateralSegmentLandmarks")
    if isinstance(body_names, (str, bytes)):
        raise TypeError("body_names must be a sequence of names")
    names = tuple(body_names)
    if not names:
        raise ValueError("body_names must not be empty")
    for index, name in enumerate(names):
        _nonempty_string(name, f"body_names[{index}]")
    if len(set(names)) != len(names):
        raise ValueError("body_names must be unique")

    positions = _numeric_array(body_positions, "body_positions")
    if positions.ndim != 3 or positions.shape != (positions.shape[0], len(names), 3):
        raise ValueError(
            f"body_positions must have shape (T, {len(names)}, 3), got {positions.shape}"
        )
    if positions.shape[0] == 0:
        raise ValueError("body_positions must contain at least one frame")
    if validity_mask is None:
        valid = np.ones(positions.shape[:2], dtype=np.bool_)
    else:
        valid = np.asarray(_immutable_bool_array(validity_mask, "validity_mask"))
        if valid.shape != positions.shape[:2]:
            raise ValueError(
                f"validity_mask must have shape {positions.shape[:2]}, got {valid.shape}"
            )

    index_by_name = {name: index for index, name in enumerate(names)}
    extracted_m: dict[str, float] = {}
    for segment in ("torso", "thigh", "shin", "upper_arm", "forearm"):
        side_medians: list[float] = []
        for first_name, second_name in getattr(landmark_pairs, segment):
            unknown = [
                name for name in (first_name, second_name) if name not in index_by_name
            ]
            if unknown:
                raise ValueError(f"{segment} landmark pair references unknown names: {unknown}")
            first = index_by_name[first_name]
            second = index_by_name[second_name]
            pair_valid = valid[:, first] & valid[:, second]
            if not np.any(pair_valid):
                raise ValueError(
                    f"{segment} landmark pair {(first_name, second_name)!r} has no valid frames"
                )
            lengths = np.linalg.norm(
                positions[pair_valid, first] - positions[pair_valid, second], axis=-1
            )
            median = float(np.median(lengths))
            if not math.isfinite(median) or median <= 0.0:
                raise ValueError(
                    f"{segment} landmark pair {(first_name, second_name)!r} has zero length"
                )
            side_medians.append(median)
        extracted_m[f"{segment}_m"] = float(np.mean(side_medians))

    return BodySegmentScales.from_lengths(
        normalization_length_m=normalization_length_m,
        **extracted_m,
    )


@dataclass(frozen=True)
class FootContactProtocol:
    """Declared probabilistic contact model for left/right foot landmarks."""

    left_foot_body: str = "left_foot"
    right_foot_body: str = "right_foot"
    speed_threshold_m_s: float = 0.20
    height_clearance_m: float = 0.05
    speed_softness_m_s: float = 0.05
    height_softness_m: float = 0.01
    ground_height_m: float = 0.0
    method: str = "sigmoid_speed_height_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_foot_body",
            _nonempty_string(self.left_foot_body, "left_foot_body"),
        )
        object.__setattr__(
            self,
            "right_foot_body",
            _nonempty_string(self.right_foot_body, "right_foot_body"),
        )
        if self.left_foot_body == self.right_foot_body:
            raise ValueError("left_foot_body and right_foot_body must be distinct")
        for name in ("speed_threshold_m_s", "height_clearance_m"):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        for name in ("speed_softness_m_s", "height_softness_m"):
            object.__setattr__(
                self, name, _finite_float(getattr(self, name), name, positive=True)
            )
        object.__setattr__(
            self,
            "ground_height_m",
            _finite_float(self.ground_height_m, "ground_height_m"),
        )
        if self.method != "sigmoid_speed_height_v1":
            raise ValueError("method must be 'sigmoid_speed_height_v1'")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def derive_foot_contact_probabilities(
    *,
    body_positions: Any,
    body_names: Sequence[str],
    timestamps: Any,
    protocol: FootContactProtocol | None = None,
    validity_mask: Any | None = None,
) -> np.ndarray:
    """Derive left/right contact probabilities from world speed and sole height.

    The returned columns follow :data:`CONTACT_LABELS`.  Each contact score is
    the product of a soft speed-threshold score and a soft height-clearance
    score, making the output continuous and bounded in ``[0, 1]``.
    """

    parsed_protocol = FootContactProtocol() if protocol is None else protocol
    if not isinstance(parsed_protocol, FootContactProtocol):
        raise TypeError("protocol must be FootContactProtocol")
    if isinstance(body_names, (str, bytes)):
        raise TypeError("body_names must be a sequence of names")
    names = tuple(body_names)
    if len(set(names)) != len(names):
        raise ValueError("body_names must be unique")
    for index, name in enumerate(names):
        _nonempty_string(name, f"body_names[{index}]")
    positions = _numeric_array(body_positions, "body_positions")
    if positions.ndim != 3 or positions.shape != (positions.shape[0], len(names), 3):
        raise ValueError(
            f"body_positions must have shape (T, {len(names)}, 3), got {positions.shape}"
        )
    if positions.shape[0] == 0:
        raise ValueError("body_positions must contain at least one frame")
    time = _numeric_array(timestamps, "timestamps")
    if time.shape != (positions.shape[0],):
        raise ValueError(f"timestamps must have shape {(positions.shape[0],)}, got {time.shape}")
    if not math.isclose(float(time[0]), 0.0, rel_tol=0.0, abs_tol=_TIME_ATOL):
        raise ValueError("timestamps must start at zero")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    if validity_mask is None:
        valid = np.ones(positions.shape[:2], dtype=np.bool_)
    else:
        valid = np.asarray(_immutable_bool_array(validity_mask, "validity_mask"))
        if valid.shape != positions.shape[:2]:
            raise ValueError(
                f"validity_mask must have shape {positions.shape[:2]}, got {valid.shape}"
            )

    index_by_name = {name: index for index, name in enumerate(names)}
    feet: list[np.ndarray] = []
    for name in (parsed_protocol.left_foot_body, parsed_protocol.right_foot_body):
        if name not in index_by_name:
            raise ValueError(f"contact protocol references unknown foot body {name!r}")
        foot_index = index_by_name[name]
        if not np.all(valid[:, foot_index]):
            invalid_frames = np.flatnonzero(~valid[:, foot_index]).tolist()
            raise ValueError(
                f"cannot derive contacts: foot body {name!r} is invalid at frames "
                f"{invalid_frames}"
            )
        feet.append(positions[:, foot_index])
    foot_positions = np.stack(feet, axis=1)
    if time.size == 1:
        speed = np.zeros((1, len(CONTACT_LABELS)), dtype=np.float64)
    else:
        velocity = np.gradient(foot_positions, time, axis=0, edge_order=1)
        speed = np.linalg.norm(velocity, axis=-1)
    height = foot_positions[..., 2] - parsed_protocol.ground_height_m
    speed_score = _sigmoid(
        (parsed_protocol.speed_threshold_m_s - speed)
        / parsed_protocol.speed_softness_m_s
    )
    height_score = _sigmoid(
        (parsed_protocol.height_clearance_m - height)
        / parsed_protocol.height_softness_m
    )
    return np.clip(speed_score * height_score, 0.0, 1.0)


@dataclass(frozen=True, eq=False)
class MotionTimebase:
    """Source sampling and the explicit, strictly uniform 50 Hz target timeline.

    Target timestamps contain only complete 20 ms ticks within the source duration.
    A non-grid-aligned source tail is deliberately retained only in
    ``source_timestamps`` and is never represented by a shorter target interval.
    """

    source_fps: float
    source_timestamps: np.ndarray
    target_fps: float
    target_timestamps: np.ndarray
    position_interpolation: str
    position_interpolation_applied: str
    quaternion_interpolation: str
    contact_interpolation: str = CONTACT_INTERPOLATION
    validity_interpolation: str = VALIDITY_INTERPOLATION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_fps", _finite_float(self.source_fps, "source_fps", positive=True)
        )
        object.__setattr__(
            self, "target_fps", _finite_float(self.target_fps, "target_fps", positive=True)
        )
        object.__setattr__(
            self,
            "source_timestamps",
            _immutable_float_array(self.source_timestamps, "source_timestamps"),
        )
        object.__setattr__(
            self,
            "target_timestamps",
            _immutable_float_array(self.target_timestamps, "target_timestamps"),
        )
        self.validate()

    def validate(self) -> None:
        if not math.isclose(self.target_fps, TARGET_FPS, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"target_fps must be exactly {TARGET_FPS:g} Hz")
        for name, values in (
            ("source_timestamps", self.source_timestamps),
            ("target_timestamps", self.target_timestamps),
        ):
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"{name} must be a non-empty one-dimensional array")
            if not math.isclose(float(values[0]), 0.0, rel_tol=0.0, abs_tol=_TIME_ATOL):
                raise ValueError(f"{name} must start at zero")
            if values.size > 1 and np.any(np.diff(values) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing")

        if self.source_timestamps.size > 1:
            source_step = 1.0 / self.source_fps
            if not np.allclose(
                np.diff(self.source_timestamps), source_step, rtol=0.0, atol=_TIME_ATOL
            ):
                raise ValueError("source_timestamps are inconsistent with source_fps")
        if self.target_timestamps.size > 1:
            target_steps = np.diff(self.target_timestamps)
            nominal_step = 1.0 / TARGET_FPS
            if not np.allclose(target_steps, nominal_step, rtol=0.0, atol=_TIME_ATOL):
                raise ValueError("target_timestamps must be a strictly uniform 50 Hz grid")
        source_endpoint = float(self.source_timestamps[-1])
        if float(self.target_timestamps[-1]) > source_endpoint:
            raise ValueError("target_timestamps may not extrapolate beyond the source duration")
        expected_target = _target_timestamps(source_endpoint, TARGET_FPS)
        if not np.array_equal(self.target_timestamps, expected_target):
            raise ValueError(
                "target_timestamps must contain every complete 50 Hz tick without "
                "exceeding the source duration"
            )

        if self.position_interpolation != POSITION_INTERPOLATION:
            raise ValueError(
                f"position_interpolation must be {POSITION_INTERPOLATION!r}"
            )
        if self.position_interpolation_applied not in {
            POSITION_INTERPOLATION,
            SHORT_SEQUENCE_POSITION_FALLBACK,
        }:
            raise ValueError("unsupported position_interpolation_applied")
        if (
            self.position_interpolation_applied == POSITION_INTERPOLATION
            and self.source_timestamps.size < 4
        ):
            raise ValueError("cubic_spline requires at least four source samples")
        if (
            self.position_interpolation_applied == SHORT_SEQUENCE_POSITION_FALLBACK
            and self.source_timestamps.size >= 4
        ):
            raise ValueError("linear fallback is reserved for fewer than four source samples")
        if self.quaternion_interpolation != QUATERNION_INTERPOLATION:
            raise ValueError(
                f"quaternion_interpolation must be {QUATERNION_INTERPOLATION!r}"
            )
        if self.contact_interpolation != CONTACT_INTERPOLATION:
            raise ValueError(f"contact_interpolation must be {CONTACT_INTERPOLATION!r}")
        if self.validity_interpolation != VALIDITY_INTERPOLATION:
            raise ValueError(f"validity_interpolation must be {VALIDITY_INTERPOLATION!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_fps": self.source_fps,
            "source_timestamps": self.source_timestamps.tolist(),
            "target_fps": self.target_fps,
            "target_timestamps": self.target_timestamps.tolist(),
            "position_interpolation": self.position_interpolation,
            "position_interpolation_applied": self.position_interpolation_applied,
            "quaternion_interpolation": self.quaternion_interpolation,
            "contact_interpolation": self.contact_interpolation,
            "validity_interpolation": self.validity_interpolation,
        }


def _target_timestamps(duration: float, fps: float = TARGET_FPS) -> np.ndarray:
    if duration < 0.0 or not math.isfinite(duration):
        raise ValueError("duration must be finite and nonnegative")
    if duration == 0.0:
        return np.asarray([0.0], dtype=np.float64)
    step = 1.0 / fps
    whole_steps = int(math.floor(duration * fps))
    return np.arange(whole_steps + 1, dtype=np.float64) * step


def _interpolate_numeric(
    values: np.ndarray,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    *,
    cubic: bool,
) -> np.ndarray:
    if values.shape[0] == 1:
        return np.repeat(values, target_timestamps.size, axis=0)
    if cubic:
        output = np.asarray(
            CubicSpline(source_timestamps, values, axis=0)(target_timestamps),
            dtype=np.float64,
        )
    else:
        flattened = values.reshape(values.shape[0], -1)
        interpolated = np.empty((target_timestamps.size, flattened.shape[1]), dtype=np.float64)
        for column in range(flattened.shape[1]):
            interpolated[:, column] = np.interp(
                target_timestamps, source_timestamps, flattened[:, column]
            )
        output = interpolated.reshape((target_timestamps.size,) + values.shape[1:])
    output[0] = values[0]
    if math.isclose(
        float(target_timestamps[-1]),
        float(source_timestamps[-1]),
        rel_tol=0.0,
        abs_tol=_TIME_ATOL,
    ):
        output[-1] = values[-1]
    return output


def _interpolate_quaternions(
    quaternions: np.ndarray,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    if quaternions.shape[0] == 1:
        return np.repeat(quaternions, target_timestamps.size, axis=0)
    flattened = quaternions.reshape(quaternions.shape[0], -1, 4)
    output = np.empty((target_timestamps.size, flattened.shape[1], 4), dtype=np.float64)
    for stream in range(flattened.shape[1]):
        xyzw = flattened[:, stream, (1, 2, 3, 0)]
        rotations = Rotation.from_quat(xyzw)
        resampled_xyzw = Slerp(source_timestamps, rotations)(target_timestamps).as_quat()
        output[:, stream] = resampled_xyzw[:, (3, 0, 1, 2)]
    norms = np.linalg.norm(output, axis=-1, keepdims=True)
    output /= norms
    output[0] = flattened[0]
    if math.isclose(
        float(target_timestamps[-1]),
        float(source_timestamps[-1]),
        rel_tol=0.0,
        abs_tol=_TIME_ATOL,
    ):
        output[-1] = flattened[-1]
    return output.reshape((target_timestamps.size,) + quaternions.shape[1:])


def _update_normalized_buffer(
    digest: Any,
    name: str,
    value: np.ndarray,
    *,
    boolean: bool = False,
) -> None:
    label = name.encode("utf-8")
    digest.update(struct.pack("<I", len(label)))
    digest.update(label)
    digest.update(struct.pack("<I", value.ndim))
    digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
    if boolean:
        normalized = np.ascontiguousarray(value, dtype=np.uint8)
        digest.update(b"u1")
    else:
        absolute = np.abs(value)
        float32_max = float(np.finfo(np.float32).max)
        if np.any(absolute > float32_max):
            raise ValueError(
                f"{name} contains values outside the finite canonical float32 range"
            )
        normalized = np.ascontiguousarray(value, dtype=np.dtype("<f4"))
        if not np.isfinite(normalized).all():
            # The range check above should make this unreachable, but fail closed if a
            # platform conversion behaves differently.
            raise ValueError(f"{name} cannot be represented by canonical finite float32 values")
        if np.any((value != 0.0) & (normalized == 0.0)):
            raise ValueError(
                f"{name} contains nonzero values below the canonical float32 range"
            )
        # Treat signed zero as one canonical scalar.
        normalized[normalized == 0.0] = 0.0
        digest.update(b"f4")
    digest.update(normalized.tobytes(order="C"))


@dataclass(frozen=True, eq=False)
class HumanMotionSpec:
    """Immutable, validated human motion sampled on an explicit 50 Hz timeline."""

    schema_version: str
    source: MotionSource
    provenance: MotionProvenance
    timebase: MotionTimebase
    frames: HumanFrameConvention
    body_names: tuple[str, ...]
    segment_scales: BodySegmentScales
    contact_protocol: FootContactProtocol
    contact_origin: str
    root_position: np.ndarray
    root_orientation_wxyz: np.ndarray
    body_positions: np.ndarray
    body_orientations_wxyz: np.ndarray
    contacts: np.ndarray
    validity_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source, MotionSource):
            raise TypeError("source must be a MotionSource")
        if not isinstance(self.provenance, MotionProvenance):
            raise TypeError("provenance must be MotionProvenance")
        if not isinstance(self.timebase, MotionTimebase):
            raise TypeError("timebase must be a MotionTimebase")
        if not isinstance(self.frames, HumanFrameConvention):
            raise TypeError("frames must be a HumanFrameConvention")
        if not isinstance(self.segment_scales, BodySegmentScales):
            raise TypeError("segment_scales must be BodySegmentScales")
        if not isinstance(self.contact_protocol, FootContactProtocol):
            raise TypeError("contact_protocol must be FootContactProtocol")
        if self.contact_origin not in {"derived", "provided"}:
            raise ValueError("contact_origin must be 'derived' or 'provided'")

        names = tuple(self.body_names)
        for index, name in enumerate(names):
            _nonempty_string(name, f"body_names[{index}]")
        object.__setattr__(self, "body_names", names)
        object.__setattr__(
            self, "root_position", _immutable_float_array(self.root_position, "root_position")
        )
        object.__setattr__(
            self,
            "root_orientation_wxyz",
            _canonical_quaternions(self.root_orientation_wxyz, "root_orientation_wxyz"),
        )
        object.__setattr__(
            self, "body_positions", _immutable_float_array(self.body_positions, "body_positions")
        )
        object.__setattr__(
            self,
            "body_orientations_wxyz",
            _canonical_quaternions(self.body_orientations_wxyz, "body_orientations_wxyz"),
        )
        object.__setattr__(
            self, "contacts", _immutable_float_array(self.contacts, "contacts")
        )
        object.__setattr__(
            self, "validity_mask", _immutable_bool_array(self.validity_mask, "validity_mask")
        )
        self.validate()

    @classmethod
    def from_source(
        cls,
        *,
        source: MotionSource,
        provenance: MotionProvenance,
        source_fps: float,
        body_names: Sequence[str],
        segment_scales: BodySegmentScales,
        root_position: Any,
        root_orientation_wxyz: Any,
        body_positions: Any,
        body_orientations_wxyz: Any,
        validity_mask: Any,
        contacts: Any | None = None,
        contact_protocol: FootContactProtocol | None = None,
        frames: HumanFrameConvention | None = None,
        target_fps: float = TARGET_FPS,
    ) -> "HumanMotionSpec":
        """Validate source samples and resample them to the canonical target timeline.

        Positions use SciPy's cubic spline with a declared linear fallback for fewer
        than four source frames.  Quaternions use ``Rotation``/``Slerp``.  Contact
        probabilities are interpolated linearly.  Canonical resampling fails closed
        unless every source body sample is valid; the emitted validity mask is therefore
        all true rather than an interpolation of missing-data state.
        """

        parsed_source_fps = _finite_float(source_fps, "source_fps", positive=True)
        parsed_target_fps = _finite_float(target_fps, "target_fps", positive=True)
        if not math.isclose(parsed_target_fps, TARGET_FPS, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"target_fps must be exactly {TARGET_FPS:g} Hz")
        if not isinstance(source, MotionSource):
            raise TypeError("source must be a MotionSource")
        if not isinstance(provenance, MotionProvenance):
            raise TypeError("provenance must be MotionProvenance")
        if not isinstance(segment_scales, BodySegmentScales):
            raise TypeError("segment_scales must be BodySegmentScales")
        parsed_frames = HumanFrameConvention() if frames is None else frames
        if not isinstance(parsed_frames, HumanFrameConvention):
            raise TypeError("frames must be a HumanFrameConvention")

        names = tuple(body_names)
        if not names:
            raise ValueError("body_names must not be empty")
        for index, name in enumerate(names):
            _nonempty_string(name, f"body_names[{index}]")
        if len(set(names)) != len(names):
            raise ValueError("body_names must be unique")

        source_root_position = _numeric_array(root_position, "root_position")
        source_root_orientation = np.asarray(
            _canonical_quaternions(root_orientation_wxyz, "root_orientation_wxyz")
        )
        source_body_positions = _numeric_array(body_positions, "body_positions")
        source_body_orientations = np.asarray(
            _canonical_quaternions(body_orientations_wxyz, "body_orientations_wxyz")
        )
        source_validity = np.asarray(_immutable_bool_array(validity_mask, "validity_mask"))

        source_frames = source_root_position.shape[0] if source_root_position.ndim >= 1 else 0
        bodies = len(names)
        expected_shapes = {
            "root_position": (source_frames, 3),
            "root_orientation_wxyz": (source_frames, 4),
            "body_positions": (source_frames, bodies, 3),
            "body_orientations_wxyz": (source_frames, bodies, 4),
            "validity_mask": (source_frames, bodies),
        }
        values = {
            "root_position": source_root_position,
            "root_orientation_wxyz": source_root_orientation,
            "body_positions": source_body_positions,
            "body_orientations_wxyz": source_body_orientations,
            "validity_mask": source_validity,
        }
        if source_frames == 0:
            raise ValueError("source motion must contain at least one frame")
        for name, expected in expected_shapes.items():
            if values[name].shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {values[name].shape}")
        if not np.all(source_validity):
            invalid = np.argwhere(~source_validity)
            preview = [tuple(int(index) for index in row) for row in invalid[:8]]
            suffix = "" if invalid.shape[0] <= 8 else f" (+{invalid.shape[0] - 8} more)"
            raise ValueError(
                "canonical resampling requires every source body sample to be valid; "
                f"invalid (frame, body) entries: {preview}{suffix}"
            )
        if provenance.source_frame_count != source_frames:
            raise ValueError(
                "provenance clip_range length must match the number of source frames"
            )

        parsed_contact_protocol = (
            FootContactProtocol() if contact_protocol is None else contact_protocol
        )
        if not isinstance(parsed_contact_protocol, FootContactProtocol):
            raise TypeError("contact_protocol must be FootContactProtocol")
        source_timestamps = np.arange(source_frames, dtype=np.float64) / parsed_source_fps
        if contacts is None:
            source_contacts = derive_foot_contact_probabilities(
                body_positions=source_body_positions,
                body_names=names,
                timestamps=source_timestamps,
                protocol=parsed_contact_protocol,
                validity_mask=source_validity,
            )
            contact_origin = "derived"
        else:
            source_contacts = _numeric_array(contacts, "contacts")
            expected_contact_shape = (source_frames, len(CONTACT_LABELS))
            if source_contacts.shape != expected_contact_shape:
                raise ValueError(
                    f"contacts must have shape {expected_contact_shape}, "
                    f"got {source_contacts.shape}"
                )
            contact_origin = "provided"
        if np.any((source_contacts < 0.0) | (source_contacts > 1.0)):
            raise ValueError("contacts must lie in the closed interval [0, 1]")

        target_timestamps = _target_timestamps(float(source_timestamps[-1]), parsed_target_fps)
        use_cubic = source_frames >= 4
        applied = (
            POSITION_INTERPOLATION if use_cubic else SHORT_SEQUENCE_POSITION_FALLBACK
        )
        timebase = MotionTimebase(
            source_fps=parsed_source_fps,
            source_timestamps=source_timestamps,
            target_fps=parsed_target_fps,
            target_timestamps=target_timestamps,
            position_interpolation=POSITION_INTERPOLATION,
            position_interpolation_applied=applied,
            quaternion_interpolation=QUATERNION_INTERPOLATION,
        )

        target_root_position = _interpolate_numeric(
            source_root_position,
            source_timestamps,
            target_timestamps,
            cubic=use_cubic,
        )
        target_body_positions = _interpolate_numeric(
            source_body_positions,
            source_timestamps,
            target_timestamps,
            cubic=use_cubic,
        )
        target_root_orientation = _interpolate_quaternions(
            source_root_orientation,
            source_timestamps,
            target_timestamps,
        )
        target_body_orientations = _interpolate_quaternions(
            source_body_orientations,
            source_timestamps,
            target_timestamps,
        )
        target_contacts = np.clip(
            _interpolate_numeric(
                source_contacts,
                source_timestamps,
                target_timestamps,
                cubic=False,
            ),
            0.0,
            1.0,
        )
        target_validity = np.ones((target_timestamps.size, bodies), dtype=np.bool_)

        return cls(
            schema_version=SCHEMA_VERSION,
            source=source,
            provenance=provenance,
            timebase=timebase,
            frames=parsed_frames,
            body_names=names,
            segment_scales=segment_scales,
            contact_protocol=parsed_contact_protocol,
            contact_origin=contact_origin,
            root_position=target_root_position,
            root_orientation_wxyz=target_root_orientation,
            body_positions=target_body_positions,
            body_orientations_wxyz=target_body_orientations,
            contacts=target_contacts,
            validity_mask=target_validity,
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported HumanMotionSpec schema {self.schema_version!r}")
        if not self.body_names:
            raise ValueError("body_names must not be empty")
        if len(set(self.body_names)) != len(self.body_names):
            raise ValueError("body_names must be unique")
        if self.provenance.source_frame_count != self.timebase.source_timestamps.size:
            raise ValueError(
                "provenance clip_range length must match the source timeline"
            )
        unknown_feet = sorted(
            {
                self.contact_protocol.left_foot_body,
                self.contact_protocol.right_foot_body,
            }
            - set(self.body_names)
        )
        if unknown_feet:
            raise ValueError(f"contact protocol references unknown foot bodies: {unknown_feet}")
        if not np.all(self.validity_mask):
            raise ValueError(
                "canonical HumanMotionSpec validity_mask must contain only valid samples"
            )

        frames = self.timebase.target_timestamps.size
        bodies = len(self.body_names)
        expected_shapes = {
            "root_position": (frames, 3),
            "root_orientation_wxyz": (frames, 4),
            "body_positions": (frames, bodies, 3),
            "body_orientations_wxyz": (frames, bodies, 4),
            "contacts": (frames, len(CONTACT_LABELS)),
            "validity_mask": (frames, bodies),
        }
        for name, expected in expected_shapes.items():
            actual = getattr(self, name).shape
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}, got {actual}")
        if np.any((self.contacts < 0.0) | (self.contacts > 1.0)):
            raise ValueError("contacts must lie in the closed interval [0, 1]")
        # Quaternion norms were checked before canonicalising signs; repeat here so
        # validate() remains meaningful after deserialisation or future constructors.
        for name in ("root_orientation_wxyz", "body_orientations_wxyz"):
            norms = np.linalg.norm(getattr(self, name), axis=-1)
            if not np.allclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_NORM_ATOL):
                raise ValueError(f"{name} must contain unit quaternions")
        # Validation also guarantees that the integrity digest can be computed without
        # lossy overflow or underflow in the canonical float32 buffer representation.
        _ = self.buffer_sha256

    @property
    def buffer_sha256(self) -> str:
        """SHA-256 over canonical, labelled flat tensor buffers."""

        digest = hashlib.sha256()
        digest.update(b"snmr.human_motion.normalized_buffers.v1\0")
        buffers = (
            ("source_timestamps", self.timebase.source_timestamps, False),
            ("target_timestamps", self.timebase.target_timestamps, False),
            ("segment_scales", self.segment_scales.as_buffer(), False),
            ("root_position", self.root_position, False),
            ("root_orientation_wxyz", self.root_orientation_wxyz, False),
            ("body_positions", self.body_positions, False),
            ("body_orientations_wxyz", self.body_orientations_wxyz, False),
            ("contacts", self.contacts, False),
            ("validity_mask", self.validity_mask, True),
        )
        for name, value, boolean in buffers:
            _update_normalized_buffer(digest, name, value, boolean=boolean)
        return digest.hexdigest()

    @property
    def integrity_sha256(self) -> str:
        """Alias making the role of :attr:`buffer_sha256` explicit in manifests."""

        return self.buffer_sha256

    def _spec_hash_payload(self) -> dict[str, Any]:
        """Metadata payload transitively binding every tensor through its digest."""

        return {
            "schema_version": self.schema_version,
            "source": asdict(self.source),
            "provenance": self.provenance.to_dict(),
            "timebase": self.timebase.to_dict(),
            "frames": asdict(self.frames),
            "body_names": list(self.body_names),
            "segment_scales": asdict(self.segment_scales),
            "contact_protocol": asdict(self.contact_protocol),
            "contact_origin": self.contact_origin,
            "buffer_sha256": self.buffer_sha256,
        }

    @property
    def spec_sha256(self) -> str:
        """SHA-256 binding provenance, conventions, protocols, and motion buffers."""

        encoded = json.dumps(
            self._spec_hash_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": asdict(self.source),
            "provenance": self.provenance.to_dict(),
            "timebase": self.timebase.to_dict(),
            "frames": asdict(self.frames),
            "body_names": list(self.body_names),
            "segment_scales": asdict(self.segment_scales),
            "contact_protocol": asdict(self.contact_protocol),
            "contact_origin": self.contact_origin,
            "root_position": self.root_position.tolist(),
            "root_orientation_wxyz": self.root_orientation_wxyz.tolist(),
            "body_positions": self.body_positions.tolist(),
            "body_orientations_wxyz": self.body_orientations_wxyz.tolist(),
            "contacts": self.contacts.tolist(),
            "validity_mask": self.validity_mask.tolist(),
            "buffer_sha256": self.buffer_sha256,
            "spec_sha256": self.spec_sha256,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanMotionSpec":
        data = _require_mapping(payload, "HumanMotionSpec")
        expected = {
            "schema_version",
            "source",
            "provenance",
            "timebase",
            "frames",
            "body_names",
            "segment_scales",
            "contact_protocol",
            "contact_origin",
            "root_position",
            "root_orientation_wxyz",
            "body_positions",
            "body_orientations_wxyz",
            "contacts",
            "validity_mask",
            "buffer_sha256",
            "spec_sha256",
        }
        _require_exact_keys(data, expected, "HumanMotionSpec")

        source_data = _require_mapping(data["source"], "source")
        _require_exact_keys(source_data, {"dataset", "sequence_id", "sha256"}, "source")
        provenance_data = _require_mapping(data["provenance"], "provenance")
        _require_exact_keys(
            provenance_data,
            {
                "motion_id",
                "source_subject",
                "clip_range",
                "split_id",
                "transformations",
                "preprocessing_version",
            },
            "provenance",
        )
        frames_data = _require_mapping(data["frames"], "frames")
        _require_exact_keys(
            frames_data,
            {
                "world_up_axis",
                "world_forward_axis",
                "root_reference",
                "root_height_convention",
                "length_unit",
                "angle_unit",
                "quaternion_convention",
            },
            "frames",
        )
        scales_data = _require_mapping(data["segment_scales"], "segment_scales")
        _require_exact_keys(
            scales_data,
            {
                "normalization_length_m",
                "torso",
                "thigh",
                "shin",
                "upper_arm",
                "forearm",
            },
            "segment_scales",
        )
        contact_protocol_data = _require_mapping(data["contact_protocol"], "contact_protocol")
        _require_exact_keys(
            contact_protocol_data,
            {
                "left_foot_body",
                "right_foot_body",
                "speed_threshold_m_s",
                "height_clearance_m",
                "speed_softness_m_s",
                "height_softness_m",
                "ground_height_m",
                "method",
            },
            "contact_protocol",
        )
        timebase_data = _require_mapping(data["timebase"], "timebase")
        _require_exact_keys(
            timebase_data,
            {
                "source_fps",
                "source_timestamps",
                "target_fps",
                "target_timestamps",
                "position_interpolation",
                "position_interpolation_applied",
                "quaternion_interpolation",
                "contact_interpolation",
                "validity_interpolation",
            },
            "timebase",
        )

        expected_digest = data["buffer_sha256"]
        if not isinstance(expected_digest, str) or _SHA256_RE.fullmatch(expected_digest) is None:
            raise ValueError("buffer_sha256 must be a 64-character hexadecimal digest")
        expected_spec_digest = data["spec_sha256"]
        if (
            not isinstance(expected_spec_digest, str)
            or _SHA256_RE.fullmatch(expected_spec_digest) is None
        ):
            raise ValueError("spec_sha256 must be a 64-character hexadecimal digest")
        spec = cls(
            schema_version=str(data["schema_version"]),
            source=MotionSource(**source_data),
            provenance=MotionProvenance(
                **{
                    **provenance_data,
                    "clip_range": tuple(provenance_data["clip_range"]),
                    "transformations": tuple(provenance_data["transformations"]),
                }
            ),
            timebase=MotionTimebase(**timebase_data),
            frames=HumanFrameConvention(**frames_data),
            body_names=tuple(data["body_names"]),
            segment_scales=BodySegmentScales(**scales_data),
            contact_protocol=FootContactProtocol(**contact_protocol_data),
            contact_origin=str(data["contact_origin"]),
            root_position=data["root_position"],
            root_orientation_wxyz=data["root_orientation_wxyz"],
            body_positions=data["body_positions"],
            body_orientations_wxyz=data["body_orientations_wxyz"],
            contacts=data["contacts"],
            validity_mask=data["validity_mask"],
        )
        if not hmac.compare_digest(spec.buffer_sha256, expected_digest.lower()):
            raise ValueError("buffer_sha256 does not match the normalized motion buffers")
        if not hmac.compare_digest(spec.spec_sha256, expected_spec_digest.lower()):
            raise ValueError("spec_sha256 does not match the motion metadata and provenance")
        return spec

    @classmethod
    def from_json(cls, encoded: str | bytes | bytearray) -> "HumanMotionSpec":
        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON field {key!r}")
                result[key] = value
            return result

        payload = json.loads(encoded, object_pairs_hook=reject_duplicate_keys)
        return cls.from_dict(_require_mapping(payload, "JSON root"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HumanMotionSpec):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# Concise aliases for callers that prefer schema-oriented terminology.
SourceIdentity = MotionSource
FrameConvention = HumanFrameConvention
SegmentScaleDescriptors = BodySegmentScales
Timebase = MotionTimebase


__all__ = [
    "BilateralSegmentLandmarks",
    "BodySegmentScales",
    "CONTACT_LABELS",
    "FootContactProtocol",
    "FrameConvention",
    "HumanFrameConvention",
    "HumanMotionSpec",
    "MotionProvenance",
    "MotionSource",
    "MotionTimebase",
    "SCHEMA_VERSION",
    "SegmentScaleDescriptors",
    "SourceIdentity",
    "TARGET_FPS",
    "Timebase",
    "derive_foot_contact_probabilities",
    "extract_body_segment_scales",
]
