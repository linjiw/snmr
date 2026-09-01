"""Canonical, hash-bound GMR teacher trajectories for amortization experiments.

``HumanMotionSpec`` owns the source-motion identity and the canonical 50 Hz
timeline.  This module binds the robot side of an existing pair artifact to that
same timeline without treating the pair container as the human source identity.

The interpolation contract is deliberately narrow and explicit:

* root translation uses SciPy's cubic spline (linear for fewer than four samples);
* root orientation uses quaternion SLERP in the package-wide ``wxyz`` convention;
* scalar joint coordinates use linear interpolation, which preserves in-limit
  source intervals instead of allowing cubic overshoot.

The result is a supervised teacher target, not a semantic evaluation metric.  GMR
may train an amortized model, but teacher-independent semantic metrics remain the
ruler for the held-out embodiment claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import struct
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .motion_spec import HumanMotionSpec, TARGET_FPS
from .robot_spec import RobotSpec


TEACHER_MOTION_SCHEMA_VERSION = "snmr.teacher-motion.v1"
ROOT_POSITION_INTERPOLATION = "scipy-cubic-spline-not-a-knot"
SHORT_ROOT_POSITION_INTERPOLATION = "linear-short-sequence-fallback"
ROOT_ORIENTATION_INTERPOLATION = "scipy-slerp-wxyz"
JOINT_POSITION_INTERPOLATION = "linear-limit-preserving"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_QUATERNION_NORM_ATOL = 1e-5
_TIME_ATOL = 1e-12
_LIMIT_ATOL = 1e-6


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.lower()) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest")
    return value.lower()


def _immutable_float32(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    as_float64 = np.asarray(raw, dtype=np.float64, order="C")
    if not np.isfinite(as_float64).all():
        raise ValueError(f"{name} must contain only finite values")
    limit = np.finfo(np.float32).max
    if np.any(np.abs(as_float64) > limit):
        raise ValueError(f"{name} cannot be represented as finite float32")
    cast = np.asarray(as_float64, dtype="<f4", order="C")
    if not np.isfinite(cast).all():
        raise ValueError(f"{name} became non-finite during float32 normalization")
    frozen = np.frombuffer(cast.tobytes(order="C"), dtype="<f4")
    return frozen.reshape(cast.shape)


def _immutable_float64(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    cast = np.asarray(raw, dtype="<f8", order="C")
    if not np.isfinite(cast).all():
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.frombuffer(cast.tobytes(order="C"), dtype="<f8")
    return frozen.reshape(cast.shape)


def _canonical_quaternions(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64, order="C")
    if result.ndim != 2 or result.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (T, 4)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(result, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=_QUATERNION_NORM_ATOL):
        raise ValueError(f"{name} must contain unit wxyz quaternions")
    normalized = result / norms[:, None]
    # Canonicalize the double-cover sign without breaking temporal continuity.
    for frame in range(1, normalized.shape[0]):
        if float(np.dot(normalized[frame - 1], normalized[frame])) < 0.0:
            normalized[frame] *= -1.0
    first_nonzero = np.flatnonzero(np.abs(normalized[0]) > np.finfo(np.float64).eps)
    if first_nonzero.size and normalized[0, first_nonzero[0]] < 0.0:
        normalized *= -1.0
    return normalized


def _interpolate_numeric(
    values: np.ndarray,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    *,
    cubic: bool,
) -> np.ndarray:
    if source_timestamps.size == 1:
        return np.repeat(values, target_timestamps.size, axis=0)
    if cubic:
        return np.asarray(
            CubicSpline(source_timestamps, values, axis=0)(target_timestamps),
            dtype=np.float64,
        )
    flattened = values.reshape(values.shape[0], -1)
    output = np.empty((target_timestamps.size, flattened.shape[1]), dtype=np.float64)
    for column in range(flattened.shape[1]):
        output[:, column] = np.interp(
            target_timestamps, source_timestamps, flattened[:, column]
        )
    return output.reshape((target_timestamps.size,) + values.shape[1:])


def _interpolate_quaternions(
    quaternions_wxyz: np.ndarray,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    if source_timestamps.size == 1:
        return np.repeat(quaternions_wxyz, target_timestamps.size, axis=0)
    rotations = Rotation.from_quat(quaternions_wxyz[:, (1, 2, 3, 0)])
    xyzw = Slerp(source_timestamps, rotations)(target_timestamps).as_quat()
    return _canonical_quaternions(xyzw[:, (3, 0, 1, 2)], "resampled root orientation")


def _update_buffer_hash(
    digest: Any,
    name: str,
    value: np.ndarray,
    *,
    dtype: str = "<f4",
) -> None:
    label = name.encode("utf-8")
    digest.update(struct.pack("<I", len(label)))
    digest.update(label)
    digest.update(struct.pack("<I", value.ndim))
    digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
    normalized = np.array(value, dtype=dtype, order="C", copy=True)
    normalized[normalized == 0.0] = 0.0
    digest.update(normalized.tobytes(order="C"))


@dataclass(frozen=True, eq=False)
class CanonicalTeacherMotion:
    """One robot teacher trajectory aligned exactly to a HumanMotionSpec."""

    timestamps_s: Any
    root_position_m: Any
    root_orientation_wxyz: Any
    joint_positions_rad: Any
    joint_names: Sequence[str]
    human_motion_spec_sha256: str
    robot_spec_sha256: str
    robot_asset_sha256: str
    pair_artifact_sha256: str
    source_fps: float
    root_position_interpolation_applied: str
    schema_version: str = TEACHER_MOTION_SCHEMA_VERSION
    root_position_interpolation: str = ROOT_POSITION_INTERPOLATION
    root_orientation_interpolation: str = ROOT_ORIENTATION_INTERPOLATION
    joint_position_interpolation: str = JOINT_POSITION_INTERPOLATION

    def __post_init__(self) -> None:
        if self.schema_version != TEACHER_MOTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported teacher-motion schema {self.schema_version!r}")
        expected_methods = {
            "root_position_interpolation": ROOT_POSITION_INTERPOLATION,
            "root_orientation_interpolation": ROOT_ORIENTATION_INTERPOLATION,
            "joint_position_interpolation": JOINT_POSITION_INTERPOLATION,
        }
        for field, expected in expected_methods.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} must equal {expected!r}")
        if self.root_position_interpolation_applied not in {
            ROOT_POSITION_INTERPOLATION,
            SHORT_ROOT_POSITION_INTERPOLATION,
        }:
            raise ValueError("unknown applied root-position interpolation")

        fps = float(self.source_fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("source_fps must be finite and positive")
        if not isinstance(self.joint_names, Sequence) or isinstance(
            self.joint_names, (str, bytes)
        ):
            raise TypeError("joint_names must be an explicit sequence")
        names = tuple(self.joint_names)
        if not names or any(
            not isinstance(name, str) or not name or name != name.strip() for name in names
        ):
            raise ValueError("joint_names must contain non-empty trimmed strings")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")

        timestamps = _immutable_float64(self.timestamps_s, "timestamps_s")
        root_position = _immutable_float32(self.root_position_m, "root_position_m")
        root_orientation = _immutable_float32(
            _canonical_quaternions(self.root_orientation_wxyz, "root_orientation_wxyz"),
            "root_orientation_wxyz",
        )
        joints = _immutable_float32(self.joint_positions_rad, "joint_positions_rad")
        frames = timestamps.size if timestamps.ndim == 1 else 0
        if frames == 0:
            raise ValueError("timestamps_s must be a non-empty one-dimensional array")
        if not math.isclose(float(timestamps[0]), 0.0, rel_tol=0.0, abs_tol=_TIME_ATOL):
            raise ValueError("timestamps_s must start at zero")
        if frames > 1:
            steps = np.diff(timestamps.astype(np.float64))
            if np.any(steps <= 0.0) or not np.allclose(
                steps, 1.0 / TARGET_FPS, rtol=0.0, atol=2e-8
            ):
                raise ValueError("timestamps_s must be a strictly uniform 50 Hz grid")
        if root_position.shape != (frames, 3):
            raise ValueError(f"root_position_m must have shape {(frames, 3)}")
        if root_orientation.shape != (frames, 4):
            raise ValueError(f"root_orientation_wxyz must have shape {(frames, 4)}")
        if joints.shape != (frames, len(names)):
            raise ValueError(
                f"joint_positions_rad must have shape {(frames, len(names))}"
            )

        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "root_position_m", root_position)
        object.__setattr__(self, "root_orientation_wxyz", root_orientation)
        object.__setattr__(self, "joint_positions_rad", joints)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "source_fps", fps)
        for field in (
            "human_motion_spec_sha256",
            "robot_spec_sha256",
            "robot_asset_sha256",
            "pair_artifact_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))

    @property
    def buffer_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"snmr.teacher-motion.normalized-float32-buffers.v1\0")
        for value in (
            self.schema_version,
            self.human_motion_spec_sha256,
            self.robot_spec_sha256,
            self.robot_asset_sha256,
            self.pair_artifact_sha256,
            *self.joint_names,
        ):
            encoded = value.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
        for value in (
            self.root_position_interpolation,
            self.root_position_interpolation_applied,
            self.root_orientation_interpolation,
            self.joint_position_interpolation,
        ):
            digest.update(value.encode("utf-8") + b"\0")
        digest.update(struct.pack("<d", self.source_fps))
        _update_buffer_hash(digest, "timestamps_s", self.timestamps_s, dtype="<f8")
        _update_buffer_hash(digest, "root_position_m", self.root_position_m)
        _update_buffer_hash(
            digest, "root_orientation_wxyz", self.root_orientation_wxyz
        )
        _update_buffer_hash(digest, "joint_positions_rad", self.joint_positions_rad)
        return digest.hexdigest()


def resample_teacher_qpos(
    qpos: Any,
    *,
    source_fps: float,
    human_motion: HumanMotionSpec,
    robot_spec: RobotSpec,
    pair_artifact_sha256: str,
) -> CanonicalTeacherMotion:
    """Resample a selected pair-NPZ qpos slice onto ``human_motion``'s timeline."""

    if not isinstance(human_motion, HumanMotionSpec):
        raise TypeError("human_motion must be a HumanMotionSpec")
    if not isinstance(robot_spec, RobotSpec):
        raise TypeError("robot_spec must be a RobotSpec")
    human_motion.validate()
    robot_spec.validate()
    parsed_fps = float(source_fps)
    if not math.isfinite(parsed_fps) or parsed_fps <= 0.0:
        raise ValueError("source_fps must be finite and positive")
    if not math.isclose(
        parsed_fps, human_motion.timebase.source_fps, rel_tol=0.0, abs_tol=_TIME_ATOL
    ):
        raise ValueError("source_fps must match HumanMotionSpec.timebase.source_fps")

    source = np.asarray(qpos)
    if source.dtype.kind not in "fiu":
        raise TypeError("qpos must contain real numeric values")
    source = np.asarray(source, dtype=np.float64, order="C")
    expected_frames = human_motion.provenance.source_frame_count
    joint_names = tuple(joint.name for joint in robot_spec.joints)
    expected_shape = (expected_frames, 7 + len(joint_names))
    if source.shape != expected_shape:
        raise ValueError(f"qpos must have shape {expected_shape}, got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("qpos must contain only finite values")

    root_orientation = _canonical_quaternions(source[:, 3:7], "qpos root orientation")
    joint_positions = source[:, 7:]
    lower = np.asarray([joint.lower_limit for joint in robot_spec.joints])
    upper = np.asarray([joint.upper_limit for joint in robot_spec.joints])
    if np.any(joint_positions < lower - _LIMIT_ATOL) or np.any(
        joint_positions > upper + _LIMIT_ATOL
    ):
        lower_excess = float(np.max(lower[None] - joint_positions))
        upper_excess = float(np.max(joint_positions - upper[None]))
        raise ValueError(
            "source teacher joint positions exceed RobotSpec limits "
            f"(lower={max(lower_excess, 0.0):g}, upper={max(upper_excess, 0.0):g})"
        )

    source_timestamps = human_motion.timebase.source_timestamps
    target_timestamps = human_motion.timebase.target_timestamps
    use_cubic = expected_frames >= 4
    target_root_position = _interpolate_numeric(
        source[:, :3], source_timestamps, target_timestamps, cubic=use_cubic
    )
    target_root_orientation = _interpolate_quaternions(
        root_orientation, source_timestamps, target_timestamps
    )
    target_joints = _interpolate_numeric(
        joint_positions, source_timestamps, target_timestamps, cubic=False
    )
    if np.any(target_joints < lower - _LIMIT_ATOL) or np.any(
        target_joints > upper + _LIMIT_ATOL
    ):
        raise AssertionError("linear interpolation unexpectedly left joint limits")

    return CanonicalTeacherMotion(
        timestamps_s=target_timestamps,
        root_position_m=target_root_position,
        root_orientation_wxyz=target_root_orientation,
        joint_positions_rad=target_joints,
        joint_names=joint_names,
        human_motion_spec_sha256=human_motion.spec_sha256,
        robot_spec_sha256=robot_spec.spec_hash,
        robot_asset_sha256=robot_spec.asset_sha256,
        pair_artifact_sha256=pair_artifact_sha256,
        source_fps=parsed_fps,
        root_position_interpolation_applied=(
            ROOT_POSITION_INTERPOLATION
            if use_cubic
            else SHORT_ROOT_POSITION_INTERPOLATION
        ),
    )


__all__ = [
    "CanonicalTeacherMotion",
    "JOINT_POSITION_INTERPOLATION",
    "ROOT_ORIENTATION_INTERPOLATION",
    "ROOT_POSITION_INTERPOLATION",
    "SHORT_ROOT_POSITION_INTERPOLATION",
    "TEACHER_MOTION_SCHEMA_VERSION",
    "resample_teacher_qpos",
]
