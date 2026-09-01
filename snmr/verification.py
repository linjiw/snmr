"""Backend-neutral evidence contract for physics-verified retargeting.

Simulator adapters live in their own environments.  They exchange JSON reports described
here instead of importing Isaac Lab, Newton, or Holosoma into the SNMR training process.
The helpers deliberately preserve failure intervals: clip-level success alone is not a
sufficient signal for localized repair or preference learning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


REPORT_SCHEMA_VERSION = "snmr.physics-verification.v0.1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _validate_sha256(value: str | None, label: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.lower()) is None:
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest")


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a JSON boolean")
    return value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _strict_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _strict_string(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rollout_contract_hash(
    *,
    seconds: float,
    start_frame: int,
    fall_root_z_m: float,
    root_xy_limit_m: float,
    tilt_limit_rad: float,
    saturation_threshold: float,
    min_failure_frames: int,
    merge_gap_frames: int,
) -> str:
    """Hash backend-independent rollout-window and termination semantics."""

    numeric = {
        "seconds": _strict_float(seconds, "seconds"),
        "fall_root_z_m": _strict_float(fall_root_z_m, "fall_root_z_m"),
        "root_xy_limit_m": _strict_float(root_xy_limit_m, "root_xy_limit_m"),
        "tilt_limit_rad": _strict_float(tilt_limit_rad, "tilt_limit_rad"),
        "saturation_threshold": _strict_float(
            saturation_threshold, "saturation_threshold"
        ),
    }
    if any(value <= 0.0 for name, value in numeric.items() if name != "saturation_threshold"):
        raise ValueError("rollout durations and physical thresholds must be positive")
    if not 0.0 <= numeric["saturation_threshold"] <= 1.0:
        raise ValueError("saturation_threshold must lie in [0, 1]")
    parsed_start = _strict_int(start_frame, "start_frame")
    parsed_min = _strict_int(min_failure_frames, "min_failure_frames")
    parsed_gap = _strict_int(merge_gap_frames, "merge_gap_frames")
    if parsed_start < 0 or parsed_min <= 0 or parsed_gap < 0:
        raise ValueError("invalid rollout frame/count parameter")
    return canonical_hash({
        "seconds": numeric["seconds"],
        "start_frame": parsed_start,
        "fall_root_z_m": numeric["fall_root_z_m"],
        "root_xy_limit_m": numeric["root_xy_limit_m"],
        "tilt_limit_rad": numeric["tilt_limit_rad"],
        "saturation_threshold": numeric["saturation_threshold"],
        "min_failure_frames": parsed_min,
        "merge_gap_frames": parsed_gap,
    })


@dataclass(frozen=True)
class BackendIdentity:
    engine: str
    solver: str
    engine_version: str
    solver_version: str | None = None
    build: str | None = None
    quaternion_convention: str = "wxyz"


@dataclass(frozen=True)
class FailureInterval:
    start_frame: int
    end_frame: int
    failure_type: str
    severity: float
    links: tuple[str, ...] = ()
    joints: tuple[str, ...] = ()
    evidence: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class VerificationReport:
    schema_version: str
    candidate_id: str
    motion_sha256: str
    robot_spec_hash: str
    robot_kinematic_hash: str
    robot_dynamics_hash: str
    verification_level: str
    passed: bool
    fps: float
    num_frames: int
    backend: BackendIdentity
    metrics: tuple[tuple[str, float], ...]
    failure_intervals: tuple[FailureInterval, ...] = ()
    controller_hash: str | None = None
    seed: int | None = None
    config_hash: str | None = None
    overflow_detected: bool = False

    def validate(self) -> None:
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported report schema {self.schema_version!r}")
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        for label, value in (
            ("motion_sha256", self.motion_sha256),
            ("robot_spec_hash", self.robot_spec_hash),
            ("robot_kinematic_hash", self.robot_kinematic_hash),
            ("robot_dynamics_hash", self.robot_dynamics_hash),
        ):
            _validate_sha256(value, label)
        _validate_sha256(self.controller_hash, "controller_hash", optional=True)
        _validate_sha256(self.config_hash, "config_hash", optional=True)
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        if not isinstance(self.overflow_detected, bool):
            raise TypeError("overflow_detected must be a boolean")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        if isinstance(self.num_frames, bool) or not isinstance(self.num_frames, (int, np.integer)):
            raise TypeError("num_frames must be an integer")
        if self.num_frames < 0:
            raise ValueError("fps must be positive and num_frames nonnegative")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
                raise TypeError("seed must be an integer or null")
            if self.seed < 0:
                raise ValueError("seed must be nonnegative")
        for label in ("engine", "solver", "engine_version", "quaternion_convention"):
            value = getattr(self.backend, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"backend {label} must be a non-empty string")
        if self.backend.quaternion_convention not in {"wxyz", "xyzw"}:
            raise ValueError("backend quaternion_convention must be 'wxyz' or 'xyzw'")
        metric_names = [name for name, _ in self.metrics]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric names must be unique")
        for name, value in self.metrics:
            if not isinstance(name, str) or not name or not math.isfinite(value):
                raise ValueError(f"invalid metric {name!r}={value}")
        previous_end_by_type: dict[str, int] = {}
        for interval in sorted(
            self.failure_intervals,
            key=lambda item: (item.failure_type, item.start_frame, item.end_frame),
        ):
            if (
                isinstance(interval.start_frame, bool)
                or not isinstance(interval.start_frame, (int, np.integer))
                or isinstance(interval.end_frame, bool)
                or not isinstance(interval.end_frame, (int, np.integer))
            ):
                raise TypeError("failure interval frame indices must be integers")
            if not 0 <= interval.start_frame <= interval.end_frame < self.num_frames:
                raise ValueError(f"failure interval outside clip: {interval}")
            if not math.isfinite(interval.severity) or interval.severity < 0.0:
                raise ValueError(f"invalid failure severity: {interval.severity}")
            if not isinstance(interval.failure_type, str) or not interval.failure_type.strip():
                raise ValueError("failure_type must be a non-empty string")
            for label, names in (("links", interval.links), ("joints", interval.joints)):
                if any(not isinstance(name, str) or not name.strip() for name in names):
                    raise ValueError(f"failure interval {label} must contain non-empty strings")
            evidence_names: set[str] = set()
            for name, value in interval.evidence:
                if not isinstance(name, str) or not name or not math.isfinite(value):
                    raise ValueError(f"invalid failure evidence {name!r}={value}")
                if name in evidence_names:
                    raise ValueError("failure evidence names must be unique")
                evidence_names.add(name)
            # Different failure types may co-occur, but one signal must not emit overlaps.
            previous_end = previous_end_by_type.get(interval.failure_type, -1)
            if interval.start_frame <= previous_end:
                raise ValueError("same-type failure intervals must not overlap")
            previous_end_by_type[interval.failure_type] = interval.end_frame

    def metric_dict(self) -> dict[str, float]:
        return dict(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VerificationReport":
        fps = _strict_float(payload["fps"], "fps")
        num_frames = _strict_int(payload["num_frames"], "num_frames")
        seed_raw = payload.get("seed")
        seed = None if seed_raw is None else _strict_int(seed_raw, "seed")
        backend_payload = payload["backend"]
        if not isinstance(backend_payload, Mapping):
            raise TypeError("backend must be an object")
        backend = BackendIdentity(
            engine=_strict_string(backend_payload.get("engine"), "backend.engine"),
            solver=_strict_string(backend_payload.get("solver"), "backend.solver"),
            engine_version=_strict_string(
                backend_payload.get("engine_version"), "backend.engine_version"
            ),
            solver_version=_strict_string(
                backend_payload.get("solver_version"),
                "backend.solver_version",
                optional=True,
            ),
            build=_strict_string(backend_payload.get("build"), "backend.build", optional=True),
            quaternion_convention=_strict_string(
                backend_payload.get("quaternion_convention", "wxyz"),
                "backend.quaternion_convention",
            ),
        )
        intervals: list[FailureInterval] = []
        for index, interval in enumerate(payload.get("failure_intervals", ())):
            if not isinstance(interval, Mapping):
                raise TypeError(f"failure_intervals[{index}] must be an object")
            intervals.append(FailureInterval(
                start_frame=_strict_int(
                    interval.get("start_frame"), f"failure_intervals[{index}].start_frame"
                ),
                end_frame=_strict_int(
                    interval.get("end_frame"), f"failure_intervals[{index}].end_frame"
                ),
                failure_type=_strict_string(
                    interval.get("failure_type"), f"failure_intervals[{index}].failure_type"
                ),
                severity=_strict_float(
                    interval.get("severity"), f"failure_intervals[{index}].severity"
                ),
                links=tuple(interval.get("links", ())),
                joints=tuple(interval.get("joints", ())),
                evidence=tuple(
                    (
                        _strict_string(name, f"failure_intervals[{index}].evidence.name"),
                        _strict_float(value, f"failure_intervals[{index}].evidence.value"),
                    )
                    for name, value in interval.get("evidence", ())
                ),
            ))
        report = cls(
            schema_version=_strict_string(payload["schema_version"], "schema_version"),
            candidate_id=_strict_string(payload["candidate_id"], "candidate_id"),
            motion_sha256=_strict_string(payload["motion_sha256"], "motion_sha256"),
            robot_spec_hash=_strict_string(payload["robot_spec_hash"], "robot_spec_hash"),
            robot_kinematic_hash=_strict_string(
                payload["robot_kinematic_hash"], "robot_kinematic_hash"
            ),
            robot_dynamics_hash=_strict_string(
                payload["robot_dynamics_hash"], "robot_dynamics_hash"
            ),
            verification_level=_strict_string(
                payload["verification_level"], "verification_level"
            ),
            passed=_strict_bool(payload["passed"], "passed"),
            fps=fps,
            num_frames=num_frames,
            backend=backend,
            metrics=tuple(
                (
                    _strict_string(name, "metric name"),
                    _strict_float(value, f"metric {name!r}"),
                )
                for name, value in payload["metrics"]
            ),
            failure_intervals=tuple(intervals),
            controller_hash=payload.get("controller_hash"),
            seed=seed,
            config_hash=payload.get("config_hash"),
            overflow_detected=_strict_bool(
                payload.get("overflow_detected", False), "overflow_detected"
            ),
        )
        report.validate()
        return report

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class CrossSolverComparison:
    candidate_id: str
    same_candidate_contract: bool
    pass_agreement: bool
    failure_type_jaccard: float
    first_failure_delta_seconds: float | None
    metric_deltas: tuple[tuple[str, float], ...]


def localize_threshold_failures(
    values: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    failure_type: str,
    comparison: str = "greater",
    min_frames: int = 1,
    merge_gap_frames: int = 0,
    links: Sequence[str] = (),
    joints: Sequence[str] = (),
) -> tuple[FailureInterval, ...]:
    """Turn a per-frame diagnostic into contiguous, repair-addressable intervals."""

    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("failure signal contains NaN/Inf")
    if min_frames <= 0 or merge_gap_frames < 0:
        raise ValueError("min_frames must be positive and merge_gap_frames nonnegative")
    if comparison == "greater":
        failed = signal > threshold
        severity = np.maximum(signal - threshold, 0.0)
    elif comparison == "less":
        failed = signal < threshold
        severity = np.maximum(threshold - signal, 0.0)
    else:
        raise ValueError("comparison must be 'greater' or 'less'")

    raw: list[tuple[int, int]] = []
    start: int | None = None
    for frame, is_failed in enumerate(failed.tolist() + [False]):
        if is_failed and start is None:
            start = frame
        elif not is_failed and start is not None:
            raw.append((start, frame - 1))
            start = None

    # Reject isolated spikes before gap-merging; otherwise a one-frame outlier can extend an
    # otherwise valid repair interval merely because it sits one frame away.
    raw = [(start, end) for start, end in raw if end - start + 1 >= min_frames]

    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] - 1 <= merge_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    intervals = []
    for start, end in merged:
        intervals.append(FailureInterval(
            start_frame=start,
            end_frame=end,
            failure_type=failure_type,
            severity=float(severity[start:end + 1].max()),
            links=tuple(links),
            joints=tuple(joints),
            evidence=(("threshold", float(threshold)), ("peak", float(signal[start:end + 1].max()))),
        ))
    return tuple(intervals)


def offset_failure_intervals(
    intervals: Sequence[FailureInterval],
    frame_offset: int,
) -> tuple[FailureInterval, ...]:
    """Translate window-local failure intervals into source-clip frame coordinates."""

    if isinstance(frame_offset, bool) or not isinstance(frame_offset, (int, np.integer)):
        raise TypeError("frame_offset must be an integer")
    if frame_offset < 0:
        raise ValueError("frame_offset must be nonnegative")
    return tuple(FailureInterval(
        start_frame=interval.start_frame + int(frame_offset),
        end_frame=interval.end_frame + int(frame_offset),
        failure_type=interval.failure_type,
        severity=interval.severity,
        links=interval.links,
        joints=interval.joints,
        evidence=interval.evidence,
    ) for interval in intervals)


def compare_solver_reports(
    left: VerificationReport,
    right: VerificationReport,
    *,
    metric_names: Sequence[str] = (
        "survival_fraction",
        "mean_dof_error_rad",
        "torque_saturation_fraction",
    ),
) -> CrossSolverComparison:
    """Compare two backends without collapsing disagreements into one opaque score."""

    left.validate()
    right.validate()
    same_contract = (
        left.candidate_id == right.candidate_id
        and left.motion_sha256 == right.motion_sha256
        and left.robot_spec_hash == right.robot_spec_hash
        and left.controller_hash == right.controller_hash
        and left.config_hash == right.config_hash
        and left.fps == right.fps
        and left.num_frames == right.num_frames
        and left.seed == right.seed
    )
    left_types = {interval.failure_type for interval in left.failure_intervals}
    right_types = {interval.failure_type for interval in right.failure_intervals}
    union = left_types | right_types
    jaccard = 1.0 if not union else len(left_types & right_types) / len(union)

    def first_failure(report: VerificationReport) -> float | None:
        if not report.failure_intervals:
            return None
        return min(item.start_frame for item in report.failure_intervals) / report.fps

    left_first, right_first = first_failure(left), first_failure(right)
    first_delta = None
    if left_first is not None and right_first is not None:
        first_delta = right_first - left_first
    lm, rm = left.metric_dict(), right.metric_dict()
    deltas = tuple(
        (name, rm[name] - lm[name])
        for name in metric_names
        if name in lm and name in rm
    )
    return CrossSolverComparison(
        candidate_id=left.candidate_id,
        same_candidate_contract=same_contract,
        pass_agreement=left.passed == right.passed,
        failure_type_jaccard=float(jaccard),
        first_failure_delta_seconds=first_delta,
        metric_deltas=deltas,
    )
