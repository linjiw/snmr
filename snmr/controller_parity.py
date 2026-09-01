"""Fail-closed contracts for the G0 controller and standing-state parity gate.

This module is deliberately simulator independent.  Backend-specific workers extract
the exact asset and controller data they consume, then pass normalized arrays here for
strict comparison.  A source-only comparison is useful evidence, but cannot pass G0
unless every required live/runtime evidence item is explicitly present and true.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


CONTROLLER_PARITY_SCHEMA_VERSION = "snmr.g0_controller_parity.v0.1"
CONTROLLER_CONTRACT_HASH_SCHEMA_VERSION = "snmr.controller-contract.v0.1"
JOINT_AXIS_ATOL = 1.0e-12
JOINT_SCALAR_ATOL = 1.0e-9
TIMING_ATOL = 1.0e-12

REQUIRED_SOURCE_EVIDENCE: tuple[str, ...] = (
    "shared_position_target_formula",
    "isaac_effort_target_adapter",
    "mujoco_effort_target_adapter",
    "isaac_simulation_dt_binding",
    "mujoco_simulation_dt_binding",
    "control_decimation_binding",
    "isaac_standing_state_binding",
    "mujoco_standing_state_binding",
    "shared_motion_reset_binding",
)

REQUIRED_LIVE_EVIDENCE: tuple[str, ...] = (
    "physx_runtime_joint_mapping",
    "physx_live_nominal_state",
    "mujoco_live_nominal_state",
    "physx_live_controller_parameters",
    "mujoco_live_controller_parameters",
    "paired_reset_same_motion_seed",
    "registered_usd_matches_tracker_training_asset",
    "tracker_training_holosoma_revision",
    "same_frozen_checkpoint_both_backends",
)


def _float_vector(values: Sequence[float], *, size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return result


def _axis_array(values: Sequence[Sequence[float]], *, size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size, 3):
        raise ValueError(f"{label} must have shape ({size}, 3), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError(f"{label} contains a zero axis")
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return result


def _freeze_json(value: object) -> object:
    """Convert validated JSON data to deeply immutable tuples and scalars."""

    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("reset_noise mapping keys must be strings")
        return tuple((key, _freeze_json(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("reset_noise contains a non-finite float")
        return value
    raise TypeError(f"reset_noise contains unsupported JSON value {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            for item in value
        ):
            return {item[0]: _thaw_json(item[1]) for item in value}
        return [_thaw_json(item) for item in value]
    return value


def _optional_float_vector(
    values: Sequence[float] | None, *, size: int, label: str
) -> np.ndarray | None:
    return None if values is None else _float_vector(values, size=size, label=label)


def _joint_names(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{label} must contain non-empty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate names")
    return result


@dataclass(frozen=True)
class JointModelContract:
    """Normalized joint fields extracted from one robot asset or recipe."""

    source: str
    joint_names: tuple[str, ...]
    joint_types: tuple[str, ...]
    axes: np.ndarray
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    effort_limits: np.ndarray
    velocity_limits: np.ndarray | None = None
    motor_effort_limits: np.ndarray | None = None
    joint_friction: np.ndarray | None = None

    @classmethod
    def from_sequences(
        cls,
        *,
        source: str,
        joint_names: Sequence[str],
        joint_types: Sequence[str],
        axes: Sequence[Sequence[float]],
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        effort_limits: Sequence[float],
        velocity_limits: Sequence[float] | None = None,
        motor_effort_limits: Sequence[float] | None = None,
        joint_friction: Sequence[float] | None = None,
    ) -> "JointModelContract":
        names = _joint_names(joint_names, label=f"{source}.joint_names")
        count = len(names)
        types = tuple(str(value) for value in joint_types)
        if len(types) != count or any(not value for value in types):
            raise ValueError(f"{source}.joint_types must contain one non-empty value per joint")
        lower = _float_vector(lower_limits, size=count, label=f"{source}.lower_limits")
        upper = _float_vector(upper_limits, size=count, label=f"{source}.upper_limits")
        if np.any(upper <= lower):
            raise ValueError(f"{source} contains an invalid joint range")
        effort = _float_vector(effort_limits, size=count, label=f"{source}.effort_limits")
        if np.any(effort <= 0.0):
            raise ValueError(f"{source}.effort_limits must be positive")
        velocity = _optional_float_vector(
            velocity_limits, size=count, label=f"{source}.velocity_limits"
        )
        if velocity is not None and np.any(velocity <= 0.0):
            raise ValueError(f"{source}.velocity_limits must be positive")
        motor = _optional_float_vector(
            motor_effort_limits, size=count, label=f"{source}.motor_effort_limits"
        )
        if motor is not None and np.any(motor <= 0.0):
            raise ValueError(f"{source}.motor_effort_limits must be positive")
        friction = _optional_float_vector(
            joint_friction, size=count, label=f"{source}.joint_friction"
        )
        if friction is not None and np.any(friction < 0.0):
            raise ValueError(f"{source}.joint_friction must be non-negative")
        return cls(
            source=str(source),
            joint_names=names,
            joint_types=types,
            axes=_axis_array(axes, size=count, label=f"{source}.axes"),
            lower_limits=lower,
            upper_limits=upper,
            effort_limits=effort,
            velocity_limits=velocity,
            motor_effort_limits=motor,
            joint_friction=friction,
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "source": self.source,
            "joint_names": list(self.joint_names),
            "joint_types": list(self.joint_types),
            "axes": self.axes.tolist(),
            "lower_limits_rad": self.lower_limits.tolist(),
            "upper_limits_rad": self.upper_limits.tolist(),
            "effort_limits_nm": self.effort_limits.tolist(),
            "velocity_limits_rad_s": (
                None if self.velocity_limits is None else self.velocity_limits.tolist()
            ),
            "motor_effort_limits_nm": (
                None if self.motor_effort_limits is None else self.motor_effort_limits.tolist()
            ),
            "joint_friction_nm": (
                None if self.joint_friction is None else self.joint_friction.tolist()
            ),
        }


@dataclass(frozen=True)
class ControllerRecipe:
    """Effective per-joint controller, timing, nominal, and reset configuration."""

    source: str
    backend: str
    joint_names: tuple[str, ...]
    nominal_q: np.ndarray
    root_position: np.ndarray
    root_quaternion_xyzw: np.ndarray
    root_linear_velocity: np.ndarray
    root_angular_velocity: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    effort_limits: np.ndarray
    velocity_limits: np.ndarray
    joint_friction: np.ndarray
    action_scales: np.ndarray
    action_scale_base: float
    action_scale_by_effort_over_kp: bool
    control_type: str
    clip_actions: bool
    action_clip_value: float
    clip_torques: bool
    simulation_dt: float
    control_dt: float
    decimation: int
    latency_enabled: bool
    latency_step_range: tuple[int, int]
    reset_semantics: str
    reset_noise: object

    @classmethod
    def from_sequences(
        cls,
        *,
        source: str,
        backend: str,
        joint_names: Sequence[str],
        nominal_q: Sequence[float],
        root_position: Sequence[float],
        root_quaternion_xyzw: Sequence[float],
        root_linear_velocity: Sequence[float],
        root_angular_velocity: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        lower_limits: Sequence[float],
        upper_limits: Sequence[float],
        effort_limits: Sequence[float],
        velocity_limits: Sequence[float],
        joint_friction: Sequence[float],
        action_scales: Sequence[float],
        action_scale_base: float,
        action_scale_by_effort_over_kp: bool,
        control_type: str,
        clip_actions: bool,
        action_clip_value: float,
        clip_torques: bool,
        simulation_dt: float,
        control_dt: float,
        decimation: int,
        latency_enabled: bool,
        latency_step_range: Sequence[int],
        reset_semantics: str,
        reset_noise: Mapping[str, object],
    ) -> "ControllerRecipe":
        names = _joint_names(joint_names, label=f"{source}.joint_names")
        count = len(names)
        if isinstance(decimation, bool) or int(decimation) != decimation or decimation <= 0:
            raise ValueError("decimation must be a positive integer")
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in latency_step_range):
            raise TypeError("latency_step_range must contain integers")
        latency = tuple(int(value) for value in latency_step_range)
        if len(latency) != 2 or latency[0] < 0 or latency[1] < latency[0]:
            raise ValueError("latency_step_range must be a non-negative [lower, upper] pair")
        sim_dt = float(simulation_dt)
        ctrl_dt = float(control_dt)
        if not np.isfinite(sim_dt) or sim_dt <= 0.0:
            raise ValueError("simulation_dt must be finite and positive")
        if not np.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
            raise ValueError("control_dt must be finite and positive")
        if abs(ctrl_dt - int(decimation) * sim_dt) > TIMING_ATOL:
            raise ValueError("control_dt must equal decimation * simulation_dt")
        base_scale = float(action_scale_base)
        clip_value = float(action_clip_value)
        if not np.isfinite(base_scale) or base_scale < 0.0:
            raise ValueError("action_scale_base must be finite and non-negative")
        if not np.isfinite(clip_value) or clip_value <= 0.0:
            raise ValueError("action_clip_value must be finite and positive")
        root_quaternion = _float_vector(
            root_quaternion_xyzw, size=4, label=f"{source}.root_quaternion_xyzw"
        )
        if abs(float(np.linalg.norm(root_quaternion)) - 1.0) > 1.0e-9:
            raise ValueError("root quaternion must be unit length")
        kp_array = _float_vector(kp, size=count, label=f"{source}.kp")
        kd_array = _float_vector(kd, size=count, label=f"{source}.kd")
        if np.any(kp_array <= 0.0) or np.any(kd_array < 0.0):
            raise ValueError("P gains must be positive and D gains must be non-negative")
        effort_array = _float_vector(
            effort_limits, size=count, label=f"{source}.effort_limits"
        )
        lower_array = _float_vector(
            lower_limits, size=count, label=f"{source}.lower_limits"
        )
        upper_array = _float_vector(
            upper_limits, size=count, label=f"{source}.upper_limits"
        )
        if np.any(upper_array <= lower_array):
            raise ValueError(f"{source} contains an invalid configured joint range")
        velocity_array = _float_vector(
            velocity_limits, size=count, label=f"{source}.velocity_limits"
        )
        if np.any(effort_array <= 0.0) or np.any(velocity_array <= 0.0):
            raise ValueError("effort and velocity limits must be positive")
        friction_array = _float_vector(
            joint_friction, size=count, label=f"{source}.joint_friction"
        )
        if np.any(friction_array < 0.0):
            raise ValueError("joint friction must be non-negative")
        action_scale_array = _float_vector(
            action_scales, size=count, label=f"{source}.action_scales"
        )
        if np.any(action_scale_array < 0.0):
            raise ValueError("action scales must be non-negative")
        if not str(reset_semantics):
            raise ValueError("reset_semantics must be non-empty")
        for name, value in (
            ("action_scale_by_effort_over_kp", action_scale_by_effort_over_kp),
            ("clip_actions", clip_actions),
            ("clip_torques", clip_torques),
            ("latency_enabled", latency_enabled),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean")
        immutable_reset_noise = _freeze_json(dict(reset_noise))
        # Round-trip once here to prove the immutable representation stays JSON-safe.
        json.dumps(_thaw_json(immutable_reset_noise), sort_keys=True, allow_nan=False)
        return cls(
            source=str(source),
            backend=str(backend),
            joint_names=names,
            nominal_q=_float_vector(nominal_q, size=count, label=f"{source}.nominal_q"),
            root_position=_float_vector(root_position, size=3, label=f"{source}.root_position"),
            root_quaternion_xyzw=root_quaternion,
            root_linear_velocity=_float_vector(
                root_linear_velocity, size=3, label=f"{source}.root_linear_velocity"
            ),
            root_angular_velocity=_float_vector(
                root_angular_velocity, size=3, label=f"{source}.root_angular_velocity"
            ),
            kp=kp_array,
            kd=kd_array,
            lower_limits=lower_array,
            upper_limits=upper_array,
            effort_limits=effort_array,
            velocity_limits=velocity_array,
            joint_friction=friction_array,
            action_scales=action_scale_array,
            action_scale_base=base_scale,
            action_scale_by_effort_over_kp=bool(action_scale_by_effort_over_kp),
            control_type=str(control_type),
            clip_actions=bool(clip_actions),
            action_clip_value=clip_value,
            clip_torques=bool(clip_torques),
            simulation_dt=sim_dt,
            control_dt=ctrl_dt,
            decimation=int(decimation),
            latency_enabled=bool(latency_enabled),
            latency_step_range=(latency[0], latency[1]),
            reset_semantics=str(reset_semantics),
            reset_noise=immutable_reset_noise,
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "source": self.source,
            "backend": self.backend,
            "joint_names": list(self.joint_names),
            "nominal_q_rad": self.nominal_q.tolist(),
            "root_position_m": self.root_position.tolist(),
            "root_quaternion_xyzw": self.root_quaternion_xyzw.tolist(),
            "root_linear_velocity_m_s": self.root_linear_velocity.tolist(),
            "root_angular_velocity_rad_s": self.root_angular_velocity.tolist(),
            "kp_nm_per_rad": self.kp.tolist(),
            "kd_nm_s_per_rad": self.kd.tolist(),
            "lower_limits_rad": self.lower_limits.tolist(),
            "upper_limits_rad": self.upper_limits.tolist(),
            "effort_limits_nm": self.effort_limits.tolist(),
            "velocity_limits_rad_s": self.velocity_limits.tolist(),
            "joint_friction_nm": self.joint_friction.tolist(),
            "action_scales_rad_per_unit_action": self.action_scales.tolist(),
            "action_scale_base": self.action_scale_base,
            "action_scale_by_effort_over_kp": self.action_scale_by_effort_over_kp,
            "control_type": self.control_type,
            "clip_actions": self.clip_actions,
            "action_clip_value": self.action_clip_value,
            "clip_torques": self.clip_torques,
            "simulation_dt_s": self.simulation_dt,
            "control_dt_s": self.control_dt,
            "decimation": self.decimation,
            "latency_enabled": self.latency_enabled,
            "latency_step_range": list(self.latency_step_range),
            "reset_semantics": self.reset_semantics,
            "reset_noise": _thaw_json(self.reset_noise),
            "position_target_formula": "q_target = nominal_q + clipped_delayed_action * action_scale",
            "torque_formula": "tau = kp * (q_target - q) - kd * qdot; then effort clip",
        }


def expand_joint_pattern_values(
    joint_names: Sequence[str], patterns: Mapping[str, float], *, label: str
) -> np.ndarray:
    """Expand Holosoma substring-keyed gain dictionaries without ambiguity."""

    result: list[float] = []
    for joint_name in joint_names:
        matches = [(key, value) for key, value in patterns.items() if str(key) in str(joint_name)]
        if len(matches) != 1:
            raise ValueError(
                f"{label}: joint {joint_name!r} matched {len(matches)} patterns: "
                f"{[key for key, _ in matches]}"
            )
        result.append(float(matches[0][1]))
    return np.ascontiguousarray(result, dtype=np.float64)


def compute_action_scales(
    *, base_scale: float, effort_limits: Sequence[float], kp: Sequence[float], by_effort_over_kp: bool
) -> np.ndarray:
    effort = np.asarray(effort_limits, dtype=np.float64)
    gains = np.asarray(kp, dtype=np.float64)
    if effort.ndim != 1 or gains.shape != effort.shape or effort.size == 0:
        raise ValueError("effort_limits and kp must be equal non-empty vectors")
    if not np.isfinite(effort).all() or not np.isfinite(gains).all() or np.any(gains <= 0.0):
        raise ValueError("effort limits and gains must be finite; gains must be positive")
    scale = float(base_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("base_scale must be finite and non-negative")
    values = scale * effort / gains if by_effort_over_kp else np.full_like(gains, scale)
    return np.ascontiguousarray(values, dtype=np.float64)


def _parse_numbers(value: str | None, *, count: int, label: str) -> tuple[float, ...]:
    if value is None:
        raise ValueError(f"missing {label}")
    try:
        result = tuple(float(item) for item in value.split())
    except ValueError as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if len(result) != count or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {count} finite numbers")
    return result


def parse_urdf_joint_model(data: bytes, *, source: str = "urdf") -> JointModelContract:
    """Extract ordered non-fixed URDF joints from the exact supplied bytes."""

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"invalid URDF XML: {exc}") from exc
    names: list[str] = []
    types: list[str] = []
    axes: list[tuple[float, ...]] = []
    lower: list[float] = []
    upper: list[float] = []
    effort: list[float] = []
    velocity: list[float] = []
    for joint in root.findall("joint"):
        joint_type = str(joint.get("type", ""))
        if joint_type == "fixed":
            continue
        name = str(joint.get("name", ""))
        if joint_type not in {"revolute", "continuous"}:
            raise ValueError(f"unsupported URDF joint type for {name!r}: {joint_type!r}")
        if joint_type == "continuous":
            raise ValueError(f"continuous URDF joint {name!r} has no finite G0 sampling range")
        axis = joint.find("axis")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"URDF joint {name!r} is missing limits")
        names.append(name)
        types.append("revolute")
        axes.append(
            _parse_numbers(
                None if axis is None else axis.get("xyz"),
                count=3,
                label=f"URDF axis for {name}",
            )
        )
        lower.append(float(limit.get("lower", "nan")))
        upper.append(float(limit.get("upper", "nan")))
        effort.append(float(limit.get("effort", "nan")))
        velocity.append(float(limit.get("velocity", "nan")))
    return JointModelContract.from_sequences(
        source=source,
        joint_names=names,
        joint_types=types,
        axes=axes,
        lower_limits=lower,
        upper_limits=upper,
        effort_limits=effort,
        velocity_limits=velocity,
    )


def parse_mjcf_joint_model(data: bytes, *, source: str = "mjcf") -> JointModelContract:
    """Extract resolved hinge defaults and motor limits from one robot MJCF buffer.

    The G0 G1 model declares every moving joint with a named default class.  This
    parser resolves nested ``<default>`` joint attributes and explicit overrides,
    while refusing incomplete or unsupported joints instead of guessing compiled
    MuJoCo defaults that could change the contract.
    """

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"invalid MJCF XML: {exc}") from exc

    class_joint_attributes: dict[str, dict[str, str]] = {}

    def visit_default(element: ET.Element, inherited: Mapping[str, str]) -> None:
        resolved = dict(inherited)
        direct_joint = next((child for child in element if child.tag == "joint"), None)
        if direct_joint is not None:
            resolved.update({str(key): str(value) for key, value in direct_joint.attrib.items()})
        class_name = element.get("class")
        if class_name is not None:
            if class_name in class_joint_attributes:
                raise ValueError(f"duplicate MJCF default class {class_name!r}")
            class_joint_attributes[class_name] = dict(resolved)
        for child in element:
            if child.tag == "default":
                visit_default(child, resolved)

    for default in root.findall("default"):
        visit_default(default, {})

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF is missing worldbody")
    names: list[str] = []
    types: list[str] = []
    axes: list[tuple[float, ...]] = []
    lower: list[float] = []
    upper: list[float] = []
    effort: list[float] = []
    friction: list[float] = []
    for joint in worldbody.iter("joint"):
        name = str(joint.get("name", ""))
        joint_type = str(joint.get("type", "hinge"))
        if joint_type == "free":
            continue
        if joint_type != "hinge":
            raise ValueError(f"unsupported MJCF joint type for {name!r}: {joint_type!r}")
        class_name = joint.get("class")
        if class_name is None or class_name not in class_joint_attributes:
            raise ValueError(f"MJCF hinge {name!r} lacks a resolvable default class")
        attributes = dict(class_joint_attributes[class_name])
        attributes.update({str(key): str(value) for key, value in joint.attrib.items()})
        joint_range = _parse_numbers(
            attributes.get("range"), count=2, label=f"MJCF range for {name}"
        )
        force_range = _parse_numbers(
            attributes.get("actuatorfrcrange"),
            count=2,
            label=f"MJCF actuator force range for {name}",
        )
        if abs(force_range[0] + force_range[1]) > JOINT_SCALAR_ATOL:
            raise ValueError(f"MJCF actuator force range for {name!r} is not symmetric")
        names.append(name)
        types.append("revolute")
        axes.append(
            _parse_numbers(attributes.get("axis"), count=3, label=f"MJCF axis for {name}")
        )
        lower.append(joint_range[0])
        upper.append(joint_range[1])
        effort.append(abs(force_range[1]))
        friction_value = float(attributes.get("frictionloss", "nan"))
        if not np.isfinite(friction_value) or friction_value < 0.0:
            raise ValueError(f"invalid MJCF frictionloss for {name!r}")
        friction.append(friction_value)

    motor_by_joint: dict[str, float] = {}
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("MJCF is missing actuators")
    for motor in actuator.findall("motor"):
        joint_name = str(motor.get("joint", ""))
        control_range = _parse_numbers(
            motor.get("ctrlrange"), count=2, label=f"MJCF motor range for {joint_name}"
        )
        if abs(control_range[0] + control_range[1]) > JOINT_SCALAR_ATOL:
            raise ValueError(f"MJCF motor range for {joint_name!r} is not symmetric")
        if joint_name in motor_by_joint:
            raise ValueError(f"multiple MJCF motors target joint {joint_name!r}")
        motor_by_joint[joint_name] = abs(control_range[1])
    missing_motors = [name for name in names if name not in motor_by_joint]
    extras = sorted(set(motor_by_joint).difference(names))
    if missing_motors or extras:
        raise ValueError(f"MJCF motor mapping mismatch: missing={missing_motors}, extra={extras}")
    return JointModelContract.from_sequences(
        source=source,
        joint_names=names,
        joint_types=types,
        axes=axes,
        lower_limits=lower,
        upper_limits=upper,
        effort_limits=effort,
        velocity_limits=None,
        motor_effort_limits=[motor_by_joint[name] for name in names],
        joint_friction=friction,
    )


def canonical_contract_sha256(payload: Mapping[str, object]) -> str:
    """Hash a normalized contract with an explicit domain and canonical JSON."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update((CONTROLLER_CONTRACT_HASH_SCHEMA_VERSION + "\0").encode("ascii"))
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _check(name: str, passed: bool, **details: object) -> dict[str, object]:
    return {"name": name, "status": "pass" if passed else "fail", **details}


def _missing(name: str, **details: object) -> dict[str, object]:
    return {"name": name, "status": "missing", **details}


def _compare_vectors(
    name: str,
    first: np.ndarray,
    second: np.ndarray,
    *,
    joint_names: Sequence[str],
    atol: float,
    units: str,
) -> dict[str, object]:
    if first.shape != second.shape:
        return _check(name, False, first_shape=list(first.shape), second_shape=list(second.shape))
    error = np.abs(first - second)
    flat_index = int(np.argmax(error))
    maximum = float(error.reshape(-1)[flat_index])
    joint_index = flat_index // max(1, int(np.prod(error.shape[1:])))
    if len(joint_names) == first.shape[0] and joint_index < len(joint_names):
        worst_name = str(joint_names[joint_index])
    else:
        worst_name = f"component_{flat_index}"
    return _check(
        name,
        bool(np.all(error <= atol)),
        atol=atol,
        tolerance_semantics="absolute_error_less_than_or_equal",
        max_absolute_error=maximum,
        units=units,
        worst_joint_index=joint_index,
        worst_joint_name=worst_name,
    )


def _compare_recipe_pair(first: ControllerRecipe, second: ControllerRecipe) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    checks.append(
        _check(
            "paired_recipe.joint_names_order",
            first.joint_names == second.joint_names,
            first=list(first.joint_names),
            second=list(second.joint_names),
        )
    )
    for field, units in (
        ("nominal_q", "rad"),
        ("root_position", "m"),
        ("root_quaternion_xyzw", "unitless"),
        ("root_linear_velocity", "m/s"),
        ("root_angular_velocity", "rad/s"),
        ("kp", "N*m/rad"),
        ("kd", "N*m*s/rad"),
        ("lower_limits", "rad"),
        ("upper_limits", "rad"),
        ("effort_limits", "N*m"),
        ("velocity_limits", "rad/s"),
        ("joint_friction", "N*m"),
        ("action_scales", "rad/action"),
    ):
        left = getattr(first, field)
        right = getattr(second, field)
        joint_names = first.joint_names if left.shape[0] == len(first.joint_names) else (field,)
        checks.append(
            _compare_vectors(
                f"paired_recipe.{field}",
                left,
                right,
                joint_names=joint_names,
                atol=JOINT_SCALAR_ATOL,
                units=units,
            )
        )
    scalar_fields = (
        "action_scale_base",
        "action_scale_by_effort_over_kp",
        "control_type",
        "clip_actions",
        "action_clip_value",
        "clip_torques",
        "decimation",
        "latency_enabled",
        "latency_step_range",
        "reset_semantics",
        "reset_noise",
    )
    for field in scalar_fields:
        left = getattr(first, field)
        right = getattr(second, field)
        checks.append(_check(f"paired_recipe.{field}", left == right, first=left, second=right))
    checks.append(
        _check(
            "paired_recipe.simulation_dt",
            abs(first.simulation_dt - second.simulation_dt) <= TIMING_ATOL,
            first_s=first.simulation_dt,
            second_s=second.simulation_dt,
            atol_s=TIMING_ATOL,
        )
    )
    checks.append(
        _check(
            "paired_recipe.control_dt",
            abs(first.control_dt - second.control_dt) <= TIMING_ATOL,
            first_s=first.control_dt,
            second_s=second.control_dt,
            atol_s=TIMING_ATOL,
        )
    )
    return checks


def evaluate_controller_parity(
    *,
    physx_recipe: ControllerRecipe,
    mujoco_recipe: ControllerRecipe,
    urdf: JointModelContract,
    mjcf: JointModelContract,
    source_evidence: Mapping[str, bool | None],
    live_evidence: Mapping[str, bool | None],
) -> dict[str, object]:
    """Evaluate the strict G0 controller contract and fail closed on missing evidence."""

    checks = _compare_recipe_pair(physx_recipe, mujoco_recipe)
    canonical_names = physx_recipe.joint_names
    for model in (urdf, mjcf):
        checks.append(
            _check(
                f"{model.source}.joint_names_order",
                model.joint_names == canonical_names,
                expected=list(canonical_names),
                observed=list(model.joint_names),
            )
        )
        checks.append(
            _check(
                f"{model.source}.joint_types",
                model.joint_types == tuple("revolute" for _ in canonical_names),
                expected=["revolute"] * len(canonical_names),
                observed=list(model.joint_types),
            )
        )

    if urdf.joint_names == mjcf.joint_names:
        checks.append(
            _compare_vectors(
                "asset.axes",
                urdf.axes,
                mjcf.axes,
                joint_names=canonical_names,
                atol=JOINT_AXIS_ATOL,
                units="unitless",
            )
        )
        checks.append(
            _compare_vectors(
                "asset.lower_limits",
                urdf.lower_limits,
                mjcf.lower_limits,
                joint_names=canonical_names,
                atol=JOINT_SCALAR_ATOL,
                units="rad",
            )
        )
        checks.append(
            _compare_vectors(
                "asset.upper_limits",
                urdf.upper_limits,
                mjcf.upper_limits,
                joint_names=canonical_names,
                atol=JOINT_SCALAR_ATOL,
                units="rad",
            )
        )
    else:
        for name in ("asset.axes", "asset.lower_limits", "asset.upper_limits"):
            checks.append(_check(name, False, reason="joint order differs; numerical comparison unsafe"))

    for model in (urdf, mjcf):
        checks.append(
            _compare_vectors(
                f"{model.source}.configured_lower_limits",
                physx_recipe.lower_limits,
                model.lower_limits,
                joint_names=canonical_names,
                atol=JOINT_SCALAR_ATOL,
                units="rad",
            )
        )
        checks.append(
            _compare_vectors(
                f"{model.source}.configured_upper_limits",
                physx_recipe.upper_limits,
                model.upper_limits,
                joint_names=canonical_names,
                atol=JOINT_SCALAR_ATOL,
                units="rad",
            )
        )
        checks.append(
            _compare_vectors(
                f"{model.source}.configured_effort_limits",
                physx_recipe.effort_limits,
                model.effort_limits,
                joint_names=canonical_names,
                atol=JOINT_SCALAR_ATOL,
                units="N*m",
            )
        )
    if mjcf.joint_friction is None:
        checks.append(_missing("mjcf.joint_friction"))
    else:
        checks.append(
            _compare_vectors(
                "mjcf.configured_joint_friction",
                physx_recipe.joint_friction,
                mjcf.joint_friction,
                joint_names=canonical_names,
                atol=JOINT_SCALAR_ATOL,
                units="N*m",
            )
        )
    if mjcf.motor_effort_limits is None:
        checks.append(_missing("mjcf.motor_effort_limits"))
    else:
        checks.append(
            _compare_vectors(
                "mjcf.motor_effort_limits",
                physx_recipe.effort_limits,
                mjcf.motor_effort_limits,
                joint_names=canonical_names,
                atol=JOINT_SCALAR_ATOL,
                units="N*m",
            )
        )
    if urdf.velocity_limits is None:
        checks.append(_missing("urdf.velocity_limits"))
    else:
        checks.append(
            _compare_vectors(
                "urdf.configured_velocity_limits",
                physx_recipe.velocity_limits,
                urdf.velocity_limits,
                joint_names=canonical_names,
                atol=JOINT_SCALAR_ATOL,
                units="rad/s",
            )
        )
    if mjcf.velocity_limits is None:
        checks.append(
            _missing(
                "mjcf.velocity_limits",
                reason="MJCF has no explicit joint velocity-limit field; runtime enforcement is unverified",
            )
        )
    else:
        checks.append(
            _compare_vectors(
                "mjcf.configured_velocity_limits",
                physx_recipe.velocity_limits,
                mjcf.velocity_limits,
                joint_names=canonical_names,
                atol=JOINT_SCALAR_ATOL,
                units="rad/s",
            )
        )

    for model in (urdf, mjcf):
        names_match = model.joint_names == canonical_names
        within = bool(
            names_match
            and np.all(physx_recipe.nominal_q >= model.lower_limits)
            and np.all(physx_recipe.nominal_q <= model.upper_limits)
        )
        checks.append(
            _check(
                f"{model.source}.nominal_q_within_limits",
                within,
                nominal_source=physx_recipe.source,
            )
        )

    for name in REQUIRED_SOURCE_EVIDENCE:
        value = source_evidence.get(name)
        checks.append(
            _missing(f"source_evidence.{name}")
            if value is None
            else _check(f"source_evidence.{name}", bool(value))
        )
    for name in REQUIRED_LIVE_EVIDENCE:
        value = live_evidence.get(name)
        checks.append(
            _missing(f"live_evidence.{name}")
            if value is None
            else _check(f"live_evidence.{name}", bool(value))
        )

    failed = tuple(str(item["name"]) for item in checks if item["status"] == "fail")
    missing = tuple(str(item["name"]) for item in checks if item["status"] == "missing")
    passed = not failed and not missing
    manifest: dict[str, object] = {
        "schema_version": CONTROLLER_PARITY_SCHEMA_VERSION,
        "thresholds": {
            "joint_axis_atol": JOINT_AXIS_ATOL,
            "joint_scalar_atol": JOINT_SCALAR_ATOL,
            "timing_atol_s": TIMING_ATOL,
            "rtol": 0.0,
        },
        "checks": checks,
        "failed_checks": list(failed),
        "missing_checks": list(missing),
        "complete": not missing,
        "g0_controller_contract_pass": passed,
        "g0_pass": passed,
    }
    normalized = {
        "physx_recipe": physx_recipe.to_manifest(),
        "mujoco_recipe": mujoco_recipe.to_manifest(),
        "urdf": urdf.to_manifest(),
        "mjcf": mjcf.to_manifest(),
        "source_evidence": dict(source_evidence),
        "live_evidence": dict(live_evidence),
    }
    manifest["normalized_contract"] = normalized
    manifest["normalized_contract_hash_schema"] = CONTROLLER_CONTRACT_HASH_SCHEMA_VERSION
    manifest["normalized_contract_sha256"] = canonical_contract_sha256(normalized)
    return manifest
