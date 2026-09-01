"""Tests for the simulator-independent G0 controller parity contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from snmr.controller_parity import (
    CONTROLLER_CONTRACT_HASH_SCHEMA_VERSION,
    JOINT_SCALAR_ATOL,
    REQUIRED_LIVE_EVIDENCE,
    REQUIRED_SOURCE_EVIDENCE,
    ControllerRecipe,
    JointModelContract,
    canonical_contract_sha256,
    compute_action_scales,
    evaluate_controller_parity,
    expand_joint_pattern_values,
    parse_mjcf_joint_model,
    parse_urdf_joint_model,
)


NAMES = ("hip_joint", "knee_joint")
AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _model(
    source: str,
    *,
    lower: tuple[float, float] = (-1.0, -0.5),
    motor: tuple[float, float] | None = (10.0, 20.0),
    velocity: tuple[float, float] | None = (5.0, 6.0),
    friction: tuple[float, float] | None = (0.0, 0.0),
) -> JointModelContract:
    return JointModelContract.from_sequences(
        source=source,
        joint_names=NAMES,
        joint_types=("revolute", "revolute"),
        axes=AXES,
        lower_limits=lower,
        upper_limits=(1.0, 1.5),
        effort_limits=(10.0, 20.0),
        velocity_limits=velocity,
        motor_effort_limits=motor,
        joint_friction=friction,
    )


def _recipe(
    source: str,
    backend: str,
    *,
    action_scales: tuple[float, float] = (0.25, 0.25),
) -> ControllerRecipe:
    return ControllerRecipe.from_sequences(
        source=source,
        backend=backend,
        joint_names=NAMES,
        nominal_q=(0.0, 0.2),
        root_position=(0.0, 0.0, 0.76),
        root_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        root_linear_velocity=(0.0, 0.0, 0.0),
        root_angular_velocity=(0.0, 0.0, 0.0),
        kp=(40.0, 80.0),
        kd=(2.0, 4.0),
        lower_limits=(-1.0, -0.5),
        upper_limits=(1.0, 1.5),
        effort_limits=(10.0, 20.0),
        velocity_limits=(5.0, 6.0),
        joint_friction=(0.0, 0.0),
        action_scales=action_scales,
        action_scale_base=0.25,
        action_scale_by_effort_over_kp=False,
        control_type="P",
        clip_actions=True,
        action_clip_value=100.0,
        clip_torques=True,
        simulation_dt=0.005,
        control_dt=0.02,
        decimation=4,
        latency_enabled=False,
        latency_step_range=(0, 1),
        reset_semantics="motion_reference_plus_noise",
        reset_noise={"dof_pos": 0.1, "root_pos": [0.05, 0.05, 0.01]},
    )


def _evidence(keys: tuple[str, ...], value: bool | None = True) -> dict[str, bool | None]:
    return {key: value for key in keys}


def _evaluate(
    *,
    physx: ControllerRecipe | None = None,
    mujoco: ControllerRecipe | None = None,
    urdf: JointModelContract | None = None,
    mjcf: JointModelContract | None = None,
    source_evidence: dict[str, bool | None] | None = None,
    live_evidence: dict[str, bool | None] | None = None,
) -> dict[str, object]:
    return evaluate_controller_parity(
        physx_recipe=physx or _recipe("physx", "isaac"),
        mujoco_recipe=mujoco or _recipe("mujoco", "mujoco"),
        urdf=urdf or _model("urdf", motor=None),
        mjcf=mjcf or _model("mjcf"),
        source_evidence=source_evidence or _evidence(REQUIRED_SOURCE_EVIDENCE),
        live_evidence=live_evidence or _evidence(REQUIRED_LIVE_EVIDENCE),
    )


def _check(report: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in report["checks"] if item["name"] == name)


def test_complete_equal_contract_passes_only_with_all_live_evidence() -> None:
    report = _evaluate()

    assert report["complete"] is True
    assert report["g0_controller_contract_pass"] is True
    assert report["g0_pass"] is True
    assert report["failed_checks"] == []
    assert report["missing_checks"] == []
    assert len(report["normalized_contract_sha256"]) == 64


def test_motor_effort_mismatch_is_a_named_hard_failure() -> None:
    report = _evaluate(mjcf=_model("mjcf", motor=(8.0, 20.0)))

    assert report["complete"] is True
    assert report["g0_controller_contract_pass"] is False
    assert "mjcf.motor_effort_limits" in report["failed_checks"]
    check = _check(report, "mjcf.motor_effort_limits")
    assert check["max_absolute_error"] == pytest.approx(2.0)
    assert check["worst_joint_name"] == "hip_joint"


def test_asset_joint_friction_mismatch_is_a_named_hard_failure() -> None:
    report = _evaluate(mjcf=_model("mjcf", friction=(0.1, 0.1)))

    assert "mjcf.configured_joint_friction" in report["failed_checks"]
    check = _check(report, "mjcf.configured_joint_friction")
    assert check["max_absolute_error"] == pytest.approx(0.1)
    assert report["g0_pass"] is False


def test_missing_live_observation_fails_closed_without_becoming_failure() -> None:
    live = _evidence(REQUIRED_LIVE_EVIDENCE)
    live["paired_reset_same_motion_seed"] = None
    report = _evaluate(live_evidence=live)

    assert report["complete"] is False
    assert report["failed_checks"] == []
    assert "live_evidence.paired_reset_same_motion_seed" in report["missing_checks"]
    assert report["g0_pass"] is False


def test_joint_limit_comparison_uses_frozen_strict_absolute_tolerance() -> None:
    passing = _evaluate(mjcf=_model("mjcf", lower=(-1.0 + JOINT_SCALAR_ATOL, -0.5)))
    failing = _evaluate(mjcf=_model("mjcf", lower=(-1.0 + 2.0 * JOINT_SCALAR_ATOL, -0.5)))

    assert _check(passing, "asset.lower_limits")["status"] == "pass"
    assert _check(failing, "asset.lower_limits")["status"] == "fail"


def test_pair_action_scale_drift_is_not_hidden_by_shared_formula() -> None:
    report = _evaluate(mujoco=_recipe("mujoco", "mujoco", action_scales=(0.1, 0.2)))

    assert _check(report, "paired_recipe.action_scales")["status"] == "fail"
    assert report["g0_controller_contract_pass"] is False


def test_pattern_gain_expansion_and_effort_over_kp_action_scale() -> None:
    kp = expand_joint_pattern_values(
        ("left_hip_pitch_joint", "left_knee_joint"),
        {"hip_pitch": 40.0, "knee": 100.0},
        label="kp",
    )
    scales = compute_action_scales(
        base_scale=0.25,
        effort_limits=(80.0, 100.0),
        kp=kp,
        by_effort_over_kp=True,
    )
    assert kp.tolist() == [40.0, 100.0]
    assert scales.tolist() == pytest.approx([0.5, 0.25])

    with pytest.raises(ValueError, match="matched 0 patterns"):
        expand_joint_pattern_values(("ankle",), {"hip": 1.0}, label="kp")
    with pytest.raises(ValueError, match="matched 2 patterns"):
        expand_joint_pattern_values(("hip_pitch",), {"hip": 1.0, "hip_pitch": 2.0}, label="kp")


def test_urdf_and_nested_mjcf_default_parsers_preserve_order_and_limits() -> None:
    urdf_bytes = b"""
    <robot name="synthetic">
      <link name="base"/><link name="hip"/><link name="knee"/>
      <joint name="fixed_mount" type="fixed"><parent link="base"/><child link="hip"/></joint>
      <joint name="hip_joint" type="revolute">
        <parent link="base"/><child link="hip"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" effort="10" velocity="5"/>
      </joint>
      <joint name="knee_joint" type="revolute">
        <parent link="hip"/><child link="knee"/><axis xyz="0 1 0"/>
        <limit lower="-0.5" upper="1.5" effort="20" velocity="6"/>
      </joint>
    </robot>
    """
    mjcf_bytes = b"""
    <mujoco model="synthetic">
      <default><default class="robot"><joint actuatorfrcrange="-10 10" frictionloss="0"/>
        <default class="hip"><joint axis="1 0 0" range="-1 1"/></default>
        <default class="knee"><joint axis="0 1 0" range="-0.5 1.5" actuatorfrcrange="-20 20"/></default>
      </default></default>
      <worldbody><body name="base">
        <joint name="hip_joint" class="hip"/>
        <body name="leg"><joint name="knee_joint" class="knee"/></body>
      </body></worldbody>
      <actuator>
        <motor name="hip" joint="hip_joint" ctrlrange="-10 10"/>
        <motor name="knee" joint="knee_joint" ctrlrange="-20 20"/>
      </actuator>
    </mujoco>
    """

    urdf = parse_urdf_joint_model(urdf_bytes)
    mjcf = parse_mjcf_joint_model(mjcf_bytes)

    assert urdf.joint_names == NAMES
    assert mjcf.joint_names == NAMES
    assert np.array_equal(urdf.axes, mjcf.axes)
    assert np.array_equal(urdf.lower_limits, mjcf.lower_limits)
    assert mjcf.effort_limits.tolist() == [10.0, 20.0]
    assert mjcf.motor_effort_limits.tolist() == [10.0, 20.0]
    assert mjcf.joint_friction.tolist() == [0.0, 0.0]
    assert mjcf.velocity_limits is None


def test_contract_hash_is_domain_separated_deterministic_and_content_bound() -> None:
    payload = {"a": 1, "nested": {"b": [2.0, 3.0]}}
    first = canonical_contract_sha256(payload)
    reordered = canonical_contract_sha256({"nested": {"b": [2.0, 3.0]}, "a": 1})
    changed = canonical_contract_sha256({"a": 1, "nested": {"b": [2.0, 3.1]}})

    assert CONTROLLER_CONTRACT_HASH_SCHEMA_VERSION == "snmr.controller-contract.v0.1"
    assert first == reordered
    assert first != changed
    assert len(first) == 64


def test_controller_contract_arrays_and_nested_reset_metadata_are_immutable() -> None:
    recipe = _recipe("physx", "isaac")
    assert recipe.nominal_q.flags.writeable is False
    with pytest.raises(ValueError):
        recipe.nominal_q[0] = 1.0
    with pytest.raises(TypeError):
        recipe.reset_noise[0] = ("tampered", True)  # type: ignore[index]
    assert recipe.to_manifest()["reset_noise"] == {
        "dof_pos": 0.1,
        "root_pos": [0.05, 0.05, 0.01],
    }


def _load_report_script():
    path = Path(__file__).resolve().parents[1] / "scripts/g0_controller_parity.py"
    spec = importlib.util.spec_from_file_location("snmr_test_g0_controller_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_writer_refuses_overwrite(tmp_path: Path) -> None:
    module = _load_report_script()
    output = tmp_path / "report.json"
    module._write_json_exclusive(output, {"status": "first"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}

    with pytest.raises(FileExistsError):
        module._write_json_exclusive(output, {"status": "second"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}


def test_mujoco_nominal_source_binding_requires_write_after_reset() -> None:
    module = _load_report_script()
    reset = b"mujoco.mj_resetData(self.root_model, self.root_data)"
    nominal = b"self._set_initial_joint_angles()"

    assert module._mujoco_nominal_source_binding(nominal + b"\n" + reset) is False
    assert module._mujoco_nominal_source_binding(reset + b"\n" + nominal) is True
    assert module._mujoco_nominal_source_binding(reset) is False
