from dataclasses import replace
import math

import pytest

from snmr.verification import (
    BackendIdentity,
    FailureInterval,
    REPORT_SCHEMA_VERSION,
    VerificationReport,
    compare_solver_reports,
    localize_threshold_failures,
    offset_failure_intervals,
    rollout_contract_hash,
)


DIGEST = "0" * 64


def make_report(*, engine: str, passed: bool, intervals=(), metrics=()):
    return VerificationReport(
        schema_version=REPORT_SCHEMA_VERSION,
        candidate_id="clip:torque_x1",
        motion_sha256=DIGEST,
        robot_spec_hash=DIGEST,
        robot_kinematic_hash=DIGEST,
        robot_dynamics_hash=DIGEST,
        verification_level="L2",
        passed=passed,
        fps=50.0,
        num_frames=20,
        backend=BackendIdentity(engine=engine, solver=engine, engine_version="1"),
        metrics=tuple(metrics),
        failure_intervals=tuple(intervals),
        controller_hash=DIGEST,
        config_hash=DIGEST,
        seed=0,
    )


def test_localizer_filters_short_runs_and_merges_small_gaps():
    intervals = localize_threshold_failures(
        [0.0, 0.2, 0.3, 0.0, 0.4, 0.5, 0.0, 0.9, 0.0],
        threshold=0.1,
        failure_type="torque_deficit",
        min_frames=2,
        merge_gap_frames=1,
        joints=("knee",),
    )
    assert len(intervals) == 1
    assert (intervals[0].start_frame, intervals[0].end_frame) == (1, 5)
    assert intervals[0].severity == pytest.approx(0.4)
    assert intervals[0].joints == ("knee",)


def test_cross_solver_comparison_preserves_disagreement():
    left = make_report(
        engine="physx",
        passed=True,
        metrics=(("survival_fraction", 1.0), ("mean_dof_error_rad", 0.1)),
    )
    right = make_report(
        engine="newton",
        passed=False,
        intervals=(FailureInterval(10, 12, "fall", 1.0),),
        metrics=(("survival_fraction", 0.5), ("mean_dof_error_rad", 0.2)),
    )
    comparison = compare_solver_reports(left, right)
    assert VerificationReport.from_dict(left.to_dict()) == left
    assert comparison.same_candidate_contract
    assert not comparison.pass_agreement
    assert comparison.failure_type_jaccard == 0.0
    assert dict(comparison.metric_deltas) == pytest.approx({
        "survival_fraction": -0.5,
        "mean_dof_error_rad": 0.1,
    })


def test_report_rejects_overlapping_same_type_intervals():
    report = make_report(
        engine="newton",
        passed=False,
        intervals=(
            FailureInterval(1, 4, "torque", 0.2),
            FailureInterval(4, 6, "torque", 0.3),
        ),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        report.validate()


@pytest.mark.parametrize("field", [
    "motion_sha256",
    "robot_spec_hash",
    "robot_kinematic_hash",
    "robot_dynamics_hash",
    "controller_hash",
    "config_hash",
])
def test_report_rejects_nonhexadecimal_digests(field):
    report = make_report(engine="mujoco", passed=True)
    payload = report.to_dict()
    payload[field] = "z" * 64
    with pytest.raises(ValueError, match="hexadecimal"):
        VerificationReport.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("passed", "false", TypeError),
        ("overflow_detected", 0, TypeError),
        ("fps", math.nan, ValueError),
        ("fps", math.inf, ValueError),
        ("num_frames", 2.5, TypeError),
        ("seed", True, TypeError),
    ],
)
def test_report_deserialization_rejects_coercion_and_nonfinite_values(field, value, error):
    payload = make_report(engine="mujoco", passed=True).to_dict()
    payload[field] = value
    with pytest.raises(error):
        VerificationReport.from_dict(payload)


def test_report_rejects_invalid_backend_identity():
    report = make_report(engine="mujoco", passed=True)
    with pytest.raises(ValueError, match="quaternion_convention"):
        replace(report, backend=replace(report.backend, quaternion_convention="scalar_last")).validate()
    with pytest.raises(ValueError, match="non-empty"):
        replace(report, backend=replace(report.backend, solver="")).validate()


def test_report_deserialization_rejects_fractional_interval_and_string_metric():
    report = make_report(
        engine="mujoco",
        passed=False,
        intervals=(FailureInterval(1, 2, "fall", 1.0),),
        metrics=(("survival_fraction", 0.5),),
    )
    payload = report.to_dict()
    payload["failure_intervals"][0]["start_frame"] = 1.5
    with pytest.raises(TypeError, match="integer"):
        VerificationReport.from_dict(payload)

    payload = report.to_dict()
    payload["metrics"] = [("survival_fraction", "0.5")]
    with pytest.raises(TypeError, match="JSON number"):
        VerificationReport.from_dict(payload)


@pytest.mark.parametrize(
    "changed",
    [
        {"config_hash": "1" * 64},
        {"fps": 60.0},
        {"num_frames": 21},
        {"seed": 1},
    ],
)
def test_cross_solver_contract_rejects_evaluation_mismatch(changed):
    left = make_report(engine="physx", passed=True)
    right = replace(make_report(engine="newton", passed=True), **changed)
    assert not compare_solver_reports(left, right).same_candidate_contract


def test_failure_interval_offset_translates_window_coordinates():
    intervals = offset_failure_intervals(
        (FailureInterval(1, 3, "saturation", 0.5),),
        17,
    )
    assert (intervals[0].start_frame, intervals[0].end_frame) == (18, 20)
    with pytest.raises(ValueError, match="nonnegative"):
        offset_failure_intervals(intervals, -1)


def test_rollout_contract_hash_is_backend_independent_and_parameter_sensitive():
    values = dict(
        seconds=1.0,
        start_frame=0,
        fall_root_z_m=0.35,
        root_xy_limit_m=0.5,
        tilt_limit_rad=math.radians(60.0),
        saturation_threshold=0.05,
        min_failure_frames=2,
        merge_gap_frames=1,
    )
    first = rollout_contract_hash(**values)
    assert first == rollout_contract_hash(**values)
    assert first != rollout_contract_hash(**{**values, "start_frame": 1})
    with pytest.raises(ValueError, match="saturation_threshold"):
        rollout_contract_hash(**{**values, "saturation_threshold": 1.1})
