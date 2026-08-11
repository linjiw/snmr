import json

import pytest

from scripts.summarize_e70_loopback_repeats import summarize_reports


def _report(path, *, candidate="abc", passes=True, height=0.75):
    report = {
        "onnx_sha256": candidate,
        "runtime_variant": "production WBT with default safety-policy handoff",
        "thresholds": {"minimum_base_height_m": 0.45},
        "physical_robot_commands_sent": 0,
        "passes": passes,
        "phases": {
            "safe_hold": {
                "minimum_base_height_m": height,
                "minimum_up_axis_z": 0.9,
                "minimum_measured_joint_limit_margin_rad": 0.1,
                "minimum_commanded_joint_limit_margin_rad": 0.2,
                "maximum_abs_joint_velocity_rad_s": 3.0,
                "maximum_abs_torque_nm": 20.0,
                "nonfinite_samples": 0,
                "measured_joint_limit_violation_samples": 0,
                "commanded_joint_limit_violation_samples": 0,
                "velocity_limit_violation_samples": 0,
                "torque_limit_violation_samples": 0,
            }
        },
    }
    path.write_text(json.dumps(report))
    return path


def test_summary_requires_three_matching_passing_reports(tmp_path):
    paths = [_report(tmp_path / f"report_{index}.json") for index in range(3)]
    summary = summarize_reports(paths)
    assert summary["passes"]
    assert summary["observed_repeats"] == 3
    assert summary["phase_worst_case"]["safe_hold"]["minimum_base_height_m"] == 0.75


def test_summary_fails_closed_on_too_few_or_mismatched_candidates(tmp_path):
    paths = [_report(tmp_path / f"report_{index}.json") for index in range(2)]
    with pytest.raises(ValueError, match="at least 3"):
        summarize_reports(paths)
    paths.append(_report(tmp_path / "report_2.json", candidate="different"))
    with pytest.raises(ValueError, match="one ONNX"):
        summarize_reports(paths)
