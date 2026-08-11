#!/usr/bin/env python
"""Bind repeated E70 loopback qualifications into one fail-closed summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


PROTOCOL = "E70 repeated loopback qualification summary v1"
MINIMUM_REPEATS = 3


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_reports(paths: list[pathlib.Path]) -> dict[str, Any]:
    if len(paths) < MINIMUM_REPEATS:
        raise ValueError(f"at least {MINIMUM_REPEATS} loopback reports are required")
    reports = [json.loads(path.read_text()) for path in paths]
    candidate_hashes = {report.get("onnx_sha256") for report in reports}
    variants = {report.get("runtime_variant") for report in reports}
    thresholds = {json.dumps(report.get("thresholds"), sort_keys=True) for report in reports}
    phase_names = [tuple(report.get("phases", {})) for report in reports]
    if len(candidate_hashes) != 1 or None in candidate_hashes:
        raise ValueError("repeat reports do not identify one ONNX candidate")
    if variants != {"production WBT with default safety-policy handoff"}:
        raise ValueError("repeat reports are not the registered safety-handoff variant")
    if len(thresholds) != 1:
        raise ValueError("loopback thresholds differ across repeats")
    if len(set(phase_names)) != 1 or not phase_names[0]:
        raise ValueError("loopback phases differ across repeats")

    phase_summary: dict[str, Any] = {}
    for phase in phase_names[0]:
        values = [report["phases"][phase] for report in reports]
        phase_summary[phase] = {
            "minimum_base_height_m": min(v["minimum_base_height_m"] for v in values),
            "minimum_up_axis_z": min(v["minimum_up_axis_z"] for v in values),
            "minimum_measured_joint_limit_margin_rad": min(
                v["minimum_measured_joint_limit_margin_rad"] for v in values
            ),
            "minimum_commanded_joint_limit_margin_rad": min(
                v["minimum_commanded_joint_limit_margin_rad"] for v in values
            ),
            "maximum_abs_joint_velocity_rad_s": max(
                v["maximum_abs_joint_velocity_rad_s"] for v in values
            ),
            "maximum_abs_torque_nm": max(v["maximum_abs_torque_nm"] for v in values),
            "total_nonfinite_samples": sum(v["nonfinite_samples"] for v in values),
            "total_measured_joint_limit_violation_samples": sum(
                v["measured_joint_limit_violation_samples"] for v in values
            ),
            "total_commanded_joint_limit_violation_samples": sum(
                v["commanded_joint_limit_violation_samples"] for v in values
            ),
            "total_velocity_limit_violation_samples": sum(
                v["velocity_limit_violation_samples"] for v in values
            ),
            "total_torque_limit_violation_samples": sum(
                v["torque_limit_violation_samples"] for v in values
            ),
        }

    report_entries = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "passes": bool(report.get("passes")),
            "physical_robot_commands_sent": report.get(
                "physical_robot_commands_sent"
            ),
        }
        for path, report in zip(paths, reports, strict=True)
    ]
    passes = all(
        entry["passes"] and entry["physical_robot_commands_sent"] == 0
        for entry in report_entries
    )
    return {
        "protocol": PROTOCOL,
        "required_repeats": MINIMUM_REPEATS,
        "observed_repeats": len(reports),
        "candidate_sha256": next(iter(candidate_hashes)),
        "runtime_variant": next(iter(variants)),
        "thresholds": reports[0]["thresholds"],
        "reports": report_entries,
        "phase_worst_case": phase_summary,
        "passes": passes,
        "interpretation": (
            "pre-hardware production loopback handoff gate passes"
            if passes
            else "pre-hardware production loopback handoff gate fails"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    summary = summarize_reports(args.reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"passes": summary["passes"], "repeats": len(args.reports)}))
    raise SystemExit(0 if summary["passes"] else 3)


if __name__ == "__main__":
    main()
