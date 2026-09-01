#!/usr/bin/env python
"""Falsifiable RobotSpec/dynamics-twin pilot using the local MuJoCo verifier.

This is deliberately not an RL experiment.  It establishes the prerequisite that one
fixed motion and one fixed kinematic tree produce different, localized physics evidence
when only actuator strength changes.  A retargeter trained solely on GMR labels cannot
learn this intervention because the target trajectory is identical across twins.

The generated JSON uses ``snmr.physics-verification.v0.1`` and can be compared with a
Newton/MJWarp or Isaac Lab/PhysX report for the same candidate.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.paths import g1_mjcf, holosoma_sample_npz  # noqa: E402
from snmr.provenance import (  # noqa: E402
    ArtifactSnapshot,
    MjcfBundleSnapshot,
    source_revision_manifest,
)
from snmr.robot_spec import ControlSpec, RobotSpec, SemanticManifest  # noqa: E402
from snmr.verification import (  # noqa: E402
    BackendIdentity,
    FailureInterval,
    REPORT_SCHEMA_VERSION,
    VerificationReport,
    canonical_hash,
    localize_threshold_failures,
    offset_failure_intervals,
    rollout_contract_hash,
)
from trackability_proxy import (  # noqa: E402
    FALL_ROOT_Z_M,
    HOLOSOMA_G1_DAMPING,
    HOLOSOMA_G1_EFFORT,
    HOLOSOMA_G1_STIFFNESS,
    ROOT_XY_DEV_M,
    TILT_LIMIT_RAD,
    _match_gain,
    replay,
)


def _exact_joint_values(names: list[str], table: dict[str, float]) -> dict[str, float]:
    return {name: float(_match_gain(name, table)) for name in names}


def _reference_pd_demand(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    limits: np.ndarray,
) -> dict[str, object]:
    """Controller-independent demand on the fixed reference, without rollout state drift."""

    requested = kp[None, :] * (joint_pos[1:] - joint_pos[:-1]) - kd[None, :] * joint_vel[:-1]
    ratio = np.abs(requested) / limits[None, :]
    return {
        "saturation_per_frame": (ratio > 1.0).mean(axis=1),
        "max_ratio_per_frame": ratio.max(axis=1),
        "saturation_fraction": float((ratio > 1.0).mean()),
        "max_requested_ratio": float(ratio.max(initial=0.0)),
    }


def _finite_metrics(values: dict[str, float | None]) -> tuple[tuple[str, float], ...]:
    metrics: list[tuple[str, float]] = []
    for name, value in values.items():
        if value is None:
            continue
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite verification metric {name}={value}")
        metrics.append((name, parsed))
    return tuple(sorted(metrics))


def run(args: argparse.Namespace) -> dict[str, object]:
    """Capture inputs in this worker, then run only against those frozen buffers."""

    motion_path = Path(args.npz).expanduser().resolve()
    mjcf_path = Path(args.mjcf).expanduser().resolve()
    motion_snapshot = ArtifactSnapshot.capture(motion_path)
    asset_snapshot = MjcfBundleSnapshot.capture(mjcf_path)
    revisions = source_revision_manifest(
        snmr_path=ROOT,
        newton_path=getattr(args, "newton_root", None),
        isaac_lab_path=getattr(args, "isaac_lab_root", None),
    )
    with tempfile.TemporaryDirectory(prefix="snmr-dynamics-twins-") as temporary:
        runtime_root = Path(temporary)
        runtime_motion = motion_snapshot.materialize(runtime_root / "motion" / motion_path.name)
        runtime_mjcf = asset_snapshot.materialize(runtime_root / "robot-asset")
        return _run_captured(
            args,
            motion_path=motion_path,
            mjcf_path=mjcf_path,
            runtime_motion=runtime_motion,
            runtime_mjcf=runtime_mjcf,
            motion_snapshot=motion_snapshot,
            asset_snapshot=asset_snapshot,
            revisions=revisions,
        )


def _run_captured(
    args: argparse.Namespace,
    *,
    motion_path: Path,
    mjcf_path: Path,
    runtime_motion: Path,
    runtime_mjcf: Path,
    motion_snapshot: ArtifactSnapshot,
    asset_snapshot: MjcfBundleSnapshot,
    revisions: dict[str, object],
) -> dict[str, object]:
    # NumPy consumes the exact in-memory buffer that was hashed above.  Path-only MuJoCo
    # APIs consume private materializations of the same captured buffers.
    data = np.load(io.BytesIO(motion_snapshot.data), allow_pickle=True)
    joint_names = [str(name) for name in data["joint_names"].tolist()]
    kp_by_name = _exact_joint_values(joint_names, HOLOSOMA_G1_STIFFNESS)
    kd_by_name = _exact_joint_values(joint_names, HOLOSOMA_G1_DAMPING)
    effort_by_name = _exact_joint_values(joint_names, HOLOSOMA_G1_EFFORT)

    spec = RobotSpec.from_mjcf(
        runtime_mjcf,
        SemanticManifest(
            root_link="pelvis",
            torso_link="torso_link",
            left_hand_link="left_rubber_hand_link",
            right_hand_link="right_rubber_hand_link",
            left_foot_link="left_ankle_roll_link",
            right_foot_link="right_ankle_roll_link",
            allowed_contact_links=("left_ankle_roll_link", "right_ankle_roll_link"),
            symmetry_pairs=(
                ("left_ankle_roll_link", "right_ankle_roll_link"),
                ("left_rubber_hand_link", "right_rubber_hand_link"),
            ),
        ),
        control=ControlSpec(control_dt=1.0 / args.control_hz, latency_seconds=0.0),
        torque_limits=effort_by_name,
        kp=kp_by_name,
        kd=kd_by_name,
    )
    if spec.asset_sha256 != asset_snapshot.entrypoint_snapshot.sha256:
        raise ValueError("RobotSpec parsed bytes do not match the worker's MJCF snapshot")
    if [joint.name for joint in spec.joints] != joint_names:
        raise ValueError("RobotSpec joint order does not match the motion contract")

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)[:, 7:]
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)[:, 6:]
    kp = np.asarray([kp_by_name[name] for name in joint_names])
    kd = np.asarray([kd_by_name[name] for name in joint_names])
    base_limits = np.asarray([effort_by_name[name] for name in joint_names])
    scales = sorted(set(float(value) for value in args.effort_scales.split(",")))
    if not scales or any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("--effort-scales must contain positive finite values")

    motion_hash = motion_snapshot.sha256
    candidate_config_hash = rollout_contract_hash(
        seconds=args.seconds,
        start_frame=args.start_frame,
        fall_root_z_m=FALL_ROOT_Z_M,
        root_xy_limit_m=ROOT_XY_DEV_M,
        tilt_limit_rad=TILT_LIMIT_RAD,
        saturation_threshold=args.saturation_threshold,
        min_failure_frames=args.min_failure_frames,
        merge_gap_frames=args.merge_gap_frames,
    )
    backend_config_hash = canonical_hash({
        "control_hz": args.control_hz,
        "torque_ratio_threshold": args.torque_ratio_threshold,
        "simulation_dt": spec.runtime.simulation_dt,
    })
    controller_hash = canonical_hash({"kp": kp.tolist(), "kd": kd.tolist()})
    reports: dict[str, object] = {}
    twins: dict[float, RobotSpec] = {}
    demand_by_scale: dict[float, dict[str, object]] = {}
    rollout_by_scale: dict[float, dict[str, object]] = {}

    for scale in scales:
        twin = spec.dynamics_twin(torque_scale=scale)
        twins[scale] = twin
        demand = _reference_pd_demand(joint_pos, joint_vel, kp, kd, base_limits * scale)
        demand_by_scale[scale] = demand

        started = time.perf_counter()
        rollout = replay(
            str(runtime_motion),
            str(runtime_mjcf),
            seconds_max=args.seconds,
            control_hz=args.control_hz,
            start_frame=args.start_frame,
            effort_scale=scale,
            return_trace=True,
        )
        rollout_by_scale[scale] = rollout
        elapsed = time.perf_counter() - started

        intervals = list(localize_threshold_failures(
            demand["max_ratio_per_frame"],
            threshold=args.torque_ratio_threshold,
            failure_type="reference_low_torque_margin",
            min_frames=args.min_failure_frames,
            merge_gap_frames=args.merge_gap_frames,
            joints=tuple(joint_names),
        ))
        rollout_trace = rollout["trace"]
        intervals.extend(offset_failure_intervals(localize_threshold_failures(
            rollout_trace["torque_saturation_fraction"],
            threshold=args.saturation_threshold,
            failure_type="rollout_torque_saturation",
            min_frames=args.min_failure_frames,
            merge_gap_frames=args.merge_gap_frames,
            joints=tuple(joint_names),
        ), args.start_frame))
        if rollout["diverged"]:
            failure_frame = rollout.get("failure_frame")
            if failure_frame is None:
                failure_frame = args.start_frame + max(len(rollout_trace["root_height_m"]) - 1, 0)
            intervals.append(FailureInterval(
                start_frame=int(failure_frame),
                end_frame=int(failure_frame),
                failure_type="rollout_divergence",
                severity=1.0,
                evidence=(("survival_time_s", float(rollout["survival_time_s"])),),
            ))

        report = VerificationReport(
            schema_version=REPORT_SCHEMA_VERSION,
            candidate_id=f"{motion_path.stem}:torque_x{scale:g}",
            motion_sha256=motion_hash,
            robot_spec_hash=twin.spec_hash,
            robot_kinematic_hash=twin.kinematic_hash,
            robot_dynamics_hash=twin.dynamics_hash,
            verification_level="L1_reference_demand+L2_open_loop_pd",
            passed=not bool(rollout["diverged"]),
            fps=float(data["fps"].reshape(-1)[0]),
            num_frames=int(joint_pos.shape[0]),
            backend=BackendIdentity(
                engine="mujoco",
                solver="mujoco_cpu",
                engine_version=mujoco.__version__,
            ),
            metrics=_finite_metrics({
                "reference_torque_saturation_fraction": demand["saturation_fraction"],
                "reference_max_requested_torque_ratio": demand["max_requested_ratio"],
                "survival_fraction": rollout["survived_fraction"],
                "survival_time_s": rollout["survival_time_s"],
                "mean_dof_error_rad": (
                    rollout["mean_dof_err_rad"]
                    if math.isfinite(float(rollout["mean_dof_err_rad"])) else None
                ),
                "mean_root_height_error_m": (
                    rollout["mean_root_height_err_m"]
                    if math.isfinite(float(rollout["mean_root_height_err_m"])) else None
                ),
                "torque_saturation_fraction": rollout["torque_saturation_fraction"],
                "max_requested_torque_ratio": rollout["max_requested_torque_ratio"],
                "wall_time_s": elapsed,
            }),
            failure_intervals=tuple(sorted(
                intervals,
                key=lambda item: (item.start_frame, item.failure_type),
            )),
            controller_hash=controller_hash,
            seed=0,
            config_hash=candidate_config_hash,
        )
        report.validate()
        report_payload = report.to_dict()
        report_payload.update(revisions)
        report_payload.update({
            "asset_sha256": asset_snapshot.entrypoint_snapshot.sha256,
            "asset_bundle_sha256": asset_snapshot.sha256,
            "backend_config_hash": backend_config_hash,
        })
        reports[f"{scale:g}"] = report_payload

    low, high = scales[0], scales[-1]
    low_features = twins[low].model_features().node_features
    high_features = twins[high].model_features().node_features
    max_ratio_low = float(demand_by_scale[low]["max_requested_ratio"])
    max_ratio_high = float(demand_by_scale[high]["max_requested_ratio"])
    gates = {
        "same_kinematic_hash": len({twin.kinematic_hash for twin in twins.values()}) == 1,
        "distinct_dynamics_hashes": len({twin.dynamics_hash for twin in twins.values()}) == len(twins),
        "identity_free_features_change_with_torque": bool(
            np.max(np.abs(low_features - high_features)) > 1e-8
        ),
        "reference_demand_responds_monotonically": bool(
            all(
                float(demand_by_scale[a]["max_requested_ratio"])
                > float(demand_by_scale[b]["max_requested_ratio"])
                for a, b in zip(scales, scales[1:])
            )
        ),
        "inverse_scale_ratio_matches": bool(np.isclose(
            max_ratio_low / max_ratio_high,
            high / low,
            rtol=1e-10,
            atol=1e-10,
        )),
        "localized_failure_emitted": any(
            interval["failure_type"] == "reference_low_torque_margin"
            for interval in reports[f"{low:g}"]["failure_intervals"]
        ),
    }
    result = {
        "experiment": "dynamics_twins_v0.1",
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
        },
        "motion": str(motion_path),
        "motion_sha256": motion_hash,
        "mjcf": str(mjcf_path),
        "asset_sha256": asset_snapshot.entrypoint_snapshot.sha256,
        "asset_bundle_sha256": asset_snapshot.sha256,
        "controller_hash": controller_hash,
        "config_hash": candidate_config_hash,
        "backend_config_hash": backend_config_hash,
        "input_artifacts": {
            "motion": motion_snapshot.manifest(),
            "robot_asset": asset_snapshot.manifest(),
        },
        **revisions,
        "base_spec": spec.to_dict(),
        "scales": scales,
        "reports": reports,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "interpretation_boundary": (
            "This validates the RobotSpec intervention and verifier signal, not learned "
            "dynamics adaptation. Open-loop PD is a diagnostic, not a tracker-independent oracle."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", default=str(holosoma_sample_npz()))
    parser.add_argument("--mjcf", default=str(g1_mjcf()))
    parser.add_argument("--effort-scales", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--saturation-threshold", type=float, default=0.05)
    parser.add_argument(
        "--torque-ratio-threshold",
        type=float,
        default=0.5,
        help="localize frames whose fixed-reference PD demand exceeds this fraction of effort",
    )
    parser.add_argument("--min-failure-frames", type=int, default=2)
    parser.add_argument("--merge-gap-frames", type=int, default=1)
    parser.add_argument(
        "--newton-root",
        type=Path,
        help="optional Newton checkout for exact revision binding (null when unavailable)",
    )
    parser.add_argument(
        "--isaac-lab-root",
        type=Path,
        help="optional Isaac Lab checkout for exact revision binding (null when unavailable)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.out}")
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for scale, report in result["reports"].items():
        metrics = dict(report["metrics"])
        print(
            f"torque x{scale}: reference max ratio={metrics['reference_max_requested_torque_ratio']:.3f}, "
            f"reference saturation={metrics['reference_torque_saturation_fraction']:.3f}, "
            f"PD survival={metrics['survival_time_s']:.2f}s"
        )
    print("gates", json.dumps(result["gates"], sort_keys=True))
    print(f"wrote {args.out}")
    if not result["all_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
