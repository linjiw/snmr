#!/usr/bin/env python
"""Run one dynamics-twin reference through Newton's MuJoCo-Warp solver.

Run this script with the pinned Newton environment, not SNMR's training environment:

    /home/robotixx/newton/.venv/bin/python scripts/verify_newton_pd.py \
      --pilot-json autoresearch/.../dynamics_twins_mujoco.json --scale 0.5 --out report.json

The runner uses MuJoCo-Warp's own contact path, direct joint torques, deterministic
zero-order-hold targets, explicit wxyz<->xyzw conversion, and overflow auditing.  It does
not claim differentiability: Newton's ``SolverMuJoCo`` is not differentiable.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import platform
import re
import sys
import tempfile
import time

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.provenance import (  # noqa: E402
    ArtifactMismatchError,
    ArtifactSnapshot,
    MjcfBundleSnapshot,
    source_revision_manifest,
)
from snmr.robot_spec import RobotSpec  # noqa: E402
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


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@wp.kernel(enable_backward=False)
def _pd_forces(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    kp: wp.array(dtype=wp.float32),
    kd: wp.array(dtype=wp.float32),
    limit: wp.array(dtype=wp.float32),
    joint_f: wp.array(dtype=wp.float32),
    saturated_count: wp.array(dtype=wp.int32),
    max_requested_ratio: wp.array(dtype=wp.float32),
):
    joint_id = wp.tid()
    requested = kp[joint_id] * (target[joint_id] - joint_q[7 + joint_id])
    requested -= kd[joint_id] * joint_qd[6 + joint_id]
    ratio = wp.abs(requested) / limit[joint_id]
    if ratio > 1.0:
        wp.atomic_add(saturated_count, 0, 1)
    wp.atomic_max(max_requested_ratio, 0, ratio)
    joint_f[6 + joint_id] = wp.clamp(requested, -limit[joint_id], limit[joint_id])


def _wxyz_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quat)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _tilt_from_xyzw(quat: np.ndarray) -> float:
    x, y, z, w = (float(value) for value in quat)
    rzz = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(rzz, -1.0, 1.0)))


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


def _candidate_config_hash(args: argparse.Namespace) -> str:
    return rollout_contract_hash(
        seconds=args.seconds,
        start_frame=args.start_frame,
        fall_root_z_m=args.fall_root_z,
        root_xy_limit_m=args.root_xy_limit,
        tilt_limit_rad=math.radians(args.tilt_limit_deg),
        saturation_threshold=args.saturation_threshold,
        min_failure_frames=args.min_failure_frames,
        merge_gap_frames=args.merge_gap_frames,
    )


def _manifest_path(value: object, pilot_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = pilot_path.parent / path
    return path.resolve()


def _require_digest(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.lower()) is None:
        raise ValueError(f"pilot manifest requires a 64-character hexadecimal {key}")
    return value.lower()


def run_with_provenance(
    args: argparse.Namespace,
) -> tuple[VerificationReport, dict[str, object]]:
    """Re-capture every input in the Newton worker and return report metadata."""

    import mujoco_warp
    import newton
    from newton.solvers import SolverMuJoCo

    pilot_path = Path(args.pilot_json).expanduser().resolve()
    pilot_snapshot = ArtifactSnapshot.capture(pilot_path)
    pilot = json.loads(pilot_snapshot.data)
    if not isinstance(pilot, dict):
        raise ValueError("pilot JSON must contain an object")
    base_spec = RobotSpec.from_dict(pilot["base_spec"])
    motion_path = _manifest_path(pilot["motion"], pilot_path)
    mjcf_path = _manifest_path(pilot["mjcf"], pilot_path)
    motion_snapshot = ArtifactSnapshot.capture(motion_path)
    asset_snapshot = MjcfBundleSnapshot.capture(mjcf_path)

    expected_motion = _require_digest(pilot, "motion_sha256")
    if motion_snapshot.sha256 != expected_motion:
        raise ArtifactMismatchError(
            f"Newton worker motion hash mismatch: manifest {expected_motion}, "
            f"worker {motion_snapshot.sha256}"
        )
    expected_asset = pilot.get("asset_sha256", base_spec.asset_sha256)
    if (
        not isinstance(expected_asset, str)
        or _SHA256_RE.fullmatch(expected_asset.lower()) is None
    ):
        raise ValueError(
            "pilot manifest requires a 64-character hexadecimal asset_sha256"
        )
    expected_asset = expected_asset.lower()
    actual_asset = asset_snapshot.entrypoint_snapshot.sha256
    if actual_asset != expected_asset or actual_asset != base_spec.asset_sha256:
        raise ArtifactMismatchError(
            "Newton worker MJCF entrypoint hash differs from pilot/base RobotSpec: "
            f"manifest={expected_asset}, spec={base_spec.asset_sha256}, worker={actual_asset}"
        )
    expected_bundle = _require_digest(pilot, "asset_bundle_sha256")
    if expected_bundle != asset_snapshot.sha256:
        raise ArtifactMismatchError(
            f"Newton worker asset-bundle hash mismatch: manifest {expected_bundle}, "
            f"worker {asset_snapshot.sha256}"
        )

    revisions = source_revision_manifest(
        snmr_path=ROOT,
        newton_path=newton.__file__,
        isaac_lab_path=getattr(args, "isaac_lab_root", None),
    )
    artifacts: dict[str, object] = {
        "pilot_manifest": pilot_snapshot.manifest(),
        "motion": motion_snapshot.manifest(),
        "robot_asset": asset_snapshot.manifest(),
    }
    with tempfile.TemporaryDirectory(prefix="snmr-newton-verifier-") as temporary:
        runtime_root = Path(temporary)
        runtime_mjcf = asset_snapshot.materialize(runtime_root / "robot-asset")
        report = _run_captured(
            args,
            pilot=pilot,
            base_spec=base_spec,
            motion_path=motion_path,
            runtime_mjcf=runtime_mjcf,
            motion_snapshot=motion_snapshot,
            revisions=revisions,
            newton=newton,
            mujoco_warp=mujoco_warp,
            solver_class=SolverMuJoCo,
        )
    metadata: dict[str, object] = {
        **revisions,
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "warp": str(getattr(wp, "__version__", "unknown")),
            "newton": str(getattr(newton, "__version__", "unknown")),
            "mujoco_warp": str(getattr(mujoco_warp, "__version__", "unknown")),
        },
        "asset_sha256": actual_asset,
        "asset_bundle_sha256": asset_snapshot.sha256,
        "backend_config_hash": canonical_hash({
            "iterations": args.iterations,
            "ls_iterations": args.ls_iterations,
            "njmax": args.njmax,
            "nconmax": args.nconmax,
            "simulation_dt": base_spec.runtime.simulation_dt,
            "control_dt": base_spec.control.control_dt,
        }),
        "input_artifacts": artifacts,
    }
    return report, metadata


def run(args: argparse.Namespace) -> VerificationReport:
    """Backward-compatible report-only entrypoint."""

    report, _ = run_with_provenance(args)
    return report


def _run_captured(
    args: argparse.Namespace,
    *,
    pilot: dict[str, object],
    base_spec: RobotSpec,
    motion_path: Path,
    runtime_mjcf: Path,
    motion_snapshot: ArtifactSnapshot,
    revisions: dict[str, object],
    newton: object,
    mujoco_warp: object,
    solver_class: object,
) -> VerificationReport:
    scale = float(args.scale)
    spec = base_spec.dynamics_twin(torque_scale=scale)
    data = np.load(io.BytesIO(motion_snapshot.data), allow_pickle=True)
    joint_names = [str(name) for name in data["joint_names"].tolist()]
    if [joint.name for joint in spec.joints] != joint_names:
        raise ValueError("RobotSpec and motion joint order differ")
    if any(joint.kp is None or joint.kd is None or joint.torque_limit is None
           for joint in spec.joints):
        raise ValueError("Newton PD verification requires kp, kd, and torque_limit for every joint")

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if not np.isfinite(joint_pos).all() or not np.isfinite(joint_vel).all():
        raise ValueError("motion joint buffers must be finite")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("motion FPS must be finite and positive")
    if not math.isclose(fps, 1.0 / spec.control.control_dt, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("motion FPS and RobotSpec control_dt differ")
    n_substeps = round(spec.control.control_dt / spec.runtime.simulation_dt)
    if n_substeps <= 0 or not math.isclose(
        n_substeps * spec.runtime.simulation_dt,
        spec.control.control_dt,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("simulation_dt must divide control_dt")

    newton.use_coord_layout_targets = True
    wp.init()
    robot = newton.ModelBuilder()
    solver_class.register_custom_attributes(robot)
    robot.add_mjcf(
        str(runtime_mjcf),
        ignore_names=["ground", "floor"],
        parse_visuals=False,
        ctrl_direct=True,
    )
    builder = newton.ModelBuilder()
    solver_class.register_custom_attributes(builder)
    builder.add_ground_plane()
    builder.add_world(robot)
    model = builder.finalize()
    if model.joint_coord_count != joint_pos.shape[1] or model.joint_dof_count != joint_vel.shape[1]:
        raise ValueError(
            f"Newton state widths {model.joint_coord_count}/{model.joint_dof_count} do not match "
            f"motion {joint_pos.shape[1]}/{joint_vel.shape[1]}"
        )
    solver = solver_class(
        model,
        iterations=args.iterations,
        ls_iterations=args.ls_iterations,
        njmax=args.njmax,
        nconmax=args.nconmax,
        update_data_interval=1,
        disable_sensors=True,
        use_mujoco_contacts=True,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    control.joint_f.zero_()

    start_frame = int(args.start_frame)
    if not 0 <= start_frame < joint_pos.shape[0] - 1:
        raise ValueError("start_frame outside motion")
    initial_q = joint_pos[start_frame].copy()
    # Newton 1.6 uses xyzw generalized-coordinate quaternions; SNMR artifacts are wxyz.
    initial_q[3:7] = joint_pos[start_frame, [4, 5, 6, 3]]
    initial_qd = joint_vel[start_frame].copy()
    initial_qd[3:6] = _wxyz_rotation_matrix(joint_pos[start_frame, 3:7]).T @ initial_qd[3:6]
    wp.copy(state_0.joint_q, wp.array(initial_q, dtype=wp.float32, device=model.device))
    wp.copy(state_0.joint_qd, wp.array(initial_qd, dtype=wp.float32, device=model.device))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    if not np.isfinite(state_0.joint_q.numpy()).all() or not np.isfinite(
        state_0.joint_qd.numpy()
    ).all():
        raise ValueError("non-finite initial Newton state")

    kp = wp.array([joint.kp for joint in spec.joints], dtype=wp.float32, device=model.device)
    kd = wp.array([joint.kd for joint in spec.joints], dtype=wp.float32, device=model.device)
    effort = wp.array(
        [joint.torque_limit for joint in spec.joints], dtype=wp.float32, device=model.device
    )
    target = wp.zeros(len(spec.joints), dtype=wp.float32, device=model.device)
    saturated_count = wp.zeros(1, dtype=wp.int32, device=model.device)
    max_requested_ratio = wp.zeros(1, dtype=wp.float32, device=model.device)

    ticks_requested = min(
        int(round(args.seconds * fps)),
        joint_pos.shape[0] - 1 - start_frame,
    )
    ticks_alive = 0
    diverge_reason: str | None = None
    failure_frame: int | None = None
    dof_errors: list[float] = []
    root_height_errors: list[float] = []
    saturation_trace: list[float] = []
    max_ratio_trace: list[float] = []
    total_saturated = 0
    total_torque_samples = 0
    global_max_ratio = 0.0
    started = time.perf_counter()
    for tick in range(ticks_requested):
        reference = joint_pos[start_frame + tick + 1]
        wp.copy(target, wp.array(reference[7:], dtype=wp.float32, device=model.device))
        saturated_count.zero_()
        max_requested_ratio.zero_()
        for _ in range(n_substeps):
            wp.launch(
                _pd_forces,
                dim=len(spec.joints),
                inputs=(
                    state_0.joint_q,
                    state_0.joint_qd,
                    target,
                    kp,
                    kd,
                    effort,
                    control.joint_f,
                    saturated_count,
                    max_requested_ratio,
                ),
                device=model.device,
            )
            state_0.clear_forces()
            # None is intentional: SolverMuJoCo runs its own collision path in this mode.
            solver.step(state_0, state_1, control, None, spec.runtime.simulation_dt)
            state_0, state_1 = state_1, state_0

        state_q = state_0.joint_q.numpy()
        state_qd = state_0.joint_qd.numpy()
        tick_saturated = int(saturated_count.numpy()[0])
        tick_samples = n_substeps * len(spec.joints)
        tick_max_ratio = float(max_requested_ratio.numpy()[0])
        if (
            not np.isfinite(state_q).all()
            or not np.isfinite(state_qd).all()
            or not math.isfinite(tick_max_ratio)
        ):
            diverge_reason = "non-finite simulator or controller state"
            failure_frame = start_frame + tick
            break
        total_saturated += tick_saturated
        total_torque_samples += tick_samples
        global_max_ratio = max(global_max_ratio, tick_max_ratio)
        saturation_trace.append(tick_saturated / tick_samples)
        max_ratio_trace.append(tick_max_ratio)

        z = float(state_q[2])
        xy_deviation = float(np.linalg.norm(state_q[:2] - reference[:2]))
        tilt = _tilt_from_xyzw(state_q[3:7])
        dof_error = float(np.mean(np.abs(state_q[7:] - reference[7:])))
        if z < args.fall_root_z:
            diverge_reason = f"root z {z:.3f} < {args.fall_root_z}"
        elif xy_deviation > args.root_xy_limit:
            diverge_reason = f"root xy deviation {xy_deviation:.3f} > {args.root_xy_limit}"
        elif tilt > math.radians(args.tilt_limit_deg):
            diverge_reason = f"tilt {math.degrees(tilt):.1f} deg > {args.tilt_limit_deg}"
        if diverge_reason is not None:
            failure_frame = start_frame + tick
            break
        ticks_alive += 1
        dof_errors.append(dof_error)
        root_height_errors.append(abs(z - float(reference[2])))
    wp.synchronize()
    wall_time = time.perf_counter() - started

    overflow = getattr(solver.mjw_data, "overflow", None)
    overflow_detected = bool(np.any(overflow.numpy())) if overflow is not None else False
    intervals = list(offset_failure_intervals(localize_threshold_failures(
        saturation_trace,
        threshold=args.saturation_threshold,
        failure_type="rollout_torque_saturation",
        min_frames=args.min_failure_frames,
        merge_gap_frames=args.merge_gap_frames,
        joints=tuple(joint_names),
    ), start_frame))
    if diverge_reason is not None:
        frame = failure_frame if failure_frame is not None else start_frame
        intervals.append(FailureInterval(
            start_frame=frame,
            end_frame=frame,
            failure_type="rollout_divergence",
            severity=1.0,
            evidence=(("survival_time_s", ticks_alive / fps),),
        ))

    engine_version = str(getattr(newton, "__version__", "unknown"))
    solver_version = str(getattr(mujoco_warp, "__version__", "unknown"))
    report = VerificationReport(
        schema_version=REPORT_SCHEMA_VERSION,
        candidate_id=f"{motion_path.stem}:torque_x{scale:g}",
        motion_sha256=motion_snapshot.sha256,
        robot_spec_hash=spec.spec_hash,
        robot_kinematic_hash=spec.kinematic_hash,
        robot_dynamics_hash=spec.dynamics_hash,
        verification_level="L2_open_loop_pd",
        passed=diverge_reason is None and not overflow_detected,
        fps=fps,
        num_frames=int(joint_pos.shape[0]),
        backend=BackendIdentity(
            engine="newton",
            solver="mujoco_warp",
            engine_version=engine_version,
            solver_version=solver_version,
            build=(
                f"{revisions['newton_repo_root']}@{revisions['newton_commit']}"
                f"{'+dirty' if revisions['newton_dirty'] else ''}"
                if revisions["newton_repo_status"] == "available"
                else None
            ),
            quaternion_convention="xyzw",
        ),
        metrics=_finite_metrics({
            "survival_fraction": ticks_alive / ticks_requested if ticks_requested else 0.0,
            "survival_time_s": ticks_alive / fps,
            "mean_dof_error_rad": float(np.mean(dof_errors)) if dof_errors else None,
            "mean_root_height_error_m": (
                float(np.mean(root_height_errors)) if root_height_errors else None
            ),
            "torque_saturation_fraction": (
                total_saturated / total_torque_samples if total_torque_samples else 0.0
            ),
            "max_requested_torque_ratio": global_max_ratio,
            "wall_time_s": wall_time,
        }),
        failure_intervals=tuple(sorted(intervals, key=lambda item: (item.start_frame, item.failure_type))),
        controller_hash=canonical_hash({
            "kp": [joint.kp for joint in spec.joints],
            "kd": [joint.kd for joint in spec.joints],
        }),
        seed=0,
        config_hash=_candidate_config_hash(args),
        overflow_detected=overflow_detected,
    )
    report.validate()
    if report.config_hash != _require_digest(pilot, "config_hash"):
        raise ArtifactMismatchError("Newton candidate/evaluation config differs from pilot")
    if report.controller_hash != _require_digest(pilot, "controller_hash"):
        raise ArtifactMismatchError("Newton controller hash differs from pilot")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-json", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--ls-iterations", type=int, default=10)
    parser.add_argument("--njmax", type=int, default=4096)
    parser.add_argument("--nconmax", type=int, default=2048)
    parser.add_argument("--fall-root-z", type=float, default=0.35)
    parser.add_argument("--root-xy-limit", type=float, default=0.5)
    parser.add_argument("--tilt-limit-deg", type=float, default=60.0)
    parser.add_argument("--saturation-threshold", type=float, default=0.05)
    parser.add_argument("--min-failure-frames", type=int, default=2)
    parser.add_argument("--merge-gap-frames", type=int, default=1)
    parser.add_argument(
        "--isaac-lab-root",
        type=Path,
        help="optional Isaac Lab checkout for exact revision binding (null when unavailable)",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.out}")
    report, provenance = run_with_provenance(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    # VerificationReport.from_dict deliberately ignores extension fields, preserving the
    # v0.1 comparison interface while every generated worker report carries full provenance.
    payload.update(provenance)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    metrics = report.metric_dict()
    print(
        f"{report.candidate_id}: pass={report.passed}, "
        f"survival={metrics['survival_time_s']:.2f}s, "
        f"saturation={metrics['torque_saturation_fraction']:.3f}, "
        f"overflow={report.overflow_detected}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
