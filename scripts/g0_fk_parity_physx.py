#!/usr/bin/env python3
"""Run the independently hashed Isaac Lab/PhysX half of the G0 FK parity gate.

Launch this script with the Holosoma ``hssim`` Python environment.  It never substitutes
an offline URDF kinematics library for PhysX: poses come from Isaac Lab's live
``Articulation`` view after joint state writes.  Any startup, asset, mapping, provenance,
or sample-contract error produces a JSON report with ``g0_pass=false``.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from snmr.fk_parity import (  # noqa: E402
    FK_PARITY_SCHEMA_VERSION,
    G1_KEY_LINKS,
    ORIENTATION_THRESHOLD_RAD,
    POSITION_THRESHOLD_M,
    capture_directory_bundle,
    capture_urdf_bundle,
    compare_fk_poses,
    evaluate_fixed_link_frames,
    float64_buffer_sha256,
    joint_sample_payload_sha256,
    materialize_memory_bundle,
    resolve_fixed_link_frames,
    wxyz_to_xyzw,
)
from snmr.provenance import source_revision_manifest  # noqa: E402


DEFAULT_HOLOSOMA = Path("/home/robotixx/holosoma")
DEFAULT_ROBOT_ROOT = DEFAULT_HOLOSOMA / "src/holosoma/holosoma/data/robots"
DEFAULT_URDF = DEFAULT_ROBOT_ROOT / "g1/g1_29dof.urdf"
DEFAULT_USD_ROOT = DEFAULT_ROBOT_ROOT / "converted_rank0"
DEFAULT_USD_ENTRYPOINT = "g1_29dof.usd"
DEFAULT_ISAAC_LAB = Path("/home/robotixx/.holosoma_deps/IsaacLab")
DEFAULT_NEWTON = Path("/home/robotixx/newton")


def _progress(stage: str, started_at: float) -> None:
    """Emit a flushed phase marker so bounded external runs identify the blocking call."""

    print(
        json.dumps(
            {
                "g0_progress": stage,
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _cleanup_simulation_context(sim: object, started_at: float) -> None:
    """Release Isaac Lab state before ``SimulationApp.close`` closes the USD stage.

    Isaac Lab's environment shutdown path for a normal (not in-memory) stage clears
    callbacks and the singleton without calling ``SimulationContext.stop`` or ``clear``.
    Those two methods synchronously update the Kit timeline and can themselves block in
    this headless worker.  Releasing the singleton is enough to detach Isaac Lab's
    context ownership before ``SimulationApp.close`` performs the authoritative stage
    close.
    """

    _progress("simulation_cleanup_start", started_at)
    sim.clear_all_callbacks()
    _progress("simulation_callbacks_clear_complete", started_at)
    sim.clear_instance()
    _progress("simulation_cleanup_complete", started_at)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a MuJoCo G0 reference against live Isaac Lab/PhysX FK."
    )
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--usd-root", type=Path, default=DEFAULT_USD_ROOT)
    parser.add_argument("--usd-entrypoint", default=DEFAULT_USD_ENTRYPOINT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--snmr-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON)
    parser.add_argument("--isaac-lab-root", type=Path, default=DEFAULT_ISAAC_LAB)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _capture_bytes(path: Path) -> tuple[bytes, dict[str, object]]:
    source = path.expanduser().resolve()
    data = source.read_bytes()
    return data, {
        "source_path": str(source),
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _load_reference(
    npz_bytes: bytes,
    manifest_bytes: bytes,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MuJoCo reference JSON: {exc}") from exc
    if manifest.get("schema_version") != FK_PARITY_SCHEMA_VERSION:
        raise ValueError(f"unsupported reference schema: {manifest.get('schema_version')!r}")
    if manifest.get("worker") != "mujoco_reference":
        raise ValueError("reference JSON was not produced by the MuJoCo G0 worker")
    if manifest.get("g0_evaluated") is not False or manifest.get("g0_pass") is not False:
        raise ValueError("MuJoCo-only reference must be explicitly unevaluated and fail-closed")
    expected_npz_hash = manifest.get("output", {}).get("npz_sha256")
    observed_npz_hash = hashlib.sha256(npz_bytes).hexdigest()
    if expected_npz_hash != observed_npz_hash:
        raise ValueError(
            f"reference NPZ hash mismatch: expected {expected_npz_hash!r}, got {observed_npz_hash}"
        )

    required = {
        "samples",
        "joint_names",
        "joint_lower",
        "joint_upper",
        "key_link_names",
        "mujoco_positions",
        "mujoco_quaternions_xyzw",
    }
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"reference NPZ is missing fields: {sorted(missing)}")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in required}
    samples = arrays["samples"]
    lower = arrays["joint_lower"]
    upper = arrays["joint_upper"]
    joint_names = tuple(str(name) for name in arrays["joint_names"].tolist())
    key_names = tuple(str(name) for name in arrays["key_link_names"].tolist())
    if key_names != tuple(name for name, _ in G1_KEY_LINKS):
        raise ValueError(f"unexpected key-link contract: {key_names}")
    sample_buffer_hash = float64_buffer_sha256(samples)
    sample_payload_hash = joint_sample_payload_sha256(joint_names, lower, upper, samples)
    sampling = manifest.get("sampling", {})
    if sample_buffer_hash != sampling.get("sample_buffer_sha256"):
        raise ValueError("sample memory buffer does not match reference manifest")
    if sample_payload_hash != sampling.get("sample_payload_sha256"):
        raise ValueError("joint names/bounds/sample payload does not match reference manifest")
    if list(samples.shape) != sampling.get("shape"):
        raise ValueError("sample shape does not match reference manifest")
    if samples.shape[1] != len(joint_names):
        raise ValueError("joint-name count does not match sample width")
    if lower.shape != (len(joint_names),) or upper.shape != (len(joint_names),):
        raise ValueError("joint-limit shape does not match joint names")
    if arrays["mujoco_positions"].shape != (samples.shape[0], len(key_names), 3):
        raise ValueError("MuJoCo position tensor has the wrong shape")
    if arrays["mujoco_quaternions_xyzw"].shape != (samples.shape[0], len(key_names), 4):
        raise ValueError("MuJoCo quaternion tensor has the wrong shape")
    thresholds = manifest.get("thresholds", {})
    if thresholds.get("position_m") != POSITION_THRESHOLD_M:
        raise ValueError("reference position threshold differs from frozen G0 threshold")
    if thresholds.get("orientation_geodesic_rad") != ORIENTATION_THRESHOLD_RAD:
        raise ValueError("reference orientation threshold differs from frozen G0 threshold")
    if thresholds.get("semantics") != "strict_less_than":
        raise ValueError("reference threshold semantics are not strict less-than")
    return manifest, {
        **arrays,
        "joint_names_tuple": joint_names,
        "key_names_tuple": key_names,
        "sample_buffer_sha256": sample_buffer_hash,
        "sample_payload_sha256": sample_payload_hash,
    }


def _frame_manifest(frame: object) -> dict[str, object]:
    return {
        "semantic_name": frame.semantic_name,
        "target_link": frame.target_link,
        "runtime_body": frame.runtime_body,
        "local_position": list(frame.local_position),
        "local_quaternion_xyzw": list(frame.local_quaternion_xyzw),
    }


def _run_physx(
    args: argparse.Namespace,
    materialized_usd: Path,
    urdf_bytes: bytes,
    reference: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return live PhysX key poses; simulator imports occur only after AppLauncher."""

    started_at = time.perf_counter()
    _progress("physx_imports_start", started_at)

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass

    _progress("physx_imports_complete", started_at)

    samples = reference["samples"]
    reference_joint_names = reference["joint_names_tuple"]
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(materialized_usd),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={},
    )

    @configclass
    class FKSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = robot_cfg

    _progress("simulation_context_start", started_at)
    sim = SimulationContext(sim_utils.SimulationCfg(device=args.device))
    # Publish the context immediately so ``main`` can release it before app shutdown,
    # including when scene creation or live pose collection raises.
    args._g0_simulation_context = sim
    args._g0_simulation_started_at = started_at
    _progress("simulation_context_complete", started_at)
    scene = InteractiveScene(FKSceneCfg(num_envs=batch_size, env_spacing=2.0))
    _progress("interactive_scene_complete", started_at)
    sim.reset()
    _progress("simulation_reset_complete", started_at)
    robot = scene["robot"]
    if robot.num_instances != batch_size:
        raise RuntimeError(
            f"PhysX articulation count mismatch: {robot.num_instances} != {batch_size}"
        )
    runtime_joint_names = tuple(str(name) for name in robot.joint_names)
    if len(runtime_joint_names) != 29 or set(runtime_joint_names) != set(reference_joint_names):
        raise ValueError(
            "PhysX joint names do not exactly match the 29-joint MuJoCo sample contract; "
            f"missing={sorted(set(reference_joint_names) - set(runtime_joint_names))}, "
            f"extra={sorted(set(runtime_joint_names) - set(reference_joint_names))}"
        )
    if len(set(runtime_joint_names)) != len(runtime_joint_names):
        raise ValueError("PhysX joint names are not unique")
    runtime_body_names = tuple(str(name) for name in robot.body_names)
    frames = resolve_fixed_link_frames(urdf_bytes, G1_KEY_LINKS, runtime_body_names)

    input_index = {name: index for index, name in enumerate(reference_joint_names)}
    runtime_from_input = np.asarray(
        [input_index[name] for name in runtime_joint_names], dtype=np.int64
    )
    physx_samples = np.ascontiguousarray(samples[:, runtime_from_input], dtype=np.float32)
    physx_sample_buffer_hash = hashlib.sha256(
        memoryview(physx_samples).cast("B")
    ).hexdigest()
    num_samples = samples.shape[0]
    num_keys = len(frames)
    output_positions = np.empty((num_samples, num_keys, 3), dtype=np.float64)
    output_quaternions = np.empty((num_samples, num_keys, 4), dtype=np.float64)

    root_pose = torch.zeros((batch_size, 7), device=sim.device, dtype=torch.float32)
    root_pose[:, :3] = scene.env_origins
    root_pose[:, 3] = 1.0  # Isaac Lab root quaternion is wxyz.
    joint_velocity = torch.zeros(
        (batch_size, len(runtime_joint_names)), device=sim.device, dtype=torch.float32
    )
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(
        torch.zeros((batch_size, 6), device=sim.device, dtype=torch.float32)
    )

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        active = stop - start
        runtime_positions = np.zeros((batch_size, len(runtime_joint_names)), dtype=np.float32)
        runtime_positions[:active] = physx_samples[start:stop]
        joint_position = torch.as_tensor(runtime_positions, device=sim.device)
        robot.write_joint_state_to_sim(joint_position, joint_velocity)
        # This is a kinematic propagation call, not a dynamics rollout/step.
        sim.forward()
        if start == 0:
            _progress("first_joint_state_forward_complete", started_at)
        body_positions = robot.data.body_link_pos_w[:active].detach().cpu().numpy().astype(
            np.float64
        )
        body_positions -= scene.env_origins[:active].detach().cpu().numpy().astype(np.float64)[
            :, None, :
        ]
        body_quaternions = wxyz_to_xyzw(
            robot.data.body_link_quat_w[:active].detach().cpu().numpy().astype(np.float64)
        )
        key_positions, key_quaternions = evaluate_fixed_link_frames(
            body_positions, body_quaternions, runtime_body_names, frames
        )
        output_positions[start:stop] = key_positions
        output_quaternions[start:stop] = key_quaternions

    _progress("all_physx_poses_complete", started_at)

    runtime = {
        "device": str(sim.device),
        "batch_size": batch_size,
        "num_batches": (num_samples + batch_size - 1) // batch_size,
        "joint_names": list(runtime_joint_names),
        "joint_input_dtype": physx_samples.dtype.str,
        "joint_input_shape": list(physx_samples.shape),
        "joint_input_buffer_sha256": physx_sample_buffer_hash,
        "joint_input_order": "runtime_joint_names",
        "source_float64_sample_buffer_sha256": reference["sample_buffer_sha256"],
        "body_names": list(runtime_body_names),
        "key_frame_resolution": [_frame_manifest(frame) for frame in frames],
        "pose_source": "isaaclab.assets.Articulation.data.body_link_{pos,quat}_w",
        "propagation": "SimulationContext.forward_without_dynamics_step",
        "torch": torch.__version__,
    }
    return output_positions, output_quaternions, runtime


def main() -> int:
    parser = _base_parser()
    # Parse enough to guarantee an output location if Isaac Lab itself cannot import.
    preliminary, _ = parser.parse_known_args()
    out_json = preliminary.out_json.expanduser().resolve()
    if out_json.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {out_json}")
    revisions = source_revision_manifest(
        snmr_path=preliminary.snmr_root,
        newton_path=preliminary.newton_root,
        isaac_lab_path=preliminary.isaac_lab_root,
    )
    report: dict[str, object] = {
        "schema_version": FK_PARITY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "worker": "physx_comparison",
        "backend": "isaac_lab_physx",
        "status": "initializing",
        "g0_evaluated": False,
        "g0_pass": False,
        "command": [sys.executable, *sys.argv],
        "thresholds": {
            "position_m": POSITION_THRESHOLD_M,
            "orientation_geodesic_rad": ORIENTATION_THRESHOLD_RAD,
            "semantics": "strict_less_than",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        **revisions,
    }
    app_launcher = None
    args = None
    usd_temporary = None
    try:
        reference_npz_bytes, reference_npz_input = _capture_bytes(preliminary.reference_npz)
        reference_json_bytes, reference_json_input = _capture_bytes(preliminary.reference_json)
        mujoco_manifest, reference = _load_reference(
            reference_npz_bytes, reference_json_bytes
        )
        urdf_manifest, urdf_bytes = capture_urdf_bundle(preliminary.urdf)
        expected_urdf_hash = mujoco_manifest.get("inputs", {}).get("urdf_bundle", {}).get("sha256")
        if urdf_manifest["sha256"] != expected_urdf_hash:
            raise ValueError(
                "PhysX worker's independently captured URDF bundle differs from MuJoCo worker: "
                f"{urdf_manifest['sha256']} != {expected_urdf_hash}"
            )
        usd_manifest, usd_members = capture_directory_bundle(
            preliminary.usd_root, entrypoint=preliminary.usd_entrypoint
        )
        report["inputs"] = {
            "reference_npz": reference_npz_input,
            "reference_json": reference_json_input,
            "urdf_bundle": urdf_manifest,
            "usd_bundle": usd_manifest,
            "sample_buffer_sha256": reference["sample_buffer_sha256"],
            "sample_payload_sha256": reference["sample_payload_sha256"],
        }
        report["sampling"] = {
            "num_samples": int(reference["samples"].shape[0]),
            "num_joints": int(reference["samples"].shape[1]),
            "dtype": reference["samples"].dtype.str,
            "shape": list(reference["samples"].shape),
            "seed": mujoco_manifest.get("sampling", {}).get("seed"),
        }

        try:
            from isaaclab.app import AppLauncher
        except Exception as exc:
            raise RuntimeError(
                "Isaac Lab AppLauncher is unavailable; run this worker with Holosoma's hssim Python"
            ) from exc
        AppLauncher.add_app_launcher_args(parser)
        parser.set_defaults(headless=True)
        args = parser.parse_args()
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")

        # Keep the private USD materialization alive until after Isaac Sim closes; some
        # Kit shutdown callbacks may still resolve layer paths.
        usd_temporary = tempfile.TemporaryDirectory(prefix="snmr-g0-physx-")
        materialized_root = materialize_memory_bundle(
            usd_members, Path(usd_temporary.name) / "usd"
        )
        materialized_usd = materialized_root / args.usd_entrypoint
        launch_started_at = time.perf_counter()
        _progress("app_launcher_start", launch_started_at)
        app_launcher = AppLauncher(args)
        _progress("app_launcher_complete", launch_started_at)
        simulation_app = app_launcher.app
        physx_positions, physx_quaternions, runtime = _run_physx(
            args, materialized_usd, urdf_bytes, reference
        )
        report["environment"]["isaac_sim_app_running"] = bool(
            simulation_app.is_running()
        )

        result = compare_fk_poses(
            reference["mujoco_positions"],
            reference["mujoco_quaternions_xyzw"],
            physx_positions,
            physx_quaternions,
        )
        protocol_sample_count_pass = bool(result.num_samples == 10_000)
        comparison_manifest = result.to_manifest(reference["key_names_tuple"])
        comparison_manifest["protocol_sample_count_pass"] = protocol_sample_count_pass
        report["runtime"] = runtime
        report["comparison"] = comparison_manifest
        report["g0_evaluated"] = True
        report["g0_pass"] = bool(result.g0_pass and protocol_sample_count_pass)
        if not protocol_sample_count_pass:
            report["status"] = "protocol_incomplete"
            report["gate_reason"] = (
                f"comparison ran for {result.num_samples} samples; G0 requires exactly 10000"
            )
            exit_code = 2
        elif result.g0_pass:
            report["status"] = "pass"
            report["gate_reason"] = "Both strict maximum-error thresholds passed in live PhysX."
            exit_code = 0
        else:
            report["status"] = "fail"
            report["gate_reason"] = "At least one strict maximum-error threshold failed."
            exit_code = 1
    except BaseException as exc:
        report["status"] = "backend_error"
        report["g0_evaluated"] = False
        report["g0_pass"] = False
        report["gate_reason"] = "PhysX comparison did not complete; G0 fails closed."
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 2
    finally:
        # Journal a fail-closed report before touching Kit teardown.  If an external
        # timeout kills a genuinely stuck cleanup call, the requested output still says
        # that G0 was not evaluated.  A successful teardown atomically replaces it below.
        if app_launcher is not None:
            pending_report = copy.deepcopy(report)
            pending_report["status"] = "backend_error"
            pending_report["g0_evaluated"] = False
            pending_report["g0_pass"] = False
            pending_report["gate_reason"] = (
                "Live PhysX work finished, but simulator teardown has not completed; "
                "G0 fails closed."
            )
            pending_report["teardown"] = {
                "simulation_context_released": False,
                "stage_closed": False,
                "application_close_invoked_after_manifest": False,
            }
            pending_report["completed_at"] = _utc_now()
            _write_json_atomic(out_json, pending_report)

        simulation_context_released = False
        stage_closed = False
        simulation_context = (
            getattr(args, "_g0_simulation_context", None) if args is not None else None
        )
        if simulation_context is not None:
            try:
                _cleanup_simulation_context(
                    simulation_context,
                    getattr(args, "_g0_simulation_started_at", time.perf_counter()),
                )
                simulation_context_released = True
            except BaseException as simulation_cleanup_exc:
                report["simulation_cleanup_error"] = {
                    "type": type(simulation_cleanup_exc).__name__,
                    "message": str(simulation_cleanup_exc),
                }
                report["status"] = "backend_error"
                report["g0_evaluated"] = False
                report["g0_pass"] = False
                exit_code = 2

        if app_launcher is not None:
            try:
                stage_close_started_at = time.perf_counter()
                _progress("stage_close_start", stage_close_started_at)
                if not app_launcher.app.context.can_close_stage():
                    raise RuntimeError("Isaac Sim reports that the live USD stage cannot close")
                app_launcher.app.context.close_stage()
                stage_closed = True
                _progress("stage_close_complete", stage_close_started_at)
            except BaseException as stage_close_exc:
                report["stage_close_error"] = {
                    "type": type(stage_close_exc).__name__,
                    "message": str(stage_close_exc),
                }
                report["status"] = "backend_error"
                report["g0_evaluated"] = False
                report["g0_pass"] = False
                exit_code = 2
        if usd_temporary is not None:
            try:
                usd_temporary.cleanup()
            except BaseException as cleanup_exc:
                report["cleanup_error"] = {
                    "type": type(cleanup_exc).__name__,
                    "message": str(cleanup_exc),
                }
                report["status"] = "backend_error"
                report["g0_evaluated"] = False
                report["g0_pass"] = False
                exit_code = 2
        report["teardown"] = {
            "simulation_context_released": simulation_context_released,
            "stage_closed": stage_closed,
            "application_close_invoked_after_manifest": app_launcher is not None,
        }
        report["completed_at"] = _utc_now()
        _write_json_atomic(out_json, report)

        print(
            json.dumps(
                {
                    "manifest": str(out_json),
                    "status": report["status"],
                    "g0_pass": (
                        False
                        if report["status"] == "backend_error"
                        else report["g0_pass"]
                    ),
                }
            ),
            flush=True,
        )

        # Isaac Sim 5.1's framework shutdown terminates the embedded Python runtime,
        # so every durable result must be written before this call.  ``post_quit``
        # preserves the manifest-derived process status instead of the default zero.
        if app_launcher is not None:
            try:
                app_launcher.app.app.post_quit(exit_code)
                app_launcher.app.close()
            except BaseException as close_exc:
                report["close_error"] = {
                    "type": type(close_exc).__name__,
                    "message": str(close_exc),
                }
                report["status"] = "backend_error"
                report["g0_evaluated"] = False
                report["g0_pass"] = False
                report["teardown"]["application_close_invoked_after_manifest"] = True
                report["completed_at"] = _utc_now()
                _write_json_atomic(out_json, report)
                exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
