#!/usr/bin/env python3
"""Generate the deterministic MuJoCo half of the G0 G1 FK parity sweep.

This worker deliberately cannot pass G0.  It writes a hash-bound NPZ reference and a
JSON manifest whose gate remains unevaluated until ``g0_fk_parity_physx.py`` compares the
same 10,000 configurations in the independently launched Isaac Lab worker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

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
    evaluate_fixed_link_frames,
    float64_buffer_sha256,
    joint_sample_payload_sha256,
    resolve_fixed_link_frames,
    sample_uniform_joint_positions,
    materialize_memory_bundle,
    wxyz_to_xyzw,
)
from snmr.provenance import source_revision_manifest  # noqa: E402


DEFAULT_HOLOSOMA = Path("/home/robotixx/holosoma")
DEFAULT_ROBOT_ROOT = (
    DEFAULT_HOLOSOMA / "src/holosoma/holosoma/data/robots"
)
DEFAULT_MJCF = DEFAULT_ROBOT_ROOT / "g1/scenes/scene_g1_29dof_wbt_plane.xml"
DEFAULT_URDF = DEFAULT_ROBOT_ROOT / "g1/g1_29dof.urdf"
DEFAULT_ISAAC_LAB = Path("/home/robotixx/.holosoma_deps/IsaacLab")
DEFAULT_NEWTON = Path("/home/robotixx/newton")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a hash-bound MuJoCo FK reference for the G0 G1 parity gate."
    )
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument(
        "--mjcf-bundle-root",
        type=Path,
        help="Root copied into private memory-backed materialization (defaults to the G1 directory).",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Defaults to OUT_NPZ with a .json suffix.",
    )
    parser.add_argument("--snmr-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--newton-root", type=Path, default=DEFAULT_NEWTON)
    parser.add_argument("--isaac-lab-root", type=Path, default=DEFAULT_ISAAC_LAB)
    return parser.parse_args()


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


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as stream:
            data = stream.read()
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(data).hexdigest()


def _manifest_frame(frame: object) -> dict[str, object]:
    return {
        "semantic_name": frame.semantic_name,
        "target_link": frame.target_link,
        "runtime_body": frame.runtime_body,
        "local_position": list(frame.local_position),
        "local_quaternion_xyzw": list(frame.local_quaternion_xyzw),
    }


def main() -> int:
    args = parse_args()
    import mujoco

    out_npz = args.out_npz.expanduser().resolve()
    out_json = (
        args.out_json.expanduser().resolve()
        if args.out_json is not None
        else out_npz.with_suffix(".json")
    )
    if out_npz == out_json:
        raise ValueError("NPZ and JSON output paths must differ")
    existing = [path for path in (out_npz, out_json) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")

    # Capture all files into memory before either parser consumes them.  The model is
    # loaded from a private materialization of these exact bytes, not the source path.
    mjcf_source = args.mjcf.expanduser().resolve()
    mjcf_root = (
        args.mjcf_bundle_root.expanduser().resolve()
        if args.mjcf_bundle_root is not None
        else (mjcf_source.parent.parent if mjcf_source.parent.name == "scenes" else mjcf_source.parent)
    )
    try:
        mjcf_entrypoint = mjcf_source.relative_to(mjcf_root).as_posix()
    except ValueError as exc:
        raise ValueError("MJCF entrypoint must be inside --mjcf-bundle-root") from exc
    mjcf_manifest, mjcf_members = capture_directory_bundle(
        mjcf_root, entrypoint=mjcf_entrypoint
    )
    urdf_manifest, urdf_bytes = capture_urdf_bundle(args.urdf)
    revisions = source_revision_manifest(
        snmr_path=args.snmr_root,
        newton_path=args.newton_root,
        isaac_lab_path=args.isaac_lab_root,
    )

    with tempfile.TemporaryDirectory(prefix="snmr-g0-mujoco-") as temporary:
        materialized_root = materialize_memory_bundle(mjcf_members, Path(temporary) / "mjcf")
        materialized_mjcf = materialized_root / mjcf_entrypoint
        model = mujoco.MjModel.from_xml_path(str(materialized_mjcf))
        data = mujoco.MjData(model)

        hinge_joint_ids = [
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        free_joint_ids = [
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(hinge_joint_ids) != 29:
            raise ValueError(f"expected 29 hinge joints, found {len(hinge_joint_ids)}")
        if len(free_joint_ids) != 1:
            raise ValueError(f"expected one floating root joint, found {len(free_joint_ids)}")
        joint_names = tuple(
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
            for joint_id in hinge_joint_ids
        )
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("MuJoCo hinge joint names are not unique")
        lower = np.ascontiguousarray(model.jnt_range[hinge_joint_ids, 0], dtype=np.float64)
        upper = np.ascontiguousarray(model.jnt_range[hinge_joint_ids, 1], dtype=np.float64)
        samples = sample_uniform_joint_positions(
            lower, upper, num_samples=args.num_samples, seed=args.seed
        )

        body_ids = tuple(range(1, model.nbody))
        body_names = tuple(
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            for body_id in body_ids
        )
        if any(name == "None" for name in body_names):
            raise ValueError("all non-world MuJoCo bodies must be named")
        frames = resolve_fixed_link_frames(urdf_bytes, G1_KEY_LINKS, body_names)

        body_positions = np.empty((samples.shape[0], len(body_ids), 3), dtype=np.float64)
        body_quaternions_xyzw = np.empty(
            (samples.shape[0], len(body_ids), 4), dtype=np.float64
        )
        hinge_qpos_addresses = np.asarray(
            [model.jnt_qposadr[joint_id] for joint_id in hinge_joint_ids], dtype=np.int64
        )
        root_qpos_address = int(model.jnt_qposadr[free_joint_ids[0]])
        for sample_index, joint_position in enumerate(samples):
            data.qpos[:] = model.qpos0
            # Canonical pelvis root: world origin and identity orientation (wxyz).
            data.qpos[root_qpos_address : root_qpos_address + 7] = (
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            data.qpos[hinge_qpos_addresses] = joint_position
            mujoco.mj_forward(model, data)
            body_positions[sample_index] = data.xpos[np.asarray(body_ids)]
            body_quaternions_xyzw[sample_index] = wxyz_to_xyzw(
                data.xquat[np.asarray(body_ids)]
            )

    key_positions, key_quaternions_xyzw = evaluate_fixed_link_frames(
        body_positions, body_quaternions_xyzw, body_names, frames
    )
    key_names = tuple(semantic_name for semantic_name, _ in G1_KEY_LINKS)
    sample_buffer_hash = float64_buffer_sha256(samples)
    sample_payload_hash = joint_sample_payload_sha256(joint_names, lower, upper, samples)
    arrays = {
        "samples": samples,
        "joint_names": np.asarray(joint_names),
        "joint_lower": lower,
        "joint_upper": upper,
        "key_link_names": np.asarray(key_names),
        "mujoco_positions": key_positions,
        "mujoco_quaternions_xyzw": key_quaternions_xyzw,
    }
    npz_hash = _write_npz_atomic(out_npz, arrays)

    manifest: dict[str, object] = {
        "schema_version": FK_PARITY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "worker": "mujoco_reference",
        "backend": "mujoco_cpu",
        "status": "reference_generated",
        "g0_evaluated": False,
        "g0_pass": False,
        "gate_reason": "PhysX comparison has not been run; a reference alone cannot pass G0.",
        "command": [sys.executable, *sys.argv],
        "thresholds": {
            "position_m": POSITION_THRESHOLD_M,
            "orientation_geodesic_rad": ORIENTATION_THRESHOLD_RAD,
            "semantics": "strict_less_than",
        },
        "sampling": {
            "distribution": "independent_uniform_closed_bounds",
            "seed": args.seed,
            "num_samples": samples.shape[0],
            "num_joints": samples.shape[1],
            "dtype": samples.dtype.str,
            "shape": list(samples.shape),
            "sample_buffer_sha256": sample_buffer_hash,
            "sample_payload_sha256": sample_payload_hash,
        },
        "coordinate_contract": {
            "world_up": "+Z",
            "world_forward": "+X",
            "root_position": [0.0, 0.0, 0.0],
            "root_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "output_quaternion_convention": "xyzw",
            "linear_unit": "meter",
            "angular_unit": "radian",
        },
        "joint_names": list(joint_names),
        "key_link_names": list(key_names),
        "key_frame_resolution": [_manifest_frame(frame) for frame in frames],
        "inputs": {
            "mjcf_bundle": mjcf_manifest,
            "urdf_bundle": urdf_manifest,
        },
        "output": {
            "npz_path": str(out_npz),
            "npz_sha256": npz_hash,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
        },
        **revisions,
    }
    _write_json_atomic(out_json, manifest)
    print(json.dumps({"npz": str(out_npz), "manifest": str(out_json), **manifest["sampling"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
