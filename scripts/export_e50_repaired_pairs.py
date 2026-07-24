#!/usr/bin/env python3
"""E50 Stage-A: turn a repair recording into scored, physics-repaired reference segments.

Input: the multi-env rollout npz written by ``snmr.integration.wbt_repair.RepairRecordingCallback``
(50 Hz policy rate, one phase-stratified rollout per env — starts fixed by holosoma's
``wbt_metrics`` callback) plus the WBT reference npz the policy tracked (50 Hz) and, optionally,
the wbt_metrics report JSON to select only COMPLETED rollouts.

Output:
  1. ``<out>/segments.npz`` — per accepted env: simulated qpos-format trajectory
     (root_pos, root_quat wxyz, dof_pos), the reference frame indices it covers, and the
     simulator contact mask (feet force > threshold).
  2. ``<out>/stage_a_metrics.json`` — the preregistered Stage-A readouts
     (docs/E50_PHYSICS_REPAIRED_TEACHER_PROTOCOL.md §2): rollout-vs-reference stance-foot speed,
     penetration, MPJPE, coverage — computed with snmr.metrics on the SAME code path used for
     teacher/SNMR benchmarking, using the SIMULATOR contact mask (no estimated mask anywhere).

Quaternion convention: recording emits root_quat_xyzw; snmr-internal is wxyz — converted at load.
Run in .venv-snmr (torch + mujoco; no holosoma import needed).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from snmr import paths  # noqa: E402
from snmr.metrics import contact_motion_metrics  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

CONTACT_FORCE_THRESHOLD_N = 1.0  # holosoma undesired-contact convention
SETTLE_SKIP_STEPS = 25           # drop first 0.5 s after episode start (settle-in), 50 Hz


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def _heading_local_body_pos(
    kin, root_pos: torch.Tensor, root_quat_wxyz: torch.Tensor, dof_pos: torch.Tensor
) -> torch.Tensor:
    """FK body positions expressed in the per-frame root heading frame (T, B, 3).

    holosoma's MotionCommand continuously re-anchors the reference to the robot's CURRENT
    xy + yaw (managers/command/terms/wbt.py:845-877), so the policy only ever tracks
    relative pose and global xy drift is free by design. Fidelity of a rollout to its
    reference must therefore be measured heading-locally: subtract the root xy, rotate by
    -yaw. Height stays absolute (the tracker does control absolute z).
    """
    body_pos, _ = kin.forward_kinematics(root_pos, root_quat_wxyz, dof_pos)
    w, x, y, z = root_quat_wxyz.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    rel = body_pos - torch.cat(
        [root_pos[:, :2], torch.zeros_like(root_pos[:, :1])], dim=-1
    ).unsqueeze(1)
    out = rel.clone()
    out[..., 0] = c[:, None] * rel[..., 0] - s[:, None] * rel[..., 1]
    out[..., 1] = s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]
    return out


def load_recording(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        rec = {k: np.asarray(data[k]) for k in data.files if k != "_metadata_json"}
        meta = json.loads(str(data["_metadata_json"]))
    rec["meta"] = meta
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True,
                        help="WBT reference npz the policy tracked (50 Hz)")
    parser.add_argument("--report", type=pathlib.Path, default=None,
                        help="wbt_metrics report JSON; if given, keep only completed rollouts")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--mjcf", type=pathlib.Path, default=None)
    parser.add_argument("--mpjpe-gate-m", type=float, default=0.05)
    args = parser.parse_args()

    rec = load_recording(args.recording)
    meta = rec["meta"]
    fps = 1.0 / float(meta["dt"])
    T, N = rec["dof_pos"].shape[:2]

    ref = np.load(args.reference)
    ref_joint_pos = np.asarray(ref["joint_pos"])          # (Tm, 36) qpos layout
    ref_root_pos = ref_joint_pos[:, :3]
    ref_root_quat_wxyz = ref_joint_pos[:, 3:7]
    ref_dof = ref_joint_pos[:, 7:]

    completed = np.ones(N, dtype=bool)
    if args.report is not None:
        report = json.loads(args.report.read_text())
        completed = np.array(
            [bool(r["completed"]) for r in sorted(report["rollouts"], key=lambda r: r["env_id"])]
        )
        if completed.shape[0] != N:
            raise ValueError(f"report has {completed.shape[0]} rollouts, recording has {N} envs")

    kin = RobotKinematics(str(args.mjcf or paths.g1_mjcf()))
    foot_names = [n for n in kin.body_names if "ankle_roll_link" in n]
    # recording's feet_contact_force columns follow meta["foot_body_names"] (sim body order);
    # re-order them to match foot_names (FK order) so mask column i gates foot body i
    rec_foot_names = list(meta["foot_body_names"])
    if set(rec_foot_names) != set(foot_names):
        raise ValueError(f"foot sets differ: recording {rec_foot_names} vs FK {foot_names}")
    foot_col = [rec_foot_names.index(n) for n in foot_names]

    args.out.mkdir(parents=True, exist_ok=True)
    accepted, per_env_rows = [], []
    seg_arrays: dict[str, np.ndarray] = {}
    # aggregate stance-speed via weighted mean over contact frames, not mean-of-means
    agg = {"sim": {"speed_num": 0.0, "frames": 0}, "teacher": {"speed_num": 0.0, "frames": 0}}
    mpjpes, penetrations = [], []

    for env in range(N):
        if not completed[env]:
            continue
        # settle-skip, then the recorded horizon for this env
        sl = slice(SETTLE_SKIP_STEPS, T)
        steps = rec["time_steps"][sl, env].astype(np.int64)
        root_pos = torch.from_numpy(rec["root_pos"][sl, env]).double()
        root_quat = torch.from_numpy(_xyzw_to_wxyz(rec["root_quat_xyzw"][sl, env])).double()
        dof_pos = torch.from_numpy(rec["dof_pos"][sl, env]).double()
        contact = torch.from_numpy(
            rec["feet_contact_force"][sl, env][:, foot_col] > CONTACT_FORCE_THRESHOLD_N
        )

        reference = (
            torch.from_numpy(ref_root_pos[steps]).double(),
            torch.from_numpy(ref_root_quat_wxyz[steps]).double(),
            torch.from_numpy(ref_dof[steps]).double(),
        )
        # fidelity: heading-local MPJPE (what the WBT reward actually controls — the command
        # re-anchors to current robot xy+yaw every step, so world xy drift is free by design)
        local_sim = _heading_local_body_pos(kin, root_pos, root_quat, dof_pos)
        local_ref = _heading_local_body_pos(kin, *reference)
        mpjpe = float((local_sim - local_ref).norm(dim=-1).mean())

        # physical metrics in the world frame (frame-invariant): stance speed on the simulator
        # contact mask for sim and teacher alike (H-A(i)), penetration from foot heights
        foot_idx = [kin.body_index(n) for n in foot_names]
        sim_body, _ = kin.forward_kinematics(root_pos, root_quat, dof_pos)
        teacher_body, _ = kin.forward_kinematics(*reference)
        cm_sim = contact_motion_metrics(sim_body[:, foot_idx, :], fps, contact)
        cm_teacher = contact_motion_metrics(teacher_body[:, foot_idx, :], fps, contact)
        foot_z = sim_body[:, foot_idx, 2]
        penetration_fraction = float((foot_z < -0.01).any(dim=-1).float().mean())
        n_contact = int(contact.sum())
        if n_contact:  # stance_speed_ms is None for contact-free segments
            agg["sim"]["speed_num"] += cm_sim["stance_speed_ms"] * n_contact
            agg["teacher"]["speed_num"] += cm_teacher["stance_speed_ms"] * n_contact
            for key in agg:
                agg[key]["frames"] += n_contact
        mpjpes.append(mpjpe)
        penetrations.append(penetration_fraction)

        row = {
            "env_id": env,
            "frames": int(steps.shape[0]),
            "ref_start": int(steps[0]),
            "ref_end": int(steps[-1]),
            "mpjpe_m": mpjpe,
            "stance_speed_sim_ms": cm_sim["stance_speed_ms"],
            "stance_speed_teacher_ms": cm_teacher["stance_speed_ms"],
            "penetration_fraction": penetration_fraction,
            "mpjpe_gate_pass": bool(mpjpe <= args.mpjpe_gate_m),
        }
        per_env_rows.append(row)
        if row["mpjpe_gate_pass"]:
            accepted.append(env)
            seg_arrays[f"env{env}_root_pos"] = root_pos.numpy().astype(np.float32)
            seg_arrays[f"env{env}_root_quat_wxyz"] = root_quat.numpy().astype(np.float32)
            seg_arrays[f"env{env}_dof_pos"] = dof_pos.numpy().astype(np.float32)
            seg_arrays[f"env{env}_ref_steps"] = steps
            seg_arrays[f"env{env}_contact_mask"] = contact.numpy()

    covered = np.zeros(ref_joint_pos.shape[0], dtype=bool)
    for env in accepted:
        covered[seg_arrays[f"env{env}_ref_steps"]] = True

    speed_sim = agg["sim"]["speed_num"] / max(agg["sim"]["frames"], 1)
    speed_teacher = agg["teacher"]["speed_num"] / max(agg["teacher"]["frames"], 1)
    stage_a = {
        "recording": str(args.recording),
        "reference": str(args.reference),
        "num_envs": N,
        "num_completed": int(completed.sum()),
        "num_accepted": len(accepted),
        "contact_frames": agg["sim"]["frames"],
        "stance_speed_sim_ms": speed_sim,
        "stance_speed_teacher_ms": speed_teacher,
        "stance_speed_ratio": speed_sim / speed_teacher if speed_teacher > 0 else None,
        "mpjpe_mean_m": float(np.mean(mpjpes)) if mpjpes else None,
        "mpjpe_p95_m": float(np.percentile(mpjpes, 95)) if mpjpes else None,
        "penetration_fraction_mean": float(np.mean(penetrations)) if penetrations else None,
        "reference_frame_coverage": float(covered.mean()),
        "gates": {
            "H-A(i)_speed_ratio_le_0.5": (speed_teacher > 0 and speed_sim / speed_teacher <= 0.5),
            "H-A(ii)_penetration_approx_0": bool(np.mean(penetrations) < 0.01) if penetrations else False,
            "H-A(iii)_mpjpe_le_gate": bool(np.mean(mpjpes) <= args.mpjpe_gate_m) if mpjpes else False,
            "coverage_ge_0.8_report_only": bool(covered.mean() >= 0.8),
        },
        "per_env": per_env_rows,
    }
    (args.out / "stage_a_metrics.json").write_text(json.dumps(stage_a, indent=2) + "\n")
    seg_arrays["accepted_env_ids"] = np.asarray(accepted, dtype=np.int64)
    seg_arrays["_meta_json"] = np.array(json.dumps({**meta, "settle_skip": SETTLE_SKIP_STEPS}))
    np.savez_compressed(args.out / "segments.npz", **seg_arrays)

    print(json.dumps({k: v for k, v in stage_a.items() if k != "per_env"}, indent=2))


if __name__ == "__main__":
    main()
