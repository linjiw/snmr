#!/usr/bin/env python
"""Export SNMR-retargeted motion as a holosoma whole-body-tracking training NPZ (design step N5).

Takes a trained Phase-1 checkpoint + a LAFAN1 pair NPZ, runs human→latent→robot inference, and
replays the resulting qpos through MuJoCo FK (mirroring holosoma's ``convert_data_format_mj.py``) to
produce the full training schema consumed by holosoma's ``MotionLoader``:

    fps, joint_pos (T,7+D), joint_vel (T,6+D), body_pos_w (T,B,3), body_quat_w (T,B,4 wxyz),
    body_lin_vel_w, body_ang_vel_w, joint_names, body_names

Interpolates 30 fps → --output_fps (default 50) with lerp/slerp; velocities via finite differences
(SO(3) log-map for angular). The output is schema-validated against the real holosoma sample NPZ.

    python scripts/export_wbt_npz.py --ckpt runs/phase1_g1/ckpt.pt \
        --pair ../data/pairs/unitree_g1/walk1_subject5.npz --out exports/walk1_subject5_snmr.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.data import local_root_to_world  # noqa: E402
from snmr.human import human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

from snmr.paths import g1_mjcf as _g1_mjcf_path  # noqa: E402
from snmr.paths import holosoma_sample_npz  # noqa: E402

REQUIRED_KEYS = [
    "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w", "joint_names", "body_names",
]


def slerp_np(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Minimal wxyz slerp for resampling (vectorised over rows)."""
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    near = sin < 1e-6
    w0 = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(near, 1.0, sin))
    w1 = np.where(near, t, np.sin(t * theta) / np.where(near, 1.0, sin))
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def resample_qpos(qpos: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    T = qpos.shape[0]
    duration = (T - 1) / src_fps
    n_out = int(duration * dst_fps) + 1
    times = np.linspace(0.0, duration, n_out)
    src_idx = times * src_fps
    lo = np.floor(src_idx).astype(int).clip(max=T - 1)
    hi = (lo + 1).clip(max=T - 1)
    frac = (src_idx - lo)[:, None]

    out = np.empty((n_out, qpos.shape[1]), dtype=np.float64)
    # linear for pos + dof
    for sl in (slice(0, 3), slice(7, qpos.shape[1])):
        out[:, sl] = (1 - frac) * qpos[lo, sl] + frac * qpos[hi, sl]
    # slerp for the root quat
    out[:, 3:7] = np.stack(
        [slerp_np(qpos[l, 3:7], qpos[h, 3:7], f) for l, h, f in zip(lo, hi, frac[:, 0])]
    )
    return out


def mujoco_replay(model_path: str, qpos: np.ndarray, fps: float) -> dict:
    """Replay qpos through MuJoCo FK; velocities by finite differences (mirrors holosoma converter)."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    T = qpos.shape[0]
    B = m.nbody
    body_pos = np.zeros((T, B, 3))
    body_quat = np.zeros((T, B, 4))
    for t in range(T):
        d.qpos[:] = qpos[t]
        mujoco.mj_forward(m, d)
        body_pos[t] = d.xpos
        body_quat[t] = d.xquat  # wxyz

    dt = 1.0 / fps
    lin_vel = np.gradient(body_pos, dt, axis=0)
    # angular velocity via quaternion finite difference: w ≈ 2 * (dq/dt) ∘ q^-1 (vector part)
    ang_vel = np.zeros((T, B, 3))
    q0 = body_quat[:-1]
    q1 = body_quat[1:]
    # relative rotation q_rel = q1 * conj(q0), take log map
    w0, xyz0 = q0[..., :1], q0[..., 1:]
    w1, xyz1 = q1[..., :1], q1[..., 1:]
    rel_w = w1 * w0 + (xyz1 * xyz0).sum(-1, keepdims=True)
    rel_xyz = -w1 * xyz0 + w0 * xyz1 + np.cross(xyz1, xyz0)
    flip = rel_w < 0
    rel_w = np.where(flip, -rel_w, rel_w)
    rel_xyz = np.where(flip, -rel_xyz, rel_xyz)
    norm = np.linalg.norm(rel_xyz, axis=-1, keepdims=True).clip(min=1e-12)
    angle = 2.0 * np.arctan2(norm, rel_w)
    ang_vel[1:] = (rel_xyz / norm) * angle / dt
    ang_vel[0] = ang_vel[1]

    body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(B)]
    joint_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]

    # joint_vel: root 6 (lin + ang of pelvis) + dof rates
    dof_vel = np.gradient(qpos[:, 7:], dt, axis=0)
    pelvis = body_names.index("pelvis")
    joint_vel = np.concatenate([lin_vel[:, pelvis], ang_vel[:, pelvis], dof_vel], axis=1)

    return {
        "fps": np.array([int(fps)]),
        "joint_pos": qpos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": lin_vel,
        "body_ang_vel_w": ang_vel,
        "joint_names": np.array(joint_names),
        "body_names": np.array(body_names),
    }


def validate_against_reference(out: dict, reference_npz: str) -> list[str]:
    """Schema check vs the real holosoma sample NPZ: same keys, compatible shapes/conventions."""
    ref = np.load(reference_npz, allow_pickle=True)
    problems = []
    for k in REQUIRED_KEYS:
        if k not in out:
            problems.append(f"missing key {k}")
            continue
        if k in ("fps",):
            continue
        r, o = ref[k], out[k]
        if k in ("joint_names", "body_names"):
            if list(r) != list(o):
                problems.append(f"{k} mismatch (ref {len(r)} vs out {len(o)})")
        elif r.ndim != o.ndim or r.shape[1:] != o.shape[1:]:
            problems.append(f"{k}: shape {o.shape} vs reference {r.shape}")
    # quaternion convention: unit and wxyz (w should dominate for near-upright robots)
    qn = np.linalg.norm(out["joint_pos"][:, 3:7], axis=1)
    if not np.allclose(qn, 1, atol=1e-3):
        problems.append("joint_pos root quat not unit")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--output_fps", type=float, default=50.0)
    ap.add_argument("--mjcf", default=str(_g1_mjcf_path()))
    ap.add_argument("--reference", default=str(holosoma_sample_npz()))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    # model dims come from the trainer config stored in the checkpoint (older ckpts predate the
    # size flags and used the 64/128/128 defaults)
    tcfg = state.get("config", {})
    cfg = SNMRConfig(
        latent_dim=tcfg.get("latent_dim", 64),
        enc_hidden=tcfg.get("enc_hidden", 128),
        dec_hidden=tcfg.get("dec_hidden", 128),
    )
    model = SNMR(cfg).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()

    pair = load_pair_npz(args.pair)
    rk = RobotKinematics(args.mjcf, device=args.device)
    skel = lafan1_skeleton(device=args.device)
    static = human_static_features(skel, body_pos_sample=pair["human_pos"].to(args.device))

    hp = pair["human_pos"].to(args.device)
    hq = pair["human_quat"].to(args.device)
    # trainer stores the fitted robot/human xy trajectory scale in the checkpoint (defaults to 1.0
    # for older checkpoints that predicted relative to the raw human root)
    xy_scale = float(state.get("xy_scale", 1.0))
    anchor_pos = hp[:, 0, :].clone()
    anchor_pos[:, :2] *= xy_scale
    with torch.no_grad():
        pred = model.retarget_human_to_robot(hp, hq, skel, static, rk)
        wp, wq = local_root_to_world(anchor_pos, hq[:, 0, :], pred["root_pos"], pred["root_quat"])
    qpos = torch.cat([wp, wq, pred["dof_pos"]], dim=-1).cpu().numpy().astype(np.float64)

    qpos50 = resample_qpos(qpos, src_fps=pair["fps"], dst_fps=args.output_fps)
    out = mujoco_replay(args.mjcf, qpos50, args.output_fps)

    problems = validate_against_reference(out, args.reference)
    outpath = pathlib.Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outpath, **out)
    print(f"wrote {outpath}  ({qpos50.shape[0]} frames @ {args.output_fps} fps)")
    if problems:
        print("SCHEMA WARNINGS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("schema validation vs holosoma reference NPZ: OK")


if __name__ == "__main__":
    main()
