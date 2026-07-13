#!/usr/bin/env python
"""Export a WBT training NPZ augmented with the SNMR per-frame latent z (N10/C2 data artifact).

The "does the latent benefit RL training" experiment (C2) needs the tracking policy to observe the
shared latent z_t alongside (or instead of) the raw reference. This produces a holosoma-WBT NPZ
with an EXTRA `latent_z` field (T, latent_dim) time-aligned to the 50-fps `joint_pos` frames.

Backward-compatible by construction: holosoma's MotionLoader reads only its required keys and
ignores unknown ones (verified: `managers/command/terms/wbt.py:73` selective `data[...]` reads, no
rejection of extras). So a standard `g1-29dof-wbt` run ignores `latent_z`; a z-command run
(new observation term, separate change) reads it.

Pipeline: human pair -> SNMR encode -> z (30 fps) AND decode -> qpos (30 fps); resample BOTH to
50 fps (z by linear interp — it's a smooth latent), qpos via the existing resample; MuJoCo-replay
the qpos for body kinematics; attach the resampled z as `latent_z`. The decoded motion and its
latent stay consistent because they come from the same forward pass.

    python scripts/export_wbt_with_latent.py --ckpt runs/phase2_all5/ckpt_100k_final.pt \
        --pair ../data/pairs/unitree_g1/walk1_subject5.npz --out runs/wbt_latent/walk1_z.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.data import local_root_to_world  # noqa: E402
from snmr.human import human_pose_features, human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402
from snmr.paths import g1_mjcf, holosoma_sample_npz  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

from export_wbt_npz import mujoco_replay, resample_qpos, validate_against_reference  # noqa: E402


def resample_latent(z: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linear interpolation of the (T, d) latent to dst_fps — z is smooth, no slerp needed."""
    T = z.shape[0]
    duration = (T - 1) / src_fps
    n_out = int(duration * dst_fps) + 1
    src_idx = np.linspace(0.0, duration, n_out) * src_fps
    lo = np.floor(src_idx).astype(int).clip(max=T - 1)
    hi = (lo + 1).clip(max=T - 1)
    frac = (src_idx - lo)[:, None]
    return (1 - frac) * z[lo] + frac * z[hi]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--output_fps", type=float, default=50.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    tc = state.get("config", {})
    sd = state["model"]
    model = SNMR(SNMRConfig(
        latent_dim=tc.get("latent_dim", 128), enc_hidden=tc.get("enc_hidden", 256),
        dec_hidden=tc.get("dec_hidden", 256),
        use_temporal=any(k.startswith("encoder.temporal.") for k in sd),
        predict_contact=any(k.startswith("decoder.contact_head.") for k in sd),
    )).to(args.device)
    model.load_state_dict(sd)
    model.eval()
    xy_scale = float(state.get("xy_scale", state.get("xy_scales", {}).get(args.robot, 0.875)))

    mjcf = str(g1_mjcf())
    rk = RobotKinematics(mjcf, device=args.device)
    skel = lafan1_skeleton(device=args.device)
    pair = load_pair_npz(args.pair)
    h_static = human_static_features(skel, body_pos_sample=pair["human_pos"].to(args.device))
    hp = pair["human_pos"].to(args.device)
    hq = pair["human_quat"].to(args.device)
    anchor = hp[:, 0, :].clone()
    anchor[:, :2] *= xy_scale

    with torch.no_grad():
        z = model.encode(human_pose_features(hp, hq), h_static, _adjacency(skel))  # (T, d) @30fps
        pred = model.decode(z, rk)
        wp, wq = local_root_to_world(anchor, hq[:, 0, :], pred["root_pos"], pred["root_quat"])
    qpos30 = torch.cat([wp, wq, pred["dof_pos"]], dim=-1).cpu().numpy().astype(np.float64)
    z30 = z.cpu().numpy().astype(np.float32)

    qpos50 = resample_qpos(qpos30, src_fps=pair["fps"], dst_fps=args.output_fps)
    z50 = resample_latent(z30, src_fps=pair["fps"], dst_fps=args.output_fps)
    # align lengths (resamplers round independently)
    n = min(qpos50.shape[0], z50.shape[0])
    qpos50, z50 = qpos50[:n], z50[:n]

    out = mujoco_replay(mjcf, qpos50, args.output_fps)
    out["latent_z"] = z50.astype(np.float32)

    problems = validate_against_reference(out, str(holosoma_sample_npz()))
    outpath = pathlib.Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outpath, **out)
    print(f"wrote {outpath}  ({n} frames @ {args.output_fps} fps, latent_z {z50.shape})")
    if problems:
        print("SCHEMA WARNINGS (standard keys):", problems)
        sys.exit(1)
    print("standard-schema validation OK; latent_z attached as an extra (ignored by vanilla WBT).")


if __name__ == "__main__":
    main()
