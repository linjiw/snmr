#!/usr/bin/env python
"""Overfit SNMR to a single real motion clip and report reconstruction quality vs. optimization steps.

This is the Phase-1 Gate G1 smoke demonstration: it proves the encode->latent->decode pipeline can
represent genuine motion (the prerequisite for training on a dataset). Uses the real Unitree G1 MJCF
and the holosoma whole-body-tracking NPZ that ships in this repo.

    python scripts/overfit_batch.py --frames 60 --steps 800 --lr 3e-3
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr import rotation as rot  # noqa: E402
from snmr.data import RobotMotion, load_holosoma_wbt_npz  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402
from snmr.train import fit_motion  # noqa: E402

from snmr.paths import g1_mjcf, holosoma_sample_npz  # noqa: E402

DEFAULT_MJCF = g1_mjcf()
DEFAULT_NPZ = holosoma_sample_npz()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rk = RobotKinematics(args.mjcf)
    full = load_holosoma_wbt_npz(args.npz)
    n = min(args.frames, full.num_frames)
    motion = RobotMotion(full.root_pos[:n], full.root_quat[:n], full.dof_pos[:n], full.fps)
    print(f"Robot: {rk.num_bodies} bodies, {rk.num_dof} dof | clip: {n} frames @ {motion.fps} fps")

    model = SNMR(SNMRConfig(latent_dim=64, enc_hidden=128, dec_hidden=128))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    res = fit_motion(model, rk, motion, steps=args.steps, lr=args.lr, verbose=True)

    pred = res.final_pred
    bp_pred, _ = rk.forward_kinematics(pred["root_pos"], pred["root_quat"], pred["dof_pos"])
    bp_tgt, _ = rk.forward_kinematics(motion.root_pos, motion.root_quat, motion.dof_pos)
    mpjpe = (bp_pred - bp_tgt).norm(dim=-1).mean().item()
    dof_err = (pred["dof_pos"] - motion.dof_pos).abs().mean().item()
    ang_err = rot.quat_geodesic_angle(pred["root_quat"], motion.root_quat).mean().item()
    print("\n=== final reconstruction ===")
    print(f"MPJPE (whole body):    {mpjpe*100:.2f} cm")
    print(f"mean |dof error|:      {dof_err:.4f} rad")
    print(f"root orientation err:  {ang_err:.4f} rad")


if __name__ == "__main__":
    main()
