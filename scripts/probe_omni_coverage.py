#!/usr/bin/env python3
"""E55-B: can the current SNMR latent/decoder manifold express OmniRetarget's
interaction-rich G1 poses? (data-vs-model attribution probe)

Two measurements, both cheap (fits beside a training run in <6 GB):

1. POSE NOVELTY (data-side): nearest-neighbor DoF distance from OmniRetarget poses to the
   entire LAFAN1/GMR training pose set. High novelty = the clips add genuinely new poses.
2. LATENT INVERSION (model-side): per frame, optimize z (128-d) to minimize
   ||decoder(z, G1) - qpos_dof||. If the optimized residual is small even for novel poses,
   the decoder manifold COVERS interaction-rich poses and the gap is data/conditioning:
   adding clips (+ possibly terrain input) suffices. If residual is large, the
   representation itself is the bottleneck (per-frame z can't express these poses).

Reference floor: the same inversion run on held-out LAFAN1 frames (the decoder was trained
on this distribution; its inversion residual is the achievable floor).

Run in .venv-snmr. Uses CUDA if free memory allows, else CPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_latent_contact import load_model  # noqa: E402  (loads SNMR ckpt + config)


def invert_frames(model, kin, dof_target: torch.Tensor, iters: int, lr: float) -> torch.Tensor:
    """Optimize z per frame to reproduce dof_target through the frozen decoder.

    Returns per-frame RMSE (rad) after optimization."""
    n = dof_target.shape[0]
    z = torch.zeros(n, model.cfg.latent_dim, device=dof_target.device, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        out = model.decode(z, kin)
        loss = (out["dof_pos"] - dof_target).square().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = model.decode(z, kin)
        return (out["dof_pos"] - dof_target).square().mean(dim=-1).sqrt()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phase1_g1_large/ckpt_100k_final.pt")
    ap.add_argument("--omni_glob", default="/home/ec2-user/work/retarget/data/omniretarget/*/*/*.npz")
    ap.add_argument("--pairs", default="/home/ec2-user/work/retarget/data/pairs/unitree_g1")
    ap.add_argument("--frames_per_clip", type=int, default=40)
    ap.add_argument("--max_clips", type=int, default=60)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="runs/e55b_omni_coverage.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    model.eval()
    from snmr.paths import robot_mjcf
    from snmr.robot_model import RobotKinematics
    kin = RobotKinematics(str(robot_mjcf("unitree_g1")), device=device)

    rng = np.random.default_rng(0)

    # ---- training pose bank (for novelty NN) ----
    train_files = sorted(glob.glob(f"{args.pairs}/*.npz"))
    bank = []
    for f in train_files:
        q = np.load(f, allow_pickle=True)["qpos"][:, 7:]
        idx = rng.choice(len(q), size=min(400, len(q)), replace=False)
        bank.append(q[idx])
    bank_t = torch.from_numpy(np.concatenate(bank)).float().to(device)   # (~30k, 29)
    print(f"pose bank: {bank_t.shape[0]} poses from {len(train_files)} training clips")

    def novelty(dof: torch.Tensor) -> torch.Tensor:
        # min joint-RMSE distance to the training bank, chunked
        out = []
        for chunk in dof.split(256):
            d = (chunk.unsqueeze(1) - bank_t.unsqueeze(0)).square().mean(-1).sqrt()  # (c, bank)
            out.append(d.min(dim=1).values)
        return torch.cat(out)

    results = {}

    # ---- reference floor: held-out LAFAN1 frames ----
    VAL = ["walk1_subject5", "dance2_subject4", "fight1_subject3", "jumps1_subject2"]
    ref_dof = []
    for clip in VAL:
        q = np.load(f"{args.pairs}/{clip}.npz", allow_pickle=True)["qpos"][:, 7:]
        idx = rng.choice(len(q), size=args.frames_per_clip, replace=False)
        ref_dof.append(q[idx])
    ref_dof_t = torch.from_numpy(np.concatenate(ref_dof)).float().to(device)
    ref_res = invert_frames(model, kin, ref_dof_t, args.iters, args.lr)
    ref_nov = novelty(ref_dof_t)
    results["lafan1_heldout"] = {
        "n": int(ref_dof_t.shape[0]),
        "inversion_rmse_rad_mean": float(ref_res.mean()),
        "inversion_rmse_rad_p95": float(ref_res.quantile(0.95)),
        "novelty_rad_mean": float(ref_nov.mean()),
    }
    print("floor (LAFAN1 held-out):", results["lafan1_heldout"])

    # ---- OmniRetarget subsets ----
    omni_files = sorted(glob.glob(args.omni_glob))
    subsets = {}
    for f in omni_files:
        subset = pathlib.Path(f).parts[-3]
        subsets.setdefault(subset, []).append(f)
    for subset, files in subsets.items():
        take = files[: args.max_clips]
        dofs = []
        for f in take:
            d = np.load(f, allow_pickle=True)
            q = np.asarray(d["qpos"])[:, 7:36]  # quat-first layout: dof = cols 7..36
            idx = rng.choice(len(q), size=min(args.frames_per_clip, len(q)), replace=False)
            dofs.append(q[idx])
        dof_t = torch.from_numpy(np.concatenate(dofs)).float().to(device)
        res = invert_frames(model, kin, dof_t, args.iters, args.lr)
        nov = novelty(dof_t)
        # correlation: does inversion difficulty track novelty?
        r = float(np.corrcoef(nov.cpu().numpy(), res.cpu().numpy())[0, 1])
        results[subset] = {
            "n": int(dof_t.shape[0]),
            "clips": len(take),
            "inversion_rmse_rad_mean": float(res.mean()),
            "inversion_rmse_rad_p95": float(res.quantile(0.95)),
            "novelty_rad_mean": float(nov.mean()),
            "novelty_rad_p95": float(nov.quantile(0.95)),
            "corr_novelty_inversion": r,
        }
        print(subset, results[subset])

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
