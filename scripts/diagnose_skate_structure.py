#!/usr/bin/env python
"""E26 diagnostic: is foot skate structurally the time-derivative of foot position error,
and does the E24 foot-lock fail only because contact DETECTION fails on the noisy decoded motion?

Three questions, one clip-set (held-out G1):
  A. Decompose predicted-foot stance error into per-interval DC offset + fluctuation; measure the
     fluctuation's amplitude and timescale (autocorrelation). If skate ~= 2*pi*f*amplitude, training
     losses can only fix skate via amplitude (=MPJPE) or via anchoring — the ratio is structural.
  B. Oracle foot-lock: re-run the E24 leg-IK lock but drive it with the TEACHER contact mask
     (perfect detection). If skate collapses, detection was the failure, not the lock.
  C. Deployable foot-lock: drive the lock with the HUMAN's contact flags (computable at inference
     from the input motion; feet mapped left->left, right->right). Report skate + MPJPE cost.

Usage: python scripts/diagnose_skate_structure.py --ckpt runs/phase1_g1_large/ckpt_100k_final.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.data import local_root_to_world  # noqa: E402
from snmr.human import (  # noqa: E402
    LAFAN1_BODY_NAMES,
    foot_contact_flags,
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.metrics import FOOT_BODIES, compute_metrics, detect_contact  # noqa: E402
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from benchmark import load_model  # noqa: E402
from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402


def intervals(mask_1d: torch.Tensor) -> list[tuple[int, int]]:
    m = mask_1d.to(torch.int8).tolist()
    out, start = [], None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(m)))
    return out


def dilate_mask(mask: torch.Tensor, merge_gap: int = 4, extend: int = 1) -> torch.Tensor:
    """Close gaps <= merge_gap between contact intervals and extend each by `extend` frames.

    The deployable mask (human contact flags) under-covers the teacher's stance frames (~80%
    agreement); every uncovered stance frame keeps its full raw skate. Dilation trades a little
    swing-phase over-locking (harmless: the blend ramp keeps targets near the prediction) for
    stance coverage."""
    out = mask.clone()
    T = mask.shape[0]
    for fi in range(mask.shape[1]):
        ivs = intervals(mask[:, fi])
        merged = []
        for lo, hi in ivs:
            if merged and lo - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], hi)
            else:
                merged.append((lo, hi))
        col = torch.zeros(T, dtype=torch.bool, device=mask.device)
        for lo, hi in merged:
            col[max(lo - extend, 0):min(hi + extend, T)] = True
        out[:, fi] = col
    return out


def foot_lock_masked(kin, root_pos, root_quat, dof_pos, foot_names, contact_mask,
                     iters: int = 8, lr: float = 0.5, damping: float = 1e-2,
                     blend: int = 2) -> torch.Tensor:
    """E24 foot-lock, but contact intervals come from a CALLER-SUPPLIED mask (T, F).

    Adds a linear blend ramp of `blend` frames at interval boundaries (target weight 0->1->0)
    to avoid entry/exit pops that add back velocity at the seams."""
    from snmr.footlock import _leg_dof_indices

    dof = dof_pos.clone()
    foot_idx = [kin.body_index(n) for n in foot_names]
    with torch.no_grad():
        body_pos, _ = kin.forward_kinematics(root_pos, root_quat, dof)
        feet0 = body_pos[:, foot_idx, :]

    for fi, foot in enumerate(foot_names):
        leg_dofs = torch.tensor(_leg_dof_indices(kin, foot), dtype=torch.long, device=dof.device)
        bidx = foot_idx[fi]
        for lo, hi in intervals(contact_mask[:, fi]):
            if hi - lo < 2:
                continue
            with torch.no_grad():
                seg = feet0[lo:hi, fi, :]
                target = torch.empty(3, device=dof.device, dtype=dof.dtype)
                target[:2] = seg[:, :2].median(dim=0).values
                target[2] = seg[:, 2].min()
            for t in range(lo, hi):
                # blend weight: ramp in/out at interval edges
                w = min(1.0, (t - lo + 1) / max(blend, 1), (hi - t) / max(blend, 1))
                tgt_t = w * target + (1 - w) * feet0[t, fi, :]
                q = dof[t].clone().detach().requires_grad_(True)
                for _ in range(iters):
                    bp, _ = kin.forward_kinematics(root_pos[t], root_quat[t], q)
                    err = bp[bidx] - tgt_t
                    loss = (err ** 2).sum()
                    (grad,) = torch.autograd.grad(loss, q)
                    step = torch.zeros_like(q)
                    g = grad[leg_dofs]
                    step[leg_dofs] = g / (g.norm() + damping)
                    q = (q - lr * err.detach().norm() * step).detach().requires_grad_(True)
                dof[t] = q.detach()
    return dof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=["walk1_subject5", "run2_subject1", "dance2_subject4"])
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "runs/skate_structure/diagnosis.json"))
    args = ap.parse_args()

    dev = args.device
    model, state = load_model(args.ckpt, dev)
    ctx = RobotContext(args.robot, dev)
    if "xy_scale" in state:
        ctx.xy_scale = float(state["xy_scale"])
    skel = lafan1_skeleton(device=dev)
    pairs_root = data_root() / "pairs"
    sample = load_pair_npz(str(pairs_root / args.robot / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(dev))
    h_adj = _adjacency(skel)
    feet = FOOT_BODIES[args.robot]
    foot_idx = [ctx.kin.body_index(n) for n in feet]

    report = {"ckpt": args.ckpt, "clips": {}}
    agg = {k: [] for k in ["raw_skate", "oracle_skate", "human_skate", "raw_mpjpe",
                           "oracle_mpjpe", "human_mpjpe", "fluct_amp_cm", "ac_time_s",
                           "dc_cm", "pred_contact_frac", "teacher_contact_frac", "mask_agree",
                           "human_mask_agree"]}

    for clip in args.clips:
        pair = load_pair_npz(str(pairs_root / args.robot / f"{clip}.npz"))
        hp_all, hq_all, q_all = (pair[k].to(dev) for k in ("human_pos", "human_quat", "qpos"))
        fps = pair["fps"]
        T = q_all.shape[0]
        s = max((T - args.window) // 2, 0)
        e = min(s + args.window, T)
        hp, hq, q = hp_all[s:e], hq_all[s:e], q_all[s:e]

        anchor_pos = hp[:, 0, :].clone()
        anchor_pos[:, :2] *= ctx.xy_scale
        anchor_quat = hq[:, 0, :]
        with torch.no_grad():
            feats = human_pose_features(hp, hq)
            z = model.encode(feats, h_static, h_adj)
            pred = model.decoder(z, ctx.static, ctx.adj,
                                 model.embodiment_encoder(ctx.static), ctx.kin.graph)
            wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
            pb, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
            tb, _ = ctx.kin.forward_kinematics(q[:, 0:3], q[:, 3:7], q[:, 7:])

        pfeet, tfeet = pb[:, foot_idx, :], tb[:, foot_idx, :]
        tea_mask = detect_contact(tfeet, fps=fps)                 # (T, F) teacher (oracle)
        prd_mask = detect_contact(pfeet, fps=fps)                 # decoded (E24 baseline)
        # human contact flags -> robot feet (order: left, right in both conventions)
        hflags = foot_contact_flags(hp, LAFAN1_BODY_NAMES)  # (T, C) toes = [LeftToe, RightToe]
        hum_mask = hflags[:, :2].bool() if hflags.shape[1] >= 2 else tea_mask

        # --- A. stance error structure -----------------------------------------------------
        err_xy = (pfeet - tfeet)[:, :, :2]                        # (T, F, 2)
        fl_amp, ac_times, dc_mags = [], [], []
        for fi in range(len(foot_idx)):
            for lo, hi in intervals(tea_mask[:, fi]):
                if hi - lo < 10:
                    continue
                seg = err_xy[lo:hi, fi, :]                        # (L, 2)
                dc = seg.mean(dim=0)
                fluct = seg - dc
                dc_mags.append(float(dc.norm()))
                fl_amp.append(float(fluct.norm(dim=-1).pow(2).mean().sqrt()))
                # autocorrelation 1/e time of the fluctuation (x component)
                x = fluct[:, 0] - fluct[:, 0].mean()
                if float(x.abs().max()) > 1e-9:
                    ac = np.correlate(x.cpu().numpy(), x.cpu().numpy(), "full")
                    ac = ac[len(ac) // 2:]
                    ac = ac / max(ac[0], 1e-12)
                    below = np.nonzero(ac < 1 / np.e)[0]
                    ac_times.append(float(below[0] / fps) if len(below) else (hi - lo) / fps)

        # --- B/C. foot-lock variants --------------------------------------------------------
        ref = (q[:, 0:3], q[:, 3:7], q[:, 7:])
        m_raw = compute_metrics(ctx.kin, wp, wq, pred["dof_pos"], fps, feet, reference=ref)
        dof_orc = foot_lock_masked(ctx.kin, wp, wq, pred["dof_pos"], feet, tea_mask)
        m_orc = compute_metrics(ctx.kin, wp, wq, dof_orc, fps, feet, reference=ref)
        dof_hum = foot_lock_masked(ctx.kin, wp, wq, pred["dof_pos"], feet, hum_mask)
        m_hum = compute_metrics(ctx.kin, wp, wq, dof_hum, fps, feet, reference=ref)
        hum_dil = dilate_mask(hum_mask, merge_gap=6, extend=2)
        dof_dil = foot_lock_masked(ctx.kin, wp, wq, pred["dof_pos"], feet, hum_dil, iters=12)
        m_dil = compute_metrics(ctx.kin, wp, wq, dof_dil, fps, feet, reference=ref)

        row = {
            "raw": {"skate": m_raw.foot_skate_speed_ms, "mpjpe_cm": m_raw.mpjpe_m * 100},
            "oracle_lock": {"skate": m_orc.foot_skate_speed_ms, "mpjpe_cm": m_orc.mpjpe_m * 100},
            "human_lock": {"skate": m_hum.foot_skate_speed_ms, "mpjpe_cm": m_hum.mpjpe_m * 100},
            "human_dilated_lock": {"skate": m_dil.foot_skate_speed_ms, "mpjpe_cm": m_dil.mpjpe_m * 100,
                                   "jerk": m_dil.dof_jerk, "joint_jumps": m_dil.joint_jump_fraction},
            "stance_dc_offset_cm": float(np.mean(dc_mags)) * 100 if dc_mags else None,
            "stance_fluct_rms_cm": float(np.mean(fl_amp)) * 100 if fl_amp else None,
            "fluct_autocorr_time_s": float(np.mean(ac_times)) if ac_times else None,
            "pred_contact_frac": float(prd_mask.float().mean()),
            "teacher_contact_frac": float(tea_mask.float().mean()),
            "pred_mask_agreement": float((prd_mask == tea_mask).float().mean()),
            "human_mask_agreement": float((hum_mask == tea_mask).float().mean()),
        }
        report["clips"][clip] = row
        agg["raw_skate"].append(row["raw"]["skate"]); agg["oracle_skate"].append(row["oracle_lock"]["skate"])
        agg["human_skate"].append(row["human_lock"]["skate"]); agg["raw_mpjpe"].append(row["raw"]["mpjpe_cm"])
        agg["oracle_mpjpe"].append(row["oracle_lock"]["mpjpe_cm"]); agg["human_mpjpe"].append(row["human_lock"]["mpjpe_cm"])
        if row["stance_fluct_rms_cm"]: agg["fluct_amp_cm"].append(row["stance_fluct_rms_cm"])
        if row["fluct_autocorr_time_s"]: agg["ac_time_s"].append(row["fluct_autocorr_time_s"])
        if row["stance_dc_offset_cm"]: agg["dc_cm"].append(row["stance_dc_offset_cm"])
        agg["pred_contact_frac"].append(row["pred_contact_frac"])
        agg["teacher_contact_frac"].append(row["teacher_contact_frac"])
        agg["mask_agree"].append(row["pred_mask_agreement"])
        agg["human_mask_agree"].append(row["human_mask_agreement"])
        print(f"{clip}: raw skate {row['raw']['skate']:.3f} -> oracle-lock {row['oracle_lock']['skate']:.3f} "
              f"-> human-lock {row['human_lock']['skate']:.3f} m/s | mpjpe {row['raw']['mpjpe_cm']:.1f} -> "
              f"{row['oracle_lock']['mpjpe_cm']:.1f} / {row['human_lock']['mpjpe_cm']:.1f} cm | "
              f"fluct {row['stance_fluct_rms_cm']} cm, tau {row['fluct_autocorr_time_s']} s", flush=True)

    report["mean"] = {k: float(np.mean(v)) for k, v in agg.items() if v}
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nMEANS:", json.dumps(report["mean"], indent=1))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
