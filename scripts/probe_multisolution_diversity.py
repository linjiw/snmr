#!/usr/bin/env python
"""W1 premise probe: can we manufacture DIVERSE, physically-clean solutions per window?

Flow v3 gate W1 (docs/FLOW_RETARGETING_V3_PROTOCOL.md) plans a one-to-many dataset: for each
decoded training window, K variants produced by the frozen windowed projection under jittered
configurations, all vetted by the frozen guards. That plan rests on an untested premise — that
jittering the projection ACTUALLY yields distinct solutions rather than K copies of the same
fixed point. This probe measures it directly (descriptive, no training):

  per window: decode raw SNMR output, run K=4 oracle-mask (teacher-height) projections with
  jittered {joint_delta_bound, root bounds, deviation/velocity weights, stance-anchor dilation},
  then report (a) guard compliance per variant, (b) pairwise dof-space diversity among ACCEPTED
  variants (the v3 flow's usable conditional spread), (c) stance-speed distribution.

Diversity yardstick: E44's flow endpoint diversity was 0.4% of ||z||; MModality-style dof
diversity for real one-to-many data (SafeFlow Table: 1.0-1.4 rad) is the aspirational scale.
The W1 gate needs "diversity >= 5% of raw deviation scale with >= 3/4 variants guard-clean".

    python scripts/probe_multisolution_diversity.py --out runs/latent_flow/w1_diversity_probe.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.data import local_root_to_world  # noqa: E402
from snmr.experiment import git_state, runtime_state, sha256_file, utc_now  # noqa: E402
from snmr.human import (  # noqa: E402
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.metrics import FOOT_BODIES, contact_motion_metrics, detect_contact_height_hysteresis  # noqa: E402
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402
from snmr.projection import WindowedProjectionConfig, windowed_contact_projection  # noqa: E402

from benchmark import load_model  # noqa: E402
from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402


# K=4 jittered projection configurations. The base (k0) is the frozen Gate-1b configuration;
# k1-k3 vary the correction budget and smoothness/deviation trade-off inside registered bounds.
VARIANT_CONFIGS = {
    "k0_frozen": {},
    "k1_tight": {"joint_delta_bound_rad": 0.20, "root_translation_bound_m": 0.02,
                 "deviation_weight": 0.3},
    "k2_loose": {"joint_delta_bound_rad": 0.50, "root_translation_bound_m": 0.06,
                 "deviation_weight": 0.03, "velocity_weight": 0.25},
    "k3_smooth": {"acceleration_weight": 3.0, "velocity_weight": 1.5, "extend": 2,
                  "merge_gap": 4},
}

# Exploratory (--anchor_jitter): config jitter converges because every variant is pinned to
# the SAME stance anchors (build_stance_anchors uses the original decoded feet). Anchor jitter
# instead offsets each variant's stance TARGETS by a per-interval xy displacement — different
# but equally-planted foot placements, i.e. diversity by construction if the projection can
# reach them within guards.
ANCHOR_JITTER_XY_M = {
    "a0_none": (0.0, 0.0),
    "a1_left": (-0.02, 0.0),
    "a2_right": (0.02, 0.0),
    "a3_fwd": (0.0, 0.02),
}


def guard_check(res: dict, raw: dict) -> dict[str, bool]:
    """Frozen Gate-1b relative guards (speed endpoints handled separately)."""
    checks = {
        "teacher_height_speed_le_0.08": (
            res["teacher_height_stance_speed_ms"] is not None
            and res["teacher_height_stance_speed_ms"] <= 0.08
        ),
        "mpjpe_delta_le_0.005": res["mpjpe_m"] <= raw["mpjpe_m"] + 0.005,
        "dof_jerk_le_1.2x": res["dof_jerk"] <= 1.2 * raw["dof_jerk"],
        "zero_limit_violations": res["limit_violation_fraction"] == 0.0,
        "penetration_mean_guard": res["penetration_mean_m"] <= raw["penetration_mean_m"] + 0.002,
    }
    checks["accepted"] = all(checks.values())
    return checks


def score(ctx, wp, wq, dof, reference, fps, th_mask) -> dict:
    from snmr.metrics import compute_metrics

    m = compute_metrics(ctx.kin, wp, wq, dof, fps, FOOT_BODIES[ctx.name],
                        reference=reference, contact_mask=th_mask)
    body, _ = ctx.kin.forward_kinematics(wp, wq, dof)
    cm = contact_motion_metrics(body[:, ctx.foot_idx, :], fps, th_mask)
    row = m.as_dict()
    row["teacher_height_stance_speed_ms"] = cm["stance_speed_ms"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--windows_per_clip", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--anchor_jitter", action="store_true",
                    help="exploratory: jitter stance-anchor xy targets instead of solver config")
    ap.add_argument("--out", default=str(ROOT / "runs/latent_flow/w1_diversity_probe.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outp = pathlib.Path(args.out)
    if outp.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {outp}")

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

    per_window = []
    for clip in args.clips:
        pair = load_pair_npz(str(pairs_root / args.robot / f"{clip}.npz"))
        hp_all, hq_all, q_all = (pair[k].to(dev) for k in ("human_pos", "human_quat", "qpos"))
        fps = pair["fps"]
        T = q_all.shape[0]
        with torch.no_grad():
            tb, _ = ctx.kin.forward_kinematics(q_all[:, 0:3], q_all[:, 3:7], q_all[:, 7:])
        th_mask_full = detect_contact_height_hysteresis(
            tb[:, ctx.foot_idx, :], enter_height=0.03, exit_height=0.05
        )
        starts = np.linspace(0, max(T - args.window, 0),
                             num=min(args.windows_per_clip, max(T // args.window, 1)), dtype=int)
        for s in starts:
            s = int(s)
            e = s + args.window
            hp, hq, q = hp_all[s:e], hq_all[s:e], q_all[s:e]
            anchor_pos = hp[:, 0, :].clone()
            anchor_pos[:, :2] *= ctx.xy_scale
            anchor_quat = hq[:, 0, :]
            reference = (q[:, 0:3], q[:, 3:7], q[:, 7:])
            th_mask = th_mask_full[s:e]

            with torch.no_grad():
                z = model.encode(human_pose_features(hp, hq), h_static, h_adj)
                pred = model.decoder(
                    z, ctx.static, ctx.adj, model.embodiment_encoder(ctx.static), ctx.kin.graph
                )
                wp, wq = local_root_to_world(
                    anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"]
                )
            raw_row = score(ctx, wp, wq, pred["dof_pos"], reference, fps, th_mask)

            variants = {}
            accepted_dofs = []
            variant_calls = {
                name: {"config": WindowedProjectionConfig(**ov), "anchor_offset_xy": None}
                for name, ov in VARIANT_CONFIGS.items()
            }
            if args.anchor_jitter:
                variant_calls = {
                    name: {
                        "config": WindowedProjectionConfig(),
                        "anchor_offset_xy": torch.tensor(
                            [[dx, dy], [dx, dy]], dtype=pred["dof_pos"].dtype, device=dev
                        ),
                    }
                    for name, (dx, dy) in ANCHOR_JITTER_XY_M.items()
                }
            for name, call in variant_calls.items():
                res = windowed_contact_projection(
                    ctx.kin, wp, wq, pred["dof_pos"], feet, th_mask,
                    config=call["config"], anchor_offset_xy=call["anchor_offset_xy"],
                )
                row = score(ctx, res.root_pos, res.root_quat, res.dof_pos,
                            reference, fps, th_mask)
                row["guards"] = guard_check(row, raw_row)
                row["projection_accepted_flag"] = bool(
                    res.diagnostics.get("accepted", True)
                )
                variants[name] = row
                if row["guards"]["accepted"]:
                    accepted_dofs.append(res.dof_pos)

            # pairwise dof-space diversity among guard-accepted variants (rad, MModality-style)
            div = None
            if len(accepted_dofs) >= 2:
                dists = [
                    float((accepted_dofs[i] - accepted_dofs[j]).abs().mean())
                    for i in range(len(accepted_dofs))
                    for j in range(i + 1, len(accepted_dofs))
                ]
                div = float(np.mean(dists))
            # scale yardstick: mean |projection edit| of the frozen config vs raw
            edit_scale = float((
                torch.cat(accepted_dofs) if accepted_dofs else pred["dof_pos"]
            ).new_tensor(0.0))
            if accepted_dofs:
                edit_scale = float(np.mean([
                    float((d - pred["dof_pos"]).abs().mean()) for d in accepted_dofs
                ]))

            per_window.append({
                "clip": clip, "start": s,
                "raw": raw_row,
                "variants": variants,
                "num_accepted": len(accepted_dofs),
                "accepted_pairwise_dof_diversity_rad": div,
                "mean_edit_scale_rad": edit_scale,
            })
            print(f"{clip}@{s}: accepted {len(accepted_dofs)}/4, "
                  f"diversity {div}", flush=True)

    n_windows = len(per_window)
    divs = [r["accepted_pairwise_dof_diversity_rad"] for r in per_window
            if r["accepted_pairwise_dof_diversity_rad"] is not None]
    edits = [r["mean_edit_scale_rad"] for r in per_window if r["num_accepted"]]
    summary = {
        "num_windows": n_windows,
        "windows_with_ge2_accepted": len(divs),
        "windows_with_ge3_accepted": sum(1 for r in per_window if r["num_accepted"] >= 3),
        "mean_accepted_per_window": float(np.mean([r["num_accepted"] for r in per_window])),
        "mean_pairwise_dof_diversity_rad": float(np.mean(divs)) if divs else None,
        "mean_edit_scale_rad": float(np.mean(edits)) if edits else None,
        "diversity_over_edit_ratio": (
            float(np.mean(divs)) / max(float(np.mean(edits)), 1e-9) if divs and edits else None
        ),
        "variant_acceptance": {
            name: float(np.mean([
                r["variants"][name]["guards"]["accepted"] for r in per_window
            ]))
            for name in per_window[0]["variants"]
        },
        "variant_stance_speed_mean": {
            name: float(np.mean(vals)) if (vals := [
                r["variants"][name]["teacher_height_stance_speed_ms"] for r in per_window
                if r["variants"][name]["teacher_height_stance_speed_ms"] is not None
            ]) else None
            for name in per_window[0]["variants"]
        },
    }

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "provenance": {
            "created_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "git": git_state(ROOT),
            "runtime": runtime_state(dev),
            "checkpoint": {"path": str(pathlib.Path(args.ckpt).resolve()),
                           "sha256": sha256_file(args.ckpt)},
            "variant_configs": (
                {name: {"anchor_offset_xy_m": list(off)}
                 for name, off in ANCHOR_JITTER_XY_M.items()}
                if args.anchor_jitter else
                {name: dataclasses.asdict(WindowedProjectionConfig(**ov))
                 for name, ov in VARIANT_CONFIGS.items()}
            ),
        },
        "config": vars(args),
        "summary": summary,
        "per_window": per_window,
    }, indent=2))
    print(f"\nwrote {outp}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
