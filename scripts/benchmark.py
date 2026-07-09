#!/usr/bin/env python
"""Benchmark a SNMR checkpoint against the GMR teacher on held-out clips (design doc §6).

For every robot the checkpoint supports, and every held-out clip window:
  * runs SNMR inference (human -> latent -> robot qpos, world-recomposed via the per-robot scale),
  * scores BOTH the SNMR prediction and the GMR-teacher trajectory with the identical metric code
    (`snmr.metrics`): MPJPE (SNMR vs teacher only), foot-skate, penetration, jerk, joint limits,
  * measures inference throughput (frames/s),
and writes a JSON + a human-readable markdown table to --out.

The teacher rows quantify the *reference cost* of each metric (e.g. the teacher itself has some
foot skate); SNMR should match or beat the teacher on physical-plausibility metrics while staying
close in MPJPE — that framing is the GMR-paper methodology inverted, and matches how AdaMorph/NMR
report retargeter quality.

    python scripts/benchmark.py --ckpt runs/phase1_g1_large/ckpt_100k_final.pt --robots unitree_g1
    python scripts/benchmark.py --ckpt runs/phase2_all5/ckpt.pt   # all robots in the checkpoint
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.data import local_root_to_world  # noqa: E402
from snmr.human import human_pose_features, human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.metrics import FOOT_BODIES, compute_metrics  # noqa: E402
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402

from train_phase2 import ALL_ROBOTS, ROBOT_XY_SCALE_CONFIG, VAL_CLIPS, RobotContext  # noqa: E402

REPO = ROOT.parent


def load_model(ckpt_path: str, device: str) -> tuple[SNMR, dict]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    tcfg = state.get("config", {})
    cfg = SNMRConfig(
        latent_dim=tcfg.get("latent_dim", 64),
        enc_hidden=tcfg.get("enc_hidden", 128),
        dec_hidden=tcfg.get("dec_hidden", 128),
    )
    model = SNMR(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.no_grad()
def benchmark_robot(model, ctx: RobotContext, skel, h_static, h_adj, pairs_root, device,
                    window: int, windows_per_clip: int) -> dict:
    rows_pred, rows_teacher = [], []
    throughput_frames = 0
    throughput_time = 0.0
    feet = FOOT_BODIES[ctx.name]

    for clip in VAL_CLIPS:
        pair = load_pair_npz(str(pairs_root / ctx.name / f"{clip}.npz"))
        hp_all = pair["human_pos"].to(device)
        hq_all = pair["human_quat"].to(device)
        q_all = pair["qpos"].to(device)
        fps = pair["fps"]
        T = q_all.shape[0]
        starts = np.linspace(0, max(T - window, 0),
                             num=min(windows_per_clip, max(T // window, 1)), dtype=int)
        for s in starts:
            e = int(s) + window
            hp, hq, q = hp_all[int(s):e], hq_all[int(s):e], q_all[int(s):e]
            anchor_pos = hp[:, 0, :].clone()
            anchor_pos[:, :2] *= ctx.xy_scale
            anchor_quat = hq[:, 0, :]

            t0 = time.time()
            feats = human_pose_features(hp, hq)
            z = model.encode(feats, h_static, h_adj)
            pred = model.decoder(z, ctx.static, ctx.adj,
                                 model.embodiment_encoder(ctx.static), ctx.kin.graph)
            wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
            if device == "cuda":
                torch.cuda.synchronize()
            throughput_time += time.time() - t0
            throughput_frames += hp.shape[0]

            reference = (q[:, 0:3], q[:, 3:7], q[:, 7:])
            m_pred = compute_metrics(ctx.kin, wp, wq, pred["dof_pos"], fps, feet, reference=reference)
            m_teacher = compute_metrics(ctx.kin, *reference, fps, feet, reference=None)
            rows_pred.append(m_pred.as_dict())
            rows_teacher.append(m_teacher.as_dict())

    def agg(rows):
        keys = [k for k in rows[0] if rows[0][k] is not None]
        return {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys}

    return {
        "snmr": agg(rows_pred),
        "teacher": agg(rows_teacher),
        "throughput_fps": throughput_frames / max(throughput_time, 1e-9),
        "num_windows": len(rows_pred),
    }


def to_markdown(results: dict, ckpt: str) -> str:
    lines = [f"# SNMR benchmark — `{ckpt}`", "",
             "SNMR scored against the GMR teacher on held-out clips "
             f"({', '.join(VAL_CLIPS)}). Teacher rows = the optimization baseline's own metric "
             "values (no MPJPE: it is the reference).", ""]
    metrics_order = [
        ("mpjpe_m", "MPJPE (m)"), ("dof_err_rad", "dof err (rad)"),
        ("foot_skate_speed_ms", "foot skate (m/s)"), ("foot_slide_fraction", "slide frac"),
        ("foot_skate_mann_cm", "FS-MANN (cm/f)"),
        ("penetration_mean_m", "pen. mean (m)"), ("penetration_fraction", "pen. frac"),
        ("dof_jerk", "dof jerk (rad/s³)"), ("body_jerk", "body jerk (m/s³)"),
        ("joint_jump_fraction", "joint jumps"),
        ("limit_violation_fraction", "limit viol."),
        ("limit_proximity_fraction", "limit prox."),
    ]
    for robot, res in results.items():
        lines.append(f"## {robot}  ({res['num_windows']} windows, "
                     f"{res['throughput_fps']:.0f} frames/s inference)")
        lines.append("| metric | SNMR | teacher (GMR) |")
        lines.append("|---|---|---|")
        for key, label in metrics_order:
            sv = res["snmr"].get(key)
            tv = res["teacher"].get(key)
            fmt = lambda v: "—" if v is None else (f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
            lines.append(f"| {label} | {fmt(sv)} | {fmt(tv)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--robots", nargs="+", default=None,
                    help="default: all robots with pairs data")
    ap.add_argument("--pairs_root", default=str(REPO / "data" / "pairs"))
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--windows_per_clip", type=int, default=6)
    ap.add_argument("--out", default=None, help="output prefix (default: alongside ckpt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, state = load_model(args.ckpt, args.device)
    robots = args.robots or ALL_ROBOTS
    pairs_root = pathlib.Path(args.pairs_root)

    skel = lafan1_skeleton(device=args.device)
    sample = load_pair_npz(str(pairs_root / robots[0] / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(args.device))
    h_adj = _adjacency(skel)

    results = {}
    for robot in robots:
        ctx = RobotContext(robot, args.device)
        # allow per-checkpoint scale override (phase-1 ckpts store a single fitted value)
        if "xy_scale" in state:
            ctx.xy_scale = float(state["xy_scale"])
        elif "xy_scales" in state:
            ctx.xy_scale = float(state["xy_scales"].get(robot, ROBOT_XY_SCALE_CONFIG[robot]))
        print(f"benchmarking {robot} (xy_scale={ctx.xy_scale:.4f}) ...")
        results[robot] = benchmark_robot(model, ctx, skel, h_static, h_adj, pairs_root,
                                         args.device, args.window, args.windows_per_clip)

    prefix = pathlib.Path(args.out) if args.out else pathlib.Path(args.ckpt).parent / "benchmark"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}.json", "w") as fh:
        json.dump(results, fh, indent=2)
    md = to_markdown(results, args.ckpt)
    with open(f"{prefix}.md", "w") as fh:
        fh.write(md)
    print(md)
    print(f"\nwrote {prefix}.json and {prefix}.md")


if __name__ == "__main__":
    main()
