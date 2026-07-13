#!/usr/bin/env python
"""Gate-3 diagnostic: per-robot gradient conflict on the shared trunk (audit §Gate 3).

Before scaling capacity or adding adapters, measure WHY sharing costs fidelity: for the same
human window, compute each robot's distill-loss gradient on the shared parameters and report
(a) the pairwise gradient cosine matrix, (b) per-robot gradient norms, (c) the fraction of
negative-cosine pairs. Decision rule (PCGrad/GradNorm literature): pervasive negative cosines →
gradient-surgery arm (S4); norm imbalance → loss-balancing arm (S5); neither → capacity/
conditioning arms (S2/S3).

Runs on a trained phase-2 checkpoint; gradients are evaluated at that converged point over
sampled windows (mean over samples). CPU-safe (small model).

    python scripts/diagnose_sharing_gradients.py --ckpt runs/phase2_all5/ckpt_100k_final.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.human import human_pose_features, human_static_features, lafan1_skeleton  # noqa: E402
from snmr.losses import total_loss  # noqa: E402
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from benchmark import load_model  # noqa: E402
from train_phase2 import ALL_ROBOTS, VAL_CLIPS, RobotContext, load_clips, window_teacher  # noqa: E402


def shared_param_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    """Parameters every robot's decode path uses: encoder trunk + decoder trunk + embodiment enc."""
    groups = {"encoder": [], "decoder_trunk": [], "embodiment_encoder": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            groups["encoder"].append(p)
        elif name.startswith("embodiment_encoder."):
            groups["embodiment_encoder"].append(p)
        elif name.startswith("decoder."):
            groups["decoder_trunk"].append(p)
    return groups


def flat_grad(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase2_all5/ckpt_100k_final.pt"))
    ap.add_argument("--num_windows", type=int, default=16)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "runs/phase2_all5/sharing_gradient_diagnosis.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dev = args.device
    model, state = load_model(args.ckpt, dev)
    model.train()
    ctxs = {r: RobotContext(r, dev) for r in ALL_ROBOTS}
    skel = lafan1_skeleton(device=dev)
    pairs_root = data_root() / "pairs"
    train_clips, val_clips = load_clips(pairs_root, ALL_ROBOTS, dev)
    sample_entry = next(iter(train_clips.values()))
    h_static = human_static_features(skel, body_pos_sample=sample_entry["human_pos"])
    h_adj = _adjacency(skel)

    groups = shared_param_groups(model)
    robots = ALL_ROBOTS
    # accumulators
    cos_sum = {g: np.zeros((len(robots), len(robots))) for g in groups}
    cos_cnt = {g: np.zeros((len(robots), len(robots))) for g in groups}
    norm_sum = {g: {r: 0.0 for r in robots} for g in groups}
    loss_sum = {r: 0.0 for r in robots}

    clip_list = list(train_clips.values())
    for w in range(args.num_windows):
        entry = random.choice(clip_list)
        T = entry["human_pos"].shape[0]
        s = 0 if T <= args.window else random.randint(0, T - args.window)
        e = min(s + args.window, T)
        feats = human_pose_features(entry["human_pos"][s:e], entry["human_quat"][s:e])

        grads = {g: {} for g in groups}
        for robot in robots:
            ctx = ctxs[robot]
            model.zero_grad(set_to_none=True)
            z_h = model.encode(feats, h_static, h_adj)
            teacher, anchor, q = window_teacher(entry, robot, s, e, ctx.xy_scale)
            pred = model.decoder(z_h, ctx.static, ctx.adj,
                                 model.embodiment_encoder(ctx.static), ctx.kin.graph)
            loss, _ = total_loss(pred, ctx.kin, teacher=teacher)
            loss.backward()
            loss_sum[robot] += float(loss.detach())
            for g, params in groups.items():
                v = flat_grad(params)
                grads[g][robot] = v
                norm_sum[g][robot] += float(v.norm())

        for g in groups:
            for i, ri in enumerate(robots):
                for j, rj in enumerate(robots):
                    if j <= i:
                        continue
                    a, b = grads[g][ri], grads[g][rj]
                    denom = float(a.norm() * b.norm())
                    if denom > 1e-12:
                        cos_sum[g][i, j] += float(a @ b) / denom
                        cos_cnt[g][i, j] += 1
        print(f"window {w + 1}/{args.num_windows} done", flush=True)

    report = {"ckpt": args.ckpt, "num_windows": args.num_windows, "robots": robots,
              "mean_loss": {r: loss_sum[r] / args.num_windows for r in robots}, "groups": {}}
    for g in groups:
        mat = np.where(cos_cnt[g] > 0, cos_sum[g] / np.maximum(cos_cnt[g], 1), 0.0)
        pairs = [(robots[i], robots[j], float(mat[i, j]))
                 for i in range(len(robots)) for j in range(len(robots)) if j > i]
        neg = [p for p in pairs if p[2] < 0]
        report["groups"][g] = {
            "pairwise_cosine": {f"{a}|{b}": c for a, b, c in pairs},
            "mean_cosine": float(np.mean([c for _, _, c in pairs])),
            "negative_pair_fraction": len(neg) / len(pairs),
            "mean_grad_norm": {r: norm_sum[g][r] / args.num_windows for r in robots},
        }
        norms = list(report["groups"][g]["mean_grad_norm"].values())
        report["groups"][g]["norm_imbalance_ratio"] = float(max(norms) / max(min(norms), 1e-12))

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({g: {k: v for k, v in d.items() if k != "pairwise_cosine"}
                      for g, d in report["groups"].items()}, indent=1))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
