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

from snmr.experiment import git_state, runtime_state, sha256_file, utc_now  # noqa: E402
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
    ap.add_argument(
        "--out",
        default=str(ROOT / "runs/phase2_all5/sharing_gradient_diagnosis_eval.json"),
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outp = pathlib.Path(args.out)
    if outp.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {outp}")

    provenance = {
        "created_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "git": git_state(ROOT),
        "runtime": runtime_state(args.device),
        "artifacts": {
            "checkpoint": {
                "path": str(pathlib.Path(args.ckpt).resolve()),
                "sha256": sha256_file(args.ckpt),
            },
            "evaluator": {
                "path": str(pathlib.Path(__file__).resolve()),
                "sha256": sha256_file(__file__),
            },
        },
    }

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dev = args.device
    model, state = load_model(args.ckpt, dev)
    # Gradients are needed, but training mode is not: TransformerEncoderLayer defaults to
    # dropout=0.1, which otherwise gives each robot a different random mask and contaminates
    # cross-robot cosine measurements.
    model.eval()
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
    cos_values = {
        group: {
            (robots[i], robots[j]): []
            for i in range(len(robots))
            for j in range(i + 1, len(robots))
        }
        for group in groups
    }
    norm_values = {
        group: {robot: [] for robot in robots}
        for group in groups
    }
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
                norm_values[g][robot].append(float(v.norm()))

        for g in groups:
            for i, ri in enumerate(robots):
                for j in range(i + 1, len(robots)):
                    rj = robots[j]
                    a, b = grads[g][ri], grads[g][rj]
                    denom = float(a.norm() * b.norm())
                    if denom > 1e-12:
                        cos_values[g][(ri, rj)].append(float(a @ b) / denom)
        print(f"window {w + 1}/{args.num_windows} done", flush=True)

    report = {
        "ckpt": args.ckpt,
        "provenance": provenance,
        "protocol": {
            "model_mode": "eval",
            "num_windows": args.num_windows,
            "window": args.window,
            "seed": args.seed,
            "same_human_window_for_all_robots": True,
        },
        "robots": robots,
        "mean_loss": {r: loss_sum[r] / args.num_windows for r in robots},
        "groups": {},
    }
    for g in groups:
        pair_stats = {}
        all_cosines = []
        for (robot_a, robot_b), observations in cos_values[g].items():
            values = np.asarray(observations, dtype=np.float64)
            all_cosines.extend(observations)
            negative = values[values < 0]
            pair_stats[f"{robot_a}|{robot_b}"] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90)),
                "negative_observation_fraction": float((values < 0).mean()),
                "negative_mean": (
                    float(negative.mean()) if negative.size else None
                ),
                "observations": int(values.size),
            }
        all_values = np.asarray(all_cosines, dtype=np.float64)
        all_negative = all_values[all_values < 0]
        mean_norms = {
            robot: float(np.mean(values))
            for robot, values in norm_values[g].items()
        }
        report["groups"][g] = {
            "pairwise_cosine": pair_stats,
            "mean_cosine": float(all_values.mean()),
            "median_cosine": float(np.median(all_values)),
            "negative_observation_fraction": float((all_values < 0).mean()),
            "negative_cosine_mean": (
                float(all_negative.mean()) if all_negative.size else None
            ),
            "negative_mean_pair_fraction": float(np.mean([
                values["mean"] < 0 for values in pair_stats.values()
            ])),
            "mean_grad_norm": mean_norms,
            "grad_norm_distribution": {
                robot: {
                    "median": float(np.median(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p90": float(np.percentile(values, 90)),
                }
                for robot, values in norm_values[g].items()
            },
        }
        norms = list(mean_norms.values())
        report["groups"][g]["norm_imbalance_ratio"] = float(max(norms) / max(min(norms), 1e-12))

    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2)
    tmp.replace(outp)
    print(json.dumps({g: {k: v for k, v in d.items() if k != "pairwise_cosine"}
                      for g, d in report["groups"].items()}, indent=1))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
