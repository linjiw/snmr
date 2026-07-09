#!/usr/bin/env python
"""E2: shared-latent-space analysis for a Phase-2 checkpoint.

Measures whether the latent space is genuinely embodiment-agnostic:

  1. **Cross-embodiment consistency**: for the same motion window, distance between the human
     encoding z_h and each robot-teacher encoding z_r, vs. the distance between encodings of
     *different* motions (contrast ratio should be >> 1 the other way: same-motion distances small).
  2. **Cross-embodiment retrieval (mAP / top-1)**: query with robot A's encoding of a window,
     retrieve among robot B's encodings of many windows — the matching window should rank first.
     Chance top-1 = 1/N_windows.

Uses held-out clips only.

    python scripts/eval_latent_space.py --ckpt runs/phase2_all5/ckpt.pt
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import random
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.data import RobotMotion, robot_pose_features  # noqa: E402
from snmr.human import human_pose_features, human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from train_phase2 import ALL_ROBOTS, VAL_CLIPS, RobotContext  # noqa: E402

REPO = ROOT.parent


@torch.no_grad()
def collect_encodings(model, ctxs, skel, h_static, h_adj, pairs_root, device, window, per_clip):
    """Return {encoder_name: (N, z)} where rows align across encoders (same motion windows)."""
    encs = {name: [] for name in ["human", *ctxs]}
    for clip in VAL_CLIPS:
        pair = {r: load_pair_npz(str(pairs_root / r / f"{clip}.npz")) for r in ctxs}
        first = next(iter(pair.values()))
        T = first["human_pos"].shape[0]
        starts = np.linspace(0, T - window, num=per_clip, dtype=int)
        for s in starts:
            e = int(s) + window
            hp = first["human_pos"][int(s):e].to(device)
            hq = first["human_quat"][int(s):e].to(device)
            feats = human_pose_features(hp, hq)
            z = model.encode(feats, h_static, h_adj).mean(dim=0)  # window-mean latent
            encs["human"].append(z)
            for r, ctx in ctxs.items():
                q = pair[r]["qpos"][int(s):e].to(device)
                motion = RobotMotion(q[:, 0:3], q[:, 3:7], q[:, 7:], fps=30.0)
                rf = robot_pose_features(ctx.kin, motion)
                zr = model.encode(rf, ctx.static, ctx.adj).mean(dim=0)
                encs[r].append(zr)
    return {k: torch.stack(v) for k, v in encs.items()}


def retrieval_metrics(za: torch.Tensor, zb: torch.Tensor) -> tuple[float, float]:
    """Query za rows against zb rows (same row = match). Returns (top-1 acc, mAP)."""
    d = torch.cdist(za, zb)  # (N, N)
    ranks = d.argsort(dim=1)
    n = za.shape[0]
    correct = torch.arange(n, device=za.device)
    top1 = (ranks[:, 0] == correct).float().mean().item()
    # mAP with a single relevant item = mean reciprocal rank
    pos = (ranks == correct[:, None]).float().argmax(dim=1) + 1
    mrr = (1.0 / pos.float()).mean().item()
    return top1, mrr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs_root", default=str(REPO / "data" / "pairs"))
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--per_clip", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(0)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    tcfg = state.get("config", {})
    model = SNMR(SNMRConfig(
        latent_dim=tcfg.get("latent_dim", 128),
        enc_hidden=tcfg.get("enc_hidden", 256),
        dec_hidden=tcfg.get("dec_hidden", 256),
    )).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()

    ctxs = {r: RobotContext(r, args.device) for r in ALL_ROBOTS}
    skel = lafan1_skeleton(device=args.device)
    sample = load_pair_npz(str(pathlib.Path(args.pairs_root) / ALL_ROBOTS[0] / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(args.device))
    h_adj = _adjacency(skel)

    encs = collect_encodings(model, ctxs, skel, h_static, h_adj,
                             pathlib.Path(args.pairs_root), args.device, args.window, args.per_clip)
    n = encs["human"].shape[0]
    print(f"{n} aligned windows from {len(VAL_CLIPS)} held-out clips\n")

    # 1) consistency: same-motion cross-embodiment distance vs inter-motion distance
    names = list(encs)
    print("=== same-motion cross-embodiment distance / mean inter-motion distance (lower better) ===")
    for a, b in itertools.combinations(names, 2):
        same = (encs[a] - encs[b]).norm(dim=1).mean().item()
        inter = torch.cdist(encs[a], encs[b]).mean().item()
        print(f"{a:18s} <-> {b:18s}: same={same:.3f}  inter={inter:.3f}  ratio={same/inter:.3f}")

    # 2) retrieval
    print(f"\n=== cross-embodiment retrieval (chance top-1 = {1/n:.4f}) ===")
    rows = []
    for a, b in itertools.permutations(names, 2):
        top1, mrr = retrieval_metrics(encs[a], encs[b])
        rows.append((a, b, top1, mrr))
    for a, b, top1, mrr in rows:
        print(f"query {a:18s} -> gallery {b:18s}: top-1 {top1:.3f}  MRR {mrr:.3f}")
    mean_top1 = float(np.mean([r[2] for r in rows]))
    mean_mrr = float(np.mean([r[3] for r in rows]))
    print(f"\nMEAN: top-1 {mean_top1:.3f}  MRR {mean_mrr:.3f}  (chance {1/n:.4f})")


if __name__ == "__main__":
    main()
