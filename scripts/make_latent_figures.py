#!/usr/bin/env python
"""Paper figures for the shared-latent analysis (E23): dual-colored 2D embedding + CKA heatmap.

Two figures that visually make the N7 case:
  1. **Dual-colored 2D embedding** (t-SNE and UMAP side by side): window-mean latents from
     {human + 5 robots} on held-out clips, colored (a) by embodiment and (b) by motion category.
     If the space is shared, the embodiment-colored plot should intermix embodiments while the
     motion-colored plot shows clip/category structure — the visual complement to the quantitative
     E1/E3/E4 probes. (Caveat baked into the caption: t-SNE distances/sizes are not meaningful.)
  2. **CKA heatmap** (6x6, human + 5 robots): the pairwise linear-CKA matrix from E05, rendered.

CPU-only, uses the Phase-2 final checkpoint. Writes PNGs to runs/figures/.

    python scripts/make_latent_figures.py --ckpt runs/phase2_all5/ckpt_100k_final.pt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.data import RobotMotion, robot_pose_features  # noqa: E402
from snmr.human import human_pose_features, human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from analyze_latent import _motion_category, linear_cka  # noqa: E402
from train_phase2 import ALL_ROBOTS, VAL_CLIPS, RobotContext  # noqa: E402


def load_model(ckpt, device):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    tc = state.get("config", {})
    sd = state["model"]
    m = SNMR(SNMRConfig(latent_dim=tc.get("latent_dim", 128), enc_hidden=tc.get("enc_hidden", 256),
                        dec_hidden=tc.get("dec_hidden", 256),
                        use_temporal=any(k.startswith("encoder.temporal.") for k in sd),
                        predict_contact=any(k.startswith("decoder.contact_head.") for k in sd))).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def collect(model, ctxs, skel, h_static, h_adj, pairs_root, device, window, per_clip):
    """Per-encoder window-mean latents + aligned (embodiment, motion) labels."""
    encoders = ["human", *ctxs]
    X = {e: [] for e in encoders}
    emb, mot = [], []
    for clip in VAL_CLIPS:
        first = load_pair_npz(str(pairs_root / ALL_ROBOTS[0] / f"{clip}.npz"))
        T = first["human_pos"].shape[0]
        if T < window:
            continue
        cat = _motion_category(clip)
        for s in np.linspace(0, T - window, num=per_clip, dtype=int):
            e = int(s) + window
            hp = first["human_pos"][int(s):e].to(device)
            hq = first["human_quat"][int(s):e].to(device)
            X["human"].append(model.encode(human_pose_features(hp, hq), h_static, h_adj).mean(0).cpu().numpy())
            for r, ctx in ctxs.items():
                p = load_pair_npz(str(pairs_root / r / f"{clip}.npz"))
                q = p["qpos"][int(s):e].to(device)
                m = RobotMotion(q[:, 0:3], q[:, 3:7], q[:, 7:], fps=30.0)
                X[r].append(model.encode(robot_pose_features(ctx.kin, m), ctx.static, ctx.adj).mean(0).cpu().numpy())
            emb.append("human")
            mot.append(cat)
    return {k: np.stack(v) for k, v in X.items()}, encoders, emb, mot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--per_clip", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "runs" / "figures"))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt, args.device)
    ctxs = {r: RobotContext(r, args.device) for r in ALL_ROBOTS}
    skel = lafan1_skeleton(device=args.device)
    sample = load_pair_npz(str(data_root() / "pairs" / ALL_ROBOTS[0] / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(args.device))
    h_adj = _adjacency(skel)

    X, encoders, emb_h, mot_h = collect(model, ctxs, skel, h_static, h_adj,
                                        data_root() / "pairs", args.device, args.window, args.per_clip)

    # stack all encoder latents with embodiment + motion labels
    feats, emb_lab, mot_lab = [], [], []
    n_per = X["human"].shape[0]
    for e in encoders:
        feats.append(X[e])
        emb_lab += [e] * n_per
        mot_lab += mot_h
    feats = np.concatenate(feats)
    emb_lab = np.array(emb_lab)
    mot_lab = np.array(mot_lab)

    # ---- Fig 1: dual-colored t-SNE + UMAP ----
    from sklearn.manifold import TSNE
    import umap

    embeds = {
        "t-SNE": TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(feats),
        "UMAP": umap.UMAP(n_components=2, random_state=0, n_neighbors=15).fit_transform(feats),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for col, (name, Y) in enumerate(embeds.items()):
        # colored by embodiment
        for e in encoders:
            m = emb_lab == e
            axes[0, col].scatter(Y[m, 0], Y[m, 1], s=10, alpha=0.6, label=e)
        axes[0, col].set_title(f"{name} — colored by EMBODIMENT (want intermixed)")
        axes[0, col].legend(fontsize=7, markerscale=1.5)
        # colored by motion
        for c in sorted(set(mot_lab)):
            m = mot_lab == c
            axes[1, col].scatter(Y[m, 0], Y[m, 1], s=10, alpha=0.6, label=c)
        axes[1, col].set_title(f"{name} — colored by MOTION (structure = content)")
        axes[1, col].legend(fontsize=7, markerscale=1.5)
    fig.suptitle("SNMR shared latent (window-mean, held-out clips). "
                 "t-SNE inter-cluster distances/sizes are not meaningful.", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "latent_embedding_dual.png", dpi=130)
    plt.close(fig)

    # ---- Fig 2: CKA heatmap ----
    mat = np.eye(len(encoders))
    for i, a in enumerate(encoders):
        for j, b in enumerate(encoders):
            if i < j:
                mat[i, j] = mat[j, i] = linear_cka(X[a], X[b])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, vmin=0.8, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(encoders)))
    ax.set_yticks(range(len(encoders)))
    labels = [e.replace("unitree_", "").replace("_29dof", "").replace("stanford_", "")
              .replace("engineai_", "").replace("booster_", "").replace("fourier_", "") for e in encoders]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(encoders)):
        for j in range(len(encoders)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.92 else "black", fontsize=8)
    ax.set_title("Linear CKA between per-embodiment latent sets")
    fig.colorbar(im, ax=ax, label="CKA")
    fig.tight_layout()
    fig.savefig(out / "cka_heatmap.png", dpi=130)
    plt.close(fig)

    print(f"{feats.shape[0]} points ({n_per}/encoder × {len(encoders)} encoders)")
    print(f"wrote {out}/latent_embedding_dual.png and {out}/cka_heatmap.png")


if __name__ == "__main__":
    main()
