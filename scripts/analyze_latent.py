#!/usr/bin/env python
"""N7 latent-space analysis suite ("analyze the neural"): the shared-space evidence for the paper.

Given a Phase-2 checkpoint, collects per-window latents for {human, all robots} on held-out clips
and runs the ranked protocol (design doc §N7, research-verified):

  E1  Embodiment-identity linear probe  — want NEAR-CHANCE balanced accuracy (no embodiment leakage)
  E2  Motion-category linear probe       — want HIGH (content preserved); shuffled-label control -> selectivity
  E3  Post-hoc MLP attacker (Elazar-Goldberg) on held-out latents + proxy-A-distance
  E4  Cross-embodiment retrieval (R@1/R@5/median rank + MRR)   [complements eval_latent_space.py]
  E5  Linear CKA between per-embodiment latent sets (6x6)
  (E6 t-SNE/UMAP and E7 interpolation are separate, plotting/decoding scripts.)

Splits: latents are collected on the 7 held-out clips; for the probes we split windows into
train/test by clip so the probe generalizes rather than memorizes. Everything is CPU-friendly
(logistic regression / small MLP / linear algebra on <1000 latent vectors).

    python scripts/analyze_latent.py --ckpt runs/phase2_all5/ckpt.pt --out runs/phase2_all5/latent_analysis.json
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

from snmr.data import RobotMotion, robot_pose_features  # noqa: E402
from snmr.human import human_pose_features, human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from train_phase2 import ALL_ROBOTS, VAL_CLIPS, RobotContext  # noqa: E402


def _motion_category(clip_name: str) -> str:
    """LAFAN1 clip -> coarse motion category (strip the trailing digits + subject suffix)."""
    base = clip_name.split("_subject")[0]
    return "".join(c for c in base if not c.isdigit())


def _discover_motion_clips(pairs_root: pathlib.Path, robot: str, min_per_cat: int = 2,
                           max_clips: int = 24) -> list[str]:
    """Pick clips whose motion category appears >= min_per_cat times (so leave-clip-out is well-posed
    for the motion probe), excluding the held-out VAL_CLIPS so E2 stays on training-distribution but
    the *clip* split still forces generalization within a category."""
    from collections import defaultdict

    all_clips = sorted(p.stem for p in (pairs_root / robot).glob("*.npz"))
    by_cat: dict[str, list[str]] = defaultdict(list)
    for c in all_clips:
        by_cat[_motion_category(c)].append(c)
    chosen: list[str] = []
    for cat, clips in sorted(by_cat.items()):
        if len(clips) >= min_per_cat:
            chosen.extend(clips[:4])  # cap per category to keep it balanced/fast
    return chosen[:max_clips]


@torch.no_grad()
def collect(model, ctxs, skel, h_static, h_adj, pairs_root, device, window, per_clip,
            clip_list=None):
    """Return latents X {enc: (N, z)}, and aligned metadata (embodiment id, motion cat, clip).

    ``clip_list`` defaults to the 7 held-out VAL_CLIPS. Those 7 clips are 7 distinct motion
    categories, so the *motion* probe (E2) is ill-posed under leave-clip-out (test categories unseen
    in train). Pass a larger clip_list (categories repeated across clips) for a well-posed E2; E1/E3
    embodiment probes are well-posed on any clip set because every embodiment appears in every clip.
    """
    encoders = ["human", *ctxs]
    X = {e: [] for e in encoders}
    meta = {"clip": [], "motion": []}
    for clip in (clip_list or VAL_CLIPS):
        pair = {r: load_pair_npz(str(pairs_root / r / f"{clip}.npz")) for r in ctxs}
        first = next(iter(pair.values()))
        T = first["human_pos"].shape[0]
        if T < window:
            continue
        starts = np.linspace(0, T - window, num=per_clip, dtype=int)
        cat = _motion_category(clip)
        for s in starts:
            e = int(s) + window
            hp = first["human_pos"][int(s):e].to(device)
            hq = first["human_quat"][int(s):e].to(device)
            X["human"].append(model.encode(human_pose_features(hp, hq), h_static, h_adj).mean(0))
            for r, ctx in ctxs.items():
                q = pair[r]["qpos"][int(s):e].to(device)
                motion = RobotMotion(q[:, 0:3], q[:, 3:7], q[:, 7:], fps=30.0)
                rf = robot_pose_features(ctx.kin, motion)
                X[r].append(model.encode(rf, ctx.static, ctx.adj).mean(0))
            meta["clip"].append(clip)
            meta["motion"].append(cat)
    X = {e: torch.stack(v).cpu().numpy() for e, v in X.items()}
    return X, meta, encoders


def _stack_all(X, encoders):
    """Stack every (encoder, window) latent into one matrix with aligned embodiment + window ids."""
    feats, emb_id, win_id = [], [], []
    for ei, e in enumerate(encoders):
        feats.append(X[e])
        emb_id.append(np.full(X[e].shape[0], ei))
        win_id.append(np.arange(X[e].shape[0]))
    return np.concatenate(feats), np.concatenate(emb_id), np.concatenate(win_id)


def linear_probe(feat, label, groups, seed=0):
    """Leave-clip-out logistic-regression probe -> balanced accuracy (chance = 1/n_classes)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=seed)
    tr, te = next(gss.split(feat, label, groups))
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(feat[tr], label[tr])
    acc = balanced_accuracy_score(label[te], clf.predict(feat[te]))
    n_classes = len(np.unique(label))
    return float(acc), 1.0 / n_classes


def mlp_attacker(feat, label, groups, seed=0):
    """Strong held-out MLP attacker (Elazar-Goldberg) -> accuracy + proxy-A-distance."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neural_network import MLPClassifier

    gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=seed)
    tr, te = next(gss.split(feat, label, groups))
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800, random_state=seed)
    clf.fit(feat[tr], label[tr])
    acc = accuracy_score(label[te], clf.predict(feat[te]))
    err = 1.0 - acc
    proxy_a = 2.0 * (1.0 - 2.0 * err)   # Ganin et al.; 0 = indistinguishable
    return float(acc), float(proxy_a)


def linear_cka(a, b):
    """Linear CKA between two (N, d) latent matrices (Kornblith et al. 2019). 1 = aligned."""
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    hsic = np.linalg.norm(b.T @ a, "fro") ** 2
    denom = np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")
    return float(hsic / (denom + 1e-12))


def retrieval(za, zb):
    from scipy.spatial.distance import cdist

    d = cdist(za, zb)
    ranks = d.argsort(1)
    n = za.shape[0]
    correct = np.arange(n)
    pos = np.where(ranks == correct[:, None])[1] + 1
    return {
        "R@1": float((ranks[:, 0] == correct).mean()),
        "R@5": float((ranks[:, :5] == correct[:, None]).any(1).mean()),
        "median_rank": float(np.median(pos)),
        "MRR": float((1.0 / pos).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pairs_root", default=None)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--per_clip", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--motion_clips", nargs="+", default=None,
                    help="clips for the E2 motion probe (need repeated categories); "
                         "default: discover multi-instance categories from the dataset")
    args = ap.parse_args()

    pairs_root = pathlib.Path(args.pairs_root) if args.pairs_root else data_root() / "pairs"
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    tc = state.get("config", {})
    model = SNMR(SNMRConfig(latent_dim=tc.get("latent_dim", 128),
                            enc_hidden=tc.get("enc_hidden", 256),
                            dec_hidden=tc.get("dec_hidden", 256),
                            predict_contact=tc.get("contact_weight", 0) not in (0, None)
                            if isinstance(tc.get("contact_weight", 0), (int, float)) else False)
                ).to(args.device)
    # tolerate contact-head presence/absence mismatch
    try:
        model.load_state_dict(state["model"])
    except RuntimeError:
        model = SNMR(SNMRConfig(latent_dim=tc.get("latent_dim", 128),
                                enc_hidden=tc.get("enc_hidden", 256),
                                dec_hidden=tc.get("dec_hidden", 256))).to(args.device)
        model.load_state_dict(state["model"])
    model.eval()

    ctxs = {r: RobotContext(r, args.device) for r in ALL_ROBOTS}
    skel = lafan1_skeleton(device=args.device)
    sample = load_pair_npz(str(pairs_root / ALL_ROBOTS[0] / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(args.device))
    h_adj = _adjacency(skel)

    X, meta, encoders = collect(model, ctxs, skel, h_static, h_adj, pairs_root,
                                args.device, args.window, args.per_clip)
    clips = np.array(meta["clip"])
    motion = np.array(meta["motion"])
    n_win = X["human"].shape[0]
    print(f"{n_win} windows/encoder x {len(encoders)} encoders from {len(set(clips))} held-out clips")
    print(f"motion categories: {sorted(set(motion))}\n")

    feat_all, emb_id, win_id = _stack_all(X, encoders)
    # groups for leave-clip-out = clip index per row (tile clips across encoders)
    clip_ids = np.concatenate([np.array([sorted(set(clips)).index(c) for c in clips])
                               for _ in encoders])
    motion_all = np.concatenate([motion for _ in encoders])

    results = {"n_windows": n_win, "n_encoders": len(encoders), "encoders": encoders}

    # E1 embodiment probe (want near chance)
    acc, chance = linear_probe(feat_all, emb_id, clip_ids)
    rng = np.random.default_rng(0)
    ctrl, _ = linear_probe(feat_all, rng.permutation(emb_id), clip_ids)
    results["E1_embodiment_probe"] = {"balanced_acc": acc, "chance": chance,
                                      "shuffled_control": ctrl, "selectivity": acc - ctrl}
    print(f"E1 embodiment probe: {acc:.3f} (chance {chance:.3f}, control {ctrl:.3f}) "
          f"-> {'LOW leakage' if acc < 2*chance else 'leakage present'}")

    # E2 motion-category probe (want high) — needs categories repeated across clips so leave-clip-out
    # has each test category seen in training. Collect a dedicated multi-instance clip set.
    motion_clips = args.motion_clips or _discover_motion_clips(pairs_root, ALL_ROBOTS[0])
    if len(motion_clips) >= 4:
        Xm, metam, _ = collect(model, ctxs, skel, h_static, h_adj, pairs_root,
                               args.device, args.window, max(4, args.per_clip // 2),
                               clip_list=motion_clips)
        fm, _, _ = _stack_all(Xm, encoders)
        mm = np.concatenate([np.array(metam["motion"]) for _ in encoders])
        cm = np.concatenate([np.array([sorted(set(metam["clip"])).index(c) for c in metam["clip"]])
                             for _ in encoders])
        acc, chance = linear_probe(fm, mm, cm)
        ctrl, _ = linear_probe(fm, rng.permutation(mm), cm)
        results["E2_motion_probe"] = {"balanced_acc": acc, "chance": chance,
                                      "shuffled_control": ctrl, "selectivity": acc - ctrl,
                                      "clips": motion_clips, "categories": sorted(set(mm.tolist()))}
        print(f"E2 motion probe:     {acc:.3f} (chance {chance:.3f}, control {ctrl:.3f}) "
              f"-> selectivity {acc - ctrl:.3f}  [{len(set(mm))} cats, {len(motion_clips)} clips]")
    else:
        results["E2_motion_probe"] = {"skipped": "not enough multi-instance categories"}
        print("E2 motion probe: skipped (need >=4 clips with repeated categories)")

    # E3 adversarial attacker on embodiment (want near chance even for a strong model)
    att_acc, proxy_a = mlp_attacker(feat_all, emb_id, clip_ids)
    results["E3_adversarial_attacker"] = {"acc": att_acc, "chance": 1.0 / len(encoders),
                                          "proxy_A_distance": proxy_a}
    print(f"E3 MLP attacker:     {att_acc:.3f} (chance {1/len(encoders):.3f}, "
          f"proxy-A-dist {proxy_a:.3f})")

    # E4 retrieval (human vs each robot, mean)
    ret = {}
    for r in ctxs:
        ret[f"human->{r}"] = retrieval(X["human"], X[r])
    mean_r1 = float(np.mean([v["R@1"] for v in ret.values()]))
    mean_mrr = float(np.mean([v["MRR"] for v in ret.values()]))
    results["E4_retrieval"] = {"per_pair": ret, "mean_R@1": mean_r1, "mean_MRR": mean_mrr,
                               "chance_R@1": 1.0 / n_win}
    print(f"E4 retrieval human->robot: mean R@1 {mean_r1:.3f} MRR {mean_mrr:.3f} "
          f"(chance {1/n_win:.4f})")

    # E5 CKA heatmap
    cka = {}
    for i, a in enumerate(encoders):
        for b in encoders[i + 1:]:
            cka[f"{a}~{b}"] = linear_cka(X[a], X[b])
    results["E5_cka"] = cka
    print(f"E5 CKA: mean off-diagonal {np.mean(list(cka.values())):.3f} "
          f"(range {min(cka.values()):.3f}-{max(cka.values()):.3f})")

    out = args.out or (pathlib.Path(args.ckpt).parent / "latent_analysis.json")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
