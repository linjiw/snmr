#!/usr/bin/env python
"""Scale-leak diagnosis for the embodiment-readout gap (design doc §0.5, open problem #2).

The N7 analysis found the latent is aligned (CKA 0.91, retrieval R@1 0.75) yet a strong MLP
attacker reads embodiment identity at ~0.91. Two hypotheses:

  (H-scale)  the leak is mostly BODY SCALE — robots differ 0.34–0.78 m in root height and
             proportionally in limb excursions, and per-node positions/velocities carry that
             amplitude directly; no invariance loss can (or arguably should) remove it.
  (H-deep)   the leak is stylistic/structural beyond scale — per-robot IK idiosyncrasies,
             joint-limit clipping patterns — which a domain-confusion term could target.

Test: re-encode every robot's teacher motion with **height-normalized pose features** (divide all
positions/velocities by that robot's standing root height so bodies live in a common ~unit scale),
then re-run the MLP attacker. Interpretation:
  * attacker accuracy collapses toward chance  -> H-scale (report leak as scale; keep L_z as-is)
  * attacker accuracy stays high               -> H-deep  (motivate domain-confusion loss)

    python scripts/probe_scale_leak.py --ckpt runs/phase2_all5/ckpt_100k_final.pt
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
from snmr.human import load_pair_npz  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from analyze_latent import mlp_attacker, linear_probe  # noqa: E402
from train_phase2 import ALL_ROBOTS, VAL_CLIPS, RobotContext  # noqa: E402

# Standing root height per robot (from the N1 dataset integrity check; the scale proxy).
ROOT_HEIGHT = {
    "unitree_g1": 0.67, "booster_t1_29dof": 0.67, "fourier_n1": 0.68,
    "engineai_pm01": 0.78, "stanford_toddy": 0.34,
}


@torch.no_grad()
def collect_robot_latents(model, ctxs, pairs_root, device, window, per_clip, normalize_scale):
    X, emb_id, clip_id = [], [], []
    clip_names = sorted(set(VAL_CLIPS))
    for r_i, (r, ctx) in enumerate(ctxs.items()):
        for clip in VAL_CLIPS:
            pair = load_pair_npz(str(pairs_root / r / f"{clip}.npz"))
            q = pair["qpos"].to(device)
            T = q.shape[0]
            starts = np.linspace(0, max(T - window, 0), num=per_clip, dtype=int)
            for s in starts:
                e = int(s) + window
                motion = RobotMotion(q[int(s):e, 0:3], q[int(s):e, 3:7], q[int(s):e, 7:], fps=30.0)
                feats = robot_pose_features(ctx.kin, motion)  # (T, B, 12): pos3 + rot6d + vel3
                if normalize_scale:
                    h = ROOT_HEIGHT[r]
                    feats = feats.clone()
                    feats[..., 0:3] /= h    # positions
                    feats[..., 9:12] /= h   # velocities (rot6d untouched: scale-free already)
                z = model.encode(feats, ctx.static, ctx.adj).mean(0)
                X.append(z.cpu().numpy())
                emb_id.append(r_i)
                clip_id.append(clip_names.index(clip))
    return np.stack(X), np.array(emb_id), np.array(clip_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--per_clip", type=int, default=14)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    tcfg = state.get("config", {})
    sd = state["model"]
    model = SNMR(SNMRConfig(
        latent_dim=tcfg.get("latent_dim", 128), enc_hidden=tcfg.get("enc_hidden", 256),
        dec_hidden=tcfg.get("dec_hidden", 256),
        use_temporal=any(k.startswith("encoder.temporal.") for k in sd),
        predict_contact=any(k.startswith("decoder.contact_head.") for k in sd),
    )).to(args.device)
    model.load_state_dict(sd)
    model.eval()

    ctxs = {r: RobotContext(r, args.device) for r in ALL_ROBOTS}
    pairs_root = data_root() / "pairs"

    results = {}
    for tag, norm in [("raw", False), ("scale_normalized", True)]:
        X, emb, clip = collect_robot_latents(model, ctxs, pairs_root, args.device,
                                             args.window, args.per_clip, norm)
        lp_acc, chance = linear_probe(X, emb, clip)
        att_acc, proxy_a = mlp_attacker(X, emb, clip)
        results[tag] = {"linear_probe": lp_acc, "mlp_attacker": att_acc,
                        "proxy_A": proxy_a, "chance": chance}
        print(f"[{tag:17s}] linear probe {lp_acc:.3f} | MLP attacker {att_acc:.3f} "
              f"| proxy-A {proxy_a:.3f} | chance {chance:.3f}")

    drop = results["raw"]["mlp_attacker"] - results["scale_normalized"]["mlp_attacker"]
    total_above = results["raw"]["mlp_attacker"] - results["raw"]["chance"]
    frac = drop / total_above if total_above > 0 else 0.0
    verdict = ("H-scale: scale explains most of the leak"
               if frac > 0.5 else
               "H-deep: leak persists beyond scale (domain-confusion loss motivated)")
    results["verdict"] = {"attacker_drop": drop, "fraction_of_leak_explained": frac,
                          "conclusion": verdict}
    print(f"\nattacker drop from scale normalization: {drop:.3f} "
          f"({frac*100:.0f}% of the above-chance leak)\n=> {verdict}")

    out = args.out or (pathlib.Path(args.ckpt).parent / "scale_leak_probe.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
