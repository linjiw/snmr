#!/usr/bin/env python
"""Probe: is foot contact linearly decodable from the frozen SNMR latent z?

Phase-4 probe (b) of docs/WBT_LATENT_PLAN_v2.md section 5.5, motivated by the contact-aware
retargeting literature (latents that organize motion tend to expose contact) and by Gate-1b
(mask precision is the C5/C6 bottleneck; a z-linear contact readout would be a new deployable
mask candidate that needs no decoded-height heuristics).

Design:
- Encode the human motion of the G1 validation clips with the frozen Phase-2 checkpoint
  (identical recipe to scripts/export_wbt_with_latent.py) -> z_t at 30 fps.
- Labels: teacher-height hysteresis contact mask (enter 0.03 m / exit 0.05 m) on GMR teacher
  feet from FK — the same oracle labels Gate-1b treats as ground truth.
- Probes: logistic regression on z_t (per foot), with HELD-OUT-CLIP evaluation (train on
  k-1 clips, test on the held-out clip; rotate). Controls: (1) majority class, (2) the
  human-source height mask (the current deployable baseline, from LAFAN1 toe heights),
  (3) probe on z with a +-1 frame temporal context (checks if contact is in latent dynamics).
- Output: JSON with per-clip/per-foot F1 + AUC per probe and the aggregate comparison.

CPU-only; safe to run while WBT trains on the GPU.
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

from snmr.human import (  # noqa: E402
    LAFAN1_CONTACT_BODIES,
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.metrics import FOOT_BODIES, detect_contact_height_hysteresis  # noqa: E402
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402
from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402


def load_model(ckpt: str, device: str):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    tc = state.get("config", {})
    sd = state["model"]
    model = SNMR(SNMRConfig(
        latent_dim=tc.get("latent_dim", 128), enc_hidden=tc.get("enc_hidden", 256),
        dec_hidden=tc.get("dec_hidden", 256),
        use_temporal=any(k.startswith("encoder.temporal.") for k in sd),
        predict_contact=any(k.startswith("decoder.contact_head.") for k in sd),
    )).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    pred = scores >= 0.5
    out = {
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "accuracy": float((pred == labels).mean()),
        "positive_rate": float(labels.mean()),
    }
    if 0 < labels.mean() < 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["ap"] = float(average_precision_score(labels, scores))
    else:
        out["auroc"] = None
        out["ap"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phase2_all5/ckpt_100k_final.pt")
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--out", default="runs/latent_contact_probe/probe.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    dev = args.device
    model = load_model(args.ckpt, dev)
    ctx = RobotContext(args.robot, dev)
    skel = lafan1_skeleton(device=dev)
    pairs_root = data_root() / "pairs"
    foot_names = FOOT_BODIES[args.robot]
    foot_idx = [ctx.kin.body_index(n) for n in foot_names]

    per_clip: dict[str, dict] = {}
    for clip in args.clips:
        pair = load_pair_npz(str(pairs_root / args.robot / f"{clip}.npz"))
        hp = pair["human_pos"].to(dev)
        hq = pair["human_quat"].to(dev)
        q = pair["qpos"].to(dev)
        h_static = human_static_features(skel, body_pos_sample=hp)
        with torch.no_grad():
            z = model.encode(human_pose_features(hp, hq), h_static, _adjacency(skel))
            teacher_body, _ = ctx.kin.forward_kinematics(q[:, 0:3], q[:, 3:7], q[:, 7:])
        teacher_feet = teacher_body[:, foot_idx, :]
        labels = detect_contact_height_hysteresis(
            teacher_feet, enter_height=0.03, exit_height=0.05
        )
        src_idx = [pair["human_names"].index(n) for n in LAFAN1_CONTACT_BODIES]
        source_mask = detect_contact_height_hysteresis(
            hp[:, src_idx, :], enter_height=0.08, exit_height=0.10
        )
        per_clip[clip] = {
            "z": z.cpu().numpy().astype(np.float64),
            "labels": labels.cpu().numpy(),
            "source_mask": source_mask.cpu().numpy(),
        }

    def features(z: np.ndarray, context: bool) -> np.ndarray:
        if not context:
            return z
        prev = np.vstack([z[:1], z[:-1]])
        nxt = np.vstack([z[1:], z[-1:]])
        return np.hstack([z, nxt - z, z - prev])

    results: dict = {
        "checkpoint": args.ckpt,
        "robot": args.robot,
        "clips": args.clips,
        "labels": "teacher_height hysteresis enter=0.03 exit=0.05 (Gate-1b oracle)",
        "protocol": "held-out-clip rotation; probe trained on remaining clips",
        "feet": foot_names,
        "per_clip": {},
    }
    aggregate: dict[str, list[float]] = {}
    for held_out in args.clips:
        train_clips = [c for c in args.clips if c != held_out]
        clip_result: dict = {}
        for f, foot in enumerate(foot_names):
            train_y = np.concatenate([per_clip[c]["labels"][:, f] for c in train_clips])
            test_y = per_clip[held_out]["labels"][:, f]
            entry: dict = {}
            for probe_name, context in (("z_linear", False), ("z_context_linear", True)):
                train_x = np.concatenate(
                    [features(per_clip[c]["z"], context) for c in train_clips]
                )
                test_x = features(per_clip[held_out]["z"], context)
                if train_y.min() == train_y.max():
                    entry[probe_name] = None
                    continue
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(train_x, train_y)
                scores = clf.predict_proba(test_x)[:, 1]
                entry[probe_name] = binary_metrics(test_y, scores)
            entry["source_height_mask"] = binary_metrics(
                test_y, per_clip[held_out]["source_mask"][:, f].astype(np.float64)
            )
            entry["majority"] = binary_metrics(
                test_y,
                np.full_like(test_y, test_y.mean() >= 0.5, dtype=np.float64),
            )
            clip_result[foot] = entry
            for probe_name in ("z_linear", "z_context_linear", "source_height_mask"):
                metrics = entry.get(probe_name)
                if metrics is not None:
                    aggregate.setdefault(probe_name, []).append(metrics["f1"])
        results["per_clip"][held_out] = clip_result

    results["aggregate_f1_mean"] = {
        probe: float(np.mean(v)) for probe, v in aggregate.items()
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["aggregate_f1_mean"], indent=2))
    for clip, feet in results["per_clip"].items():
        for foot, entry in feet.items():
            z_f1 = entry["z_linear"]["f1"] if entry["z_linear"] else None
            zc_f1 = entry["z_context_linear"]["f1"] if entry["z_context_linear"] else None
            src_f1 = entry["source_height_mask"]["f1"]
            print(f"{clip:24s} {foot:28s} z={z_f1} z_ctx={zc_f1} source={src_f1:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
