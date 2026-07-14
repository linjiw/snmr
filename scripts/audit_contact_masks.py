#!/usr/bin/env python
"""Gate 1b pre-study audit: candidate contact-mask quality vs the teacher-height oracle.

Runs BEFORE any Gate 1b projection so mask quality and projection outcome can be attributed
separately (docs/NEURAL_RETARGETING_RESEARCH_fable.md section 4.1). No solver is invoked.

Measures, on the frozen 42-window / 7-held-out-clip protocol:

1. Ground-normalization gap: teacher-height hysteresis labels computed with clip-local vs
   192-frame-window-local vs 64-frame-training-window-local minima (the C1 head was TRAINED on
   64-frame window-local labels; the evaluator scores against clip-local labels).
2. Candidate deployable masks vs the clip-local teacher-height oracle: precision / recall / F1 /
   IoU / prevalence, aggregate and per clip:
   - M0 source_contact (human height+speed heuristic, the E29 deployable baseline)
   - M1 decoded_height (height hysteresis on the E03 checkpoint's decoded feet, enter 0.03 /
     exit 0.05 — E24 showed decoded heights are correct; only xy velocity is wrong)
   - M2 predicted (C1 checkpoint's contact head): plain threshold 0.5 and a probability
     hysteresis variant (enter >=0.6, exit <0.4). The frozen audit rule selects whichever
     variant has higher aggregate F1 for the single registered projection arm.

The masks here are diagnostic quality numbers; the projection endpoints remain those frozen in
the Gate 1b protocol (teacher-height AND source-contact stance speed, coprimary).
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
from snmr.experiment import git_state, runtime_state, sha256_file, utc_now  # noqa: E402
from snmr.human import (  # noqa: E402
    LAFAN1_CONTACT_BODIES,
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.metrics import (  # noqa: E402
    FOOT_BODIES,
    detect_contact,
    detect_contact_height_hysteresis,
)
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from benchmark import load_model  # noqa: E402
from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402


def mask_agreement(candidate: torch.Tensor, oracle: torch.Tensor) -> dict:
    """Binary classification metrics treating the oracle mask as ground truth."""
    cand = candidate.bool()
    orac = oracle.bool()
    tp = float((cand & orac).sum())
    fp = float((cand & ~orac).sum())
    fn = float((~cand & orac).sum())
    tn = float((~cand & ~orac).sum())
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and precision + recall > 0
        else float("nan")
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "candidate_prevalence": (tp + fp) / n if n else float("nan"),
        "oracle_prevalence": (tp + fn) / n if n else float("nan"),
        "agreement": (tp + tn) / n if n else float("nan"),
        "samples": n,
    }


def probability_hysteresis(
    probs: torch.Tensor, enter: float = 0.6, exit_: float = 0.4
) -> torch.Tensor:
    """Stateful thresholding of per-frame contact probabilities (T, F)."""
    if exit_ > enter:
        raise ValueError("exit threshold must not exceed enter threshold")
    contact = torch.zeros_like(probs, dtype=torch.bool)
    if probs.shape[0] == 0:
        return contact
    state = probs[0] >= enter
    contact[0] = state
    for t in range(1, probs.shape[0]):
        state = torch.where(state, probs[t] >= exit_, probs[t] >= enter)
        contact[t] = state
    return contact


def pooled(rows: list[dict]) -> dict:
    """Sample-weighted pooling of per-window agreement rows via confusion recomposition."""
    keys = ("precision", "recall", "f1", "iou", "candidate_prevalence", "oracle_prevalence")
    total = sum(r["samples"] for r in rows)
    out = {}
    for key in keys:
        vals = [(r[key], r["samples"]) for r in rows if r[key] == r[key]]
        w = sum(s for _, s in vals)
        out[key] = sum(v * s for v, s in vals) / w if w else float("nan")
    out["samples"] = total
    out["windows"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument(
        "--mask_ckpt",
        default=str(ROOT / "runs/gate1_g1/screen/c1_bce_seed0/ckpt.pt"),
        help="checkpoint providing the trained contact head (Gate 1 C1)",
    )
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--windows_per_clip", type=int, default=6)
    ap.add_argument("--train_window", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "runs/gate1b/mask_audit.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outp = pathlib.Path(args.out)
    if outp.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {outp}")

    dev = args.device
    model, state = load_model(args.ckpt, dev)
    mask_model, mask_state = load_model(args.mask_ckpt, dev)
    if mask_model.decoder.contact_head is None:
        raise ValueError("--mask_ckpt must contain a trained contact head")

    ctx = RobotContext(args.robot, dev)
    if "xy_scale" in state:
        ctx.xy_scale = float(state["xy_scale"])
    skel = lafan1_skeleton(device=dev)
    pairs_root = data_root() / "pairs"
    sample = load_pair_npz(str(pairs_root / args.robot / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(dev))
    h_adj = _adjacency(skel)
    feet = FOOT_BODIES[args.robot]
    foot_idx = [ctx.kin.body_index(name) for name in feet]

    candidates = [
        "source_contact",
        "decoded_height",
        "predicted_t0.5",
        "predicted_hyst",
        "teacher_height_window192",
        "teacher_height_window64",
    ]
    agreement_rows: dict[str, list[tuple[str, dict]]] = {c: [] for c in candidates}
    ground_gap_rows = []

    for clip in args.clips:
        pair = load_pair_npz(str(pairs_root / args.robot / f"{clip}.npz"))
        hp_all, hq_all, q_all = (pair[k].to(dev) for k in ("human_pos", "human_quat", "qpos"))
        fps = pair["fps"]
        T = q_all.shape[0]
        with torch.no_grad():
            teacher_body, _ = ctx.kin.forward_kinematics(
                q_all[:, 0:3], q_all[:, 3:7], q_all[:, 7:]
            )
        teacher_feet_all = teacher_body[:, foot_idx, :]
        source_idx = [pair["human_names"].index(name) for name in LAFAN1_CONTACT_BODIES]
        source_feet = hp_all[:, source_idx, :]

        # Oracle: clip-local ground normalization, exactly as benchmark.py / eval_footlock.py.
        oracle_full = detect_contact_height_hysteresis(
            teacher_feet_all, enter_height=0.03, exit_height=0.05
        )
        source_contact_full = detect_contact(
            source_feet, fps, height_threshold=0.08, speed_threshold=0.24
        )
        # 64-frame training-window-local labels, as train_phase1.factorized_contact_mask saw them.
        train_local = torch.zeros_like(oracle_full)
        for s in range(0, T, args.train_window):
            seg = teacher_feet_all[s : s + args.train_window]
            train_local[s : s + args.train_window] = detect_contact_height_hysteresis(
                seg, enter_height=0.03, exit_height=0.05
            )

        starts = np.linspace(
            0,
            max(T - args.window, 0),
            num=min(args.windows_per_clip, max(T // args.window, 1)),
            dtype=int,
        )
        for start in starts:
            start = int(start)
            end = start + args.window
            hp, hq = hp_all[start:end], hq_all[start:end]
            anchor_pos = hp[:, 0, :].clone()
            anchor_pos[:, :2] *= ctx.xy_scale
            anchor_quat = hq[:, 0, :]
            with torch.no_grad():
                feats = human_pose_features(hp, hq)
                z = model.encode(feats, h_static, h_adj)
                pred = model.decoder(
                    z, ctx.static, ctx.adj, model.embodiment_encoder(ctx.static), ctx.kin.graph
                )
                wp, wq = local_root_to_world(
                    anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"]
                )
                body, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
                decoded_feet = body[:, foot_idx, :]
                # C1 head forward pass on the same human window (cross-model mask).
                z_m = mask_model.encode(feats, h_static, h_adj)
                pred_m = mask_model.decoder(
                    z_m,
                    ctx.static,
                    ctx.adj,
                    mask_model.embodiment_encoder(ctx.static),
                    ctx.kin.graph,
                )
            probs = torch.sigmoid(pred_m["contact_logits"][:, foot_idx])

            oracle = oracle_full[start:end]
            # 192-frame window-local teacher-height labels (what a window-scoped evaluator
            # would produce) to bound the normalization sensitivity.
            oracle_w192 = detect_contact_height_hysteresis(
                teacher_feet_all[start:end], enter_height=0.03, exit_height=0.05
            )
            window_masks = {
                "source_contact": source_contact_full[start:end],
                "decoded_height": detect_contact_height_hysteresis(
                    decoded_feet, enter_height=0.03, exit_height=0.05
                ),
                "predicted_t0.5": probs >= 0.5,
                "predicted_hyst": probability_hysteresis(probs),
                "teacher_height_window192": oracle_w192,
                "teacher_height_window64": train_local[start:end],
            }
            for name, mask in window_masks.items():
                agreement_rows[name].append((clip, mask_agreement(mask, oracle)))

            # Ground-normalization gap: per-foot min-height offset window vs clip.
            clip_min = teacher_feet_all[..., 2].min(dim=0).values
            win_min = teacher_feet_all[start:end, :, 2].min(dim=0).values
            ground_gap_rows.append(
                {
                    "clip": clip,
                    "start": start,
                    "window192_minus_clip_min_m": (win_min - clip_min).tolist(),
                }
            )

    aggregate = {}
    per_clip = {}
    for name in candidates:
        rows = [m for _, m in agreement_rows[name]]
        aggregate[name] = pooled(rows)
        per_clip[name] = {
            clip: pooled([m for c, m in agreement_rows[name] if c == clip])
            for clip in args.clips
        }

    gap_values = np.array(
        [v for row in ground_gap_rows for v in row["window192_minus_clip_min_m"]]
    )
    ground_gap = {
        "window192_minus_clip_min_m": {
            "mean": float(gap_values.mean()),
            "median": float(np.median(gap_values)),
            "p90": float(np.percentile(gap_values, 90)),
            "max": float(gap_values.max()),
            "fraction_above_enter_threshold_0.03": float((gap_values > 0.03).mean()),
        },
        "per_window": ground_gap_rows,
    }

    # Frozen audit rule: the single registered M2 projection variant is the predicted-mask
    # variant with the higher aggregate F1.
    m2_choice = max(
        ("predicted_t0.5", "predicted_hyst"), key=lambda k: aggregate[k]["f1"]
    )

    out = {
        "created_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "git": git_state(ROOT),
        "runtime": runtime_state(dev),
        "artifacts": {
            "base_checkpoint": {"path": args.ckpt, "sha256": sha256_file(args.ckpt)},
            "mask_checkpoint": {
                "path": args.mask_ckpt,
                "sha256": sha256_file(args.mask_ckpt),
            },
            "auditor": {"path": __file__, "sha256": sha256_file(__file__)},
        },
        "protocol": {
            "robot": args.robot,
            "clips": args.clips,
            "window": args.window,
            "windows_per_clip_max": args.windows_per_clip,
            "train_window": args.train_window,
            "oracle": "teacher-height hysteresis enter=0.03 exit=0.05, clip-local ground",
            "mask_definitions": {
                "source_contact": "source height<0.08m and speed<0.24m/s (clip-local)",
                "decoded_height": (
                    "height hysteresis enter=0.03 exit=0.05 on the base checkpoint's decoded "
                    "feet, 192-frame-window-local ground"
                ),
                "predicted_t0.5": "C1 contact head sigmoid >= 0.5",
                "predicted_hyst": "C1 contact head probability hysteresis enter=0.6 exit=0.4",
                "teacher_height_window192": "oracle definition with 192-frame-local ground",
                "teacher_height_window64": "oracle definition with 64-frame-local ground",
            },
            "m2_selection_rule": "higher aggregate F1 between predicted_t0.5 and predicted_hyst",
        },
        "aggregate": aggregate,
        "per_clip": per_clip,
        "ground_normalization": ground_gap,
        "m2_selected_variant": m2_choice,
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    tmp.replace(outp)

    concise = {
        name: {
            k: round(v, 4) if isinstance(v, float) and v == v else v
            for k, v in aggregate[name].items()
        }
        for name in candidates
    }
    print("AGGREGATE (vs clip-local teacher-height oracle):")
    print(json.dumps(concise, indent=1))
    print("GROUND GAP:", json.dumps(ground_gap["window192_minus_clip_min_m"], indent=1))
    print("M2 selected variant:", m2_choice)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
