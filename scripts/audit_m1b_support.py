#!/usr/bin/env python
"""Audit M1b clip-ground mask support on the frozen Gate-1b windows.

This is a support-only audit: it does not invoke the projection solver. It reproduces M1b's
full-clip decoded ground and scores its mask against the clip-local teacher-height oracle on
all 42 frozen windows. The earliest support-bearing window in frozen protocol order is selected
for a plumbing smoke; zero-support windows remain failures in the full M1b study.
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
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.metrics import FOOT_BODIES, detect_contact_height_hysteresis  # noqa: E402
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from benchmark import load_model  # noqa: E402
from eval_footlock import decoded_clip_ground_height  # noqa: E402
from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402


def mask_agreement(candidate: torch.Tensor, oracle: torch.Tensor) -> dict:
    """Return exact micro classification metrics and confusion counts."""
    if candidate.shape != oracle.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} != oracle shape {tuple(oracle.shape)}"
        )
    cand = candidate.bool()
    orac = oracle.bool()
    tp = int((cand & orac).sum())
    fp = int((cand & ~orac).sum())
    fn = int((~cand & orac).sum())
    tn = int((~cand & ~orac).sum())
    samples = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "candidate_prevalence": (tp + fp) / samples if samples else None,
        "oracle_prevalence": (tp + fn) / samples if samples else None,
        "candidate_samples": tp + fp,
        "oracle_samples": tp + fn,
        "samples": samples,
    }


def select_smoke_window(rows: list[dict]) -> dict | None:
    """Select the earliest support-bearing window in frozen protocol order."""
    return next((row for row in rows if row["candidate_samples"] > 0), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--windows_per_clip", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--out",
        default=str(ROOT / "runs/gate1b/m1b_support_audit.json"),
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outp = pathlib.Path(args.out)
    if outp.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {outp}")

    model, state = load_model(args.ckpt, args.device)
    ctx = RobotContext(args.robot, args.device)
    if "xy_scale" in state:
        ctx.xy_scale = float(state["xy_scale"])
    skel = lafan1_skeleton(device=args.device)
    pairs_root = data_root() / "pairs"
    sample = load_pair_npz(str(pairs_root / args.robot / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(
        skel, body_pos_sample=sample["human_pos"].to(args.device)
    )
    h_adj = _adjacency(skel)
    feet = FOOT_BODIES[args.robot]
    foot_indices = [ctx.kin.body_index(name) for name in feet]

    window_rows = []
    clip_masks: dict[str, list[torch.Tensor]] = {}
    clip_oracles: dict[str, list[torch.Tensor]] = {}
    decoded_grounds = {}

    for clip in args.clips:
        pair = load_pair_npz(str(pairs_root / args.robot / f"{clip}.npz"))
        hp_all, hq_all, q_all = (
            pair[key].to(args.device) for key in ("human_pos", "human_quat", "qpos")
        )
        with torch.no_grad():
            teacher_body, _ = ctx.kin.forward_kinematics(
                q_all[:, 0:3], q_all[:, 3:7], q_all[:, 7:]
            )
        teacher_feet_all = teacher_body[:, foot_indices, :]
        oracle_full = detect_contact_height_hysteresis(
            teacher_feet_all, enter_height=0.03, exit_height=0.05
        )
        decoded_ground = decoded_clip_ground_height(
            model,
            ctx,
            hp_all,
            hq_all,
            h_static,
            h_adj,
            foot_indices,
            window=args.window,
        )
        decoded_grounds[clip] = {
            foot: float(decoded_ground[index])
            for index, foot in enumerate(feet)
        }
        clip_masks[clip] = []
        clip_oracles[clip] = []

        starts = np.linspace(
            0,
            max(q_all.shape[0] - args.window, 0),
            num=min(args.windows_per_clip, max(q_all.shape[0] // args.window, 1)),
            dtype=int,
        )
        for start_value in starts:
            start = int(start_value)
            end = start + args.window
            hp = hp_all[start:end]
            hq = hq_all[start:end]
            anchor_pos = hp[:, 0, :].clone()
            anchor_pos[:, :2] *= ctx.xy_scale
            with torch.no_grad():
                feats = human_pose_features(hp, hq)
                z = model.encode(feats, h_static, h_adj)
                pred = model.decoder(
                    z,
                    ctx.static,
                    ctx.adj,
                    model.embodiment_encoder(ctx.static),
                    ctx.kin.graph,
                )
                root_pos, root_quat = local_root_to_world(
                    anchor_pos,
                    hq[:, 0, :],
                    pred["root_pos"],
                    pred["root_quat"],
                )
                decoded_body, _ = ctx.kin.forward_kinematics(
                    root_pos, root_quat, pred["dof_pos"]
                )
            decoded_feet = decoded_body[:, foot_indices, :]
            mask = detect_contact_height_hysteresis(
                decoded_feet,
                enter_height=0.03,
                exit_height=0.05,
                ground_z=decoded_ground,
            )
            oracle = oracle_full[start:end]
            agreement = mask_agreement(mask, oracle)
            per_foot_samples = {
                foot: int(mask[:, index].sum())
                for index, foot in enumerate(feet)
            }
            window_rows.append(
                {
                    "clip": clip,
                    "start": start,
                    "end": end,
                    **agreement,
                    "per_foot_candidate_samples": per_foot_samples,
                    "decoded_window_min_minus_clip_ground_m": {
                        foot: float(
                            decoded_feet[:, index, 2].min() - decoded_ground[index]
                        )
                        for index, foot in enumerate(feet)
                    },
                }
            )
            clip_masks[clip].append(mask.cpu())
            clip_oracles[clip].append(oracle.cpu())

    all_masks = torch.cat(
        [mask for clip in args.clips for mask in clip_masks[clip]], dim=0
    )
    all_oracles = torch.cat(
        [mask for clip in args.clips for mask in clip_oracles[clip]], dim=0
    )
    aggregate = mask_agreement(all_masks, all_oracles)
    aggregate.update(
        {
            "windows": len(window_rows),
            "support_bearing_windows": sum(
                row["candidate_samples"] > 0 for row in window_rows
            ),
            "zero_support_windows": sum(
                row["candidate_samples"] == 0 for row in window_rows
            ),
        }
    )
    per_clip = {
        clip: {
            **mask_agreement(
                torch.cat(clip_masks[clip], dim=0),
                torch.cat(clip_oracles[clip], dim=0),
            ),
            "windows": len(clip_masks[clip]),
            "support_bearing_windows": sum(
                row["candidate_samples"] > 0
                for row in window_rows
                if row["clip"] == clip
            ),
        }
        for clip in args.clips
    }
    smoke_row = select_smoke_window(window_rows)
    smoke_window = (
        {
            "selection_rule": "first support-bearing window in frozen clip/window order",
            "clip": smoke_row["clip"],
            "start": smoke_row["start"],
            "end": smoke_row["end"],
            "candidate_samples": smoke_row["candidate_samples"],
        }
        if smoke_row is not None
        else None
    )

    out = {
        "created_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "git": git_state(ROOT),
        "runtime": runtime_state(args.device),
        "artifacts": {
            "checkpoint": {"path": args.ckpt, "sha256": sha256_file(args.ckpt)},
            "auditor": {"path": __file__, "sha256": sha256_file(__file__)},
            "evaluator": {
                "path": str(ROOT / "scripts" / "eval_footlock.py"),
                "sha256": sha256_file(ROOT / "scripts" / "eval_footlock.py"),
            },
        },
        "protocol": {
            "robot": args.robot,
            "clips": args.clips,
            "window": args.window,
            "windows_per_clip_max": args.windows_per_clip,
            "mask": (
                "decoded-foot height hysteresis enter=0.03 exit=0.05 relative to per-foot "
                "minimum from consecutive full-clip decoded tiles"
            ),
            "oracle": "teacher-height hysteresis enter=0.03 exit=0.05, clip-local ground",
            "smoke_selection": "first support-bearing window in frozen clip/window order",
            "projection_invoked": False,
        },
        "aggregate": aggregate,
        "per_clip": per_clip,
        "per_window": window_rows,
        "decoded_clip_ground_m": decoded_grounds,
        "smoke_window": smoke_window,
        "decision": (
            "ready_for_support_bearing_smoke"
            if smoke_window is not None
            else "no_support_full_run_still_required"
        ),
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    tmp.replace(outp)

    print("AGGREGATE:", json.dumps(aggregate, indent=1))
    print("SMOKE WINDOW:", json.dumps(smoke_window, indent=1))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
