#!/usr/bin/env python
"""Gate 1b pre-study audit: candidate contact-mask quality vs the teacher-height oracle.

Runs BEFORE any Gate 1b projection so mask quality and projection outcome can be attributed
separately (docs/NEURAL_RETARGETING_RESEARCH_fable.md section 4.1). No solver is invoked.

Measures, on the frozen 42-window / 7-held-out-clip protocol:

1. Ground-normalization gap: teacher-height hysteresis labels computed with clip-local vs
   192-frame-window-local vs 64-frame-training-window-local minima (the C1 head was TRAINED on
   64-frame window-local labels; the evaluator scores against clip-local labels).
2. Candidate deployable masks vs the clip-local teacher-height oracle:
   - M0 source_contact (human height+speed heuristic, the E29 deployable baseline)
   - M1 decoded_height (height hysteresis on the E03 checkpoint's decoded feet, enter 0.03 /
     exit 0.05 — E24 showed decoded heights are correct; only xy velocity is wrong)
   - M2 predicted (C1 checkpoint's contact head): plain threshold 0.5 and a probability
     hysteresis variant (enter >=0.6, exit <0.4).

Version 2 (2026-09-10) — count-first metrics
--------------------------------------------
The 2026-07-14 version computed F1 per window through precision and recall, returned NaN for
zero-overlap windows and for oracle-empty windows, and then averaged the surviving per-window
ratios with NaN dropped independently per metric (despite a docstring claiming confusion
recomposition).  That inflates F1 and makes the reported P/R/F1 triple describe no single
confusion matrix.  This version:

* stores TP/FP/FN/TN for every (clip, window) and every (clip, window, foot);
* reports **micro** scores from summed counts, with F1 = 2TP/(2TP+FP+FN) whenever that
  denominator is positive (zero overlap with errors is F1 = 0, not NaN);
* names the all-empty convention explicitly (``--all_empty_policy``);
* reports labelled **macro** averages (unit, weights, eligible denominator, undefined policy);
* reports a **frame-level** micro estimand that counts overlapping window frames once, next to
  the window-weighted one;
* reproduces the legacy aggregate from the same counts, labelled ``legacy_ratio_mean``, so old
  and new numbers can be reconciled — it is not to be used for new claims;
* never overwrites the frozen ``runs/gate1b/mask_audit.json`` (default output is a new path);
* can run without the two checkpoints (``--allow_missing_checkpoints``): the model-dependent
  masks are then skipped and listed with a reason, and the checkpoint-free masks
  (``source_contact``, the two teacher-height ground variants) are recomputed from raw pairs.

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

from snmr.contact_audit import (  # noqa: E402
    ConfusionCounts,
    WindowMasks,
    audit_windows,
    frame_level_counts,
    per_column_counts,
    scores_from_counts,
)
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

from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402

FROZEN_PREDECESSOR = ROOT / "runs/gate1b/mask_audit.json"
DEFAULT_OUT = ROOT / "runs/gate1b/mask_audit_v2.json"

CANDIDATES = (
    "source_contact",
    "decoded_height",
    "predicted_t0.5",
    "predicted_hyst",
    "teacher_height_window192",
    "teacher_height_window64",
)
MODEL_MASKS = {
    "decoded_height": "base checkpoint (--ckpt)",
    "predicted_t0.5": "mask checkpoint (--mask_ckpt)",
    "predicted_hyst": "mask checkpoint (--mask_ckpt)",
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


def _load_optional_checkpoint(path: str, device: str, allow_missing: bool):
    """Return ``(model, state)`` or ``(None, reason)`` when the file is absent and allowed."""
    p = pathlib.Path(path)
    if not p.is_file():
        if not allow_missing:
            raise FileNotFoundError(
                f"checkpoint {p} is absent; pass --allow_missing_checkpoints to skip the "
                "model-dependent masks"
            )
        return None, f"checkpoint absent on this host: {p}"
    from benchmark import load_model  # noqa: E402  (imports torch model code)

    return load_model(str(p), device)


def _summarize_clip(rows: list[dict], windows: list[WindowMasks], all_empty: str) -> dict:
    out = audit_windows(rows, all_empty=all_empty)  # type: ignore[arg-type]
    frame = frame_level_counts(windows)
    out["frame_level"] = {
        **frame,
        "scores": scores_from_counts(ConfusionCounts.from_dict(frame["counts"]), all_empty),  # type: ignore[arg-type]
        "estimand": "micro over the union of window frames, each frame counted once",
    }
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
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--allow_missing_checkpoints",
        action="store_true",
        help="skip model-dependent masks when a checkpoint is absent (recorded in the output)",
    )
    ap.add_argument(
        "--all_empty_policy",
        choices=("undefined", "one", "zero"),
        default="undefined",
        help="F1/IoU convention for windows with TP=FP=FN=0 (the only undefined case)",
    )
    args = ap.parse_args()

    outp = pathlib.Path(args.out)
    if outp.resolve() == FROZEN_PREDECESSOR.resolve():
        raise FileExistsError(
            f"refusing to write over the frozen 2026-07-14 audit {FROZEN_PREDECESSOR}; "
            "choose another --out"
        )
    if outp.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {outp}")

    dev = args.device
    model, state = _load_optional_checkpoint(args.ckpt, dev, args.allow_missing_checkpoints)
    mask_model, mask_state = _load_optional_checkpoint(
        args.mask_ckpt, dev, args.allow_missing_checkpoints
    )
    skipped: dict[str, str] = {}
    if model is None:
        skipped["decoded_height"] = f"{MODEL_MASKS['decoded_height']}: {state}"
    if mask_model is None:
        for name in ("predicted_t0.5", "predicted_hyst"):
            skipped[name] = f"{MODEL_MASKS[name]}: {mask_state}"
    elif mask_model.decoder.contact_head is None:
        raise ValueError("--mask_ckpt must contain a trained contact head")

    ctx = RobotContext(args.robot, dev)
    xy_scale_source = "ROBOT_XY_SCALE_CONFIG"
    if model is not None and "xy_scale" in state:
        ctx.xy_scale = float(state["xy_scale"])
        xy_scale_source = "base checkpoint state"
    skel = lafan1_skeleton(device=dev)
    pairs_root = data_root() / "pairs"
    sample = load_pair_npz(str(pairs_root / args.robot / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(dev))
    h_adj = _adjacency(skel)
    feet = FOOT_BODIES[args.robot]
    foot_idx = [ctx.kin.body_index(name) for name in feet]

    active = [c for c in CANDIDATES if c not in skipped]
    per_window_rows: dict[str, list[dict]] = {c: [] for c in active}
    window_masks: dict[str, dict[str, list[WindowMasks]]] = {
        c: {clip: [] for clip in args.clips} for c in active
    }
    ground_gap_rows = []
    clip_inputs = {}

    for clip in args.clips:
        pair_path = pairs_root / args.robot / f"{clip}.npz"
        clip_inputs[clip] = {"path": str(pair_path), "sha256": sha256_file(pair_path)}
        pair = load_pair_npz(str(pair_path))
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
            oracle = oracle_full[start:end]
            masks: dict[str, torch.Tensor] = {
                "source_contact": source_contact_full[start:end],
                "teacher_height_window192": detect_contact_height_hysteresis(
                    teacher_feet_all[start:end], enter_height=0.03, exit_height=0.05
                ),
                "teacher_height_window64": train_local[start:end],
            }
            if model is not None or mask_model is not None:
                with torch.no_grad():
                    feats = human_pose_features(hp, hq)
                    if model is not None:
                        anchor_pos = hp[:, 0, :].clone()
                        anchor_pos[:, :2] *= ctx.xy_scale
                        anchor_quat = hq[:, 0, :]
                        z = model.encode(feats, h_static, h_adj)
                        pred = model.decoder(
                            z, ctx.static, ctx.adj, model.embodiment_encoder(ctx.static), ctx.kin.graph
                        )
                        wp, wq = local_root_to_world(
                            anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"]
                        )
                        body, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
                        masks["decoded_height"] = detect_contact_height_hysteresis(
                            body[:, foot_idx, :], enter_height=0.03, exit_height=0.05
                        )
                    if mask_model is not None:
                        z_m = mask_model.encode(feats, h_static, h_adj)
                        pred_m = mask_model.decoder(
                            z_m,
                            ctx.static,
                            ctx.adj,
                            mask_model.embodiment_encoder(ctx.static),
                            ctx.kin.graph,
                        )
                        probs = torch.sigmoid(pred_m["contact_logits"][:, foot_idx])
                        masks["predicted_t0.5"] = probs >= 0.5
                        masks["predicted_hyst"] = probability_hysteresis(probs)

            for name in active:
                mask = masks[name]
                counts = ConfusionCounts.from_masks(mask, oracle)
                per_foot = per_column_counts(mask, oracle)
                per_window_rows[name].append(
                    {
                        "clip": clip,
                        "start": start,
                        "end": end,
                        "counts": counts,
                        "per_foot": [
                            {"body": feet[f], "counts": per_foot[f].to_dict()}
                            for f in range(len(feet))
                        ],
                        "scores": scores_from_counts(counts, args.all_empty_policy),
                    }
                )
                window_masks[name][clip].append(
                    WindowMasks(start, mask.cpu().numpy(), oracle.cpu().numpy())
                )

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
    for name in active:
        rows = per_window_rows[name]
        all_windows = [w for clip in args.clips for w in window_masks[name][clip]]
        aggregate[name] = _summarize_clip(rows, all_windows, args.all_empty_policy)
        # Frame-level union across clips = sum of per-clip frame-level counts (clips never overlap).
        per_clip[name] = {
            clip: _summarize_clip(
                [r for r in rows if r["clip"] == clip],
                window_masks[name][clip],
                args.all_empty_policy,
            )
            for clip in args.clips
        }
        frame_sum = ConfusionCounts.sum(
            ConfusionCounts.from_dict(per_clip[name][clip]["frame_level"]["counts"])
            for clip in args.clips
        )
        aggregate[name]["frame_level"] = {
            "counts": frame_sum.to_dict(),
            "scores": scores_from_counts(frame_sum, args.all_empty_policy),
            "frames_union": sum(per_clip[name][c]["frame_level"]["frames_union"] for c in args.clips),
            "frames_multiply_covered": sum(
                per_clip[name][c]["frame_level"]["frames_multiply_covered"] for c in args.clips
            ),
            "overlap_disagreements": sum(
                per_clip[name][c]["frame_level"]["overlap_disagreements"] for c in args.clips
            ),
            "estimand": "micro over the union of window frames per clip, summed across clips",
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

    # M2 selection.  The frozen rule chose by legacy aggregate F1; record both the count-first
    # choice and the legacy-rule choice when the predicted masks were computed.
    m2 = {"rule_frozen_2026-07-14": "higher aggregate F1 (legacy ratio mean)", "selected": None}
    if all(n in aggregate for n in ("predicted_t0.5", "predicted_hyst")):
        micro_choice = max(
            ("predicted_t0.5", "predicted_hyst"),
            key=lambda k: (aggregate[k]["micro"]["f1"] if aggregate[k]["micro"]["f1"] is not None else -1.0),
        )
        legacy_choice = max(
            ("predicted_t0.5", "predicted_hyst"),
            key=lambda k: (
                aggregate[k]["legacy_ratio_mean"]["f1"]
                if aggregate[k]["legacy_ratio_mean"]["f1"] is not None
                else -1.0
            ),
        )
        m2.update({"selected": micro_choice, "selected_by_legacy_rule": legacy_choice,
                   "rule_v2": "higher micro F1 from summed counts"})
    else:
        m2["reason_not_selected"] = "predicted masks not computed (checkpoint absent)"

    out = {
        "created_at": utc_now(),
        "version": 2,
        "command": [sys.executable, *sys.argv],
        "git": git_state(ROOT),
        "runtime": runtime_state(dev),
        "frozen_predecessor": {
            "path": str(FROZEN_PREDECESSOR),
            "sha256": sha256_file(FROZEN_PREDECESSOR) if FROZEN_PREDECESSOR.is_file() else None,
            "note": "unchanged; v2 does not replace it",
        },
        "artifacts": {
            "base_checkpoint": {
                "path": args.ckpt,
                "sha256": sha256_file(args.ckpt) if pathlib.Path(args.ckpt).is_file() else None,
                "present": model is not None,
            },
            "mask_checkpoint": {
                "path": args.mask_ckpt,
                "sha256": sha256_file(args.mask_ckpt) if pathlib.Path(args.mask_ckpt).is_file() else None,
                "present": mask_model is not None,
            },
            "auditor": {"path": __file__, "sha256": sha256_file(__file__)},
            "robot_mjcf": {"path": str(ctx.kin.mjcf_path) if hasattr(ctx.kin, "mjcf_path") else None},
            "clip_inputs": clip_inputs,
        },
        "skipped_masks": skipped,
        "protocol": {
            "robot": args.robot,
            "clips": args.clips,
            "window": args.window,
            "windows_per_clip_max": args.windows_per_clip,
            "train_window": args.train_window,
            "xy_scale": ctx.xy_scale,
            "xy_scale_source": xy_scale_source,
            "oracle": "teacher-height hysteresis enter=0.03 exit=0.05, clip-local per-foot minimum as ground",
            "oracle_label_source_caveat": (
                "teacher-height hysteresis on the ankle body origin is a proxy for stance, "
                "not force-verified contact; see the separate label-source audit"
            ),
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
            "metric_definitions": {
                "unit": "one sample = one (frame, foot) label",
                "micro": "ratios of TP/FP/FN/TN summed over windows; F1 = 2TP/(2TP+FP+FN)",
                "frame_level": "micro over the union of window frames, each frame counted once",
                "macro_window_uniform": "uniform mean of per-window ratios; undefined windows excluded and counted",
                "legacy_ratio_mean": "the frozen 2026-07-14 aggregate reproduced from counts, for reconciliation only",
                "all_empty_policy": args.all_empty_policy,
                "undefined_representation": "JSON null (never NaN)",
            },
            "m2_selection": m2,
        },
        "aggregate": aggregate,
        "per_clip": per_clip,
        "per_window": {
            name: [
                {
                    "clip": r["clip"],
                    "start": r["start"],
                    "end": r["end"],
                    "counts": r["counts"].to_dict(),
                    "per_foot": r["per_foot"],
                }
                for r in per_window_rows[name]
            ]
            for name in active
        },
        "ground_normalization": ground_gap,
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    tmp.replace(outp)

    def _r(v):
        return round(v, 4) if isinstance(v, float) else v

    print("MICRO (summed counts) vs clip-local teacher-height oracle:")
    for name in active:
        m = aggregate[name]["micro"]
        lg = aggregate[name]["legacy_ratio_mean"]
        print(
            f"  {name:<26} P {_r(m['precision'])} R {_r(m['recall'])} F1 {_r(m['f1'])} "
            f"IoU {_r(m['iou'])} prev(cand/orac) {_r(m['candidate_prevalence'])}/{_r(m['oracle_prevalence'])} "
            f"| legacy F1 {_r(lg['f1'])} (from {lg['windows_contributing']['f1']}/{lg['windows']} windows)"
        )
    if skipped:
        print("SKIPPED:", json.dumps(skipped, indent=1))
    print("GROUND GAP:", json.dumps(ground_gap["window192_minus_clip_min_m"], indent=1))
    print("M2 selection:", json.dumps(m2))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
