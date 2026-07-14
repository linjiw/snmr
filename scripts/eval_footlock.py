#!/usr/bin/env python
"""E26b: evaluate source-mask foot locking on the frozen Gate 0 protocol.

Grid over smooth_sigma to measure the jerk/skate trade-off. This script uses the same full-clip
contact masks, fixed windows, physical guards, and clip bootstrap as ``scripts/benchmark.py``.
The result diagnoses the DLS foot-lock heuristic; it is not the full windowed C6 projection.

    python scripts/eval_footlock.py --ckpt runs/phase1_g1_large/ckpt_100k_final.pt
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
from snmr.footlock import foot_lock_masked  # noqa: E402
from snmr.human import (  # noqa: E402
    LAFAN1_CONTACT_BODIES,
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.metrics import (  # noqa: E402
    FOOT_BODIES,
    compute_metrics,
    contact_motion_metrics,
    detect_contact,
    detect_contact_height_hysteresis,
)
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from benchmark import (  # noqa: E402
    _aggregate_rows,
    _flatten_contact_metrics,
    load_model,
)
from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--windows_per_clip", type=int, default=6)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    ap.add_argument("--lock_iters", type=int, default=12)
    ap.add_argument("--lock_step_scale", type=float, default=0.5)
    ap.add_argument("--lock_damping", type=float, default=1e-2)
    ap.add_argument("--lock_blend", type=int, default=2)
    ap.add_argument("--lock_merge_gap", type=int, default=6)
    ap.add_argument("--lock_extend", type=int, default=2)
    ap.add_argument(
        "--lock_mask",
        choices=["source_contact", "source_height", "teacher_height"],
        default="source_contact",
    )
    ap.add_argument("--bootstrap_samples", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--out",
        default=str(ROOT / "runs/skate_structure/footlock_eval_dls.json"),
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
            "solver": {
                "path": str((ROOT / "snmr" / "footlock.py").resolve()),
                "sha256": sha256_file(ROOT / "snmr" / "footlock.py"),
            },
        },
    }

    dev = args.device
    model, state = load_model(args.ckpt, dev)
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

    variants = ["raw"] + [f"lock_s{sig:g}" for sig in args.sigmas]
    rows: dict[str, list[tuple[str, dict]]] = {v: [] for v in variants}

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
        full_masks = {
            "source_contact": detect_contact(
                source_feet, fps, height_threshold=0.08, speed_threshold=0.24
            ),
            "source_height": detect_contact_height_hysteresis(
                source_feet, enter_height=0.08, exit_height=0.10
            ),
            "teacher_height": detect_contact_height_hysteresis(
                teacher_feet_all, enter_height=0.03, exit_height=0.05
            ),
            "teacher_legacy": detect_contact(teacher_feet_all, fps),
        }

        starts = np.linspace(
            0,
            max(T - args.window, 0),
            num=min(args.windows_per_clip, max(T // args.window, 1)),
            dtype=int,
        )
        for start in starts:
            start = int(start)
            end = start + args.window
            hp, hq, q = hp_all[start:end], hq_all[start:end], q_all[start:end]
            anchor_pos = hp[:, 0, :].clone()
            anchor_pos[:, :2] *= ctx.xy_scale
            anchor_quat = hq[:, 0, :]
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
                wp, wq = local_root_to_world(
                    anchor_pos,
                    anchor_quat,
                    pred["root_pos"],
                    pred["root_quat"],
                )

            reference = (q[:, 0:3], q[:, 3:7], q[:, 7:])
            window_masks = {
                name: mask[start:end] for name, mask in full_masks.items()
            }
            teacher_feet = teacher_feet_all[start:end]

            def score(dof: torch.Tensor) -> dict:
                metrics = compute_metrics(
                    ctx.kin,
                    wp,
                    wq,
                    dof,
                    fps,
                    feet,
                    reference=reference,
                ).as_dict()
                with torch.no_grad():
                    body, _ = ctx.kin.forward_kinematics(wp, wq, dof)
                candidate_feet = body[:, foot_idx, :]
                for mask_name, mask in window_masks.items():
                    contact = contact_motion_metrics(
                        candidate_feet,
                        fps,
                        mask,
                        reference_foot_pos=teacher_feet,
                    )
                    metrics.update(_flatten_contact_metrics(mask_name, contact, feet))
                return metrics

            rows["raw"].append((clip, score(pred["dof_pos"])))
            for sig in args.sigmas:
                locked = foot_lock_masked(
                    ctx.kin,
                    wp,
                    wq,
                    pred["dof_pos"],
                    feet,
                    window_masks[args.lock_mask],
                    iters=args.lock_iters,
                    lr=args.lock_step_scale,
                    damping=args.lock_damping,
                    blend=args.lock_blend,
                    smooth_sigma=sig,
                    merge_gap=args.lock_merge_gap,
                    extend=args.lock_extend,
                )
                rows[f"lock_s{sig:g}"].append((clip, score(locked)))
            print(f"{clip} window {start}:{end} complete", flush=True)

    aggregates = {}
    distributions = {}
    per_clip = {}
    for variant in variants:
        aggregate, distribution, clip_rows = _aggregate_rows(
            rows[variant],
            bootstrap_samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        aggregates[variant] = aggregate
        distributions[variant] = distribution
        per_clip[variant] = clip_rows

    raw = aggregates["raw"]
    decisions = {}
    for variant in variants[1:]:
        result = aggregates[variant]
        decisions[variant] = {
            "teacher_height_speed_le_0.08": (
                result["teacher_height_stance_speed_ms"] <= 0.08
            ),
            "mpjpe_delta_le_0.005": result["mpjpe_m"] <= raw["mpjpe_m"] + 0.005,
            "absolute_mpjpe_le_0.04": result["mpjpe_m"] <= 0.04,
            "dof_jerk_le_1.2x": result["dof_jerk"] <= 1.2 * raw["dof_jerk"],
            "zero_limit_violations": result["limit_violation_fraction"] == 0.0,
            "penetration_mean_guard": (
                result["penetration_mean_m"] <= raw["penetration_mean_m"] + 0.002
            ),
            "penetration_fraction_guard": (
                result["penetration_fraction"] <= raw["penetration_fraction"] + 0.02
            ),
        }
        decisions[variant]["all_relative_guards_pass"] = all(
            value
            for key, value in decisions[variant].items()
            if key != "absolute_mpjpe_le_0.04"
        )

    out = {
        "ckpt": args.ckpt,
        "provenance": provenance,
        "protocol": {
            "robot": args.robot,
            "clips": args.clips,
            "window": args.window,
            "windows_per_clip_max": args.windows_per_clip,
            "num_windows": len(rows["raw"]),
            "lock_mask": args.lock_mask,
            "lock_mask_definitions": {
                "source_contact": "source height<0.08m and speed<0.24m/s",
                "source_height": "source height hysteresis enter=0.08m, exit=0.10m",
                "teacher_height": "teacher height hysteresis enter=0.03m, exit=0.05m",
            },
            "lock_parameters": {
                "iters": args.lock_iters,
                "step_scale": args.lock_step_scale,
                "damping": args.lock_damping,
                "blend": args.lock_blend,
                "merge_gap": args.lock_merge_gap,
                "extend": args.lock_extend,
                "smooth_sigmas": args.sigmas,
            },
            "primary_endpoint": "teacher_height_stance_speed_ms",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "summary": aggregates,
        "distributions": distributions,
        "per_clip": per_clip,
        "decisions": decisions,
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2)
    tmp.replace(outp)
    concise = {
        variant: {
            "mpjpe_cm": values["mpjpe_m"] * 100,
            "teacher_height_stance_speed_ms": values[
                "teacher_height_stance_speed_ms"
            ],
            "source_contact_stance_speed_ms": values[
                "source_contact_stance_speed_ms"
            ],
            "dof_jerk": values["dof_jerk"],
            "limit_violation_fraction": values["limit_violation_fraction"],
        }
        for variant, values in aggregates.items()
    }
    print("\nSUMMARY:", json.dumps(concise, indent=1))
    print("DECISIONS:", json.dumps(decisions, indent=1))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
