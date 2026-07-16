#!/usr/bin/env python
"""E26b: evaluate source-mask foot locking on the frozen Gate 0 protocol.

Grid over smooth_sigma to measure the jerk/skate trade-off. This script uses the same full-clip
contact masks, fixed windows, physical guards, and clip bootstrap as ``scripts/benchmark.py``.
The result diagnoses the DLS foot-lock heuristic; it is not the full windowed C6 projection.

    python scripts/eval_footlock.py --ckpt runs/phase1_g1_large/ckpt_100k_final.pt
"""

from __future__ import annotations

import argparse
import dataclasses
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
from snmr.projection import (  # noqa: E402
    WindowedProjectionConfig,
    windowed_contact_projection,
)

from benchmark import (  # noqa: E402
    _aggregate_rows,
    _flatten_contact_metrics,
    load_model,
)
from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402


def predicted_contact_mask(
    contact_logits: torch.Tensor,
    foot_indices: list[int],
    *,
    probability_threshold: float = 0.5,
) -> torch.Tensor:
    """Threshold decoder foot-contact probabilities for projection."""
    if contact_logits.ndim != 2:
        raise ValueError(
            f"contact logits must have shape (T, N), got {tuple(contact_logits.shape)}"
        )
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("contact probability threshold must be in (0, 1)")
    if not foot_indices:
        raise ValueError("at least one foot index is required")
    return torch.sigmoid(contact_logits[:, foot_indices]) >= probability_threshold


def decoded_clip_ground_height(
    model,
    ctx: RobotContext,
    hp_all: torch.Tensor,
    hq_all: torch.Tensor,
    h_static: torch.Tensor,
    h_adj: torch.Tensor,
    foot_indices: list[int],
    *,
    window: int,
) -> torch.Tensor:
    """Decode consecutive clip tiles and return each foot's minimum world height."""
    if window < 2:
        raise ValueError("window must be at least 2 frames")
    if hp_all.shape[0] != hq_all.shape[0]:
        raise ValueError("human position and quaternion clips must have equal length")
    if hp_all.shape[0] < 2:
        raise ValueError("clip must contain at least 2 frames")

    heights = []
    for seg_start in range(0, hp_all.shape[0], window):
        seg_end = min(seg_start + window, hp_all.shape[0])
        if seg_end - seg_start < 2:
            continue
        hp_seg = hp_all[seg_start:seg_end]
        hq_seg = hq_all[seg_start:seg_end]
        seg_anchor_pos = hp_seg[:, 0, :].clone()
        seg_anchor_pos[:, :2] *= ctx.xy_scale
        with torch.no_grad():
            seg_feats = human_pose_features(hp_seg, hq_seg)
            seg_z = model.encode(seg_feats, h_static, h_adj)
            seg_pred = model.decoder(
                seg_z,
                ctx.static,
                ctx.adj,
                model.embodiment_encoder(ctx.static),
                ctx.kin.graph,
            )
            seg_wp, seg_wq = local_root_to_world(
                seg_anchor_pos,
                hq_seg[:, 0, :],
                seg_pred["root_pos"],
                seg_pred["root_quat"],
            )
            seg_body, _ = ctx.kin.forward_kinematics(
                seg_wp, seg_wq, seg_pred["dof_pos"]
            )
        heights.append(seg_body[:, foot_indices, 2])
    if not heights:
        raise ValueError("clip tiling produced no decodable segments")
    return torch.cat(heights, dim=0).min(dim=0).values


def projection_decision(result: dict, raw: dict) -> dict[str, bool]:
    """Apply the frozen Gate-1b coprimary endpoints and physical guards."""
    decision = {
        "teacher_height_speed_le_0.08": (
            result["teacher_height_stance_speed_ms"] <= 0.08
        ),
        "source_contact_speed_le_0.10": (
            result["source_contact_stance_speed_ms"] <= 0.10
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
    decision["all_relative_guards_pass"] = all(
        value
        for key, value in decision.items()
        if key != "absolute_mpjpe_le_0.04"
    )
    return decision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--windows_per_clip", type=int, default=6)
    ap.add_argument(
        "--only_window_start",
        type=int,
        default=None,
        help="evaluate one explicit start frame; requires exactly one --clips value",
    )
    ap.add_argument(
        "--method",
        choices=["framewise", "windowed"],
        default="framewise",
        help="framewise E26 DLS or joint temporal C6 projection",
    )
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    ap.add_argument("--lock_iters", type=int, default=12)
    ap.add_argument("--lock_step_scale", type=float, default=0.5)
    ap.add_argument("--lock_damping", type=float, default=1e-2)
    ap.add_argument("--lock_blend", type=int, default=2)
    ap.add_argument("--lock_merge_gap", type=int, default=6)
    ap.add_argument("--lock_extend", type=int, default=2)
    ap.add_argument("--projection_iters", type=int, default=30)
    ap.add_argument("--projection_history_size", type=int, default=10)
    ap.add_argument("--projection_lr", type=float, default=1.0)
    ap.add_argument("--projection_stance_weight", type=float, default=1000.0)
    ap.add_argument("--projection_stance_velocity_weight", type=float, default=1000.0)
    ap.add_argument("--projection_deviation_weight", type=float, default=0.1)
    ap.add_argument("--projection_velocity_weight", type=float, default=0.5)
    ap.add_argument("--projection_acceleration_weight", type=float, default=1.0)
    ap.add_argument("--projection_root_translation_bound", type=float, default=0.04)
    ap.add_argument("--projection_root_yaw_bound", type=float, default=0.12)
    ap.add_argument("--projection_joint_delta_bound", type=float, default=0.35)
    ap.add_argument("--projection_merge_gap", type=int, default=0)
    ap.add_argument("--projection_extend", type=int, default=0)
    ap.add_argument("--projection_min_stance_frames", type=int, default=1)
    ap.add_argument("--projection_tolerance_grad", type=float, default=1e-7)
    ap.add_argument("--projection_tolerance_change", type=float, default=1e-9)
    ap.add_argument(
        "--lock_mask",
        choices=[
            "source_contact",
            "source_height",
            "teacher_height",
            "predicted_contact",
            "decoded_height",
            "decoded_height_clip",
        ],
        default="source_contact",
    )
    ap.add_argument(
        "--contact_probability_threshold",
        type=float,
        default=0.5,
        help="fixed decoder contact probability threshold for --lock_mask predicted_contact",
    )
    ap.add_argument(
        "--mask_ckpt",
        default=None,
        help=(
            "optional separate checkpoint whose contact head generates the predicted_contact "
            "mask (Gate 1b M2: the C1 BCE head applied to the frozen base decoder's output)"
        ),
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
    projection_config = WindowedProjectionConfig(
        max_iterations=args.projection_iters,
        history_size=args.projection_history_size,
        learning_rate=args.projection_lr,
        stance_weight=args.projection_stance_weight,
        stance_velocity_weight=args.projection_stance_velocity_weight,
        deviation_weight=args.projection_deviation_weight,
        velocity_weight=args.projection_velocity_weight,
        acceleration_weight=args.projection_acceleration_weight,
        root_translation_bound_m=args.projection_root_translation_bound,
        root_yaw_bound_rad=args.projection_root_yaw_bound,
        joint_delta_bound_rad=args.projection_joint_delta_bound,
        merge_gap=args.projection_merge_gap,
        extend=args.projection_extend,
        min_stance_frames=args.projection_min_stance_frames,
        tolerance_grad=args.projection_tolerance_grad,
        tolerance_change=args.projection_tolerance_change,
    )

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
            "mask_checkpoint": (
                {
                    "path": str(pathlib.Path(args.mask_ckpt).resolve()),
                    "sha256": sha256_file(args.mask_ckpt),
                }
                if args.mask_ckpt is not None
                else None
            ),
            "evaluator": {
                "path": str(pathlib.Path(__file__).resolve()),
                "sha256": sha256_file(__file__),
            },
            "solver": {
                "path": str(
                    (
                        ROOT
                        / "snmr"
                        / ("footlock.py" if args.method == "framewise" else "projection.py")
                    ).resolve()
                ),
                "sha256": sha256_file(
                    ROOT
                    / "snmr"
                    / ("footlock.py" if args.method == "framewise" else "projection.py")
                ),
            },
        },
    }

    dev = args.device
    model, state = load_model(args.ckpt, dev)
    mask_model = None
    if args.mask_ckpt is not None:
        if args.lock_mask != "predicted_contact":
            raise ValueError("--mask_ckpt is only meaningful with --lock_mask predicted_contact")
        mask_model, _ = load_model(args.mask_ckpt, dev)
        if mask_model.decoder.contact_head is None:
            raise ValueError("--mask_ckpt must contain a trained contact head")
    elif args.lock_mask == "predicted_contact" and model.decoder.contact_head is None:
        raise ValueError(
            "--lock_mask predicted_contact requires a checkpoint with a contact head"
        )
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

    variants = (
        ["raw"] + [f"lock_s{sig:g}" for sig in args.sigmas]
        if args.method == "framewise"
        else ["raw", "windowed"]
    )
    rows: dict[str, list[tuple[str, dict]]] = {v: [] for v in variants}
    projection_diagnostics = []

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

        decoded_clip_ground = None
        if args.lock_mask == "decoded_height_clip":
            # M1b: clip-local decoded ground — tile the whole clip in consecutive windows,
            # decode, FK, and take the per-foot minimum height. Uses no teacher signal.
            decoded_clip_ground = decoded_clip_ground_height(
                model,
                ctx,
                hp_all,
                hq_all,
                h_static,
                h_adj,
                foot_idx,
                window=args.window,
            )

        if args.only_window_start is not None:
            if len(args.clips) != 1:
                raise ValueError("--only_window_start requires exactly one --clips value")
            if not 0 <= args.only_window_start <= max(T - args.window, 0):
                raise ValueError(
                    f"--only_window_start must be in [0, {max(T - args.window, 0)}]"
                )
            starts = np.array([args.only_window_start], dtype=int)
        else:
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
            with torch.no_grad():
                decoded_body, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
            decoded_feet_window = decoded_body[:, foot_idx, :]
            window_masks["decoded_height"] = detect_contact_height_hysteresis(
                decoded_feet_window, enter_height=0.03, exit_height=0.05
            )
            if decoded_clip_ground is not None:
                # Per-foot clip-local ground; broadcast against (T, F) heights.
                window_masks["decoded_height_clip"] = detect_contact_height_hysteresis(
                    decoded_feet_window,
                    enter_height=0.03,
                    exit_height=0.05,
                    ground_z=decoded_clip_ground,
                )
            if mask_model is not None:
                with torch.no_grad():
                    z_m = mask_model.encode(feats, h_static, h_adj)
                    pred_m = mask_model.decoder(
                        z_m,
                        ctx.static,
                        ctx.adj,
                        mask_model.embodiment_encoder(ctx.static),
                        ctx.kin.graph,
                    )
                window_masks["predicted_contact"] = predicted_contact_mask(
                    pred_m["contact_logits"],
                    foot_idx,
                    probability_threshold=args.contact_probability_threshold,
                )
            elif "contact_logits" in pred:
                window_masks["predicted_contact"] = predicted_contact_mask(
                    pred["contact_logits"],
                    foot_idx,
                    probability_threshold=args.contact_probability_threshold,
                )
            teacher_feet = teacher_feet_all[start:end]

            def score(
                candidate_root_pos: torch.Tensor,
                candidate_root_quat: torch.Tensor,
                dof: torch.Tensor,
            ) -> dict:
                metrics = compute_metrics(
                    ctx.kin,
                    candidate_root_pos,
                    candidate_root_quat,
                    dof,
                    fps,
                    feet,
                    reference=reference,
                ).as_dict()
                with torch.no_grad():
                    body, _ = ctx.kin.forward_kinematics(
                        candidate_root_pos,
                        candidate_root_quat,
                        dof,
                    )
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

            rows["raw"].append((clip, score(wp, wq, pred["dof_pos"])))
            if args.method == "framewise":
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
                    rows[f"lock_s{sig:g}"].append((clip, score(wp, wq, locked)))
            else:
                projected = windowed_contact_projection(
                    ctx.kin,
                    wp,
                    wq,
                    pred["dof_pos"],
                    feet,
                    window_masks[args.lock_mask],
                    config=projection_config,
                )
                rows["windowed"].append((
                    clip,
                    score(
                        projected.root_pos,
                        projected.root_quat,
                        projected.dof_pos,
                    ),
                ))
                projection_diagnostics.append({
                    "clip": clip,
                    "start": start,
                    "end": end,
                    **projected.diagnostics,
                })
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
        decisions[variant] = projection_decision(aggregates[variant], raw)

    out = {
        "ckpt": args.ckpt,
        "provenance": provenance,
        "protocol": {
            "robot": args.robot,
            "clips": args.clips,
            "window": args.window,
            "windows_per_clip_max": args.windows_per_clip,
            "num_windows": len(rows["raw"]),
            "method": args.method,
            "lock_mask": args.lock_mask,
            "lock_mask_definitions": {
                "source_contact": "source height<0.08m and speed<0.24m/s",
                "source_height": "source height hysteresis enter=0.08m, exit=0.10m",
                "teacher_height": "teacher height hysteresis enter=0.03m, exit=0.05m",
                "predicted_contact": (
                    "decoder sigmoid probability"
                    f">={args.contact_probability_threshold:g}"
                    + (f" from mask_ckpt {args.mask_ckpt}" if args.mask_ckpt else "")
                ),
                "decoded_height": (
                    "height hysteresis enter=0.03m, exit=0.05m on the decoded feet "
                    "(window-local ground)"
                ),
                "decoded_height_clip": (
                    "height hysteresis enter=0.03m, exit=0.05m on the decoded feet, "
                    "per-foot CLIP-local decoded ground (M1b; no teacher signal)"
                ),
            },
            "contact_probability_threshold": args.contact_probability_threshold,
            "lock_parameters": {
                "iters": args.lock_iters,
                "step_scale": args.lock_step_scale,
                "damping": args.lock_damping,
                "blend": args.lock_blend,
                "merge_gap": args.lock_merge_gap,
                "extend": args.lock_extend,
                "smooth_sigmas": args.sigmas,
            } if args.method == "framewise" else None,
            "projection_parameters": (
                dataclasses.asdict(projection_config)
                if args.method == "windowed"
                else None
            ),
            "primary_endpoint": "teacher_height_stance_speed_ms",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "summary": aggregates,
        "distributions": distributions,
        "per_clip": per_clip,
        "decisions": decisions,
        "projection_diagnostics": projection_diagnostics,
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
