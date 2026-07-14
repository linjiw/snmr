#!/usr/bin/env python
"""Benchmark a SNMR checkpoint against the GMR teacher on held-out clips (design doc §6).

For every robot the checkpoint supports, and every held-out clip window:
  * runs SNMR inference (human -> latent -> robot qpos, world-recomposed via the per-robot scale),
  * scores BOTH the SNMR prediction and the GMR-teacher trajectory with the identical metric code
    (`snmr.metrics`): MPJPE (SNMR vs teacher only), foot-skate, penetration, jerk, joint limits,
  * measures inference throughput (frames/s),
and writes a JSON + a human-readable markdown table to --out.

The teacher rows quantify the *reference cost* of each metric (e.g. the teacher itself has some
foot skate); SNMR should match or beat the teacher on physical-plausibility metrics while staying
close in MPJPE — that framing is the GMR-paper methodology inverted, and matches how AdaMorph/NMR
report retargeter quality.

    python scripts/benchmark.py --ckpt runs/phase1_g1_large/ckpt_100k_final.pt --robots unitree_g1
    python scripts/benchmark.py --ckpt runs/phase2_all5/ckpt.pt   # all robots in the checkpoint
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.data import local_root_to_world  # noqa: E402
from snmr.diagnostics import binary_contact_metrics  # noqa: E402
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
    compute_metrics,
    contact_motion_metrics,
    detect_contact,
    detect_contact_height_hysteresis,
)
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402

from train_phase2 import ALL_ROBOTS, ROBOT_XY_SCALE_CONFIG, VAL_CLIPS, RobotContext  # noqa: E402

from snmr.paths import data_root  # noqa: E402


def load_model(ckpt_path: str, device: str) -> tuple[SNMR, dict]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    tcfg = state.get("config", {})
    sd = state["model"]
    # Reconstruct the exact architecture. Dim sizes come from the stored trainer args; the
    # OPTIONAL sub-modules (temporal transformer, contact head) are detected from the state dict
    # itself — config flags are unreliable across trainers (e.g. train_phase1's contact ablation
    # sets contact_weight>0 but builds no contact head).
    cfg = SNMRConfig(
        latent_dim=tcfg.get("latent_dim", 64),
        enc_hidden=tcfg.get("enc_hidden", 128),
        dec_hidden=tcfg.get("dec_hidden", 128),
        use_temporal=any(k.startswith("encoder.temporal.") for k in sd),
        # parameter-free option: not detectable from the state dict, must come from the config
        temporal_positional=tcfg.get("temporal_positional", False),
        predict_contact=any(k.startswith("decoder.contact_head.") for k in sd),
    )
    model = SNMR(cfg).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, state


def _cuda_sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


@torch.no_grad()
def _infer_window(model, ctx, h_static, h_adj, hp, hq):
    anchor_pos = hp[:, 0, :].clone()
    anchor_pos[:, :2] *= ctx.xy_scale
    anchor_quat = hq[:, 0, :]
    features = human_pose_features(hp, hq)
    latent = model.encode(features, h_static, h_adj)
    pred = model.decoder(
        latent,
        ctx.static,
        ctx.adj,
        model.embodiment_encoder(ctx.static),
        ctx.kin.graph,
    )
    world_pos, world_quat = local_root_to_world(
        anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"]
    )
    return pred, world_pos, world_quat


@torch.no_grad()
def _time_inference(call, frames: int, device: str, warmup: int, repeats: int) -> tuple[tuple, dict]:
    if repeats < 1:
        raise ValueError("timing repeats must be positive")
    output = None
    for _ in range(warmup):
        output = call()
    _cuda_sync(device)
    durations = []
    for _ in range(repeats):
        _cuda_sync(device)
        start = time.perf_counter()
        output = call()
        _cuda_sync(device)
        durations.append(time.perf_counter() - start)
    fps_samples = np.asarray([frames / max(duration, 1e-12) for duration in durations])
    return output, {
        "scope": "human tensors through features, encoder, decoder, and world recomposition",
        "warmup_repeats": warmup,
        "timed_repeats": repeats,
        "frames_per_repeat": frames,
        "fps_median": float(np.median(fps_samples)),
        "fps_p10": float(np.percentile(fps_samples, 10)),
        "fps_p90": float(np.percentile(fps_samples, 90)),
        "duration_s_median": float(np.median(durations)),
    }


def _flatten_contact_metrics(prefix: str, values: dict, foot_names: list[str]) -> dict:
    flat = {
        f"{prefix}_stance_speed_ms": values["stance_speed_ms"],
        f"{prefix}_slide_fraction": values["slide_fraction"],
        f"{prefix}_floating_mean_m": values["floating_mean_m"],
        f"{prefix}_floating_fraction": values["floating_fraction"],
        f"{prefix}_contact_prevalence": values["contact_prevalence"],
        f"{prefix}_contact_samples": values["contact_samples"],
    }
    for index, foot_name in enumerate(foot_names):
        key = foot_name.replace(" ", "_")
        flat[f"{prefix}_{key}_stance_speed_ms"] = values["per_foot_stance_speed_ms"][index]
        flat[f"{prefix}_{key}_slide_fraction"] = values["per_foot_slide_fraction"][index]
        flat[f"{prefix}_{key}_floating_fraction"] = (
            values["per_foot_floating_fraction"][index]
        )
        flat[f"{prefix}_{key}_contact_samples"] = values["per_foot_contact_samples"][index]
    return flat


def _flatten_classification(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _aggregate_rows(
    rows: list[tuple[str, dict]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict, dict]:
    scalar_keys = sorted({
        key
        for _, row in rows
        for key, value in row.items()
        if value is not None and isinstance(value, (int, float))
    })
    aggregate = {
        key: float(np.mean([row[key] for _, row in rows if row.get(key) is not None]))
        for key in scalar_keys
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for clip, row in rows:
        grouped[clip].append(row)
    per_clip = {}
    for clip, clip_rows in grouped.items():
        per_clip[clip] = {
            key: float(np.mean([row[key] for row in clip_rows if row.get(key) is not None]))
            for key in scalar_keys
            if any(row.get(key) is not None for row in clip_rows)
        }

    rng = np.random.default_rng(seed)
    distributions = {}
    for key in scalar_keys:
        values = np.asarray([
            metrics[key] for metrics in per_clip.values() if metrics.get(key) is not None
        ], dtype=np.float64)
        if values.size == 0:
            continue
        if bootstrap_samples > 0:
            indices = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
            bootstrap_means = values[indices].mean(axis=1)
            ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
        else:
            ci_low = ci_high = float(values.mean())
        distributions[key] = {
            "clip_mean": float(values.mean()),
            "clip_median": float(np.median(values)),
            "clip_p95": float(np.percentile(values, 95)),
            "bootstrap_ci95_low": float(ci_low),
            "bootstrap_ci95_high": float(ci_high),
            "num_clips": int(values.size),
        }
    return aggregate, distributions, per_clip


@torch.no_grad()
def benchmark_robot(
    model,
    ctx: RobotContext,
    skel,
    h_static,
    h_adj,
    pairs_root,
    device,
    window: int,
    windows_per_clip: int,
    timing_warmup: int,
    timing_repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    rows_pred: list[tuple[str, dict]] = []
    rows_teacher: list[tuple[str, dict]] = []
    foot_names = FOOT_BODIES[ctx.name]
    foot_idx = [ctx.kin.body_index(name) for name in foot_names]
    timing = None
    observed_fps = set()

    for clip in VAL_CLIPS:
        pair = load_pair_npz(str(pairs_root / ctx.name / f"{clip}.npz"))
        hp_all = pair["human_pos"].to(device)
        hq_all = pair["human_quat"].to(device)
        q_all = pair["qpos"].to(device)
        fps = pair["fps"]
        observed_fps.add(float(fps))
        frames = q_all.shape[0]

        reference_body, _ = ctx.kin.forward_kinematics(
            q_all[:, 0:3], q_all[:, 3:7], q_all[:, 7:]
        )
        reference_feet = reference_body[:, foot_idx, :]
        source_idx = [pair["human_names"].index(name) for name in LAFAN1_CONTACT_BODIES]
        source_feet = hp_all[:, source_idx, :]
        masks = {
            "source_contact": detect_contact(
                source_feet, fps, height_threshold=0.08, speed_threshold=0.24
            ),
            "source_height": detect_contact_height_hysteresis(
                source_feet, enter_height=0.08, exit_height=0.10
            ),
            "teacher_height": detect_contact_height_hysteresis(
                reference_feet, enter_height=0.03, exit_height=0.05
            ),
            "teacher_legacy": detect_contact(reference_feet, fps),
        }

        starts = np.linspace(
            0,
            max(frames - window, 0),
            num=min(windows_per_clip, max(frames // window, 1)),
            dtype=int,
        )
        for start_index in starts:
            start_index = int(start_index)
            end_index = start_index + window
            hp = hp_all[start_index:end_index]
            hq = hq_all[start_index:end_index]
            q = q_all[start_index:end_index]
            inference_call = lambda: _infer_window(  # noqa: E731
                model, ctx, h_static, h_adj, hp, hq
            )
            if timing is None:
                (pred, world_pos, world_quat), timing = _time_inference(
                    inference_call,
                    frames=hp.shape[0],
                    device=device,
                    warmup=timing_warmup,
                    repeats=timing_repeats,
                )
            else:
                pred, world_pos, world_quat = inference_call()

            reference = (q[:, 0:3], q[:, 3:7], q[:, 7:])
            pred_metrics = compute_metrics(
                ctx.kin,
                world_pos,
                world_quat,
                pred["dof_pos"],
                fps,
                foot_names,
                reference=reference,
            ).as_dict()
            teacher_metrics = compute_metrics(
                ctx.kin, *reference, fps, foot_names, reference=None
            ).as_dict()
            pred_body, _ = ctx.kin.forward_kinematics(
                world_pos, world_quat, pred["dof_pos"]
            )
            pred_feet = pred_body[:, foot_idx, :]
            teacher_feet = reference_feet[start_index:end_index]

            for mask_name, full_mask in masks.items():
                mask = full_mask[start_index:end_index]
                pred_contact = contact_motion_metrics(
                    pred_feet, fps, mask, reference_foot_pos=teacher_feet
                )
                teacher_contact = contact_motion_metrics(
                    teacher_feet, fps, mask, reference_foot_pos=teacher_feet
                )
                pred_metrics.update(
                    _flatten_contact_metrics(mask_name, pred_contact, foot_names)
                )
                teacher_metrics.update(
                    _flatten_contact_metrics(mask_name, teacher_contact, foot_names)
                )
                if "contact_logits" in pred:
                    classification = binary_contact_metrics(
                        pred["contact_logits"][:, foot_idx], mask
                    )
                    pred_metrics.update(_flatten_classification(
                        f"contact_head_vs_{mask_name}", classification
                    ))

            rows_pred.append((clip, pred_metrics))
            rows_teacher.append((clip, teacher_metrics))

    pred_mean, pred_distribution, pred_per_clip = _aggregate_rows(
        rows_pred, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed
    )
    teacher_mean, teacher_distribution, teacher_per_clip = _aggregate_rows(
        rows_teacher, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed
    )
    assert timing is not None
    return {
        "snmr": pred_mean,
        "teacher": teacher_mean,
        "throughput_fps": timing["fps_median"],
        "timing": timing,
        "num_windows": len(rows_pred),
        "fps": sorted(observed_fps),
        "distributions": {
            "snmr": pred_distribution,
            "teacher": teacher_distribution,
        },
        "per_clip": {
            "snmr": pred_per_clip,
            "teacher": teacher_per_clip,
        },
    }


def to_markdown(results: dict, ckpt: str) -> str:
    lines = [f"# SNMR benchmark — `{ckpt}`", "",
             "SNMR scored against the GMR teacher on held-out clips "
             f"({', '.join(VAL_CLIPS)}). Teacher rows = the optimization baseline's own metric "
             "values (no MPJPE: it is the reference).", ""]
    metrics_order = [
        ("mpjpe_m", "MPJPE (m)"), ("dof_err_rad", "dof err (rad)"),
        ("source_contact_stance_speed_ms", "stance speed, source-contact mask (m/s)"),
        ("source_height_stance_speed_ms", "stance speed, source-height mask (m/s)"),
        ("teacher_height_stance_speed_ms", "stance speed, teacher-height mask (m/s)"),
        ("teacher_legacy_stance_speed_ms", "stance speed, legacy teacher mask (m/s)"),
        ("teacher_height_slide_fraction", "slide frac, teacher-height mask"),
        ("teacher_height_floating_fraction", "floating frac, teacher-height mask"),
        ("contact_head_vs_teacher_height_f1", "contact-head F1 vs teacher-height"),
        ("foot_skate_speed_ms", "legacy foot skate (m/s)"),
        ("foot_skate_mann_cm", "FS-MANN (cm/f)"),
        ("foot_height_mean_m", "foot height mean (m)"),
        ("foot_floating_mean_m", "stance floating mean (m)"),
        ("foot_floating_fraction", "foot floating frac"),
        ("penetration_mean_m", "pen. mean (m)"), ("penetration_fraction", "pen. frac"),
        ("dof_jerk", "dof jerk (rad/s³)"), ("body_jerk", "body jerk (m/s³)"),
        ("joint_jump_fraction", "joint jumps"),
        ("limit_violation_fraction", "limit viol."),
        ("limit_proximity_fraction", "limit prox."),
    ]
    for robot, res in results.items():
        timing = res["timing"]
        lines.append(f"## {robot}  ({res['num_windows']} windows, "
                     f"{timing['fps_median']:.0f} frames/s median inference; "
                     f"p10/p90 {timing['fps_p10']:.0f}/{timing['fps_p90']:.0f})")
        lines.append("| metric | SNMR | teacher (GMR) |")
        lines.append("|---|---|---|")
        for key, label in metrics_order:
            sv = res["snmr"].get(key)
            tv = res["teacher"].get(key)
            fmt = lambda v: "—" if v is None else (f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
            lines.append(f"| {label} | {fmt(sv)} | {fmt(tv)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--robots", nargs="+", default=None,
                    help="default: all robots with pairs data")
    ap.add_argument("--pairs_root", default=None, help="default: <data_root>/pairs")
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--windows_per_clip", type=int, default=6)
    ap.add_argument("--timing_warmup", type=int, default=10)
    ap.add_argument("--timing_repeats", type=int, default=30)
    ap.add_argument("--bootstrap_samples", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output prefix (default: alongside ckpt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, state = load_model(args.ckpt, args.device)
    robots = args.robots or ALL_ROBOTS
    pairs_root = pathlib.Path(args.pairs_root) if args.pairs_root else data_root() / "pairs"

    skel = lafan1_skeleton(device=args.device)
    sample = load_pair_npz(str(pairs_root / robots[0] / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(args.device))
    h_adj = _adjacency(skel)

    results = {}
    for robot in robots:
        ctx = RobotContext(robot, args.device)
        # allow per-checkpoint scale override (phase-1 ckpts store a single fitted value)
        if "xy_scale" in state:
            ctx.xy_scale = float(state["xy_scale"])
        elif "xy_scales" in state:
            ctx.xy_scale = float(state["xy_scales"].get(robot, ROBOT_XY_SCALE_CONFIG[robot]))
        print(f"benchmarking {robot} (xy_scale={ctx.xy_scale:.4f}) ...")
        results[robot] = benchmark_robot(model, ctx, skel, h_static, h_adj, pairs_root,
                                         args.device, args.window, args.windows_per_clip,
                                         args.timing_warmup, args.timing_repeats,
                                         args.bootstrap_samples, args.bootstrap_seed)

    prefix = pathlib.Path(args.out) if args.out else pathlib.Path(args.ckpt).parent / "benchmark"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "evaluator": "scripts/benchmark.py",
        "evaluator_sha256": sha256_file(__file__),
        "evaluated_utc": utc_now(),
        "git": git_state(ROOT),
        "runtime": runtime_state(args.device),
        "checkpoint": {
            "path": str(pathlib.Path(args.ckpt).resolve()),
            "sha256": sha256_file(args.ckpt),
            "step": state.get("step"),
            "complete": state.get("step") == state.get("config", {}).get("steps"),
        },
        "window_frames": args.window,
        "windows_per_clip_max": args.windows_per_clip,
        "clips": list(VAL_CLIPS),
        "timing": {
            "warmup_repeats": args.timing_warmup,
            "timed_repeats": args.timing_repeats,
            "cuda_sync_before_and_after": torch.device(args.device).type == "cuda",
            "scope": "human tensors through features, encoder, decoder, and world recomposition",
            "teacher_timing": "not measured by this evaluator",
        },
        "contact_definitions": {
            "source_contact": {
                "source": "LAFAN1 LeftToe/RightToe",
                "velocity_gated": True,
                "relative_height": True,
                "height_threshold_m": 0.08,
                "source_speed_threshold_ms": 0.24,
            },
            "source_height": {
                "source": "LAFAN1 LeftToe/RightToe",
                "velocity_gated": False,
                "relative_height": True,
                "enter_height_m": 0.08,
                "exit_height_m": 0.10,
            },
            "teacher_height": {
                "source": "GMR teacher robot feet",
                "velocity_gated": False,
                "relative_height": True,
                "enter_height_m": 0.03,
                "exit_height_m": 0.05,
            },
            "teacher_legacy": {
                "source": "GMR teacher robot feet",
                "velocity_gated": True,
                "relative_height": True,
                "height_threshold_m": 0.03,
                "speed_threshold_ms": 0.3,
            },
        },
        "slide_speed_threshold_ms": 0.3,
        "bootstrap": {
            "unit": "clip",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "confidence": 0.95,
        },
    }
    with open(f"{prefix}.json", "w") as fh:
        json.dump({"_protocol": protocol, **results}, fh, indent=2)
    md = to_markdown(results, args.ckpt)
    with open(f"{prefix}.md", "w") as fh:
        fh.write(md)
    print(md)
    print(f"\nwrote {prefix}.json and {prefix}.md")


if __name__ == "__main__":
    main()
