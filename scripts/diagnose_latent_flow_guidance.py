#!/usr/bin/env python
"""Post-hoc diagnostics for the E44 F2 guidance failure (descriptive, not a new gate).

E44 left two candidate explanations for "guidance nearly inert":

  H1 (scale): the raw clamped cost gradient is tiny/badly scaled relative to the learned
      velocity field, so alpha*grad never meaningfully perturbs the trajectory. Predicts:
      grad-norm << velocity-norm per step, low clamp saturation, and a NORM-MATCHED guidance
      probe (rescale grad to a fixed fraction of |v|) moves the endpoint much further.

  H2 (contraction): the z_h-target flow learned a near-Dirac conditional transport
      p(z1|c) ≈ δ_{c}; a contractive field actively annihilates perturbations, so ANY guidance
      applied before the final steps is undone. Predicts: injected perturbations shrink through
      the remaining integration, samples from different noise collapse to the same endpoint,
      and guidance restricted to LATE u (where the field has less time to contract it) does
      relatively more per unit push than early guidance.

This script measures both on the frozen F1 checkpoint + eval windows and writes one JSON:

  per-step norms        |v|, |grad_raw|, |grad_clamped|, clamp saturation fraction, cos(v, grad)
  perturbation decay    inject eps at u in {0.2, 0.5, 0.8}; report |Δz1| / |eps|
  sample diversity      pairwise endpoint distance across 8 noise draws, same condition
  norm-matched probe    guidance with grad rescaled to rho*|v|, rho in {0.1, 0.3, 1.0};
                        report endpoint skate + MPJPE vs the E44 clamped-alpha rows
  descent direction     cos(guidance direction, Adam z-descent displacement direction)

    python scripts/diagnose_latent_flow_guidance.py \
        --flow_ckpt runs/latent_flow/f1_zh/flow_ckpt.pt \
        --out runs/latent_flow/guidance_diagnostics.json
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
from snmr.flow import PhysicsCostConfig, make_decoded_physics_cost  # noqa: E402
from snmr.human import (  # noqa: E402
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.metrics import contact_motion_metrics, detect_contact_height_hysteresis  # noqa: E402
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root  # noqa: E402

from benchmark import load_model  # noqa: E402
from eval_latent_flow import load_flow, make_decode_fn, z_descent  # noqa: E402
from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402


def euler_with_probes(
    flow, cond, cost_fn, num_steps: int, z0: torch.Tensor,
    guidance_mode: str = "none",       # none | clamped | norm_matched
    alpha_end: float = 3.0,
    rho: float = 0.3,                   # norm-matched: |push| = rho * |v| per step
    grad_clamp: float = 0.2,
    skip_final: float = 0.9,
    late_only: bool = False,            # apply guidance only for u >= 0.5
) -> tuple[torch.Tensor, list[dict]]:
    """Euler integration that records per-step norms; mirrors snmr.flow.sample_flow."""
    z = z0.clone()
    h = 1.0 / num_steps
    steps = []
    for i in range(num_steps):
        u_val = i * h
        u = torch.full((cond.shape[0],), u_val, device=cond.device, dtype=cond.dtype)
        with torch.no_grad():
            v = flow(z, u, cond)
        rec = {"u": u_val, "v_norm": float(v.norm())}
        active = u_val < skip_final and (not late_only or u_val >= 0.5)
        if guidance_mode != "none" and active:
            z1_hat = (z + (1.0 - u_val) * v).detach().requires_grad_(True)
            with torch.enable_grad():
                cost = cost_fn(z1_hat)
                (grad,) = torch.autograd.grad(cost, z1_hat)
            clamped = grad.clamp(-grad_clamp, grad_clamp)
            rec.update({
                "grad_norm": float(grad.norm()),
                "grad_clamped_norm": float(clamped.norm()),
                "clamp_saturation": float((grad.abs() >= grad_clamp).float().mean()),
                "cos_v_grad": float(torch.nn.functional.cosine_similarity(
                    v.flatten(), grad.flatten(), dim=0
                )),
                "cost": float(cost),
            })
            if guidance_mode == "clamped":
                alpha = alpha_end * u_val  # linear 0 -> alpha_end schedule (E44 convention)
                push = alpha * clamped
            else:  # norm_matched: direction from the raw gradient, magnitude tied to |v|
                direction = grad / grad.norm().clamp_min(1e-12)
                push = rho * v.norm() * direction
            rec["push_norm"] = float(push.norm())
            rec["push_over_v"] = float(push.norm() / v.norm().clamp_min(1e-12))
            v = v - push
        z = z + h * v
        steps.append(rec)
    return z, steps


@torch.no_grad()
def perturbation_decay(flow, cond, z0, num_steps: int, inject_at: float, eps_scale: float):
    """|endpoint difference| / |injected perturbation| — <1 means the field contracts it."""
    h = 1.0 / num_steps
    z_a, z_b = z0.clone(), None
    for i in range(num_steps):
        u_val = i * h
        u = torch.full((cond.shape[0],), u_val, device=cond.device, dtype=cond.dtype)
        if z_b is None and u_val >= inject_at:
            eps = torch.randn_like(z_a) * eps_scale
            z_b = z_a + eps
            eps_norm = float(eps.norm())
        z_a = z_a + h * flow(z_a, u, cond)
        if z_b is not None:
            z_b = z_b + h * flow(z_b, u, cond)
    return float((z_b - z_a).norm()) / max(eps_norm, 1e-12)


def window_metrics(model, ctx, z, anchor_pos, anchor_quat, reference, masks, fps) -> dict:
    with torch.no_grad():
        pred = model.decoder(
            z, ctx.static, ctx.adj, model.embodiment_encoder(ctx.static), ctx.kin.graph
        )
        wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
        body, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
        ref_body, _ = ctx.kin.forward_kinematics(*reference)
        feet = body[:, ctx.foot_idx, :]
        cm = contact_motion_metrics(feet, fps, masks["teacher_height"])
    return {
        "mpjpe_m": float((body - ref_body).norm(dim=-1).mean()),
        "teacher_height_stance_speed_ms": cm["stance_speed_ms"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument("--flow_ckpt", default=str(ROOT / "runs/latent_flow/f1_zh/flow_ckpt.pt"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=["walk1_subject5", "run2_subject1", "dance2_subject4"])
    ap.add_argument("--windows_per_clip", type=int, default=2)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--sample_steps", type=int, default=25)
    ap.add_argument("--rho_grid", nargs="+", type=float, default=[0.1, 0.3, 1.0])
    ap.add_argument("--num_diversity_samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "runs/latent_flow/guidance_diagnostics.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outp = pathlib.Path(args.out)
    if outp.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {outp}")

    dev = args.device
    torch.manual_seed(args.seed)
    model, state = load_model(args.ckpt, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    flow = load_flow(args.flow_ckpt, model.cfg.latent_dim, dev)
    ctx = RobotContext(args.robot, dev)
    if "xy_scale" in state:
        ctx.xy_scale = float(state["xy_scale"])
    skel = lafan1_skeleton(device=dev)
    pairs_root = data_root() / "pairs"
    sample = load_pair_npz(str(pairs_root / args.robot / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(dev))
    h_adj = _adjacency(skel)

    windows = []
    for clip in args.clips:
        pair = load_pair_npz(str(pairs_root / args.robot / f"{clip}.npz"))
        hp_all, hq_all, q_all = (pair[k].to(dev) for k in ("human_pos", "human_quat", "qpos"))
        fps = pair["fps"]
        T = q_all.shape[0]
        with torch.no_grad():
            tb, _ = ctx.kin.forward_kinematics(q_all[:, 0:3], q_all[:, 3:7], q_all[:, 7:])
        th_mask = detect_contact_height_hysteresis(
            tb[:, ctx.foot_idx, :], enter_height=0.03, exit_height=0.05
        )
        starts = np.linspace(0, max(T - args.window, 0), num=args.windows_per_clip, dtype=int)
        for s in starts:
            s = int(s)
            e = s + args.window
            windows.append({
                "clip": clip, "start": s, "fps": fps,
                "hp": hp_all[s:e], "hq": hq_all[s:e], "q": q_all[s:e],
                "th_mask": th_mask[s:e],
            })

    per_window = []
    for w in windows:
        anchor_pos = w["hp"][:, 0, :].clone()
        anchor_pos[:, :2] *= ctx.xy_scale
        anchor_quat = w["hq"][:, 0, :]
        q = w["q"]
        reference = (q[:, 0:3], q[:, 3:7], q[:, 7:])
        masks = {"teacher_height": w["th_mask"]}
        with torch.no_grad():
            z_h = model.encode(human_pose_features(w["hp"], w["hq"]), h_static, h_adj)
        cond = z_h.unsqueeze(0)
        decode_fn = make_decode_fn(model, ctx, anchor_pos, anchor_quat, ctx.foot_idx)
        cost_cfg = PhysicsCostConfig(fps=w["fps"])
        cost_fn = make_decoded_physics_cost(decode_fn, cost_cfg)  # soft_decoded (deployable)

        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        z0 = torch.randn((1,) + z_h.shape, dtype=z_h.dtype, generator=gen).to(dev)

        # -- per-step norms under the E44 clamped-guidance configuration ------------------
        z_clamped, steps = euler_with_probes(
            flow, cond, cost_fn, args.sample_steps, z0,
            guidance_mode="clamped", alpha_end=3.0,
        )
        # -- unguided endpoint (shared z0) -------------------------------------------------
        z_none, _ = euler_with_probes(flow, cond, cost_fn, args.sample_steps, z0)

        row = {
            "clip": w["clip"], "start": w["start"],
            "steps_clamped_a3": steps,
            "endpoint_shift_clamped_a3": float((z_clamped - z_none).norm()),
            "z_h_norm": float(z_h.norm()),
        }

        # -- H2 probes: contraction + diversity --------------------------------------------
        row["perturbation_decay"] = {
            f"u{u_inj:g}": perturbation_decay(
                flow, cond, z0, args.sample_steps, inject_at=u_inj, eps_scale=1.0
            )
            for u_inj in (0.2, 0.5, 0.8)
        }
        endpoints = []
        for k in range(args.num_diversity_samples):
            gk = torch.Generator(device="cpu").manual_seed(args.seed + 1000 + k)
            z0k = torch.randn((1,) + z_h.shape, dtype=z_h.dtype, generator=gk).to(dev)
            zk, _ = euler_with_probes(flow, cond, cost_fn, args.sample_steps, z0k)
            endpoints.append(zk)
        pair_d = [
            float((endpoints[i] - endpoints[j]).norm())
            for i in range(len(endpoints)) for j in range(i + 1, len(endpoints))
        ]
        row["endpoint_diversity"] = {
            "mean_pairwise_dist": float(np.mean(pair_d)),
            "relative_to_z_h_norm": float(np.mean(pair_d)) / max(row["z_h_norm"], 1e-12),
        }

        # -- H1 probe: norm-matched guidance grid (+ a late-only variant at rho=0.3) -------
        row["variants"] = {
            "raw_decode": window_metrics(
                model, ctx, z_h, anchor_pos, anchor_quat, reference, masks, w["fps"]
            ),
            "flow_unguided": window_metrics(
                model, ctx, z_none[0], anchor_pos, anchor_quat, reference, masks, w["fps"]
            ),
            "flow_clamped_a3": window_metrics(
                model, ctx, z_clamped[0], anchor_pos, anchor_quat, reference, masks, w["fps"]
            ),
        }
        for rho in args.rho_grid:
            z_nm, _ = euler_with_probes(
                flow, cond, cost_fn, args.sample_steps, z0,
                guidance_mode="norm_matched", rho=rho,
            )
            row["variants"][f"flow_norm_matched_rho{rho:g}"] = window_metrics(
                model, ctx, z_nm[0], anchor_pos, anchor_quat, reference, masks, w["fps"]
            )
        z_late, _ = euler_with_probes(
            flow, cond, cost_fn, args.sample_steps, z0,
            guidance_mode="norm_matched", rho=0.3, late_only=True,
        )
        row["variants"]["flow_norm_matched_rho0.3_late_only"] = window_metrics(
            model, ctx, z_late[0], anchor_pos, anchor_quat, reference, masks, w["fps"]
        )

        # -- direction agreement with the working corrector --------------------------------
        z_desc = z_descent(z_h, cost_fn, iters=args.sample_steps)
        desc_dir = (z_desc - z_h).flatten()
        guid_dir = (z_clamped[0] - z_none[0]).flatten()
        row["cos_guidance_vs_descent"] = float(torch.nn.functional.cosine_similarity(
            guid_dir, desc_dir, dim=0
        )) if guid_dir.norm() > 1e-9 and desc_dir.norm() > 1e-9 else None
        row["variants"]["z_descent"] = window_metrics(
            model, ctx, z_desc, anchor_pos, anchor_quat, reference, masks, w["fps"]
        )

        per_window.append(row)
        print(f"diagnosed {w['clip']}@{w['start']}", flush=True)

    # ---- aggregate summary ---------------------------------------------------------------
    def agg_metric(variant: str, key: str):
        vals = [
            r["variants"][variant][key] for r in per_window
            if r["variants"][variant][key] is not None
        ]
        return float(np.mean(vals)) if vals else None

    variants = list(per_window[0]["variants"])
    guided_steps = [s for r in per_window for s in r["steps_clamped_a3"] if "grad_norm" in s]
    summary = {
        "grad_to_velocity_norm_ratio_median": float(np.median(
            [s["grad_clamped_norm"] / max(s["v_norm"], 1e-12) for s in guided_steps]
        )),
        "raw_grad_norm_median": float(np.median([s["grad_norm"] for s in guided_steps])),
        "clamp_saturation_median": float(np.median(
            [s["clamp_saturation"] for s in guided_steps]
        )),
        "push_over_v_median_a3": float(np.median(
            [s["push_over_v"] for s in guided_steps]
        )),
        "perturbation_decay_mean": {
            k: float(np.mean([r["perturbation_decay"][k] for r in per_window]))
            for k in per_window[0]["perturbation_decay"]
        },
        "endpoint_diversity_relative_mean": float(np.mean(
            [r["endpoint_diversity"]["relative_to_z_h_norm"] for r in per_window]
        )),
        "cos_guidance_vs_descent_mean": float(np.mean(
            [r["cos_guidance_vs_descent"] for r in per_window
             if r["cos_guidance_vs_descent"] is not None]
        )),
        "variant_means": {
            v: {
                "mpjpe_m": agg_metric(v, "mpjpe_m"),
                "teacher_height_stance_speed_ms": agg_metric(
                    v, "teacher_height_stance_speed_ms"
                ),
            }
            for v in variants
        },
    }

    result = {
        "provenance": {
            "created_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "git": git_state(ROOT),
            "runtime": runtime_state(dev),
            "checkpoint": {"path": str(pathlib.Path(args.ckpt).resolve()),
                           "sha256": sha256_file(args.ckpt)},
            "flow_checkpoint": {"path": str(pathlib.Path(args.flow_ckpt).resolve()),
                                "sha256": sha256_file(args.flow_ckpt)},
        },
        "config": vars(args),
        "summary": summary,
        "per_window": per_window,
    }
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {outp}")
    print(json.dumps({k: v for k, v in summary.items() if k != "variant_means"}, indent=2))
    for v, m in summary["variant_means"].items():
        speed = m["teacher_height_stance_speed_ms"]
        print(f"{v:38s} mpjpe {m['mpjpe_m']*100:6.2f}cm  th-stance "
              f"{speed:.3f}" if speed is not None else f"{v:38s} (no stance samples)")


if __name__ == "__main__":
    main()
