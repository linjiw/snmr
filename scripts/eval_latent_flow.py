#!/usr/bin/env python
"""Evaluate the latent-flow side project against gates F0–F2 (docs/FLOW_RETARGETING_SIDE_PROJECT.md).

Arms, all decoded through the SAME frozen SNMR decoder and scored on the SAME fixed windows:

  raw          deterministic decode of z_h (the paired baseline every guard is relative to)
  zr_oracle    decode of the frozen encoder's teacher-robot latent z_r (F0 target viability)
  z_descent    Adam directly on z (init z_h) minimizing the physics cost — the "no flow prior
               needed" null control, NFE-matched to the flow sampler
  flow         unguided flow sample conditioned on z_h (F1 fidelity)
  flow_guided  physics-guided flow sampling, grid over alpha_end x stance-weight mode (F2)

Guard set and thresholds are Gate-1b's frozen `projection_decision` conventions.

    python scripts/eval_latent_flow.py --flow_ckpt runs/latent_flow/f1_zr/flow_ckpt.pt \
        --out runs/latent_flow/f2_screen.json
    python scripts/eval_latent_flow.py --f0_only --out runs/latent_flow/f0_viability.json
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

from snmr.data import RobotMotion, local_root_to_world, robot_pose_features  # noqa: E402
from snmr.experiment import git_state, runtime_state, sha256_file, utc_now  # noqa: E402
from snmr.flow import (  # noqa: E402
    GuidanceConfig,
    LatentFlowConfig,
    LatentFlowNet,
    PhysicsCostConfig,
    make_decoded_physics_cost,
    sample_flow,
)
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

from benchmark import load_model  # noqa: E402
from train_phase2 import VAL_CLIPS, RobotContext  # noqa: E402


STANCE_MODES = ("soft_decoded", "source_gated", "teacher_height")
DEPLOYABLE_MODES = ("soft_decoded", "source_gated")


def load_flow(flow_ckpt: str, latent_dim: int, device: str) -> LatentFlowNet:
    state = torch.load(flow_ckpt, map_location=device, weights_only=False)
    fcfg = state["flow_config"]
    if fcfg["latent_dim"] != latent_dim:
        raise ValueError(
            f"flow latent_dim {fcfg['latent_dim']} != frozen SNMR latent_dim {latent_dim}"
        )
    flow = LatentFlowNet(LatentFlowConfig(
        latent_dim=fcfg["latent_dim"], hidden_dim=fcfg["hidden_dim"],
        num_layers=fcfg["num_layers"],
    )).to(device)
    flow.load_state_dict(state["flow"])
    flow.eval()
    return flow


def make_decode_fn(model, ctx, anchor_pos, anchor_quat, foot_idx):
    """Latent (B, T, latent) → (world foot positions (B,T,F,3), dof (B,T,D)); graph kept."""
    def decode(z_batch: torch.Tensor):
        feet, dofs = [], []
        for z in z_batch:  # decoder is unbatched over sequences; B is small (1) in this screen
            pred = model.decoder(
                z, ctx.static, ctx.adj, model.embodiment_encoder(ctx.static), ctx.kin.graph
            )
            wp, wq = local_root_to_world(
                anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"]
            )
            body, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
            feet.append(body[:, foot_idx, :])
            dofs.append(pred["dof_pos"])
        return torch.stack(feet), torch.stack(dofs)

    return decode


@torch.no_grad()
def decode_world(model, ctx, z, anchor_pos, anchor_quat):
    pred = model.decoder(
        z, ctx.static, ctx.adj, model.embodiment_encoder(ctx.static), ctx.kin.graph
    )
    wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
    return wp, wq, pred["dof_pos"]


def z_descent(z_h, cost_fn, iters: int, lr: float = 0.003, trust_weight: float = 10.0):
    """The F0/F2 null control: latent gradient descent on the identical cost, no flow prior.

    The trust-region term anchors the solution to z_h so the control cannot trivially win by
    wandering off the decoder manifold; iteration count is NFE-matched to the flow sampler.
    """
    z = z_h.detach().clone().unsqueeze(0).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    anchor = z_h.detach().unsqueeze(0)
    for _ in range(iters):
        opt.zero_grad()
        loss = cost_fn(z) + trust_weight * (z - anchor).square().mean()
        loss.backward()
        opt.step()
    return z.detach()[0]


def score_variant(ctx, wp, wq, dof, reference, fps, masks) -> dict:
    """Frozen-metric row: MPJPE/jerk/penetration/limits + stance speeds under both frozen masks."""
    metrics = compute_metrics(
        ctx.kin, wp, wq, dof, fps, FOOT_BODIES[ctx.name],
        reference=reference, contact_mask=masks["teacher_height"],
    )
    body, _ = ctx.kin.forward_kinematics(wp, wq, dof)
    feet = body[:, ctx.foot_idx, :]
    row = metrics.as_dict()
    for mask_name in ("teacher_height", "source_contact"):
        cm = contact_motion_metrics(feet, fps, masks[mask_name])
        row[f"{mask_name}_stance_speed_ms"] = cm["stance_speed_ms"]
    return row


def flow_decision(result: dict, raw: dict) -> dict[str, bool]:
    """Gate-1b's frozen coprimary endpoints and physical guards, applied to a flow arm."""
    decision = {
        "teacher_height_speed_le_0.08": (
            result["teacher_height_stance_speed_ms"] is not None
            and result["teacher_height_stance_speed_ms"] <= 0.08
        ),
        "source_contact_speed_le_0.10": (
            result["source_contact_stance_speed_ms"] is not None
            and result["source_contact_stance_speed_ms"] <= 0.10
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
        value for key, value in decision.items() if key != "absolute_mpjpe_le_0.04"
    )
    return decision


def aggregate(rows: list[dict]) -> dict:
    """Mean over windows, ignoring None stance speeds (empty-mask windows)."""
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float)) and r[k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument("--flow_ckpt", default=None, help="required unless --f0_only")
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--clips", nargs="+", default=VAL_CLIPS)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--windows_per_clip", type=int, default=6)
    ap.add_argument("--sample_steps", type=int, default=25)
    ap.add_argument("--alpha_grid", nargs="+", type=float, default=[1.0, 3.0, 10.0])
    ap.add_argument("--stance_modes", nargs="+", choices=STANCE_MODES,
                    default=list(STANCE_MODES))
    ap.add_argument("--skate_weight", type=float, default=1.0)
    ap.add_argument("--penetration_weight", type=float, default=1.0)
    ap.add_argument("--smooth_weight", type=float, default=0.1)
    ap.add_argument("--descent_lr", type=float, default=0.003)
    ap.add_argument("--descent_trust_weight", type=float, default=10.0)
    ap.add_argument("--f0_only", action="store_true",
                    help="only run raw / zr_oracle / z_descent (no flow checkpoint needed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "runs/latent_flow/eval_latent_flow.json"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.f0_only and args.flow_ckpt is None:
        ap.error("--flow_ckpt is required unless --f0_only")
    outp = pathlib.Path(args.out)
    if outp.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing result: {outp}")
    outp.parent.mkdir(parents=True, exist_ok=True)

    dev = args.device
    torch.manual_seed(args.seed)
    model, state = load_model(args.ckpt, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    ctx = RobotContext(args.robot, dev)
    if "xy_scale" in state:
        ctx.xy_scale = float(state["xy_scale"])
    skel = lafan1_skeleton(device=dev)
    pairs_root = data_root() / "pairs"
    sample = load_pair_npz(str(pairs_root / args.robot / f"{VAL_CLIPS[0]}.npz"))
    h_static = human_static_features(skel, body_pos_sample=sample["human_pos"].to(dev))
    h_adj = _adjacency(skel)
    foot_idx = ctx.foot_idx

    flow = None
    if not args.f0_only:
        flow = load_flow(args.flow_ckpt, model.cfg.latent_dim, dev)

    guided_variants = [
        f"flow_guided_a{alpha:g}_{mode}"
        for alpha in args.alpha_grid
        for mode in args.stance_modes
    ]
    variants = ["raw", "zr_oracle", "z_descent"]
    if flow is not None:
        variants += ["flow"] + guided_variants
    rows: dict[str, list[dict]] = {v: [] for v in variants}

    cost_cfg = PhysicsCostConfig(
        skate_weight=args.skate_weight,
        penetration_weight=args.penetration_weight,
        smooth_weight=args.smooth_weight,
    )

    for clip in args.clips:
        pair = load_pair_npz(str(pairs_root / args.robot / f"{clip}.npz"))
        hp_all, hq_all, q_all = (pair[k].to(dev) for k in ("human_pos", "human_quat", "qpos"))
        fps = pair["fps"]
        cost_cfg.fps = fps
        T = q_all.shape[0]
        with torch.no_grad():
            teacher_body, _ = ctx.kin.forward_kinematics(
                q_all[:, 0:3], q_all[:, 3:7], q_all[:, 7:]
            )
        teacher_feet_all = teacher_body[:, foot_idx, :]
        source_idx = [pair["human_names"].index(n) for n in LAFAN1_CONTACT_BODIES]
        source_feet = hp_all[:, source_idx, :]
        # full-clip masks, frozen Gate-0/1b definitions (windowed masks are slices of these)
        full_masks = {
            "source_contact": detect_contact(
                source_feet, fps, height_threshold=0.08, speed_threshold=0.24
            ),
            "teacher_height": detect_contact_height_hysteresis(
                teacher_feet_all, enter_height=0.03, exit_height=0.05
            ),
        }

        starts = np.linspace(
            0, max(T - args.window, 0),
            num=min(args.windows_per_clip, max(T // args.window, 1)), dtype=int,
        )
        for start in starts:
            start = int(start)
            end = start + args.window
            hp, hq, q = hp_all[start:end], hq_all[start:end], q_all[start:end]
            anchor_pos = hp[:, 0, :].clone()
            anchor_pos[:, :2] *= ctx.xy_scale
            anchor_quat = hq[:, 0, :]
            reference = (q[:, 0:3], q[:, 3:7], q[:, 7:])
            masks = {name: m[start:end] for name, m in full_masks.items()}

            with torch.no_grad():
                z_h = model.encode(human_pose_features(hp, hq), h_static, h_adj)
                motion = RobotMotion(q[:, 0:3], q[:, 3:7], q[:, 7:], fps=fps)
                z_r = model.encode(
                    robot_pose_features(ctx.kin, motion), ctx.static, ctx.adj
                )

            def add_row(variant: str, z: torch.Tensor) -> None:
                with torch.no_grad():
                    wp, wq, dof = decode_world(model, ctx, z, anchor_pos, anchor_quat)
                    rows[variant].append(
                        score_variant(ctx, wp, wq, dof, reference, fps, masks)
                    )

            add_row("raw", z_h)
            add_row("zr_oracle", z_r)

            decode_fn = make_decode_fn(model, ctx, anchor_pos, anchor_quat, foot_idx)

            def make_cost(mode: str):
                if mode == "teacher_height":
                    return make_decoded_physics_cost(
                        decode_fn, cost_cfg,
                        stance_weights=masks["teacher_height"].unsqueeze(0).float(),
                    )
                if mode == "source_gated":
                    return make_decoded_physics_cost(
                        decode_fn, cost_cfg,
                        stance_gate=masks["source_contact"].unsqueeze(0).float(),
                    )
                return make_decoded_physics_cost(decode_fn, cost_cfg)  # soft_decoded

            # NFE-matched null control (soft_decoded cost — the fully deployable signal)
            add_row("z_descent", z_descent(
                z_h, make_cost("soft_decoded"), iters=args.sample_steps,
                lr=args.descent_lr, trust_weight=args.descent_trust_weight,
            ))

            if flow is not None:
                clip_tag = int.from_bytes(clip.encode()[:4].ljust(4, b"\0"), "little")
                gen = torch.Generator(device="cpu").manual_seed(
                    args.seed + 7919 * start + clip_tag % 65536
                )
                z0 = torch.randn((1,) + z_h.shape, dtype=z_h.dtype, generator=gen).to(dev)
                add_row("flow", sample_flow(
                    flow, z_h.unsqueeze(0), num_steps=args.sample_steps, z0=z0
                )[0])
                for alpha in args.alpha_grid:
                    for mode in args.stance_modes:
                        guided = sample_flow(
                            flow, z_h.unsqueeze(0), num_steps=args.sample_steps,
                            z0=z0.clone(),
                            guidance=GuidanceConfig(
                                cost_fn=make_cost(mode), alpha_end=alpha,
                            ),
                        )[0]
                        add_row(f"flow_guided_a{alpha:g}_{mode}", guided)
        print(f"scored {clip}", flush=True)

    agg = {v: aggregate(r) for v, r in rows.items()}
    decisions = {
        v: flow_decision(agg[v], agg["raw"]) for v in variants if v != "raw"
    }

    # F2 verdict: some deployable-mode guided cell passes every guard AND beats the z-descent
    # control on teacher-height stance speed (at guard-compliant z_descent it must also beat it
    # only if that control itself passes; otherwise beating raw-guarded control is trivial).
    def cell_passes(v: str) -> bool:
        return decisions[v]["all_relative_guards_pass"]

    deployable_pass = [
        v for v in guided_variants
        if any(v.endswith(m) for m in DEPLOYABLE_MODES) and v in decisions and cell_passes(v)
    ]
    control_speed = agg["z_descent"]["teacher_height_stance_speed_ms"]
    f2_pass = any(
        agg[v]["teacher_height_stance_speed_ms"] is not None
        and control_speed is not None
        and agg[v]["teacher_height_stance_speed_ms"] < control_speed
        for v in deployable_pass
    )

    result = {
        "provenance": {
            "created_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "git": git_state(ROOT),
            "runtime": runtime_state(dev),
            "checkpoint": {"path": str(pathlib.Path(args.ckpt).resolve()),
                           "sha256": sha256_file(args.ckpt)},
            "flow_checkpoint": (
                {"path": str(pathlib.Path(args.flow_ckpt).resolve()),
                 "sha256": sha256_file(args.flow_ckpt)}
                if args.flow_ckpt else None
            ),
        },
        "config": vars(args),
        "aggregate": agg,
        "decisions": decisions,
        "gates": {
            "f0_zr_viable": agg["zr_oracle"]["mpjpe_m"] <= 0.03,
            "f1_fidelity_pass": (
                agg["flow"]["mpjpe_m"] <= agg["raw"]["mpjpe_m"] + 0.005
                if "flow" in agg else None
            ),
            "f2_deployable_pass_cells": deployable_pass,
            "f2_beats_z_descent_control": f2_pass if flow is not None else None,
        },
        "per_window": {v: r for v, r in rows.items()},
    }
    outp.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {outp}")
    print(f"F0 zr_oracle MPJPE {agg['zr_oracle']['mpjpe_m']*100:.2f} cm "
          f"(viable ≤ 3 cm: {result['gates']['f0_zr_viable']})")
    if flow is not None:
        print(f"F1 flow MPJPE {agg['flow']['mpjpe_m']*100:.2f} cm vs raw "
              f"{agg['raw']['mpjpe_m']*100:.2f} cm → {result['gates']['f1_fidelity_pass']}")
        print(f"F2 deployable passing cells: {deployable_pass or 'none'}; "
              f"beats z-descent control: {f2_pass}")


if __name__ == "__main__":
    main()
