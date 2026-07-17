#!/usr/bin/env python
"""V0: noise-regularize the frozen-encoder SNMR decoder (flow v2 protocol, gate V0).

See docs/FLOW_RETARGETING_V2_PROTOCOL.md §2. Hypothesis under test: the regression latent's
decoder is brittle exactly where latent-space correction pushes (off the clean-z manifold), and
a decoder finetuned under Gaussian latent augmentation makes latent descent deployably
effective (V0b gate) without losing clean fidelity (V0a guard).

Training: encoder FROZEN; decoder + embodiment encoder finetuned with the baseline Phase-1
objective (distill + limits + smooth). On a random 50% of steps the decoded latent is
`z' = z + sigma * eps * std(z)` with eps ~ N(0, I) and std(z) the per-dim latent std estimated
once from training windows (stored in the checkpoint); the distill target is unchanged, so the
decoder learns to map a noise ball around each clean latent to the same teacher answer.

    python scripts/train_v0_noise_decoder.py --sigma 0.1 --out runs/latent_flow/v0_sigma0.1

V1 extension (gate V1, --zr_decode_prob > 0): with that probability the decoded latent is the
frozen encoder's TEACHER-ROBOT encoding z_r instead of z_h (the Phase-2-deferred
`zr_decode_prob` experiment) — training the decoder to answer from robot encodings, which v1's
F0 showed it cannot do (50 cm). Noise augmentation applies to whichever latent is decoded.

    python scripts/train_v0_noise_decoder.py --sigma 0.1 --zr_decode_prob 0.5 \
        --out runs/latent_flow/v1_zr_decode
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from collections import deque

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.data import RobotMotion, local_root_to_world, robot_pose_features, world_root_to_local  # noqa: E402
from snmr.experiment import (  # noqa: E402
    RunManifest,
    capture_rng_state,
    dataset_fingerprint,
    restore_rng_state,
    sha256_file,
)
from snmr.human import human_pose_features, human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.losses import collect_loss_terms, weighted_loss, DEFAULT_LOSS_WEIGHTS  # noqa: E402
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root, robot_mjcf  # noqa: E402

from benchmark import load_model  # noqa: E402
from train_phase1 import VAL_CLIPS, split_pair_paths  # noqa: E402
from train_phase2 import RobotContext  # noqa: E402


def load_clips(pairs_dir: pathlib.Path, device: str):
    train, val = [], []
    for f in sorted(pairs_dir.glob("*.npz")):
        pair = load_pair_npz(str(f))
        item = {
            "name": f.stem,
            "human_pos": pair["human_pos"].to(device),
            "human_quat": pair["human_quat"].to(device),
            "qpos": pair["qpos"].to(device),
            "fps": pair["fps"],
        }
        (val if f.stem in VAL_CLIPS else train).append(item)
    return train, val


def window_slices(clip: dict, window: int, xy_scale: float, s: int | None = None):
    T = clip["qpos"].shape[0]
    if s is None:
        s = 0 if T <= window else random.randint(0, T - window)
    e = min(s + window, T)
    hp, hq, q = clip["human_pos"][s:e], clip["human_quat"][s:e], clip["qpos"][s:e]
    anchor_pos = hp[:, 0, :].clone()
    anchor_pos[:, :2] *= xy_scale
    anchor_quat = hq[:, 0, :]
    lp, lq = world_root_to_local(anchor_pos, anchor_quat, q[:, 0:3], q[:, 3:7])
    teacher = {"root_pos": lp, "root_quat": lq, "dof_pos": q[:, 7:]}
    return hp, hq, q, teacher, (anchor_pos, anchor_quat)


@torch.no_grad()
def estimate_latent_std(model, clips, h_static, h_adj, window: int, num_windows: int = 64):
    """Per-dim std of the frozen encoder's latent over random training windows."""
    zs = []
    for _ in range(num_windows):
        clip = random.choice(clips)
        hp, hq, *_ = window_slices(clip, window, 1.0)
        zs.append(model.encode(human_pose_features(hp, hq), h_static, h_adj))
    return torch.cat(zs, dim=0).std(dim=0)  # (latent_dim,)


@torch.no_grad()
def evaluate_clean(model, ctx, h_static, h_adj, clips, window: int, xy_scale: float,
                   windows_per_clip: int = 4) -> float:
    """Clean-z whole-body MPJPE vs teacher FK (the V0a guard metric)."""
    model.eval()
    mpjpes = []
    for clip in clips:
        T = clip["qpos"].shape[0]
        starts = np.linspace(0, max(T - window, 0),
                             num=min(windows_per_clip, max(T // window, 1)), dtype=int)
        for s in starts:
            hp, hq, q, teacher, (ap, aq) = window_slices(clip, window, xy_scale, int(s))
            z = model.encode(human_pose_features(hp, hq), h_static, h_adj)
            pred = model.decoder(
                z, ctx.static, ctx.adj, model.embodiment_encoder(ctx.static), ctx.kin.graph
            )
            wp, wq = local_root_to_world(ap, aq, pred["root_pos"], pred["root_quat"])
            bp, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
            bt, _ = ctx.kin.forward_kinematics(q[:, 0:3], q[:, 3:7], q[:, 7:])
            mpjpes.append(float((bp - bt).norm(dim=-1).mean()))
    return float(np.mean(mpjpes))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--pairs_dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs/latent_flow/v0_sigma0.1"))
    ap.add_argument("--sigma", type=float, default=0.1,
                    help="latent noise scale, in units of per-dim latent std")
    ap.add_argument("--noise_prob", type=float, default=0.5,
                    help="fraction of steps trained on noisy z (protocol-frozen at 0.5)")
    ap.add_argument("--zr_decode_prob", type=float, default=0.0,
                    help="V1: probability of decoding from the teacher-robot encoding z_r")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_dir = pathlib.Path(args.pairs_dir) if args.pairs_dir else data_root() / "pairs" / args.robot

    model, state = load_model(args.ckpt, args.device)
    base_config = dict(state.get("config", {}))
    xy_scale = float(state["xy_scale"])
    # freeze the encoder; finetune decoder + embodiment encoder
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    trainable = [p for m in (model.decoder, model.embodiment_encoder)
                 for p in m.parameters()]
    for p in trainable:
        p.requires_grad_(True)
    model.encoder.eval()
    model.decoder.train()
    model.embodiment_encoder.train()

    train_clips, val_clips = load_clips(pairs_dir, args.device)
    ctx = RobotContext(args.robot, args.device)
    ctx.xy_scale = xy_scale
    skel = lafan1_skeleton(device=args.device)
    h_static = human_static_features(skel, body_pos_sample=train_clips[0]["human_pos"])
    h_adj = _adjacency(skel)

    latent_std = estimate_latent_std(model, train_clips, h_static, h_adj, args.window)
    baseline_mpjpe = evaluate_clean(model, ctx, h_static, h_adj, val_clips,
                                    args.window, xy_scale)
    model.decoder.train()
    model.embodiment_encoder.train()
    print(f"frozen-encoder baseline clean MPJPE {baseline_mpjpe*100:.2f} cm | "
          f"latent std mean {float(latent_std.mean()):.3f} | sigma {args.sigma} "
          f"noise_prob {args.noise_prob}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.min_lr)

    start_step = 0
    ckpt_path = out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_step = st["step"]
        if "rng_state" in st:
            restore_rng_state(st["rng_state"])
        print(f"resumed from step {start_step}")

    split_paths = split_pair_paths(pairs_dir)
    split_paths["robot_assets"] = [robot_mjcf(args.robot)]
    manifest = RunManifest.start(
        out / "manifest.json",
        trainer="scripts/train_v0_noise_decoder.py",
        repo_root=ROOT,
        argv=[sys.executable, *sys.argv],
        config=vars(args),
        dataset=dataset_fingerprint(split_paths, root=pairs_dir),
        training={
            "seed": args.seed,
            "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": 1e-4},
            "lr_schedule": {"name": "CosineAnnealingLR", "t_max_steps": args.steps,
                            "eta_min": args.min_lr},
            "planned_optimizer_steps": args.steps,
            "starting_step": start_step,
            "initialization": {"path": str(pathlib.Path(args.ckpt).resolve()),
                               "sha256": sha256_file(args.ckpt)},
            "frozen_modules": ["encoder"],
            "baseline_clean_mpjpe_m": baseline_mpjpe,
            "latent_std_mean": float(latent_std.mean()),
        },
        objectives={
            "distill": {"weight": 1.0, "semantics": "unchanged Phase-1 configuration-space distill"},
            "limits": {"weight": 0.1, "semantics": "squared joint-limit excess"},
            "smooth": {"weight": 0.01, "semantics": "acceleration + jerk"},
            "latent_noise_augmentation": {
                "sigma_rel_std": args.sigma,
                "noise_prob": args.noise_prob,
                "semantics": "V0: decode z + sigma*eps*std(z) on a random fraction of steps",
            },
            "zr_decode": {
                "prob": args.zr_decode_prob,
                "active": args.zr_decode_prob > 0,
                "semantics": "V1: decode from the frozen teacher-robot encoding z_r",
            },
        },
        resume=args.resume,
    )

    log_path = out / "log.jsonl"
    t0 = time.time()
    running = deque(maxlen=500)
    for step in range(start_step, args.steps):
        clip = random.choice(train_clips)
        hp, hq, q, teacher, (ap_, aq_) = window_slices(clip, args.window, xy_scale)
        opt.zero_grad()
        with torch.no_grad():
            if args.zr_decode_prob > 0 and random.random() < args.zr_decode_prob:
                motion = RobotMotion(q[:, 0:3], q[:, 3:7], q[:, 7:], fps=clip["fps"])
                z = model.encode(
                    robot_pose_features(ctx.kin, motion), ctx.static, ctx.adj
                )
            else:
                z = model.encode(human_pose_features(hp, hq), h_static, h_adj)
        if random.random() < args.noise_prob:
            z = z + args.sigma * torch.randn_like(z) * latent_std
        pred = model.decoder(
            z, ctx.static, ctx.adj, model.embodiment_encoder(ctx.static), ctx.kin.graph
        )
        terms = collect_loss_terms(pred, ctx.kin, teacher=teacher)
        loss, _ = weighted_loss(terms, {k: DEFAULT_LOSS_WEIGHTS.get(k, 1.0) for k in terms})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        running.append(float(loss.detach()))

        if (step + 1) % args.eval_every == 0:
            mpjpe = evaluate_clean(model, ctx, h_static, h_adj, val_clips,
                                   args.window, xy_scale)
            model.decoder.train()
            model.embodiment_encoder.train()
            rec = {"step": step + 1, "train_loss": float(np.mean(running)),
                   "val_clean_mpjpe_m": mpjpe, "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"step {step+1:6d}  loss {rec['train_loss']:.4f}  "
                  f"VAL clean mpjpe {mpjpe*100:.2f} cm (baseline {baseline_mpjpe*100:.2f}, "
                  f"guard +0.2) ({rec['elapsed_s']:.0f}s)", flush=True)

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            # save in the load_model-compatible layout (full SNMR state dict + config)
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "step": step + 1,
                 "xy_scale": xy_scale,
                 "config": {**base_config, **vars(args)},
                 "latent_std": latent_std,
                 "v0": {"sigma": args.sigma, "noise_prob": args.noise_prob,
                        "baseline_clean_mpjpe_m": baseline_mpjpe},
                 "rng_state": capture_rng_state()},
                ckpt_path,
            )
            manifest.update_progress(
                step=step + 1,
                robot_exposures={args.robot: step + 1},
                checkpoint_path=ckpt_path,
            )

    final_mpjpe = evaluate_clean(model, ctx, h_static, h_adj, val_clips,
                                 args.window, xy_scale, windows_per_clip=6)
    v0a_pass = final_mpjpe <= baseline_mpjpe + 0.002
    (out / "final_eval.json").write_text(json.dumps({
        "clean_mpjpe_m": final_mpjpe,
        "baseline_clean_mpjpe_m": baseline_mpjpe,
        "v0a_fidelity_guard_pass": v0a_pass,
    }, indent=2))
    print(f"\nFINAL clean MPJPE {final_mpjpe*100:.2f} cm vs baseline "
          f"{baseline_mpjpe*100:.2f} cm | V0a guard (≤ +0.2 cm): "
          f"{'PASS' if v0a_pass else 'FAIL'}")
    manifest.update_progress(
        step=args.steps,
        robot_exposures={args.robot: args.steps},
        checkpoint_path=ckpt_path,
        complete=True,
    )


if __name__ == "__main__":
    main()
