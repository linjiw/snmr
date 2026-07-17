#!/usr/bin/env python
"""Train the latent rectified-flow head on a FROZEN SNMR checkpoint (side project F1).

See docs/FLOW_RETARGETING_SIDE_PROJECT.md. The flow learns the conditional transport
p(z_target | z_h) in the frozen 128-d latent space: condition = frozen encoder's human latent,
target = frozen encoder's teacher-robot latent (``--target z_r``, default) or the human latent
itself (``--target z_h``, the F0 fallback). SNMR weights never receive gradients.

    python scripts/train_latent_flow.py --ckpt runs/phase1_g1_large/ckpt_100k_final.pt \
        --steps 20000 --out runs/latent_flow/f1_zr
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

from snmr.data import RobotMotion, robot_pose_features  # noqa: E402
from snmr.experiment import (  # noqa: E402
    RunManifest,
    capture_rng_state,
    dataset_fingerprint,
    restore_rng_state,
    sha256_file,
)
from snmr.flow import LatentFlowConfig, LatentFlowNet, rectified_flow_loss, sample_flow  # noqa: E402
from snmr.human import human_pose_features, human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root, robot_mjcf  # noqa: E402

from benchmark import load_model  # noqa: E402
from train_phase1 import VAL_CLIPS, split_pair_paths  # noqa: E402
from train_phase2 import RobotContext  # noqa: E402


def load_clips(pairs_dir: pathlib.Path, device: str) -> tuple[list[dict], list[dict]]:
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
    if not train or not val:
        raise SystemExit(f"missing clips under {pairs_dir}")
    return train, val


@torch.no_grad()
def encode_window(model, ctx: RobotContext, h_static, h_adj, clip: dict, s: int, e: int, target: str):
    """Frozen-encoder condition/target latents for one window: (z_h, z1) each (T, latent)."""
    z_h = model.encode(
        human_pose_features(clip["human_pos"][s:e], clip["human_quat"][s:e]), h_static, h_adj
    )
    if target == "z_h":
        return z_h, z_h
    q = clip["qpos"][s:e]
    motion = RobotMotion(q[:, 0:3], q[:, 3:7], q[:, 7:], fps=clip["fps"])
    feats = robot_pose_features(ctx.kin, motion)
    z_r = model.encode(feats, ctx.static, ctx.adj)
    return z_h, z_r


@torch.no_grad()
def sample_batch(model, ctx, h_static, h_adj, clips, batch: int, window: int, target: str):
    conds, targets = [], []
    for _ in range(batch):
        clip = random.choice(clips)
        T = clip["qpos"].shape[0]
        s = 0 if T <= window else random.randint(0, T - window)
        e = min(s + window, T)
        if e - s < window:  # skip short clips' ragged tails; all LAFAN1 clips exceed 64 frames
            s, e = 0, window
        z_h, z1 = encode_window(model, ctx, h_static, h_adj, clip, s, e, target)
        conds.append(z_h)
        targets.append(z1)
    return torch.stack(conds), torch.stack(targets)


@torch.no_grad()
def evaluate_fidelity(
    flow, model, ctx, h_static, h_adj, clips, window: int, target: str,
    num_steps: int, windows_per_clip: int = 4, samples: int = 1, seed: int = 0,
) -> dict:
    """Latent-space MSE of flow samples vs the encoder target, plus decoded MPJPE vs teacher.

    MPJPE follows the trainer convention: world-frame FK of the decoded sample against the
    teacher qpos FK on the same window, decoded via the scaled-human anchor.
    """
    from snmr.data import local_root_to_world

    flow.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z_mses, mpjpes, raw_mpjpes = [], [], []
    for clip in clips:
        T = clip["qpos"].shape[0]
        starts = np.linspace(
            0, max(T - window, 0), num=min(windows_per_clip, max(T // window, 1)), dtype=int
        )
        for s in starts:
            s = int(s)
            e = s + window
            z_h, z1 = encode_window(model, ctx, h_static, h_adj, clip, s, e, target)
            anchor_pos = clip["human_pos"][s:e, 0, :].clone()
            anchor_pos[:, :2] *= ctx.xy_scale
            anchor_quat = clip["human_quat"][s:e, 0, :]
            q = clip["qpos"][s:e]
            teacher_body, _ = ctx.kin.forward_kinematics(q[:, 0:3], q[:, 3:7], q[:, 7:])

            def decode_mpjpe(z: torch.Tensor) -> float:
                pred = model.decoder(
                    z, ctx.static, ctx.adj, model.embodiment_encoder(ctx.static), ctx.kin.graph
                )
                wp, wq = local_root_to_world(
                    anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"]
                )
                body, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
                return float((body - teacher_body).norm(dim=-1).mean())

            raw_mpjpes.append(decode_mpjpe(z_h))
            window_mpjpes = []
            for _ in range(samples):
                z0 = torch.randn(
                    (1,) + z_h.shape, dtype=z_h.dtype, generator=gen
                ).to(z_h.device)
                z_hat = sample_flow(flow, z_h.unsqueeze(0), num_steps=num_steps, z0=z0)[0]
                z_mses.append(float((z_hat - z1).square().mean()))
                window_mpjpes.append(decode_mpjpe(z_hat))
            mpjpes.append(float(np.median(window_mpjpes)))
    flow.train()
    return {
        "flow_latent_mse": float(np.mean(z_mses)),
        "flow_mpjpe_m": float(np.mean(mpjpes)),
        "raw_mpjpe_m": float(np.mean(raw_mpjpes)),
        "num_windows": len(mpjpes),
    }


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/phase1_g1_large/ckpt_100k_final.pt"))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--pairs_dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs/latent_flow/f1_zr"))
    ap.add_argument("--target", choices=["z_r", "z_h"], default="z_r",
                    help="flow target latent: teacher-robot encoding (default) or F0-fallback z_h")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--sample_steps", type=int, default=25, help="Euler NFE for eval sampling")
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

    model, state = load_model(args.ckpt, args.device)  # frozen: load_model puts it in eval()
    for p in model.parameters():
        p.requires_grad_(False)
    latent_dim = model.cfg.latent_dim

    train_clips, val_clips = load_clips(pairs_dir, args.device)
    ctx = RobotContext(args.robot, args.device)
    if "xy_scale" in state:
        ctx.xy_scale = float(state["xy_scale"])
    skel = lafan1_skeleton(device=args.device)
    h_static = human_static_features(skel, body_pos_sample=train_clips[0]["human_pos"])
    h_adj = _adjacency(skel)
    print(f"frozen SNMR latent_dim={latent_dim}  train clips {len(train_clips)}  "
          f"val clips {len(val_clips)}  target={args.target}  device={args.device}")

    flow = LatentFlowNet(LatentFlowConfig(
        latent_dim=latent_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers
    )).to(args.device)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.min_lr)

    start_step = 0
    ckpt_path = out / "flow_ckpt.pt"
    if args.resume and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        flow.load_state_dict(st["flow"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_step = st["step"]
        if "rng_state" in st:
            restore_rng_state(st["rng_state"])
        print(f"resumed from step {start_step}")
    flow.train()

    split_paths = split_pair_paths(pairs_dir)
    split_paths["robot_assets"] = [robot_mjcf(args.robot)]
    manifest = RunManifest.start(
        out / "manifest.json",
        trainer="scripts/train_latent_flow.py",
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
            "frozen_snmr_checkpoint": {"path": str(pathlib.Path(args.ckpt).resolve()),
                                       "sha256": sha256_file(args.ckpt)},
            "flow_parameters": sum(p.numel() for p in flow.parameters()),
        },
        objectives={
            "rectified_flow": {
                "weight": 1.0,
                "semantics": f"CondOT conditional flow matching, target={args.target}",
            },
        },
        resume=args.resume,
    )

    log_path = out / "log.jsonl"
    t0 = time.time()
    running = deque(maxlen=500)
    for step in range(start_step, args.steps):
        cond, z1 = sample_batch(
            model, ctx, h_static, h_adj, train_clips, args.batch, args.window, args.target
        )
        opt.zero_grad()
        loss = rectified_flow_loss(flow, z1, cond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(float(loss.detach()))

        if (step + 1) % args.eval_every == 0:
            rec = evaluate_fidelity(
                flow, model, ctx, h_static, h_adj, val_clips, args.window, args.target,
                num_steps=args.sample_steps,
            )
            rec.update({
                "step": step + 1,
                "train_loss": float(np.mean(running)),
                "lr": sched.get_last_lr()[0],
                "elapsed_s": round(time.time() - t0, 1),
            })
            append_jsonl(log_path, rec)
            print(f"step {step+1:6d}  cfm {rec['train_loss']:.4f}  "
                  f"VAL flow-mpjpe {rec['flow_mpjpe_m']*100:.2f} cm "
                  f"(raw {rec['raw_mpjpe_m']*100:.2f} cm)  "
                  f"z-mse {rec['flow_latent_mse']:.4f}  ({rec['elapsed_s']:.0f}s)", flush=True)

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save(
                {"flow": flow.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "step": step + 1, "config": vars(args),
                 "flow_config": {"latent_dim": latent_dim, "hidden_dim": args.hidden_dim,
                                 "num_layers": args.num_layers},
                 "snmr_ckpt_sha256": sha256_file(args.ckpt),
                 "rng_state": capture_rng_state()},
                ckpt_path,
            )
            manifest.update_progress(
                step=step + 1,
                robot_exposures={args.robot: step + 1},
                checkpoint_path=ckpt_path,
            )

    final = evaluate_fidelity(
        flow, model, ctx, h_static, h_adj, val_clips, args.window, args.target,
        num_steps=args.sample_steps, windows_per_clip=6, samples=3,
    )
    (out / "final_eval.json").write_text(json.dumps(final, indent=2))
    print(f"\nFINAL held-out: flow MPJPE {final['flow_mpjpe_m']*100:.2f} cm | "
          f"raw {final['raw_mpjpe_m']*100:.2f} cm | "
          f"F1 gate: flow ≤ raw + 0.5 cm → "
          f"{'PASS' if final['flow_mpjpe_m'] <= final['raw_mpjpe_m'] + 0.005 else 'FAIL'}")
    manifest.update_progress(
        step=args.steps,
        robot_exposures={args.robot: args.steps},
        checkpoint_path=ckpt_path,
        complete=True,
    )


if __name__ == "__main__":
    main()
