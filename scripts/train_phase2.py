#!/usr/bin/env python
"""Phase-2 training: multi-robot shared latent (design step N4).

One SNMR model trained across multiple robots simultaneously:

  * Per step: sample a clip window; encode the human motion once -> z_h; decode z_h to K sampled
    robots and distill against each robot's GMR-teacher qpos.
  * **L_z (shared-space loss):** encode each robot's *teacher motion* (via that robot's graph +
    differentiable-FK pose features) -> z_r, and pull z_r and z_h together symmetrically. Since every
    robot's z_r is tied to the same z_h, all embodiments' encodings of the same motion converge to a
    single point in latent space — the SAME idea extended across robots.
  * **Leave-one-robot-out (LORO):** pass --holdout_robot to exclude a robot from ALL training
    (no decode, no distill, no L_z). Evaluation then decodes to it zero-shot from its MJCF-derived
    embodiment code alone. Its xy trajectory scale comes from the GMR IK config
    (human_scale_table[root] x height ratio) — a design constant, not fitted on held-out data
    (config-derived vs data-fitted scales agree to <=0.0013 across all 5 robots).

Robot roots are predicted in the scaled-human-root heading frame (Phase-1 lesson; per-robot scale).

    python scripts/train_phase2.py --steps 100000 --out runs/phase2_all5
    python scripts/train_phase2.py --steps 100000 --holdout_robot engineai_pm01 --out runs/phase2_loro_pm01
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.data import (  # noqa: E402
    RobotMotion,
    local_root_to_world,
    robot_node_static_features,
    robot_pose_features,
    world_root_to_local,
)
from snmr.human import human_pose_features, human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.losses import latent_consistency_loss, total_loss  # noqa: E402
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402
from snmr.skeleton import SkeletonGraph  # noqa: E402

REPO = ROOT.parent

ALL_ROBOTS = ["unitree_g1", "booster_t1_29dof", "fourier_n1", "engineai_pm01", "stanford_toddy"]

ROBOT_MJCF = {
    "unitree_g1": REPO / "holosoma/src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml",
    "booster_t1_29dof": REPO / "GMR/assets/booster_t1_29dof/t1_mocap.xml",
    "fourier_n1": REPO / "GMR/assets/fourier_n1/n1_mocap.xml",
    "engineai_pm01": REPO / "GMR/assets/engineai_pm01/pm_v2.xml",
    "stanford_toddy": REPO / "GMR/assets/stanford_toddy/toddy_mocap.xml",
}

# xy trajectory scale from the GMR IK config: human_scale_table[human_root] * (1.75 / height_assumption)
# (LAFAN1 loader hardcodes human height 1.75). Verified against least-squares data fits (diff <= 1.3e-3).
ROBOT_XY_SCALE_CONFIG = {
    "unitree_g1": 0.8750,
    "booster_t1_29dof": 0.5833,
    "fourier_n1": 0.7778,
    "engineai_pm01": 0.8264,
    "stanford_toddy": 0.3403,
}

VAL_CLIPS = [
    "walk1_subject5", "dance2_subject4", "fight1_subject3",
    "run2_subject1", "jumps1_subject2", "sprint1_subject4", "aiming2_subject3",
]


class RobotContext:
    """Everything static needed to decode to / encode from one robot."""

    def __init__(self, name: str, device: str):
        self.name = name
        self.kin = RobotKinematics(str(ROBOT_MJCF[name]), device=device)
        self.static = robot_node_static_features(self.kin.graph)
        self.adj = _adjacency(SkeletonGraph.from_robot_graph(self.kin.graph))
        self.xy_scale = ROBOT_XY_SCALE_CONFIG[name]


def load_clips(pairs_root: pathlib.Path, robots: list[str], device: str):
    """Load all clips; human arrays deduplicated across robots (same LAFAN1 source clip)."""
    clip_names = sorted(p.stem for p in (pairs_root / robots[0]).glob("*.npz"))
    train, val = {}, {}
    for name in clip_names:
        entry = None
        for robot in robots:
            pair = load_pair_npz(str(pairs_root / robot / f"{name}.npz"))
            if entry is None:
                entry = {
                    "human_pos": pair["human_pos"].to(device),
                    "human_quat": pair["human_quat"].to(device),
                    "qpos": {},
                }
            entry["qpos"][robot] = pair["qpos"].to(device)
        (val if name in VAL_CLIPS else train)[name] = entry
    return train, val


def window_teacher(entry: dict, robot: str, s: int, e: int, xy_scale: float):
    """Teacher dict for one robot window, root in the scaled-human-heading frame."""
    q = entry["qpos"][robot][s:e]
    anchor_pos = entry["human_pos"][s:e, 0, :].clone()
    anchor_pos[:, :2] *= xy_scale
    anchor_quat = entry["human_quat"][s:e, 0, :]
    lp, lq = world_root_to_local(anchor_pos, anchor_quat, q[:, 0:3], q[:, 3:7])
    return {"root_pos": lp, "root_quat": lq, "dof_pos": q[:, 7:]}, (anchor_pos, anchor_quat), q


def encode_robot_teacher(model: SNMR, ctx: RobotContext, q: torch.Tensor) -> torch.Tensor:
    """z of a robot's teacher motion window (input features are constants — no grad through FK)."""
    with torch.no_grad():
        motion = RobotMotion(q[:, 0:3], q[:, 3:7], q[:, 7:], fps=30.0)
        feats = robot_pose_features(ctx.kin, motion)
    return model.encode(feats, ctx.static, ctx.adj)


@torch.no_grad()
def evaluate_robot(model, ctx: RobotContext, skel, h_static, clips: dict, window: int,
                   max_windows_per_clip: int = 4) -> tuple[float, float]:
    """Held-out MPJPE (m) + dof err for one robot (world-frame FK, scaled-anchor recomposition)."""
    model.eval()
    mpjpes, dof_errs = [], []
    for entry in clips.values():
        T = entry["qpos"][ctx.name].shape[0]
        starts = np.linspace(0, max(T - window, 0),
                             num=min(max_windows_per_clip, max(T // window, 1)), dtype=int)
        for s in starts:
            e = int(s) + window
            teacher, (anchor_pos, anchor_quat), q = window_teacher(entry, ctx.name, int(s), e, ctx.xy_scale)
            feats = human_pose_features(entry["human_pos"][int(s):e], entry["human_quat"][int(s):e])
            z = model.encode(feats, h_static, _HUMAN_ADJ)
            pred = model.decoder(z, ctx.static, ctx.adj,
                                 model.embodiment_encoder(ctx.static), ctx.kin.graph)
            wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
            bp_p, _ = ctx.kin.forward_kinematics(wp, wq, pred["dof_pos"])
            bp_t, _ = ctx.kin.forward_kinematics(q[:, 0:3], q[:, 3:7], q[:, 7:])
            mpjpes.append((bp_p - bp_t).norm(dim=-1).mean().item())
            dof_errs.append((pred["dof_pos"] - q[:, 7:]).abs().mean().item())
    model.train()
    return float(np.mean(mpjpes)), float(np.mean(dof_errs))


_HUMAN_ADJ = None  # set in main() after device is known


def main() -> None:
    global _HUMAN_ADJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", nargs="+", default=ALL_ROBOTS)
    ap.add_argument("--holdout_robot", default=None,
                    help="robot excluded from ALL training; evaluated zero-shot")
    ap.add_argument("--pairs_root", default=str(REPO / "data" / "pairs"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "phase2"))
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--robots_per_step", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)  # Phase-1 lesson: 8e-4 destabilizes this size
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--latent_weight", type=float, default=1.0)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--enc_hidden", type=int, default=256)
    ap.add_argument("--dec_hidden", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--ckpt_every", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_robots = [r for r in args.robots if r != args.holdout_robot]
    eval_robots = list(args.robots)  # includes the holdout for zero-shot eval
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_clips, val_clips = load_clips(pathlib.Path(args.pairs_root), eval_robots, args.device)
    print(f"train robots: {train_robots}  holdout: {args.holdout_robot}")
    print(f"train clips: {len(train_clips)}  val clips: {len(val_clips)}  device: {args.device}")

    ctxs = {r: RobotContext(r, args.device) for r in eval_robots}
    skel = lafan1_skeleton(device=args.device)
    sample_entry = next(iter(train_clips.values()))
    h_static = human_static_features(skel, body_pos_sample=sample_entry["human_pos"])
    _HUMAN_ADJ = _adjacency(skel)

    model = SNMR(SNMRConfig(latent_dim=args.latent_dim, enc_hidden=args.enc_hidden,
                            dec_hidden=args.dec_hidden)).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.min_lr)

    start_step = 0
    ckpt_path = out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_step = state["step"]
        print(f"resumed from step {start_step}")

    log_path = out / "log.jsonl"
    clip_list = list(train_clips.values())
    t0 = time.time()
    running, running_lz = [], []

    for step in range(start_step, args.steps):
        entry = random.choice(clip_list)
        T = entry["human_pos"].shape[0]
        s = 0 if T <= args.window else random.randint(0, T - args.window)
        e = min(s + args.window, T)

        robots_k = random.sample(train_robots, min(args.robots_per_step, len(train_robots)))

        feats = human_pose_features(entry["human_pos"][s:e], entry["human_quat"][s:e])
        z_h = model.encode(feats, h_static, _HUMAN_ADJ)

        opt.zero_grad()
        loss = z_h.new_zeros(())
        lz_total = z_h.new_zeros(())
        for robot in robots_k:
            ctx = ctxs[robot]
            teacher, _, q = window_teacher(entry, robot, s, e, ctx.xy_scale)
            pred = model.decoder(z_h, ctx.static, ctx.adj,
                                 model.embodiment_encoder(ctx.static), ctx.kin.graph)
            l_robot, _ = total_loss(pred, ctx.kin, teacher=teacher)
            z_r = encode_robot_teacher(model, ctx, q)
            l_z = latent_consistency_loss(z_h, z_r)  # symmetric: grads flow into both encodings
            loss = loss + l_robot + args.latent_weight * l_z
            lz_total = lz_total + l_z
        loss = loss / len(robots_k)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(float(loss.detach()))
        running_lz.append(float(lz_total.detach()) / len(robots_k))

        if (step + 1) % args.eval_every == 0:
            rec = {
                "step": step + 1,
                "train_loss": float(np.mean(running[-500:])),
                "train_lz": float(np.mean(running_lz[-500:])),
                "lr": sched.get_last_lr()[0],
                "elapsed_s": round(time.time() - t0, 1),
            }
            msgs = []
            for robot in eval_robots:
                mpjpe, dof_err = evaluate_robot(model, ctxs[robot], skel, h_static, val_clips, args.window)
                tag = "ZS" if robot == args.holdout_robot else "  "
                rec[f"val_mpjpe_{robot}"] = mpjpe
                rec[f"val_dof_{robot}"] = dof_err
                msgs.append(f"{robot.split('_')[-1]}{tag}:{mpjpe*100:.1f}cm")
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"step {rec['step']:6d} loss {rec['train_loss']:.4f} Lz {rec['train_lz']:.4f} | "
                  + " ".join(msgs) + f" | {rec['elapsed_s']:.0f}s")

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "step": step + 1, "xy_scales": ROBOT_XY_SCALE_CONFIG, "config": vars(args)},
                ckpt_path,
            )

    print("\n=== FINAL held-out eval (denser windows) ===")
    final = {}
    for robot in eval_robots:
        mpjpe, dof_err = evaluate_robot(model, ctxs[robot], skel, h_static, val_clips,
                                        args.window, max_windows_per_clip=16)
        tag = " [ZERO-SHOT]" if robot == args.holdout_robot else ""
        final[robot] = {"mpjpe_m": mpjpe, "dof_err_rad": dof_err}
        print(f"{robot:18s}{tag}: MPJPE {mpjpe*100:.2f} cm | dof err {dof_err:.4f} rad")
    with open(out / "final_eval.json", "w") as fh:
        json.dump(final, fh, indent=2)


if __name__ == "__main__":
    main()
