#!/usr/bin/env python
"""Phase-1 training: LAFAN1 human motion → shared latent → single robot (default G1).

Design-doc step N3. Trains `SNMR.retarget_human_to_robot` on the paired dataset produced by
`make_pairs_lafan1.py`, with a clip-level train/val split, window sampling, and periodic held-out
evaluation of the Gate-G1 metric: whole-body MPJPE between the prediction's FK and the GMR teacher's
FK on unseen clips.

    python scripts/train_phase1.py --robot unitree_g1 --steps 60000 --window 64 \
        --out runs/phase1_g1

Checkpoints (model + optimizer + step) are written every --ckpt_every steps and at the end;
`--resume` continues from the latest checkpoint in --out.
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

from snmr.data import local_root_to_world, world_root_to_local  # noqa: E402
from snmr.human import human_static_features, lafan1_skeleton, load_pair_npz  # noqa: E402
from snmr.losses import total_loss  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

# The decoder's root head predicts the robot root **in the scaled-human-root heading frame**:
#  (1) world-frame root targets are unlearnable — encoder features are heading/translation invariant
#      (first smoke run: ~3.3 m val MPJPE of pure root drift);
#  (2) anchoring at the *raw* human root still leaves an unlearnable residual, because GMR scales the
#      human trajectory before IK: robot_xy ≈ s·human_xy with s≈0.875 for LAFAN1→G1 (measured residual
#      ≤1.5 cm), so (robot − human) grows with distance from the world origin (measured: 59 cm root
#      error dominating a 61 cm val MPJPE while dof-only error was 12 cm);
#  (3) therefore the anchor is s·human_xy with per-robot constant s fitted on the training set by
#      least squares (stored in the checkpoint; a known retargeting constant at inference, no leakage).
# World pose is recovered at eval/inference by composing with the known (scaled) human trajectory.

from snmr.paths import data_root, robot_mjcf  # noqa: E402

# data-consistent models resolved centrally (snmr/paths.py); see THIRD_PARTY.md
ROBOT_MJCF_KEYS = ["unitree_g1", "booster_t1_29dof", "fourier_n1", "engineai_pm01", "stanford_toddy"]

# LAFAN1 clip prefixes; we hold out whole clips (all subjects of a held-out sequence stay together
# would be even stricter, but per-clip split matches the design doc and keeps val diverse).
VAL_CLIPS = [
    "walk1_subject5", "dance2_subject4", "fight1_subject3",
    "run2_subject1", "jumps1_subject2", "sprint1_subject4", "aiming2_subject3",
]


def load_dataset(pairs_dir: pathlib.Path, device: str):
    files = sorted(pairs_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no pair NPZs in {pairs_dir}")
    train, val = [], []
    for f in files:
        pair = load_pair_npz(str(f))
        item = {
            "name": f.stem,
            "human_pos": pair["human_pos"].to(device),
            "human_quat": pair["human_quat"].to(device),
            "qpos": pair["qpos"].to(device),
        }
        (val if f.stem in VAL_CLIPS else train).append(item)
    return train, val


def fit_xy_scale(clips: list[dict]) -> float:
    """Least-squares fit of s in robot_xy ≈ s * human_root_xy over the training clips.

    This is the trajectory scale GMR applies (human_scale_table x height ratio); for LAFAN1→G1 it is
    ~0.875 with ≤1.5 cm residual. Fitted once, stored in the checkpoint, and treated as a constant of
    the retargeting config at inference.
    """
    num = 0.0
    den = 0.0
    for clip in clips:
        hx = clip["human_pos"][:, 0, :2]
        rx = clip["qpos"][:, 0:2]
        num += float((rx * hx).sum())
        den += float((hx * hx).sum())
    return num / max(den, 1e-9)


def scaled_anchor(clip: dict, s: int, e: int, xy_scale: float) -> tuple:
    anchor_pos = clip["human_pos"][s:e, 0, :].clone()  # Hips
    anchor_pos[:, :2] *= xy_scale
    anchor_quat = clip["human_quat"][s:e, 0, :]
    return anchor_pos, anchor_quat


def make_teacher(clip: dict, s: int, e: int, xy_scale: float) -> tuple:
    """Slice a window and build the teacher dict with the root pose in the scaled-human-heading frame."""
    q = clip["qpos"][s:e]
    anchor_pos, anchor_quat = scaled_anchor(clip, s, e, xy_scale)
    local_pos, local_quat = world_root_to_local(anchor_pos, anchor_quat, q[:, 0:3], q[:, 3:7])
    teacher = {"root_pos": local_pos, "root_quat": local_quat, "dof_pos": q[:, 7:]}
    return clip["human_pos"][s:e], clip["human_quat"][s:e], teacher, (anchor_pos, anchor_quat), q


def sample_window(clip: dict, window: int, xy_scale: float):
    T = clip["qpos"].shape[0]
    s = 0 if T <= window else random.randint(0, T - window)
    e = min(s + window, T)
    hp, hq, teacher, anchor, q = make_teacher(clip, s, e, xy_scale)
    return hp, hq, teacher, anchor, q


@torch.no_grad()
def evaluate(model, rk, skel, static, clips, window: int, xy_scale: float, max_windows_per_clip: int = 4):
    """Held-out whole-body MPJPE (m) between predicted-FK and teacher-FK, plus dof/root errors."""
    model.eval()
    mpjpes, dof_errs = [], []
    for clip in clips:
        T = clip["qpos"].shape[0]
        starts = np.linspace(0, max(T - window, 0), num=min(max_windows_per_clip, max(T // window, 1)), dtype=int)
        for s in starts:
            e = s + window
            hp, hq, teacher, (anchor_pos, anchor_quat), q = make_teacher(clip, int(s), int(e), xy_scale)
            pred = model.retarget_human_to_robot(hp, hq, skel, static, rk)
            # recover the predicted world root from the local prediction, then FK both in world frame
            wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
            bp_p, _ = rk.forward_kinematics(wp, wq, pred["dof_pos"])
            bp_t, _ = rk.forward_kinematics(q[:, 0:3], q[:, 3:7], teacher["dof_pos"])
            mpjpes.append((bp_p - bp_t).norm(dim=-1).mean().item())
            dof_errs.append((pred["dof_pos"] - teacher["dof_pos"]).abs().mean().item())
    model.train()
    return float(np.mean(mpjpes)), float(np.mean(dof_errs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="unitree_g1", choices=ROBOT_MJCF_KEYS)
    ap.add_argument("--pairs_dir", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "phase1_g1"))
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--enc_hidden", type=int, default=128)
    ap.add_argument("--dec_hidden", type=int, default=128)
    ap.add_argument("--contact_weight", type=float, default=0.0,
                    help="weight of the foot-contact loss (0 = off; contact mask from teacher feet)")
    ap.add_argument("--edge_contact", action="store_true",
                    help="add the EDGE self-consistency contact head (predict_contact) + BCE "
                         "supervision on top of the teacher-mask velocity loss")
    ap.add_argument("--no_temporal", action="store_true",
                    help="ablation: disable the temporal transformer over frame latents")
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

    train_clips, val_clips = load_dataset(pairs_dir, args.device)
    print(f"train clips: {len(train_clips)}  val clips: {len(val_clips)}  device: {args.device}")

    rk = RobotKinematics(str(robot_mjcf(args.robot)), device=args.device)
    skel = lafan1_skeleton(device=args.device)
    # static features from a sample clip (bone lengths are constant across LAFAN1 subjects up to scale)
    static = human_static_features(skel, body_pos_sample=train_clips[0]["human_pos"])

    xy_scale = fit_xy_scale(train_clips)
    print(f"fitted robot/human xy trajectory scale: {xy_scale:.4f}")

    model = SNMR(
        SNMRConfig(latent_dim=args.latent_dim, enc_hidden=args.enc_hidden,
                   dec_hidden=args.dec_hidden, use_temporal=not args.no_temporal,
                   predict_contact=args.edge_contact and args.contact_weight > 0)
    ).to(args.device)

    # foot-contact loss support: teacher contact masks + foot body indices (G1 by default)
    from snmr.metrics import FOOT_BODIES, detect_contact
    foot_names = FOOT_BODIES.get(args.robot)
    foot_idx = [rk.body_index(n) for n in foot_names] if foot_names else None
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
    t0 = time.time()
    running = []
    for step in range(start_step, args.steps):
        clip = random.choice(train_clips)
        hp, hq, teacher, (anchor_pos, anchor_quat), q = sample_window(clip, args.window, xy_scale)
        opt.zero_grad()
        pred = model.retarget_human_to_robot(hp, hq, skel, static, rk)
        loss, parts = total_loss(pred, rk, teacher=teacher)
        if args.contact_weight > 0 and foot_idx is not None:
            # Contact/skate must be judged in the WORLD frame (a planted foot is stationary in world
            # coordinates, not in the moving anchor frame). Contact mask = teacher's world-frame
            # feet; prediction recomposed to world before the penalty.
            from snmr.losses import (
                contact_prediction_loss,
                contact_self_consistency_loss,
                foot_contact_loss,
            )

            with torch.no_grad():
                t_body_w, _ = rk.forward_kinematics(q[:, 0:3], q[:, 3:7], q[:, 7:])
                cmask = detect_contact(t_body_w[:, foot_idx, :], fps=30.0)
            wp, wq = local_root_to_world(anchor_pos, anchor_quat, pred["root_pos"], pred["root_quat"])
            world_pred = {"root_pos": wp, "root_quat": wq, "dof_pos": pred["dof_pos"]}
            l_contact = foot_contact_loss(world_pred, rk, foot_idx, cmask)
            if args.edge_contact and "contact_logits" in pred:
                # EDGE self-consistency on the model's OWN predicted contacts + BCE supervision of
                # the contact head vs the teacher mask. Pass the LOCAL pred + anchor; the loss
                # recomposes to world internally (matches train_phase2 usage — do NOT pre-recompose).
                l_contact = l_contact \
                    + contact_self_consistency_loss(pred, rk, foot_idx,
                                                    anchor=(anchor_pos, anchor_quat)) \
                    + contact_prediction_loss(pred["contact_logits"], foot_idx, cmask)
            loss = loss + args.contact_weight * l_contact
            parts["contact"] = float(l_contact.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(float(loss.detach()))

        if (step + 1) % args.eval_every == 0:
            mpjpe, dof_err = evaluate(model, rk, skel, static, val_clips, args.window, xy_scale)
            rec = {
                "step": step + 1,
                "train_loss": float(np.mean(running[-500:])),
                "val_mpjpe_m": mpjpe,
                "val_dof_err_rad": dof_err,
                "lr": sched.get_last_lr()[0],
                "elapsed_s": round(time.time() - t0, 1),
            }
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"step {step+1:6d}  loss {rec['train_loss']:.4f}  "
                  f"VAL mpjpe {mpjpe*100:.2f} cm  dof {dof_err:.4f} rad  "
                  f"({rec['elapsed_s']:.0f}s)")

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "step": step + 1,
                 "xy_scale": xy_scale, "config": vars(args)},
                ckpt_path,
            )

    mpjpe, dof_err = evaluate(model, rk, skel, static, val_clips, args.window, xy_scale, max_windows_per_clip=16)
    print(f"\nFINAL held-out: MPJPE {mpjpe*100:.2f} cm | dof err {dof_err:.4f} rad "
          f"| Gate G1 target < 3 cm")


if __name__ == "__main__":
    main()
