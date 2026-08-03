#!/usr/bin/env python3
"""E55-R: two-teacher redo per review blocker B4 — full root supervision, clean splits.

Fixes over E55-A (which produced the paper's *preliminary* Table II):
  1. FULL ROOT: the model is supervised on the robot root pose expressed in the HUMAN
     root heading frame (rel_pos = R_h^-1 (p_robot - s*p_human_xy), rel_rot = R_h^-1
     R_robot, with s a per-pool displacement-ratio anchor scale). Per-frame z encodes
     heading-local pose only, so heading-local root targets are learnable where
     world-frame targets are not (phase-1 lesson). Eval reconstructs the world root
     from the human trajectory and reports UNPINNED world-frame MPJPE.
  2. GROUP-HELD-OUT SPLITS: omni val pools are split by sibling STEM — an entire
     sibling group (identical human motion, all z_scale/rot/trans variants) lands in
     train or val, never both. No shared-human leakage across the split (B4's core
     complaint about the tail-based split).
  3. SPECIALIST BASELINES: arms = lafan (locomotion specialist), omni (interaction
     specialist), twoteach (main). The honest Table II contrast is
     twoteach-vs-specialist on each pool, all with full root.

Pre-specified readouts (per val pool): world-frame MPJPE with predicted root
(primary), GT-root-pinned MPJPE (comparability column vs old Table II), rel-root
position/rotation errors. No promote/kill gate — this is a measurement redo; the
paper's Table II is replaced by whatever these numbers are.

Runs in .venv-snmr beside the GPU tracking job (same footprint as E55-A).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_e55a_twoteacher import load_lafan, load_omni  # noqa: E402
from snmr import rotation as rot  # noqa: E402
from snmr.human import human_pose_features  # noqa: E402
from snmr.model import SNMR, SNMRConfig  # noqa: E402
from snmr.paths import robot_mjcf  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402

STEM_RE = re.compile(r"(_z_scale_[0-9.]+|_original|_rot_\d+|_trans_\d+)")


def stem(name: str) -> str:
    return STEM_RE.sub("", name)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw-only quaternion (wxyz) from a full quaternion."""
    fwd = rot.quat_rotate(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand_as(q[..., 1:]))
    yaw = torch.atan2(fwd[..., 1], fwd[..., 0])
    zero = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], -1)


def pool_xy_scale(clips) -> float:
    """Displacement-ratio anchor scale between human root xy and robot root xy."""
    num, den = 0.0, 0.0
    for c in clips[:200]:
        h = c["human_pos"][:, 0, :2]
        r = c["qpos"][:, :2]
        dh = (h[1:] - h[:-1]).norm(dim=-1)
        dr = (r[1:] - r[:-1]).norm(dim=-1)
        num += float(dr.sum()); den += float(dh.sum())
    return num / max(den, 1e-6)


def root_targets(clip, s, e, xy_scale, device):
    """(T,3) rel_pos + (T,6) rel rot6d of robot root in the human heading frame."""
    hp = clip["human_pos"][s:e, 0, :]
    hq = clip["human_quat"][s:e, 0, :]
    rp = clip["qpos"][s:e, :3]
    rq = clip["qpos"][s:e, 3:7]
    head = heading_quat(hq)
    inv = rot.quat_conjugate(head)
    anchor = torch.cat([hp[:, :2] * xy_scale, torch.zeros_like(hp[:, :1])], -1)
    rel_pos = rot.quat_rotate(inv, rp - anchor)
    rel_rot6 = rot.quat_to_rot6d(rot.quat_mul(inv, rq))
    return rel_pos, rel_rot6, head, anchor


def split_pools(omni_tr, omni_va, rng, val_frac=0.12):
    """Re-split omni train+val by sibling stem (group-held-out). Returns (train, val)."""
    out_tr, out_va = [], []
    for tr, va in zip(omni_tr, omni_va):
        clips = tr.clips + va.clips
        by = {}
        for c in clips:
            by.setdefault(stem(c["name"]), []).append(c)
        stems = sorted(by)
        rng.shuffle(stems)
        n_val = max(2, int(len(stems) * val_frac))
        val_stems = set(stems[:n_val])
        tr.clips = [c for st in stems[n_val:] for c in by[st]]
        va.clips = [c for st in val_stems for c in by[st]]
        out_tr.append(tr); out_va.append(va)
        print(f"{tr.name}: {len(stems)} stems -> {len(tr.clips)} train / {len(va.clips)} val clips "
              f"({n_val} held-out groups, no shared human across split)")
    return out_tr, out_va


@torch.no_grad()
def eval_pools(model, kin, pools, scales, window=192, per_clip=2, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for pool in pools:
        if not pool.clips:
            continue
        m_free, m_pin, e_rp, e_rr = [], [], [], []
        for clip in pool.clips:
            for _ in range(per_clip):
                T = clip["human_pos"].shape[0]
                s = 0 if T <= window else int(rng.integers(0, T - window))
                e = min(T, s + window)
                feat = human_pose_features(clip["human_pos"][s:e], clip["human_quat"][s:e])
                z = model.encode(feat, pool.static, pool.adj)
                pred = model.decode(z, kin)
                gt_dof = clip["qpos"][s:e, 7:]
                rel_p, rel_r6, head, anchor = root_targets(clip, s, e, scales[pool.name], z.device)
                # reconstruct world root from predicted rel + human trajectory
                wp = anchor + rot.quat_rotate(head, pred["root_pos"])
                wq = rot.quat_mul(head, pred["root_quat"])
                bp_p, _ = kin.forward_kinematics(wp, wq, pred["dof_pos"])
                bp_g, _ = kin.forward_kinematics(clip["qpos"][s:e, :3], clip["qpos"][s:e, 3:7], gt_dof)
                m_free.append(float((bp_p - bp_g).norm(dim=-1).mean()))
                # GT-root-pinned (old Table II instrument, for comparability)
                bp_pp, _ = kin.forward_kinematics(clip["qpos"][s:e, :3], clip["qpos"][s:e, 3:7], pred["dof_pos"])
                m_pin.append(float((bp_pp - bp_g).norm(dim=-1).mean()))
                e_rp.append(float((pred["root_pos"] - rel_p).norm(dim=-1).mean()))
                e_rr.append(float((rot.quat_to_rot6d(pred["root_quat"]) - rel_r6).square().mean()))
        out[pool.name] = {"mpjpe_free_root": float(np.mean(m_free)),
                          "mpjpe_gt_root": float(np.mean(m_pin)),
                          "root_pos_err": float(np.mean(e_rp)),
                          "root_rot6d_mse": float(np.mean(e_rr))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["lafan", "omni", "twoteach"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch_windows", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = pathlib.Path(f"runs/e55r_{args.arm}")
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    kin = RobotKinematics(str(robot_mjcf("unitree_g1")), device=device)
    lafan_tr, lafan_va = load_lafan(device)
    omni_tr, omni_va = load_omni(device)
    omni_tr, omni_va = split_pools(omni_tr, omni_va, rng)

    train_pools = {"lafan": [lafan_tr], "omni": omni_tr,
                   "twoteach": [lafan_tr] + omni_tr}[args.arm]
    val_pools = [lafan_va] + omni_va  # every arm evaluated on every pool
    scales = {}
    for p in ([lafan_tr, lafan_va] + omni_tr + omni_va):
        base = p.name.replace("_val", "")
        if base not in scales:
            src = p if p.clips else None
            scales[base] = pool_xy_scale((src or p).clips) if (src or p).clips else 1.0
        scales[p.name] = scales[base]
    print("anchor xy scales:", {k: round(v, 3) for k, v in scales.items() if "_val" not in k})

    frames = np.array([sum(c["human_pos"].shape[0] for c in p.clips) for p in train_pools], float)
    weights = np.sqrt(frames); weights /= weights.sum()
    print(f"arm={args.arm} pools:", [(p.name, len(p.clips)) for p in train_pools],
          "weights", weights.round(3), flush=True)

    model = SNMR(SNMRConfig(latent_dim=128, enc_hidden=256, dec_hidden=256,
                            use_temporal=False)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = (out / "log.jsonl").open("a")

    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        total = 0.0
        for _ in range(args.batch_windows):
            pool = train_pools[rng.choice(len(train_pools), p=weights)]
            clip = pool.clips[rng.integers(len(pool.clips))]
            T = clip["human_pos"].shape[0]
            s = 0 if T <= args.window else int(rng.integers(0, T - args.window))
            e = min(T, s + args.window)
            feat = human_pose_features(clip["human_pos"][s:e], clip["human_quat"][s:e])
            z = model.encode(feat, pool.static, pool.adj)
            pred = model.decode(z, kin)
            qpos = clip["qpos"][s:e]
            rel_p, rel_r6, _, _ = root_targets(clip, s, e, scales[pool.name], device)
            loss = (pred["dof_pos"] - qpos[:, 7:]).square().mean() \
                 + 0.5 * (pred["root_pos"] - rel_p).square().mean() \
                 + 0.5 * (rot.quat_to_rot6d(pred["root_quat"]) - rel_r6).square().mean()
            (loss / args.batch_windows).backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                ev = eval_pools(model, kin, val_pools, scales)
            model.train()
            rec = {"step": step, "loss": total / args.batch_windows,
                   "eval": ev, "elapsed_s": round(time.time() - t0)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"model": model.state_dict(), "arm": args.arm,
                        "scales": scales}, out / "ckpt.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
