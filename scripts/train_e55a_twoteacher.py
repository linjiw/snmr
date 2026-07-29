#!/usr/bin/env python3
"""E55-A: two-teacher training — G1 specialist on LAFAN1/GMR pairs + OmniRetarget pairs.

Extends the phase-1 recipe with heterogeneous human skeletons: LAFAN1 (24 joints, real
quats) + OmniRetarget OMOMO (52 joints, synthetic quats) + OmniRetarget mocap (53 joints,
synthetic quats). The GAT encoder is skeleton-agnostic (per-node features + adjacency +
global pool), so each batch simply carries its own skeleton's adjacency/static tensors —
this run is simultaneously the data-scaling arm and the first real test of the
skeleton-agnosticism claim.

Arms (E55A_ARM):
  base      — LAFAN1 pairs only (control at matched budget; reproduces phase-1)
  twoteach  — LAFAN1 + OmniRetarget pairs (interleaved per-source batches)

Held-out eval (both arms, every eval): the 7 LAFAN1 val clips (comparability with
BENCH-v2) + 12 held-out OmniRetarget clips (6 object / 6 terrain, fixed by name below).
Success = twoteach ≤ base + 0.3 cm on LAFAN1 val (no forgetting) AND twoteach beats base
by ≥ 30% MPJPE on omni val (the new data is learnable in the shared latent).

GPU budget: model 1.5M params, batch windows chosen to stay under ~5 GB so this trains
BESIDE the E53 WBT run. Checkpoints under runs/e55a_<arm>/.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.human import (  # noqa: E402
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.model import SNMR, SNMRConfig, _adjacency  # noqa: E402
from snmr.paths import data_root, robot_mjcf  # noqa: E402
from snmr.robot_model import RobotKinematics  # noqa: E402
from snmr.skeleton import SkeletonGraph  # noqa: E402

VAL_LAFAN = ["walk1_subject5", "dance2_subject4", "fight1_subject3", "run2_subject1",
             "jumps1_subject2", "sprint1_subject4", "aiming2_subject3"]
# fixed omni holdout: 6 object subjects + 6 terrain stems (by sorted-name tail)
OMNI_VAL_TAILS = 12


def skeleton_from_parents(parents: np.ndarray, device):
    """Topologically reorder (parent-before-child) and return (skel, perm).

    perm maps NEW index -> OLD index; apply `arr[:, perm]` to reorder joint arrays."""
    n = len(parents)
    kids = {j: [] for j in range(n)}
    for j, pp in enumerate(parents):
        if pp >= 0:
            kids[int(pp)].append(j)
    order = []
    stack = [int(np.where(parents < 0)[0][0])]
    while stack:
        j = stack.pop()
        order.append(j)
        stack.extend(reversed(kids[j]))
    old_to_new = {o: i for i, o in enumerate(order)}
    new_parents = np.array(
        [old_to_new[int(parents[o])] if parents[o] >= 0 else -1 for o in order], np.int64)
    is_ee = torch.tensor([len(kids[o]) == 0 for o in order], device=device)
    skel = SkeletonGraph(
        names=[f"j{o}" for o in order],
        parent_index=torch.as_tensor(new_parents, dtype=torch.long, device=device),
        rest_offset=torch.zeros(n, 3, device=device),
        is_end_effector=is_ee,
    )
    return skel, np.array(order, np.int64)


class SourcePool:
    """One human-skeleton family: shared static features + adjacency + clip list."""

    def __init__(self, name, skel, static, adj, clips):
        self.name, self.skel, self.static, self.adj, self.clips = name, skel, static, adj, clips


def load_lafan(device):
    skel = lafan1_skeleton(device=device)
    pairs_root = data_root() / "pairs" / "unitree_g1"
    clips_tr, clips_va = [], []
    static = None
    for f in sorted(pairs_root.glob("*.npz")):
        pair = load_pair_npz(str(f))
        item = {"name": f.stem,
                "feat": None,  # lazily built below to share code path
                "human_pos": pair["human_pos"].to(device),
                "human_quat": pair["human_quat"].to(device),
                "qpos": pair["qpos"].to(device), "fps": pair["fps"]}
        (clips_va if f.stem in VAL_LAFAN else clips_tr).append(item)
    static = human_static_features(skel, body_pos_sample=clips_tr[0]["human_pos"]).to(device)
    adj = _adjacency(skel).to(device)
    return SourcePool("lafan1", skel, static, adj, clips_tr), \
           SourcePool("lafan1_val", skel, static, adj, clips_va)


def load_omni(device):
    files = sorted(glob.glob("/home/ec2-user/work/retarget/data/pairs_omni/unitree_g1/omni_*.npz"))
    by_joints = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        j = d["human_pos"].shape[1]
        by_joints.setdefault(j, []).append((f, d))
    pools_tr, pools_va = [], []
    for j, items in sorted(by_joints.items()):
        parents = np.asarray(items[0][1]["human_parents"])
        skel, perm = skeleton_from_parents(parents, device)
        sample = torch.from_numpy(
            np.asarray(items[0][1]["human_pos"], dtype=np.float32)[:, perm]).to(device)
        static = human_static_features(skel, body_pos_sample=sample).to(device)
        adj = _adjacency(skel).to(device)
        clips = []
        for f, d in items:
            clips.append({
                "name": pathlib.Path(f).stem,
                "human_pos": torch.from_numpy(np.asarray(d["human_pos"], np.float32)[:, perm]).to(device),
                "human_quat": torch.from_numpy(np.asarray(d["human_quat"], np.float32)[:, perm]).to(device),
                "qpos": torch.from_numpy(np.asarray(d["qpos"], np.float32)).to(device),
                "fps": float(d["fps"]),
            })
        val_names = {c["name"] for c in clips[-OMNI_VAL_TAILS // 2:]}
        tr = [c for c in clips if c["name"] not in val_names]
        va = [c for c in clips if c["name"] in val_names]
        pools_tr.append(SourcePool(f"omni{j}", skel, static, adj, tr))
        pools_va.append(SourcePool(f"omni{j}_val", skel, static, adj, va))
    return pools_tr, pools_va


def sample_window(pool, window, rng):
    clip = pool.clips[rng.integers(len(pool.clips))]
    T = clip["human_pos"].shape[0]
    if T <= window:
        s = 0; e = T
    else:
        s = int(rng.integers(0, T - window)); e = s + window
    feat = human_pose_features(clip["human_pos"][s:e], clip["human_quat"][s:e])
    return feat, clip["qpos"][s:e], clip["fps"]


def dof_loss(model, kin, pool, feat, qpos):
    z = model.encode(feat, pool.static, pool.adj)
    out = model.decode(z, kin)
    dof_t = qpos[:, 7:]
    l_dof = (out["dof_pos"] - dof_t).square().mean()
    # root height only (world-frame xy is heading-dependent; keep the pilot simple)
    l_rootz = (out["root_pos"][:, 2] - qpos[:, 2]).square().mean()
    return l_dof + 0.5 * l_rootz, l_dof.detach()


@torch.no_grad()
def eval_pools(model, kin, pools, window=192, per_clip=2, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for pool in pools:
        errs = []
        for clip in pool.clips:
            for _ in range(per_clip):
                T = clip["human_pos"].shape[0]
                s = 0 if T <= window else int(rng.integers(0, T - window))
                e = min(T, s + window)
                feat = human_pose_features(clip["human_pos"][s:e], clip["human_quat"][s:e])
                z = model.encode(feat, pool.static, pool.adj)
                pred = model.decode(z, kin)
                # body-space MPJPE via FK on dof (root pinned to gt for comparability)
                gt_dof = clip["qpos"][s:e, 7:]
                bp_p, _ = kin.forward_kinematics(clip["qpos"][s:e, :3], clip["qpos"][s:e, 3:7], pred["dof_pos"])
                bp_g, _ = kin.forward_kinematics(clip["qpos"][s:e, :3], clip["qpos"][s:e, 3:7], gt_dof)
                errs.append(float((bp_p - bp_g).norm(dim=-1).mean()))
        out[pool.name] = float(np.mean(errs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, help="base | twoteach (or env E55A_ARM)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--batch_windows", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import os
    arm = args.arm or os.environ.get("E55A_ARM", "twoteach")
    out = pathlib.Path(args.out or f"runs/e55a_{arm}")
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    torch.manual_seed(0)
    kin = RobotKinematics(str(robot_mjcf("unitree_g1")), device=device)
    lafan_tr, lafan_va = load_lafan(device)
    omni_tr, omni_va = load_omni(device)
    pools = [lafan_tr] + (omni_tr if arm == "twoteach" else [])
    # v3: sampling proportional to sqrt(FRAME count), not clip count. v2's sqrt-clip-count
    # gave lafan 23% of samples despite 82% of frames (omni clips are ~10x shorter),
    # producing avoidable interference (lafan-val 6-7cm vs base 3.31cm).
    frames = np.array([sum(c["human_pos"].shape[0] for c in p.clips) for p in pools], float)
    weights = np.sqrt(frames); weights /= weights.sum()
    print(f"arm={arm} pools:", [(p.name, len(p.clips)) for p in pools], "weights", weights.round(2))

    model = SNMR(SNMRConfig(latent_dim=128, enc_hidden=256, dec_hidden=256,
                            use_temporal=False)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    log = (out / "log.jsonl").open("a")

    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        total = 0.0
        for _ in range(args.batch_windows):
            pool = pools[rng.choice(len(pools), p=weights)]
            feat, qpos, fps = sample_window(pool, args.window, rng)
            loss, _ = dof_loss(model, kin, pool, feat, qpos)
            (loss / args.batch_windows).backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            model.eval()
            ev = eval_pools(model, kin, [lafan_va] + omni_va)
            model.train()
            rec = {"step": step, "loss": total / args.batch_windows,
                   **{f"val_{k}_mpjpe_m": v for k, v in ev.items()},
                   "elapsed_s": round(time.time() - t0)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"model": model.state_dict(),
                        "config": {"latent_dim": 128, "enc_hidden": 256, "dec_hidden": 256},
                        "arm": arm, "step": step}, out / "ckpt.pt")
    print(f"done: {n_par/1e6:.2f}M params, {args.steps} steps, {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
