#!/usr/bin/env python3
"""E55-A data prep: convert OmniRetarget clips into SNMR pair NPZs (second teacher).

OmniRetarget clips carry `qpos (T, 36|43)` (G1, QUAT-FIRST base: [qw qx qy qz x y z | 29])
and `human_joints (T, J, 3)` — paired human keypoints, POSITIONS ONLY, J varies by source
(52 for OMOMO robot-object, 53 for in-house mocap). Our pair format wants
[x y z | qw qx qy qz | 29] qpos plus human_pos/human_quat for the encoder.

Skeleton handling (the whole point of a skeleton-agnostic encoder):
  * Topology is INFERRED per source: pairwise joint distances with ~zero temporal std are
    rigid bones; a minimum-spanning tree over distance-std gives parents. Verified: bone-
    length std < 1 mm on real clips.
  * Root = joint 0 (pelvis convention holds in both subsets).
  * Orientation features: the encoder wants per-node quats, but only positions exist.
    We synthesize a per-joint frame from the parent->child bone direction (z-axis) and a
    heading-consistent lateral axis — deterministic, differentiable-free, and honest about
    its provenance (stored flag `human_quat_synthetic=True`). The GAT input projection can
    learn to discount it; E55-A's ablation arm drops rot features for this source entirely.

Output: <out>/<clip>.npz with human_pos (T,J,3), human_quat (T,J,4 wxyz synthetic),
human_parents (J,), qpos (T,36) pos-first, fps, robot='unitree_g1',
source='omniretarget/<subset>', human_height (estimated from skeleton).
Object trajectory columns (7 extra in robot-object) are stored as object_pose when present.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np


def infer_parents(human: np.ndarray) -> np.ndarray:
    """MST over temporal-std of pairwise distances -> parent array (root=0, parent=-1)."""
    T, J, _ = human.shape
    sample = human[:: max(1, T // 120)]
    D = np.linalg.norm(sample[:, :, None, :] - sample[:, None, :, :], axis=-1)
    cost = D.std(0) + 1e-6
    INF = 1e9
    used = np.zeros(J, bool)
    used[0] = True
    dist = cost[0].copy()
    parent = np.zeros(J, np.int64)
    parents = np.full(J, -1, np.int64)
    for _ in range(J - 1):
        j = int(np.where(used, INF, dist).argmin())
        parents[j] = parent[j]
        used[j] = True
        upd = (cost[j] < dist) & ~used
        parent = np.where(upd, j, parent)
        dist = np.where(used, dist, np.minimum(dist, cost[j]))
    return parents


def bone_length_std(human: np.ndarray, parents: np.ndarray) -> float:
    lens = []
    for j in range(1, human.shape[1]):
        p = parents[j]
        if p >= 0:
            lens.append(np.linalg.norm(human[:, j] - human[:, p], axis=-1).std())
    return float(np.mean(lens))


def synthetic_quats(human: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Per-joint wxyz quat from bone direction (z) + world-up-orthogonalized x."""
    T, J, _ = human.shape
    z = np.zeros((T, J, 3))
    for j in range(J):
        kids = np.where(parents == j)[0]
        if len(kids):
            z[:, j] = human[:, kids[0]] - human[:, j]
        elif parents[j] >= 0:
            z[:, j] = human[:, j] - human[:, parents[j]]
        else:
            z[:, j, 2] = 1.0
    z /= np.linalg.norm(z, axis=-1, keepdims=True).clip(1e-6)
    up = np.zeros_like(z)
    up[..., 2] = 1.0
    x = np.cross(up, z)
    deg = np.linalg.norm(x, axis=-1, keepdims=True)
    x = np.where(deg > 1e-3, x / deg.clip(1e-6), np.broadcast_to([1.0, 0, 0], x.shape))
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-1)  # (T,J,3,3) columns
    # rotation matrix -> wxyz quat
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    qw = np.sqrt(np.clip(1 + tr, 1e-9, None)) / 2
    qx = (m[..., 2, 1] - m[..., 1, 2]) / (4 * qw)
    qy = (m[..., 0, 2] - m[..., 2, 0]) / (4 * qw)
    qz = (m[..., 1, 0] - m[..., 0, 1]) / (4 * qw)
    q = np.stack([qw, qx, qy, qz], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni_root", default="/home/ec2-user/work/retarget/data/omniretarget")
    ap.add_argument("--subsets", nargs="+",
                    default=["robot-terrain", "robot-object", "robot-object-terrain"])
    ap.add_argument("--originals_only", action="store_true",
                    help="skip augmentation variants (rot_*/z_scale != 1.0)")
    ap.add_argument("--out", default="/home/ec2-user/work/retarget/data/pairs_omni/unitree_g1")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for subset in args.subsets:
        files = sorted(glob.glob(f"{args.omni_root}/{subset}/{subset}/*.npz"))
        if args.originals_only:
            files = [f for f in files
                     if f.endswith("_original.npz") or "_z_scale_1.0" in f]
        for f in files:
            d = np.load(f, allow_pickle=True)
            if "human_joints" not in d.files:
                continue  # some clips ship robot-only (no paired human); unusable as pairs
            qpos_raw = np.asarray(d["qpos"], dtype=np.float64)
            human = np.asarray(d["human_joints"], dtype=np.float64)
            fps = float(d["fps"])
            # quat-first -> pos-first (robot base); keep 29 dof; object cols -> aside
            base_quat, base_pos = qpos_raw[:, 0:4], qpos_raw[:, 4:7]
            dof = qpos_raw[:, 7:36]
            qpos = np.concatenate([base_pos, base_quat, dof], axis=1)
            obj = qpos_raw[:, 36:43] if qpos_raw.shape[1] >= 43 else None

            parents = infer_parents(human)
            bls = bone_length_std(human, parents)
            if bls > 0.005:
                print(f"SKIP {f}: bone-length std {bls:.4f} m (non-rigid skeleton?)")
                continue
            quats = synthetic_quats(human, parents)
            height = float(np.percentile(human[..., 2].max(axis=1)
                                         - human[..., 2].min(axis=1), 95))
            name = pathlib.Path(f).stem
            payload = dict(
                human_pos=human.astype(np.float32),
                human_quat=quats.astype(np.float32),
                human_parents=parents,
                human_quat_synthetic=np.array(True),
                qpos=qpos.astype(np.float32),
                fps=np.array(fps),
                robot=np.array("unitree_g1"),
                source=np.array(f"omniretarget/{subset}"),
                human_height=np.array(height),
            )
            if obj is not None:
                payload["object_pose"] = obj.astype(np.float32)
            np.savez_compressed(out / f"omni_{subset}_{name}.npz", **payload)
            manifest[f"omni_{subset}_{name}"] = {
                "frames": int(qpos.shape[0]), "joints": int(human.shape[1]),
                "bone_std_m": round(bls, 5), "height_m": round(height, 3),
                "has_object": obj is not None,
            }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    n = len(manifest)
    frames = sum(v["frames"] for v in manifest.values())
    print(f"converted {n} clips, {frames:,} frames -> {out}")


if __name__ == "__main__":
    main()
