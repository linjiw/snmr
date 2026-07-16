#!/usr/bin/env python
"""Batch-export GMR-reference WBT NPZs with attached SNMR latents (Phase-3 data prep).

For each clip this produces one NPZ whose standard WBT fields come from the GMR teacher qpos
passed through the standard converter (resample + MuJoCo replay — identical path to
prepare_wbt_validation's GMR branch) plus a time-aligned ``latent_z`` encoded from the paired
human motion by the frozen SNMR checkpoint (identical recipe to export_wbt_with_latent.py).

The output directory is directly usable as a holosoma ``motion_dir`` for multi-clip WBT training;
snmr.integration.wbt_latent loads and concatenates the per-clip latents in the loader's
sorted-glob order.

    python scripts/export_wbt_gmr_latent_batch.py --ckpt runs/phase2_all5/ckpt_100k_final.pt \
        --clips walk1_subject5 walk3_subject1 run2_subject1 --out runs/wbt_latent_gmr_multi
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

from snmr.human import (  # noqa: E402
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.model import _adjacency  # noqa: E402
from snmr.paths import data_root, g1_mjcf, holosoma_sample_npz  # noqa: E402

from export_wbt_npz import mujoco_replay, resample_qpos, validate_against_reference  # noqa: E402
from export_wbt_with_latent import resample_latent  # noqa: E402
from probe_latent_contact import load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--output_fps", type=float, default=50.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt, args.device)
    mjcf = str(g1_mjcf())
    skel = lafan1_skeleton(device=args.device)
    pairs = data_root() / "pairs" / args.robot
    reference = str(holosoma_sample_npz())

    manifest = {
        "ckpt": args.ckpt,
        "robot": args.robot,
        "output_fps": args.output_fps,
        "reference_source": "gmr_teacher_qpos",
        "clips": {},
    }
    failures = []
    for clip in args.clips:
        pair = load_pair_npz(str(pairs / f"{clip}.npz"))
        hp = pair["human_pos"].to(args.device)
        hq = pair["human_quat"].to(args.device)
        h_static = human_static_features(skel, body_pos_sample=hp)
        with torch.no_grad():
            z30 = (
                model.encode(human_pose_features(hp, hq), h_static, _adjacency(skel))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        q30 = pair["qpos"].cpu().numpy().astype(np.float64)

        q50 = resample_qpos(q30, src_fps=pair["fps"], dst_fps=args.output_fps)
        z50 = resample_latent(z30, src_fps=pair["fps"], dst_fps=args.output_fps)
        n = min(q50.shape[0], z50.shape[0])
        q50, z50 = q50[:n], z50[:n]

        npz = mujoco_replay(mjcf, q50, args.output_fps)
        npz["latent_z"] = z50.astype(np.float32)
        problems = validate_against_reference(npz, reference)
        path = out / f"{clip}_mj_z.npz"
        np.savez_compressed(path, **npz)
        manifest["clips"][clip] = {
            "frames": int(n),
            "latent_dim": int(z50.shape[1]),
            "schema_problems": problems,
        }
        if problems:
            failures.append(clip)
        print(f"wrote {path} ({n} frames, latent {z50.shape})")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        raise SystemExit(f"schema validation failed for: {failures}")
    print(f"manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
