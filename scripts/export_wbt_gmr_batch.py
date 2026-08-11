#!/usr/bin/env python
"""Export GMR teacher pairs to Holosoma WBT NPZs without requiring an SNMR checkpoint."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from snmr.human import load_pair_npz  # noqa: E402
from snmr.paths import data_root, g1_mjcf, holosoma_sample_npz  # noqa: E402

from export_wbt_npz import (  # noqa: E402
    mujoco_replay,
    resample_qpos,
    validate_against_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--output_fps", type=float, default=50.0)
    args = parser.parse_args()

    output = pathlib.Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    pair_root = data_root() / "pairs" / args.robot
    mjcf = str(g1_mjcf())
    reference = str(holosoma_sample_npz())
    manifest = {
        "robot": args.robot,
        "output_fps": args.output_fps,
        "reference_source": "gmr_teacher_qpos",
        "clips": {},
    }
    failed = []
    for clip in args.clips:
        pair_path = pair_root / f"{clip}.npz"
        pair = load_pair_npz(str(pair_path))
        qpos = pair["qpos"].cpu().numpy().astype(np.float64)
        qpos = resample_qpos(qpos, src_fps=pair["fps"], dst_fps=args.output_fps)
        arrays = mujoco_replay(mjcf, qpos, args.output_fps)
        problems = validate_against_reference(arrays, reference)
        path = output / f"{clip}_mj.npz"
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
        manifest["clips"][clip] = {
            "source": str(pair_path.resolve()),
            "frames": int(qpos.shape[0]),
            "schema_problems": problems,
        }
        if problems:
            failed.append(clip)
        print(f"wrote {path} ({qpos.shape[0]} frames)")
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if failed:
        raise SystemExit(f"schema validation failed for {failed}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
