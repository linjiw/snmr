#!/usr/bin/env python3
"""Measure how much clip identity each E70 command arm carries.

The shuffled arm (S) is a phase-matched MISALIGNED-REFERENCE control, not an
identity-erasing one: its donor map is a deterministic bijection, so clip
identity survives it.  This script quantifies that, and is the artifact behind
the 8.04 SD / 0.000 SD figures in the manuscript and in
docs/E70_SHUFFLE_CONTROL_AUDIT_2026-09-10.md.

Metric: the L2 distance between per-clip mean codes, after per-dimension
standardization with pool-wide statistics -- i.e. the same standardization the
trainer applies (scripts/train_e52_dagger.py), so the number describes what the
student actually receives.  A linearly decodable identity signal shows up as a
large separation; an identity-free code gives exactly zero.

Usage:
    python scripts/measure_arm_identity_separation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from snmr.integration.distillation import (
    same_phase_shuffled_latents,
    shared_time_index_latents,
)

DEFAULT_CLIPS = (
    Path("/data/robotixx/snmr-research/e69/motions/walk1_subject1_mj_z.npz"),
    Path("/data/robotixx/snmr-research/e67/motions/walk1_subject5_mj_z.npz"),
)


def _separation(codes: torch.Tensor, boundary: int) -> float:
    mean = codes.mean(0, keepdim=True)
    std = codes.std(0, keepdim=True) + 1e-6
    standardized = (codes - mean) / std
    first = standardized[:boundary].mean(0)
    second = standardized[boundary:].mean(0)
    return float(torch.linalg.norm(first - second))


STATISTIC = (
    "For each arm, standardize the concatenated per-frame codes of both clips per dimension "
    "(pool mean, pool std + 1e-6), take the per-clip mean of the standardized codes, and "
    "report the L2 norm of the difference of the two clip means, in pooled-SD units. "
    "A large value means clip identity is linearly decodable from the arm's command; "
    "exactly zero means the two clips receive identical code distributions in the mean."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, nargs=2, default=list(DEFAULT_CLIPS))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write a JSON artifact (input sha256, statistic, command, values) to this path",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    missing = [p for p in args.clips if not p.is_file()]
    if missing:
        print(f"missing frozen motion files: {missing}")
        return 2
    if args.out is not None and args.out.exists() and not args.overwrite:
        print(f"refusing to overwrite existing artifact: {args.out}")
        return 3

    first, second = (np.load(p)["latent_z"] for p in args.clips)
    latents = torch.tensor(np.concatenate([first, second]), dtype=torch.float32)
    boundary = len(first)
    starts = torch.tensor([0, boundary])
    ends = torch.tensor([boundary, len(latents)])

    print(f"clip 1: {args.clips[0].name}  {first.shape}")
    print(f"clip 2: {args.clips[1].name}  {second.shape}\n")

    arms = {
        "A  frozen SNMR latent": latents,
        "S  misaligned reference": same_phase_shuffled_latents(latents, starts, ends),
        "T  time code": shared_time_index_latents(
            starts, ends, output_dim=latents.shape[1]
        ),
    }
    values = {}
    for name, codes in arms.items():
        values[name.split()[0]] = _separation(codes, boundary)
        print(f"  {name:<26} between-clip separation = {values[name.split()[0]]:8.4f} SD")

    print(
        "\nS carries the same linearly decodable clip identity as A, because the donor\n"
        "map only relabels the two clusters. T is identity-free by construction: it is\n"
        "the assay's content-free null, and S is not."
    )

    if args.out is not None:
        import hashlib
        import json
        import sys

        from snmr.experiment import git_state, utc_now

        def sha256(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        artifact = {
            "created_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "git": git_state(Path(__file__).resolve().parents[1]),
            "generator": {"path": __file__, "sha256": sha256(Path(__file__))},
            "inputs": [
                {"path": str(p), "sha256": sha256(p), "frames": int(n), "latent_dim": int(d)}
                for p, (n, d) in zip(args.clips, (first.shape, second.shape))
            ],
            "statistic": STATISTIC,
            "arm_constructions": {
                "A": "latent_z as stored in the frozen motion files (frozen SNMR latent)",
                "S": "snmr.integration.distillation.same_phase_shuffled_latents: donor = (destination + 1) mod n at matched normalized time",
                "T": "snmr.integration.distillation.shared_time_index_latents: identical code at equal local frame index",
            },
            "between_clip_separation_sd": values,
            "interpretation_limits": (
                "Equal separation in A and S shows S carries as much linearly decodable clip identity "
                "as A under this statistic. It does not show identity was maximally available in any "
                "information-theoretic sense, and S's score does not bound what a directly supervised "
                "clip-identity-plus-phase controller could reach; that arm is unrun."
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
