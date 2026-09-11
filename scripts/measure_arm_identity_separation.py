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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, nargs=2, default=list(DEFAULT_CLIPS))
    args = parser.parse_args()

    missing = [p for p in args.clips if not p.is_file()]
    if missing:
        print(f"missing frozen motion files: {missing}")
        return 2

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
    for name, codes in arms.items():
        print(f"  {name:<26} between-clip separation = {_separation(codes, boundary):8.4f} SD")

    print(
        "\nS carries the same linearly decodable clip identity as A, because the donor\n"
        "map only relabels the two clusters. T is identity-free by construction: it is\n"
        "the assay's content-free null, and S is not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
