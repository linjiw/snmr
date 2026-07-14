#!/usr/bin/env python
"""Attach a precomputed SNMR latent to an unchanged WBT reference NPZ.

This is the controlled GMR+SNMR integration boundary: all standard WBT fields come byte-for-byte
from the GMR reference, while only ``latent_z`` is copied from an SNMR latent export with the same
timeline.

Example:
    python scripts/attach_latent_to_wbt.py \
        --reference runs/wbt_validation/gmr/walk1_subject5_mj.npz \
        --latent_source runs/wbt_latent/walk1_z.npz \
        --out runs/wbt_latent_gmr/walk1_subject5_mj_z.npz
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def build_augmented_reference(
    reference_path: str | pathlib.Path,
    latent_source_path: str | pathlib.Path,
) -> dict[str, np.ndarray]:
    """Return the reference fields plus a validated, time-aligned ``latent_z``."""
    with np.load(reference_path, allow_pickle=True) as reference:
        out = {key: np.asarray(reference[key]) for key in reference.files}
    with np.load(latent_source_path, allow_pickle=True) as latent_source:
        if "latent_z" not in latent_source.files:
            raise ValueError(f"{latent_source_path} has no latent_z field")
        latent = np.asarray(latent_source["latent_z"], dtype=np.float32)
        latent_fps = float(np.asarray(latent_source["fps"]).reshape(-1)[0])

    reference_fps = float(np.asarray(out["fps"]).reshape(-1)[0])
    if not np.isclose(reference_fps, latent_fps):
        raise ValueError(
            f"fps mismatch: reference={reference_fps}, latent={latent_fps}"
        )
    if latent.ndim != 2:
        raise ValueError(f"latent_z must have shape (T,d), got {latent.shape}")
    if latent.shape[0] != out["joint_pos"].shape[0]:
        raise ValueError(
            f"frame mismatch: reference={out['joint_pos'].shape[0]}, "
            f"latent={latent.shape[0]}"
        )
    if not np.isfinite(latent).all():
        raise ValueError("latent_z contains nonfinite values")
    out["latent_z"] = latent
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--latent_source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    augmented = build_augmented_reference(args.reference, args.latent_source)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **augmented)
    print(
        f"wrote {out_path} with unchanged reference motion and "
        f"latent_z {augmented['latent_z'].shape}"
    )


if __name__ == "__main__":
    main()
