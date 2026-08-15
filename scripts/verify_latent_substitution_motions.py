#!/usr/bin/env python3
"""Standalone verification of an E72 latent-substitution motion tree.

Independent of the generator's own bookkeeping: it re-reads the frozen sources and every
produced NPZ from disk and asserts, per arm and per clip, that

  * the file name (and therefore the loader's clip name) is preserved exactly;
  * every non-``latent_z`` array is bit-identical to the source (dtype, shape, raw bytes);
  * ``latent_z`` keeps dtype float32 and shape (T, d);
  * ``latent_z`` is finite;

and prints the per-arm L2 (Frobenius) distance between the substituted and the original
latent.  The ``control`` arm must be exactly 0.0 or the run fails.

CPU only; reads the frozen motions read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import numpy as np

from scripts.build_latent_substitution_motions import (
    DEFAULT_OUT_ROOT,
    DEFAULT_SOURCES,
    LATENT_KEY,
    latent_l2,
    sha256_file,
)

NON_LATENT_REQUIRED = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_names",
    "body_names",
)


def _load(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verify(out_root: pathlib.Path, sources: list[tuple[pathlib.Path, str | None]]) -> int:
    originals: dict[str, dict[str, np.ndarray]] = {}
    for path, expected_sha in sources:
        actual = sha256_file(path)
        if expected_sha is not None and actual != expected_sha:
            raise SystemExit(f"source hash mismatch for {path}: {actual}")
        originals[path.name] = _load(path)
        print(f"source {path.name}: sha256 {actual}")

    motions_root = out_root / "motions"
    arm_dirs = sorted(p for p in motions_root.iterdir() if p.is_dir())
    if not arm_dirs:
        raise SystemExit(f"no arm directories under {motions_root}")

    failures: list[str] = []
    print()
    print(f"{'arm':<14} {'clip':<26} {'L2(z_sub, z_orig)':>18} {'frames_changed':>15}")
    for arm_dir in arm_dirs:
        arm = arm_dir.name
        produced_names = sorted(p.name for p in arm_dir.glob("*.npz"))
        if produced_names != sorted(originals):
            failures.append(f"{arm}: clip files {produced_names} != {sorted(originals)}")
            continue
        squares = 0.0
        for name in produced_names:
            source = originals[name]
            produced = _load(arm_dir / name)
            if set(produced) != set(source):
                failures.append(f"{arm}/{name}: field set differs")
                continue
            for key in NON_LATENT_REQUIRED:
                same = (
                    produced[key].dtype == source[key].dtype
                    and produced[key].shape == source[key].shape
                    and np.ascontiguousarray(produced[key]).tobytes()
                    == np.ascontiguousarray(source[key]).tobytes()
                )
                if not same:
                    failures.append(f"{arm}/{name}: non-latent array {key} differs")
            latent = produced[LATENT_KEY]
            original = source[LATENT_KEY]
            if latent.dtype != np.float32 or latent.shape != original.shape:
                failures.append(
                    f"{arm}/{name}: latent_z is {latent.dtype}{latent.shape}, "
                    f"expected float32{original.shape}"
                )
                continue
            if not np.isfinite(latent).all():
                failures.append(f"{arm}/{name}: latent_z has nonfinite values")
            distance = latent_l2(original, latent)
            squares += distance * distance
            changed = int((latent != original).any(axis=1).sum())
            print(f"{arm:<14} {name:<26} {distance:>18.6f} {changed:>15d}")
            if arm == "control" and (distance != 0.0 or changed != 0):
                failures.append(f"control arm is not identical for {name}")
            if arm == "control":
                same_bytes = latent.tobytes() == original.tobytes()
                if not same_bytes:
                    failures.append(f"control arm latent_z is not byte-identical for {name}")
        print(f"{arm:<14} {'ALL CLIPS':<26} {np.sqrt(squares):>18.6f}")
        digest = hashlib.sha256()
        for name in produced_names:
            digest.update(bytes.fromhex(sha256_file(arm_dir / name)))
        print(f"{'':<14} {'tree digest':<26} {digest.hexdigest()}")

    print()
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"OK: {len(arm_dirs)} arms verified; every non-latent array bit-identical to source")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--source", action="append", default=None, help="PATH[:SHA256]")
    args = parser.parse_args(argv)
    if args.source:
        sources = []
        for item in args.source:
            path, _, sha = item.partition(":")
            sources.append((pathlib.Path(path), sha or None))
    else:
        sources = [(pathlib.Path(p), s) for p, s in DEFAULT_SOURCES]
    return verify(pathlib.Path(args.out_root), sources)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
