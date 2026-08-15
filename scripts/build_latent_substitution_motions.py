#!/usr/bin/env python3
"""Build E72 latent-substitution motion trees (source intervention on ``latent_z``).

Mechanism
---------
``snmr/integration/wbt_latent.py`` loads ``latent_z`` straight out of the WBT motion NPZ
(``_load_latent_npz``), and ``scripts/train_e52_dagger.py`` in eval-only mode restores
``z_mean``/``z_std`` from the student CHECKPOINT rather than recomputing them from the loaded
motions.  A latent intervention can therefore be delivered *at the source*: write a NEW motion
directory whose NPZs are identical to the frozen originals in every field except ``latent_z``
(identical physics, identical reference target, identical clip names and file names), and point
the frozen eval-only path at that directory.  No frozen file is edited.

Arms (see ``docs/E72_LATENT_SUBSTITUTION_PROTOCOL.md``)
------------------------------------------------------
``control``       ``latent_z`` copied through unchanged (byte-identical) -- mandatory control.
``shift_m0250``   ``latent_z`` shifted within-clip by -12 frames (-0.24 s at 50 Hz).
``shift_p0250``   ``latent_z`` shifted within-clip by +12 frames (+0.24 s at 50 Hz).
``shift_p0500``   ``latent_z`` shifted within-clip by +25 frames (+0.50 s at 50 Hz).
``first_frame``   ``latent_z[0]`` of the clip broadcast across the whole clip (static code).
``clip_mean``     ``mean(latent_z)`` over the clip broadcast across the whole clip (static code).

Frame offsets are derived from the NPZ's own ``fps`` field (asserted equal across clips), not
assumed.  The magnitude is truncated, never rounded up, so a labelled 0.25 s offset at 50 Hz
becomes 12 frames (0.24 s) rather than 13 -- the intervention is never larger than its label.

Boundary safety
---------------
Every clip lives in its own NPZ, so a clip is exactly one file's frame range ``[0, T)``.  Shifts
are computed as ``out[t] = z[clip(t + delta, 0, T - 1)]``: never across a clip boundary, and
clamped (held) at both edges rather than wrapped.  This matters because the frozen
``wbt_latent._gather_at_offsets`` clamps only the UPPER index; a negative index would silently
wrap at gather time.  By construction the arrays written here contain no out-of-range read.

CPU only.  This script reads the frozen motions read-only and writes to a NEW data root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
from typing import Any, Callable, Iterable

import numpy as np

LATENT_KEY = "latent_z"
FPS_KEY = "fps"
EXPECTED_FPS = 50

#: Frozen, read-only inputs.  SHA-256 is verified before the file is opened.
DEFAULT_SOURCES: tuple[tuple[str, str], ...] = (
    (
        "/data/robotixx/snmr-research/e70/motions/walk1_subject1_mj_z.npz",
        "b78f294395a5c74f37edc1c09dd6de0909d966a9b7fd948291c8dd803f7106aa",
    ),
    (
        "/data/robotixx/snmr-research/e70/motions/walk1_subject5_mj_z.npz",
        "d8de93425c14e90dce2930450d722d3eb2b6fcbb09e9c4ff3d59725025424f51",
    ),
)

DEFAULT_OUT_ROOT = "/data/robotixx/snmr-research/e72_latent_sub"

#: Paths that must never be written to (frozen experiment inputs).
FORBIDDEN_WRITE_ROOTS: tuple[str, ...] = ("/data/robotixx/snmr-research/e70",)

#: Arm specification.  ``seconds`` is the *nominal* label; the realized integer frame offset is
#: derived from the motion's own fps by magnitude truncation.
ARM_SPECS: dict[str, dict[str, Any]] = {
    "control": {"kind": "identity"},
    "shift_m0250": {"kind": "shift", "seconds": -0.25},
    "shift_p0250": {"kind": "shift", "seconds": 0.25},
    "shift_p0500": {"kind": "shift", "seconds": 0.50},
    "first_frame": {"kind": "first_frame"},
    "clip_mean": {"kind": "clip_mean"},
}


# --------------------------------------------------------------------------------------
# pure helpers (no I/O; unit-tested on synthetic arrays)
# --------------------------------------------------------------------------------------
def offset_frames(seconds: float, fps: float) -> int:
    """Integer frame offset for a nominal second offset, truncated toward zero.

    Truncation (not rounding) guarantees the realized misalignment never exceeds its label:
    at 50 Hz, 0.25 s -> 12 frames (0.24 s), 0.50 s -> 25 frames (0.50 s).
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    magnitude = int(np.floor(abs(float(seconds)) * float(fps) + 1e-9))
    return -magnitude if seconds < 0 else magnitude


def validate_latent(latent: np.ndarray) -> np.ndarray:
    """Assert the loaded latent matches what the frozen loader requires."""
    if latent.ndim != 2:
        raise ValueError(f"latent_z must have shape (T, d), got {latent.shape}")
    if latent.dtype != np.float32:
        raise ValueError(f"latent_z must be float32, got {latent.dtype}")
    if latent.shape[0] < 1:
        raise ValueError("latent_z has zero frames")
    if not np.isfinite(latent).all():
        raise ValueError("latent_z contains nonfinite values")
    return latent


def shifted_latent(latent: np.ndarray, delta: int) -> np.ndarray:
    """``out[t] = latent[clip(t + delta, 0, T - 1)]`` -- within-clip, edge-clamped, no wrap."""
    validate_latent(latent)
    frames = latent.shape[0]
    if abs(int(delta)) >= frames:
        raise ValueError(f"shift {delta} is not smaller than the clip length {frames}")
    index = np.clip(np.arange(frames, dtype=np.int64) + int(delta), 0, frames - 1)
    return np.ascontiguousarray(latent[index], dtype=np.float32)


def first_frame_latent(latent: np.ndarray) -> np.ndarray:
    """The clip's first latent frame broadcast across the clip (constant in time)."""
    validate_latent(latent)
    out = np.empty_like(latent)
    out[:] = latent[0]
    return np.ascontiguousarray(out, dtype=np.float32)


def clip_mean_latent(latent: np.ndarray) -> np.ndarray:
    """The clip-mean latent broadcast across the clip (constant in time).

    The mean is accumulated in float64 and cast once, so the constant does not depend on the
    summation order of a float32 accumulator.
    """
    validate_latent(latent)
    mean = latent.mean(axis=0, dtype=np.float64).astype(np.float32)
    out = np.empty_like(latent)
    out[:] = mean
    return np.ascontiguousarray(out, dtype=np.float32)


def substitute_latent(latent: np.ndarray, spec: dict[str, Any], fps: float) -> np.ndarray:
    """Apply one arm's substitution to a single clip's ``latent_z``."""
    validate_latent(latent)
    kind = spec["kind"]
    if kind == "identity":
        return np.ascontiguousarray(latent.copy(), dtype=np.float32)
    if kind == "shift":
        return shifted_latent(latent, offset_frames(spec["seconds"], fps))
    if kind == "first_frame":
        return first_frame_latent(latent)
    if kind == "clip_mean":
        return clip_mean_latent(latent)
    raise ValueError(f"unknown arm kind {kind!r}")


def arm_frame_offset(spec: dict[str, Any], fps: float) -> int | None:
    """Realized integer frame offset for a shift arm; ``None`` for non-shift arms."""
    if spec["kind"] != "shift":
        return None
    return offset_frames(spec["seconds"], fps)


def substituted_arrays(
    arrays: dict[str, np.ndarray], spec: dict[str, Any], fps: float
) -> dict[str, np.ndarray]:
    """Copy every array through unchanged except ``latent_z``, which is substituted."""
    if LATENT_KEY not in arrays:
        raise ValueError(f"motion has no {LATENT_KEY} field")
    out: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key == LATENT_KEY:
            out[key] = substitute_latent(validate_latent(value), spec, fps)
        else:
            out[key] = value  # copied through verbatim (same dtype, shape, bytes)
    return out


def latent_l2(original: np.ndarray, substituted: np.ndarray) -> float:
    """Frobenius distance ``||z_sub - z_orig||_2`` over the whole (T, d) clip, in float64."""
    diff = substituted.astype(np.float64) - original.astype(np.float64)
    return float(np.sqrt((diff * diff).sum()))


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_motion(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def motion_fps(arrays: dict[str, np.ndarray]) -> float:
    if FPS_KEY not in arrays:
        raise ValueError(f"motion has no {FPS_KEY} field; the frame rate cannot be derived")
    value = np.asarray(arrays[FPS_KEY]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"{FPS_KEY} must hold a single value, got shape {value.shape}")
    fps = float(value[0])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid fps {fps}")
    return fps


def assert_writable_root(out_root: pathlib.Path) -> None:
    resolved = out_root.resolve()
    for forbidden in FORBIDDEN_WRITE_ROOTS:
        forbidden_path = pathlib.Path(forbidden).resolve()
        if resolved == forbidden_path or forbidden_path in resolved.parents:
            raise ValueError(f"refusing to write inside frozen root {forbidden}: {resolved}")


def write_motion(arrays: dict[str, np.ndarray], destination: pathlib.Path) -> None:
    """Atomic compressed write (matches the sources' deflate-compressed NPZ layout)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)


def build(
    sources: Iterable[tuple[pathlib.Path, str | None]],
    out_root: pathlib.Path,
    arms: dict[str, dict[str, Any]] = ARM_SPECS,
    expected_fps: float | None = EXPECTED_FPS,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Verify the inputs, then write one motion directory per arm.  Returns a manifest dict."""
    assert_writable_root(out_root)

    loaded: list[dict[str, Any]] = []
    for path, expected_sha in sources:
        path = pathlib.Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_file(path)
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )
        arrays = load_motion(path)
        fps = motion_fps(arrays)
        validate_latent(arrays[LATENT_KEY])
        loaded.append(
            {
                "path": path,
                "sha256": actual_sha,
                "arrays": arrays,
                "fps": fps,
                "frames": int(arrays[LATENT_KEY].shape[0]),
            }
        )
        log(
            f"source {path.name}: sha256 ok, fps={fps:g}, frames={loaded[-1]['frames']}, "
            f"latent_z {arrays[LATENT_KEY].shape} {arrays[LATENT_KEY].dtype}"
        )

    if not loaded:
        raise ValueError("no source motions given")
    rates = {entry["fps"] for entry in loaded}
    if len(rates) != 1:
        raise ValueError(f"source motions disagree on fps: {sorted(rates)}")
    fps = loaded[0]["fps"]
    if expected_fps is not None and fps != float(expected_fps):
        raise ValueError(
            f"motion fps is {fps}, expected {expected_fps}; frame offsets would be "
            "mislabelled (pass --expected-fps to override deliberately)"
        )

    manifest: dict[str, Any] = {
        "protocol": "E72 latent-substitution motion generation v1",
        "fps": fps,
        "out_root": str(out_root.resolve()),
        "sources": [
            {"path": str(e["path"]), "sha256": e["sha256"], "frames": e["frames"]}
            for e in loaded
        ],
        "arms": {},
    }

    for arm, spec in arms.items():
        delta = arm_frame_offset(spec, fps)
        arm_dir = out_root / "motions" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        clips: list[dict[str, Any]] = []
        for entry in loaded:
            arrays = entry["arrays"]
            original = arrays[LATENT_KEY]
            new_arrays = substituted_arrays(arrays, spec, fps)
            new_latent = new_arrays[LATENT_KEY]
            if new_latent.shape != original.shape or new_latent.dtype != original.dtype:
                raise AssertionError("substitution changed latent_z shape or dtype")
            destination = arm_dir / entry["path"].name  # original filename preserved exactly
            write_motion(new_arrays, destination)
            clips.append(
                {
                    "clip_file": destination.name,
                    "path": str(destination),
                    "source_path": str(entry["path"]),
                    "frames": int(original.shape[0]),
                    "latent_l2": latent_l2(original, new_latent),
                    "latent_max_abs_diff": float(
                        np.abs(new_latent.astype(np.float64) - original.astype(np.float64)).max()
                    ),
                    "frames_changed": int(
                        (new_latent != original).any(axis=1).sum()
                    ),
                    "latent_sha256": sha256_array(new_latent),
                    "source_latent_sha256": sha256_array(original),
                    "file_sha256": sha256_file(destination),
                }
            )
            log(
                f"  [{arm}] {destination.name}: L2={clips[-1]['latent_l2']:.6f} "
                f"frames_changed={clips[-1]['frames_changed']}/{clips[-1]['frames']}"
            )
        aggregate = float(np.sqrt(sum(c["latent_l2"] ** 2 for c in clips)))
        manifest["arms"][arm] = {
            "kind": spec["kind"],
            "nominal_seconds": spec.get("seconds"),
            "frame_offset": delta,
            "realized_seconds": None if delta is None else delta / fps,
            "motion_dir": str(arm_dir),
            "latent_l2_all_clips": aggregate,
            "clips": clips,
        }
        log(f"arm {arm}: frame_offset={delta} aggregate L2={aggregate:.6f}")

    manifest_path = out_root / "latent_substitution_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    temporary.replace(manifest_path)
    log(f"wrote {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="PATH[:SHA256] source motion NPZ (repeatable). Defaults to the frozen E70 pair.",
    )
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        choices=sorted(ARM_SPECS),
        help="restrict to these arms (default: all)",
    )
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=float(EXPECTED_FPS),
        help="fail unless the motions declare this frame rate",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove out_root/motions before writing",
    )
    args = parser.parse_args(argv)

    if args.source:
        sources: list[tuple[pathlib.Path, str | None]] = []
        for item in args.source:
            path, _, sha = item.partition(":")
            sources.append((pathlib.Path(path), sha or None))
    else:
        sources = [(pathlib.Path(p), s) for p, s in DEFAULT_SOURCES]

    arms = (
        {name: ARM_SPECS[name] for name in args.arm} if args.arm else dict(ARM_SPECS)
    )
    out_root = pathlib.Path(args.out_root)
    if args.clean:
        assert_writable_root(out_root)
        if (out_root / "motions").exists():
            shutil.rmtree(out_root / "motions")
    build(sources, out_root, arms=arms, expected_fps=args.expected_fps)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
