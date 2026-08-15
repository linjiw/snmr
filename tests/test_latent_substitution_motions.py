"""CPU tests for the E72 latent-substitution motion generator (no GPU, no holosoma)."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from scripts.build_latent_substitution_motions import (
    ARM_SPECS,
    LATENT_KEY,
    build,
    clip_mean_latent,
    first_frame_latent,
    offset_frames,
    shifted_latent,
    substitute_latent,
    substituted_arrays,
)

FPS = 50
FRAMES = 40
LATENT_DIM = 4
NON_LATENT_KEYS = (
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


def _synthetic_motion(seed: int) -> dict[str, np.ndarray]:
    """A miniature NPZ with the same field set/dtypes as the real WBT motions."""
    rng = np.random.default_rng(seed)
    latent = np.arange(FRAMES * LATENT_DIM, dtype=np.float32).reshape(FRAMES, LATENT_DIM)
    latent = latent + float(seed)  # make each clip's latent distinguishable
    return {
        "fps": np.array([FPS], dtype=np.int64),
        "joint_pos": rng.normal(size=(FRAMES, 6)).astype(np.float64),
        "joint_vel": rng.normal(size=(FRAMES, 5)).astype(np.float64),
        "body_pos_w": rng.normal(size=(FRAMES, 3, 3)).astype(np.float64),
        "body_quat_w": rng.normal(size=(FRAMES, 3, 4)).astype(np.float64),
        "body_lin_vel_w": rng.normal(size=(FRAMES, 3, 3)).astype(np.float64),
        "body_ang_vel_w": rng.normal(size=(FRAMES, 3, 3)).astype(np.float64),
        "joint_names": np.array([f"joint_{i}" for i in range(6)], dtype="<U26"),
        "body_names": np.array([f"body_{i}" for i in range(3)], dtype="<U31"),
        LATENT_KEY: np.ascontiguousarray(latent),
    }


def _write_sources(tmp_path: pathlib.Path) -> list[tuple[pathlib.Path, None]]:
    sources = []
    for index, name in enumerate(("walk1_subject1_mj_z.npz", "walk1_subject5_mj_z.npz")):
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **_synthetic_motion(index + 1))
        sources.append((path, None))
    return sources


def _load(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture()
def built(tmp_path):
    sources = _write_sources(tmp_path)
    out_root = tmp_path / "out"
    manifest = build(sources, out_root, log=lambda _msg: None)
    return {
        "sources": {p.name: _load(p) for p, _ in sources},
        "out_root": out_root,
        "manifest": manifest,
    }


# ---------------------------------------------------------------- pure transforms


def test_offset_frames_truncates_magnitude_and_uses_the_motion_rate():
    assert offset_frames(0.25, 50) == 12  # 12.5 frames truncated, never rounded up
    assert offset_frames(-0.25, 50) == -12
    assert offset_frames(0.50, 50) == 25
    assert offset_frames(0.25, 30) == 7  # rate comes from the motion, not a constant
    assert offset_frames(0.0, 50) == 0
    with pytest.raises(ValueError):
        offset_frames(0.25, 0)


def test_shift_maps_frames_and_clamps_at_both_clip_edges():
    latent = np.arange(FRAMES * 2, dtype=np.float32).reshape(FRAMES, 2)

    forward = shifted_latent(latent, 12)
    assert np.array_equal(forward[: FRAMES - 12], latent[12:])
    # upper edge: the last 12 frames hold the clip's final latent (no wrap to frame 0)
    for t in range(FRAMES - 12, FRAMES):
        assert np.array_equal(forward[t], latent[FRAMES - 1])

    backward = shifted_latent(latent, -12)
    assert np.array_equal(backward[12:], latent[: FRAMES - 12])
    # lower edge: the first 12 frames hold the clip's first latent (no negative index)
    for t in range(12):
        assert np.array_equal(backward[t], latent[0])

    assert np.array_equal(shifted_latent(latent, 0), latent)
    with pytest.raises(ValueError):
        shifted_latent(latent, FRAMES)


def test_shift_never_reads_outside_the_clip():
    """Every output row must be one of this clip's own rows -- no wrap, no cross-clip read."""
    latent = (np.arange(FRAMES, dtype=np.float32)[:, None] + 1.0) * np.ones((1, 3), np.float32)
    for delta in (-25, -12, 12, 25):
        out = shifted_latent(latent, delta)
        assert set(np.unique(out).tolist()) <= set(np.unique(latent).tolist())
        expected = np.clip(np.arange(FRAMES) + delta, 0, FRAMES - 1)
        assert np.array_equal(out[:, 0], latent[expected, 0])


def test_static_arms_are_constant_in_time_and_equal_the_right_statistic():
    latent = np.random.default_rng(0).normal(size=(FRAMES, LATENT_DIM)).astype(np.float32)

    first = first_frame_latent(latent)
    assert np.array_equal(first, np.tile(latent[0], (FRAMES, 1)))
    assert (first == first[0]).all()
    assert np.ptp(first, axis=0).max() == 0.0

    mean = clip_mean_latent(latent)
    assert (mean == mean[0]).all()
    assert np.array_equal(mean[0], latent.mean(axis=0, dtype=np.float64).astype(np.float32))


def test_shapes_and_dtypes_are_preserved_by_every_arm():
    latent = np.random.default_rng(1).normal(size=(FRAMES, LATENT_DIM)).astype(np.float32)
    for spec in ARM_SPECS.values():
        out = substitute_latent(latent, spec, FPS)
        assert out.dtype == np.float32
        assert out.shape == (FRAMES, LATENT_DIM)
        assert out.ndim == 2
        assert np.isfinite(out).all()


def test_substitute_rejects_bad_latents():
    with pytest.raises(ValueError):
        substitute_latent(np.zeros((5,), np.float32), ARM_SPECS["control"], FPS)
    with pytest.raises(ValueError):
        substitute_latent(np.zeros((5, 2), np.float64), ARM_SPECS["control"], FPS)
    bad = np.zeros((5, 2), np.float32)
    bad[2, 1] = np.nan
    with pytest.raises(ValueError):
        substitute_latent(bad, ARM_SPECS["control"], FPS)


def test_substituted_arrays_touches_only_the_latent():
    arrays = _synthetic_motion(3)
    out = substituted_arrays(arrays, ARM_SPECS["shift_p0500"], FPS)
    assert set(out) == set(arrays)
    for key in NON_LATENT_KEYS:
        assert out[key] is arrays[key] or np.array_equal(out[key], arrays[key])
    assert not np.array_equal(out[LATENT_KEY], arrays[LATENT_KEY])


# ---------------------------------------------------------------- end-to-end tree


def test_all_arms_are_written_with_original_clip_filenames(built):
    for arm in ARM_SPECS:
        arm_dir = built["out_root"] / "motions" / arm
        names = sorted(p.name for p in arm_dir.glob("*.npz"))
        assert names == ["walk1_subject1_mj_z.npz", "walk1_subject5_mj_z.npz"]
        assert not list(arm_dir.glob("*.tmp"))


def test_control_latent_is_byte_identical_to_the_source(built):
    for name, source in built["sources"].items():
        produced = _load(built["out_root"] / "motions" / "control" / name)
        assert produced[LATENT_KEY].dtype == np.float32
        assert produced[LATENT_KEY].shape == source[LATENT_KEY].shape
        assert produced[LATENT_KEY].tobytes() == source[LATENT_KEY].tobytes()
    control = built["manifest"]["arms"]["control"]
    assert control["latent_l2_all_clips"] == 0.0
    for clip in control["clips"]:
        assert clip["latent_l2"] == 0.0
        assert clip["frames_changed"] == 0
        assert clip["latent_sha256"] == clip["source_latent_sha256"]


@pytest.mark.parametrize("arm", sorted(ARM_SPECS))
def test_every_arm_preserves_all_non_latent_arrays_bit_exactly(built, arm):
    for name, source in built["sources"].items():
        produced = _load(built["out_root"] / "motions" / arm / name)
        assert set(produced) == set(source)
        for key in NON_LATENT_KEYS:
            assert produced[key].dtype == source[key].dtype, key
            assert produced[key].shape == source[key].shape, key
            assert produced[key].tobytes() == source[key].tobytes(), key
        assert produced[LATENT_KEY].dtype == np.float32
        assert produced[LATENT_KEY].shape == source[LATENT_KEY].shape


@pytest.mark.parametrize(
    "arm,delta", [("shift_m0250", -12), ("shift_p0250", 12), ("shift_p0500", 25)]
)
def test_written_shift_arms_have_the_expected_frame_mapping(built, arm, delta):
    assert built["manifest"]["arms"][arm]["frame_offset"] == delta
    assert built["manifest"]["arms"][arm]["realized_seconds"] == delta / FPS
    for name, source in built["sources"].items():
        latent = source[LATENT_KEY]
        produced = _load(built["out_root"] / "motions" / arm / name)[LATENT_KEY]
        expected_index = np.clip(np.arange(FRAMES) + delta, 0, FRAMES - 1)
        assert np.array_equal(produced, latent[expected_index])
        # explicit edge assertions
        if delta > 0:
            assert np.array_equal(produced[-1], latent[-1])
            assert np.array_equal(produced[FRAMES - delta], latent[-1])
            assert np.array_equal(produced[0], latent[delta])
        else:
            assert np.array_equal(produced[0], latent[0])
            assert np.array_equal(produced[-delta - 1], latent[0])
            assert np.array_equal(produced[-1], latent[FRAMES - 1 + delta])


@pytest.mark.parametrize("arm", ["first_frame", "clip_mean"])
def test_written_static_arms_are_constant_along_time(built, arm):
    for name, source in built["sources"].items():
        latent = source[LATENT_KEY]
        produced = _load(built["out_root"] / "motions" / arm / name)[LATENT_KEY]
        assert (produced == produced[0]).all()
        expected = (
            latent[0]
            if arm == "first_frame"
            else latent.mean(axis=0, dtype=np.float64).astype(np.float32)
        )
        assert np.array_equal(produced[0], expected)
        assert built["manifest"]["arms"][arm]["frame_offset"] is None


def test_manifest_records_rate_sources_and_distances(built, tmp_path):
    manifest = json.loads(
        (built["out_root"] / "latent_substitution_manifest.json").read_text()
    )
    assert manifest["fps"] == float(FPS)
    assert len(manifest["sources"]) == 2
    assert all(len(source["sha256"]) == 64 for source in manifest["sources"])
    assert set(manifest["arms"]) == set(ARM_SPECS)
    for arm, report in manifest["arms"].items():
        assert (report["latent_l2_all_clips"] == 0.0) == (arm == "control")


def test_source_hash_mismatch_aborts_before_writing(tmp_path):
    sources = _write_sources(tmp_path)
    out_root = tmp_path / "out_bad"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build([(sources[0][0], "0" * 64)], out_root, log=lambda _msg: None)
    assert not (out_root / "motions").exists()


def test_refuses_to_write_into_the_frozen_e70_root(tmp_path):
    sources = _write_sources(tmp_path)
    with pytest.raises(ValueError, match="frozen root"):
        build(
            sources,
            pathlib.Path("/data/robotixx/snmr-research/e70/latent_sub"),
            log=lambda _msg: None,
        )


def test_fps_disagreement_and_unexpected_rate_are_fatal(tmp_path):
    sources = _write_sources(tmp_path)
    other = _synthetic_motion(9)
    other["fps"] = np.array([30], dtype=np.int64)
    odd = tmp_path / "src" / "walk1_subject9_mj_z.npz"
    np.savez_compressed(odd, **other)
    with pytest.raises(ValueError, match="disagree on fps"):
        build(sources + [(odd, None)], tmp_path / "o1", log=lambda _msg: None)
    with pytest.raises(ValueError, match="expected"):
        build([(odd, None)], tmp_path / "o2", expected_fps=50, log=lambda _msg: None)
