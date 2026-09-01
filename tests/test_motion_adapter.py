import hashlib
from pathlib import Path

import numpy as np
import pytest

from snmr.human import LAFAN1_BODY_NAMES
from snmr.motion_adapter import adapt_lafan1_pair_npz
from snmr.motion_spec import (
    BilateralSegmentLandmarks,
    FootContactProtocol,
    HumanFrameConvention,
    MotionProvenance,
    MotionSource,
)
from snmr.provenance import ArtifactMismatchError


def _pair_arrays(frames: int = 6) -> dict[str, np.ndarray]:
    names = tuple(LAFAN1_BODY_NAMES)
    index = {name: body_index for body_index, name in enumerate(names)}
    time = np.arange(frames, dtype=np.float64) / 25.0
    root = np.stack((1.0 + 0.2 * time, -0.3 + time**2, np.ones(frames)), axis=-1)

    offsets = np.zeros((len(names), 3), dtype=np.float64)
    offsets[index["LeftUpLeg"]] = (0.0, 0.1, -0.1)
    offsets[index["RightUpLeg"]] = (0.0, -0.1, -0.1)
    offsets[index["LeftLeg"]] = (0.0, 0.1, -0.5)
    offsets[index["RightLeg"]] = (0.0, -0.1, -0.5)
    offsets[index["LeftFoot"]] = (0.0, 0.1, -0.9)
    offsets[index["RightFoot"]] = (0.0, -0.1, -0.9)
    offsets[index["LeftToe"]] = (0.1, 0.1, -0.98)
    offsets[index["RightToe"]] = (0.1, -0.1, -0.98)
    offsets[index["LeftShoulder"]] = (0.0, 0.1, 0.5)
    offsets[index["RightShoulder"]] = (0.0, -0.1, 0.5)
    offsets[index["LeftArm"]] = (0.0, 0.25, 0.5)
    offsets[index["RightArm"]] = (0.0, -0.25, 0.5)
    offsets[index["LeftForeArm"]] = (0.0, 0.55, 0.5)
    offsets[index["RightForeArm"]] = (0.0, -0.55, 0.5)
    offsets[index["LeftHand"]] = (0.0, 0.8, 0.5)
    offsets[index["RightHand"]] = (0.0, -0.8, 0.5)

    body_positions = root[:, None, :] + offsets[None, :, :]
    body_orientations = np.zeros((frames, len(names), 4), dtype=np.float64)
    body_orientations[..., 0] = 1.0
    qpos = np.zeros((frames, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    return {
        "human_pos": body_positions.astype(np.float32),
        "human_quat": body_orientations.astype(np.float32),
        "human_names": np.asarray(names),
        "qpos": qpos.astype(np.float32),
        "fps": np.asarray(25.0),
        "robot": np.asarray("unitree_g1"),
        "human_height": np.asarray(2.0),
    }


def _write_pair(
    tmp_path: Path,
    *,
    filename: str = "filename_is_not_identity.npz",
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[Path, str, dict[str, np.ndarray]]:
    payload = _pair_arrays() if arrays is None else arrays
    path = tmp_path / filename
    np.savez_compressed(path, **payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest, payload


def _landmarks() -> BilateralSegmentLandmarks:
    return BilateralSegmentLandmarks(
        torso=(
            ("LeftUpLeg", "LeftShoulder"),
            ("RightUpLeg", "RightShoulder"),
        ),
        thigh=(("LeftUpLeg", "LeftLeg"), ("RightUpLeg", "RightLeg")),
        shin=(("LeftLeg", "LeftFoot"), ("RightLeg", "RightFoot")),
        upper_arm=(
            ("LeftArm", "LeftForeArm"),
            ("RightArm", "RightForeArm"),
        ),
        forearm=(
            ("LeftForeArm", "LeftHand"),
            ("RightForeArm", "RightHand"),
        ),
    )


def _protocol(**overrides) -> FootContactProtocol:
    values = {
        "left_foot_body": "LeftToe",
        "right_foot_body": "RightToe",
        "speed_threshold_m_s": 0.2,
        "height_clearance_m": 0.05,
        "speed_softness_m_s": 0.05,
        "height_softness_m": 0.01,
        "ground_height_m": 0.0,
    }
    values.update(overrides)
    return FootContactProtocol(**values)


def _declarations(
    *,
    clip_range: tuple[int, int] = (1, 4),
) -> tuple[MotionSource, MotionProvenance]:
    return (
        MotionSource(
            dataset="explicit-lafan1-pair-corpus",
            sequence_id="explicit/subject-7/take-42",
            sha256="ab" * 32,
        ),
        MotionProvenance(
            motion_id="explicit-motion-id",
            source_subject="subject-7",
            clip_range=clip_range,
            split_id="train-fold-a",
            transformations=("gmr_lafan1_world_kinematics", "pair_npz_export"),
            preprocessing_version="pair-adapter-test-v1",
        ),
    )


def _adapt(
    path: Path,
    digest: str,
    *,
    clip_range: tuple[int, int] = (1, 4),
    body_names: tuple[str, ...] = tuple(LAFAN1_BODY_NAMES),
    root_body_name: str = "Hips",
    contact_protocol: FootContactProtocol | None = None,
    validity_mask: np.ndarray | None = None,
):
    source, provenance = _declarations(clip_range=clip_range)
    selected_frames = clip_range[1] - clip_range[0]
    validity = (
        np.ones((selected_frames, len(LAFAN1_BODY_NAMES)), dtype=np.bool_)
        if validity_mask is None
        else validity_mask
    )
    return adapt_lafan1_pair_npz(
        path,
        pair_artifact_sha256=digest,
        source=source,
        provenance=provenance,
        body_names=body_names,
        root_body_name=root_body_name,
        landmark_pairs=_landmarks(),
        contact_protocol=_protocol() if contact_protocol is None else contact_protocol,
        frames=HumanFrameConvention(),
        validity_mask=validity,
    )


def test_pair_adapter_preserves_world_hips_samples_and_explicit_metadata(tmp_path):
    path, digest, arrays = _write_pair(tmp_path)
    adaptation = _adapt(path, digest)
    spec = adaptation.motion_spec

    assert spec.source.dataset == "explicit-lafan1-pair-corpus"
    assert spec.source.sequence_id == "explicit/subject-7/take-42"
    assert spec.source.sha256 == "ab" * 32
    assert spec.source.sequence_id not in path.name
    assert adaptation.pair_artifact_sha256 == digest
    assert adaptation.pair_artifact_size_bytes == path.stat().st_size
    assert adaptation.source_group_key == (
        "explicit-lafan1-pair-corpus",
        "explicit/subject-7/take-42",
        "ab" * 32,
    )
    assert adaptation.to_manifest()["motion_buffer_sha256"] == spec.buffer_sha256
    manifest = adaptation.to_manifest()
    assert manifest["scale_landmarks"] == _landmarks().to_dict()
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest == adaptation.to_manifest()
    assert spec.provenance.motion_id == "explicit-motion-id"
    assert spec.provenance.source_subject == "subject-7"
    assert spec.provenance.clip_range == (1, 4)
    assert spec.provenance.split_id == "train-fold-a"
    assert spec.frames == HumanFrameConvention()
    assert spec.frames.root_height_convention == "absolute_world_z"
    assert spec.body_names == tuple(LAFAN1_BODY_NAMES)
    assert spec.contact_origin == "derived"
    assert spec.contact_protocol == _protocol()

    np.testing.assert_array_equal(spec.timebase.source_timestamps, [0.0, 0.04, 0.08])
    np.testing.assert_array_equal(
        spec.timebase.target_timestamps, [0.0, 0.02, 0.04, 0.06, 0.08]
    )
    selected_positions = arrays["human_pos"][1:4]
    selected_orientations = arrays["human_quat"][1:4]
    np.testing.assert_allclose(spec.body_positions[::2], selected_positions, atol=1e-7)
    np.testing.assert_allclose(
        spec.body_orientations_wxyz[::2], selected_orientations, atol=1e-7
    )
    np.testing.assert_array_equal(spec.root_position, spec.body_positions[:, 0])
    np.testing.assert_array_equal(
        spec.root_orientation_wxyz, spec.body_orientations_wxyz[:, 0]
    )

    assert spec.segment_scales.normalization_length_m == pytest.approx(2.0)
    assert spec.segment_scales.torso == pytest.approx(0.3)
    assert spec.segment_scales.thigh == pytest.approx(0.2)
    assert spec.segment_scales.shin == pytest.approx(0.2)
    assert spec.segment_scales.upper_arm == pytest.approx(0.15)
    assert spec.segment_scales.forearm == pytest.approx(0.125)
    assert spec.contacts.shape == (5, 2)
    assert np.all((0.0 <= spec.contacts) & (spec.contacts <= 1.0))
    assert np.all(spec.validity_mask)


def test_pair_adapter_rejects_artifact_hash_or_declared_layout_drift(tmp_path):
    path, digest, arrays = _write_pair(tmp_path)
    with pytest.raises(ArtifactMismatchError, match="exact LAFAN1 pair NPZ bytes"):
        _adapt(path, "00" * 32)

    swapped_declaration = list(LAFAN1_BODY_NAMES)
    swapped_declaration[1], swapped_declaration[2] = (
        swapped_declaration[2],
        swapped_declaration[1],
    )
    with pytest.raises(ValueError, match="current 24-body LAFAN1 training order"):
        _adapt(path, digest, body_names=tuple(swapped_declaration))

    changed_names = dict(arrays)
    stored_names = arrays["human_names"].copy()
    stored_names[[1, 2]] = stored_names[[2, 1]]
    changed_names["human_names"] = stored_names
    changed_path, changed_digest, _ = _write_pair(
        tmp_path, filename="stored_layout_drift.npz", arrays=changed_names
    )
    with pytest.raises(ValueError, match="pair human_names do not match"):
        _adapt(changed_path, changed_digest)


def test_pair_adapter_rejects_out_of_range_clip_and_wrong_root(tmp_path):
    path, digest, _ = _write_pair(tmp_path)
    with pytest.raises(ValueError, match="exceeds pair length"):
        _adapt(path, digest, clip_range=(1, 8))
    with pytest.raises(ValueError, match="root_body_name must be 'Hips' at index 0"):
        _adapt(path, digest, root_body_name="Spine")


def test_pair_adapter_rejects_invalid_samples_before_resampling_or_contacts(tmp_path):
    path, digest, _ = _write_pair(tmp_path)
    validity = np.ones((3, len(LAFAN1_BODY_NAMES)), dtype=np.bool_)
    validity[1, LAFAN1_BODY_NAMES.index("LeftToe")] = False
    with pytest.raises(ValueError, match="requires every source body sample to be valid"):
        _adapt(path, digest, validity_mask=validity)

    unknown_foot = _protocol(left_foot_body="UndeclaredLeftFoot")
    with pytest.raises(ValueError, match="unknown foot body"):
        _adapt(path, digest, contact_protocol=unknown_foot)


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    [
        (
            "human_pos",
            lambda value: np.where(np.indices(value.shape)[0] == 2, np.nan, value),
            "finite",
        ),
        (
            "human_quat",
            lambda value: value * 2.0,
            "unit quaternions",
        ),
        (
            "human_quat",
            lambda value: value[:, :-1],
            "human_quat must have shape",
        ),
    ],
)
def test_pair_adapter_fails_closed_on_noncanonical_pair_tensors(
    tmp_path, field, mutate, message
):
    arrays = _pair_arrays()
    arrays[field] = mutate(arrays[field])
    path, digest, _ = _write_pair(tmp_path, arrays=arrays)
    with pytest.raises(ValueError, match=message):
        _adapt(path, digest)


def test_pair_artifact_identity_is_separate_from_raw_human_source_identity(tmp_path):
    first_arrays = _pair_arrays()
    second_arrays = _pair_arrays()
    second_arrays["robot"] = np.asarray("different_robot")
    second_arrays["qpos"] = np.ones_like(second_arrays["qpos"])
    first_path, first_digest, _ = _write_pair(
        tmp_path, filename="first_robot_pair.npz", arrays=first_arrays
    )
    second_path, second_digest, _ = _write_pair(
        tmp_path, filename="second_robot_pair.npz", arrays=second_arrays
    )

    first = _adapt(first_path, first_digest)
    second = _adapt(second_path, second_digest)

    assert first.pair_artifact_sha256 != second.pair_artifact_sha256
    assert first.source_group_key == second.source_group_key
    assert first.motion_spec.source == second.motion_spec.source
    assert first.motion_spec.buffer_sha256 == second.motion_spec.buffer_sha256
    assert first.motion_spec.spec_sha256 == second.motion_spec.spec_sha256
