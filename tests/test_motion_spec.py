import json
import warnings

import numpy as np
import pytest

from snmr.motion_spec import (
    BilateralSegmentLandmarks,
    BodySegmentScales,
    FootContactProtocol,
    HumanFrameConvention,
    HumanMotionSpec,
    MotionProvenance,
    MotionSource,
    MotionTimebase,
    POSITION_INTERPOLATION,
    SCHEMA_VERSION,
    SHORT_SEQUENCE_POSITION_FALLBACK,
    TARGET_FPS,
    derive_foot_contact_probabilities,
    extract_body_segment_scales,
)


def _source_motion(
    *,
    frames: int = 5,
    dtype: np.dtype = np.dtype(np.float64),
) -> dict[str, np.ndarray]:
    time = np.arange(frames, dtype=np.float64) / 25.0
    root_position = np.stack((time, time**2, 0.9 + 0.1 * time), axis=-1)

    yaw = np.linspace(0.0, np.pi / 2.0, frames)
    root_orientation = np.stack(
        (np.cos(yaw / 2.0), np.zeros(frames), np.zeros(frames), np.sin(yaw / 2.0)),
        axis=-1,
    )
    offsets = np.asarray(
        [[0.0, 0.0, 0.0], [-0.1, 0.0, -0.8], [0.1, 0.0, -0.8]],
        dtype=np.float64,
    )
    body_positions = root_position[:, None, :] + offsets[None, :, :]
    body_orientations = np.repeat(root_orientation[:, None, :], 3, axis=1)
    contacts = np.stack(
        (np.linspace(1.0, 0.0, frames), np.linspace(0.0, 1.0, frames)), axis=-1
    )
    validity = np.ones((frames, 3), dtype=np.bool_)
    return {
        "root_position": root_position.astype(dtype),
        "root_orientation_wxyz": root_orientation.astype(dtype),
        "body_positions": body_positions.astype(dtype),
        "body_orientations_wxyz": body_orientations.astype(dtype),
        "contacts": contacts.astype(dtype),
        "validity_mask": validity,
    }


def _scales() -> BodySegmentScales:
    return BodySegmentScales.from_lengths(
        normalization_length_m=1.8,
        torso_m=0.54,
        thigh_m=0.45,
        shin_m=0.43,
        upper_arm_m=0.30,
        forearm_m=0.27,
    )


def _provenance(frames: int, motion_id: str) -> MotionProvenance:
    return MotionProvenance(
        motion_id=motion_id,
        source_subject=None,
        clip_range=(0, frames),
        split_id="test",
        transformations=("canonicalize",),
        preprocessing_version="motion-test-v1",
    )


def _make_spec(*, frames: int = 5, dtype: np.dtype = np.dtype(np.float64)) -> HumanMotionSpec:
    return HumanMotionSpec.from_source(
        source=MotionSource(dataset="unit-test", sequence_id="clip/001", sha256="ab" * 32),
        provenance=MotionProvenance(
            motion_id="unit-test:clip/001",
            source_subject="subject-01",
            clip_range=(10, 10 + frames),
            split_id="test",
            transformations=("z_up_x_forward", "pelvis_origin"),
            preprocessing_version="motion-test-v1",
        ),
        source_fps=25.0,
        body_names=("pelvis", "left_foot", "right_foot"),
        segment_scales=_scales(),
        **_source_motion(frames=frames, dtype=dtype),
    )


def _replace_motion_array(
    spec: HumanMotionSpec,
    field: str,
    value: np.ndarray,
) -> HumanMotionSpec:
    fields = {
        "schema_version": spec.schema_version,
        "source": spec.source,
        "provenance": spec.provenance,
        "timebase": spec.timebase,
        "frames": spec.frames,
        "body_names": spec.body_names,
        "segment_scales": spec.segment_scales,
        "contact_protocol": spec.contact_protocol,
        "contact_origin": spec.contact_origin,
        "root_position": spec.root_position,
        "root_orientation_wxyz": spec.root_orientation_wxyz,
        "body_positions": spec.body_positions,
        "body_orientations_wxyz": spec.body_orientations_wxyz,
        "contacts": spec.contacts,
        "validity_mask": spec.validity_mask,
    }
    fields[field] = value
    return HumanMotionSpec(**fields)


def test_motion_spec_json_round_trip_hash_and_immutability():
    spec = _make_spec()
    restored = HumanMotionSpec.from_json(spec.to_json(indent=2))

    assert restored == spec
    assert restored.buffer_sha256 == spec.buffer_sha256
    assert restored.integrity_sha256 == spec.buffer_sha256
    assert restored.spec_sha256 == spec.spec_sha256
    assert len(spec.buffer_sha256) == 64
    assert len(spec.spec_sha256) == 64
    assert spec.schema_version == SCHEMA_VERSION
    assert spec.frames == HumanFrameConvention()
    assert spec.frames.root_height_convention == "absolute_world_z"
    assert spec.provenance.motion_id == "unit-test:clip/001"
    assert spec.provenance.clip_range == (10, 15)
    assert spec.contact_origin == "provided"
    assert spec.timebase.target_fps == TARGET_FPS
    assert spec.segment_scales.torso == pytest.approx(0.3)
    assert not spec.root_position.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        spec.root_position[0, 0] = 3.0

    tampered = json.loads(spec.to_json())
    tampered["root_position"][1][0] += 0.25
    with pytest.raises(ValueError, match="buffer_sha256 does not match"):
        HumanMotionSpec.from_dict(tampered)

    provenance_tampered = json.loads(spec.to_json())
    provenance_tampered["provenance"]["split_id"] = "train"
    with pytest.raises(ValueError, match="spec_sha256 does not match"):
        HumanMotionSpec.from_dict(provenance_tampered)


def test_normalized_buffer_hash_is_layout_and_dtype_stable_and_mutation_sensitive():
    spec = _make_spec()
    float32_fortran = np.asfortranarray(spec.body_positions.astype(np.float32))
    equivalent = _replace_motion_array(spec, "body_positions", float32_fortran)
    assert equivalent.body_positions.flags.c_contiguous
    assert equivalent.body_positions.dtype == np.float64
    assert equivalent.buffer_sha256 == spec.buffer_sha256

    changed_position = spec.body_positions.copy()
    changed_position[2, 1, 0] += 0.125
    changed = _replace_motion_array(spec, "body_positions", changed_position)
    assert changed.buffer_sha256 != spec.buffer_sha256


def test_normalized_buffer_hash_fails_closed_before_float32_overflow_without_warning():
    spec = _make_spec()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for value in (1e200, 1e300):
            unrepresentable = spec.root_position.copy()
            unrepresentable[1, 0] = value
            with pytest.raises(ValueError, match="finite canonical float32 range"):
                _replace_motion_array(spec, "root_position", unrepresentable)
    assert caught == []


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    [
        (
            "root_position",
            lambda value: np.where(np.indices(value.shape)[0] == 0, np.nan, value),
            "finite",
        ),
        (
            "root_orientation_wxyz",
            lambda value: value * 2.0,
            "unit quaternions",
        ),
        (
            "body_positions",
            lambda value: value[:, :-1],
            "body_positions must have shape",
        ),
        (
            "contacts",
            lambda value: value + 1.1,
            "closed interval",
        ),
        (
            "validity_mask",
            lambda value: value.astype(np.float32),
            "boolean dtype",
        ),
    ],
)
def test_source_validation_rejects_nonfinite_shape_quaternion_contact_and_mask(
    field, mutate, message
):
    arrays = _source_motion()
    arrays[field] = mutate(arrays[field])
    with pytest.raises((TypeError, ValueError), match=message):
        HumanMotionSpec.from_source(
            source=MotionSource(dataset="unit-test", sequence_id="bad", sha256="01" * 32),
            provenance=_provenance(arrays["root_position"].shape[0], "bad"),
            source_fps=25.0,
            body_names=("pelvis", "left_foot", "right_foot"),
            segment_scales=_scales(),
            **arrays,
        )


def test_schema_rejects_noncanonical_frames_hash_and_timeline():
    with pytest.raises(ValueError, match="world_up_axis"):
        HumanFrameConvention(world_up_axis="y")
    with pytest.raises(ValueError, match="root_height_convention"):
        HumanFrameConvention(root_height_convention="pelvis_relative")
    with pytest.raises(ValueError, match="64-character"):
        MotionSource(dataset="unit-test", sequence_id="clip", sha256="short")
    with pytest.raises(ValueError, match="0 <= start < stop"):
        MotionProvenance(
            motion_id="bad-range",
            source_subject=None,
            clip_range=(4, 4),
            split_id="test",
            transformations=(),
            preprocessing_version="v1",
        )


def test_resampling_fails_closed_on_any_invalid_source_body_sample():
    arrays = _source_motion(frames=5)
    arrays["validity_mask"][2, 0] = False
    with pytest.raises(
        ValueError,
        match="canonical resampling requires every source body sample to be valid",
    ):
        HumanMotionSpec.from_source(
            source=MotionSource(dataset="unit-test", sequence_id="invalid", sha256="12" * 32),
            provenance=_provenance(5, "invalid"),
            source_fps=25.0,
            body_names=("pelvis", "left_foot", "right_foot"),
            segment_scales=_scales(),
            **arrays,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        MotionTimebase(
            source_fps=25.0,
            source_timestamps=np.asarray([0.0, 0.04, 0.08]),
            target_fps=50.0,
            target_timestamps=np.asarray([0.0, 0.02, 0.02, 0.08]),
            position_interpolation=POSITION_INTERPOLATION,
            position_interpolation_applied=SHORT_SEQUENCE_POSITION_FALLBACK,
            quaternion_interpolation="slerp",
        )


def test_resampling_uses_explicit_50hz_grid_slerp_and_preserves_endpoints():
    arrays = _source_motion(frames=4)
    spec = HumanMotionSpec.from_source(
        source=MotionSource(dataset="unit-test", sequence_id="cubic", sha256="23" * 32),
        provenance=_provenance(4, "cubic"),
        source_fps=25.0,
        body_names=("pelvis", "left_foot", "right_foot"),
        segment_scales=_scales(),
        **arrays,
    )

    np.testing.assert_allclose(spec.timebase.target_timestamps, np.arange(7) / 50.0, atol=1e-12)
    assert spec.timebase.position_interpolation == POSITION_INTERPOLATION
    assert spec.timebase.position_interpolation_applied == POSITION_INTERPOLATION
    np.testing.assert_allclose(spec.root_position[0], arrays["root_position"][0], atol=1e-12)
    np.testing.assert_allclose(spec.root_position[-1], arrays["root_position"][-1], atol=1e-12)
    np.testing.assert_allclose(spec.body_positions[0], arrays["body_positions"][0], atol=1e-12)
    np.testing.assert_allclose(spec.body_positions[-1], arrays["body_positions"][-1], atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(spec.root_orientation_wxyz, axis=-1), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(spec.body_orientations_wxyz, axis=-1), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        spec.root_orientation_wxyz[[0, -1]],
        arrays["root_orientation_wxyz"][[0, -1]],
        atol=1e-12,
    )


def test_extract_body_segment_scales_from_declared_bilateral_landmarks():
    body_names = (
        "left_hip",
        "right_hip",
        "left_shoulder",
        "right_shoulder",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    )
    point = {
        "left_hip": (-0.1, 0.0, 1.0),
        "right_hip": (0.1, 0.0, 1.0),
        "left_shoulder": (-0.1, 0.0, 1.5),
        "right_shoulder": (0.1, 0.0, 1.5),
        "left_knee": (-0.1, 0.0, 0.6),
        "right_knee": (0.1, 0.0, 0.6),
        "left_ankle": (-0.1, 0.0, 0.2),
        "right_ankle": (0.1, 0.0, 0.2),
        "left_elbow": (-0.4, 0.0, 1.5),
        "right_elbow": (0.4, 0.0, 1.5),
        "left_wrist": (-0.65, 0.0, 1.5),
        "right_wrist": (0.65, 0.0, 1.5),
    }
    one_frame = np.asarray([point[name] for name in body_names], dtype=np.float64)
    positions = np.repeat(one_frame[None, :, :], 3, axis=0)
    validity = np.ones(positions.shape[:2], dtype=np.bool_)
    validity[1, body_names.index("right_elbow")] = False
    landmarks = BilateralSegmentLandmarks(
        torso=(
            ("left_hip", "left_shoulder"),
            ("right_hip", "right_shoulder"),
        ),
        thigh=(("left_hip", "left_knee"), ("right_hip", "right_knee")),
        shin=(("left_knee", "left_ankle"), ("right_knee", "right_ankle")),
        upper_arm=(
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
        ),
        forearm=(("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")),
    )

    scales = extract_body_segment_scales(
        body_positions=positions,
        body_names=body_names,
        landmark_pairs=landmarks,
        normalization_length_m=2.0,
        validity_mask=validity,
    )
    assert scales.torso == pytest.approx(0.25)
    assert scales.thigh == pytest.approx(0.20)
    assert scales.shin == pytest.approx(0.20)
    assert scales.upper_arm == pytest.approx(0.15)
    assert scales.forearm == pytest.approx(0.125)
    assert BodySegmentScales.from_landmarks(
        body_positions=positions,
        body_names=body_names,
        landmark_pairs=landmarks,
        normalization_length_m=2.0,
        validity_mask=validity,
    ) == scales


def test_probabilistic_foot_contacts_use_declared_speed_and_height_protocol():
    timestamps = np.arange(5, dtype=np.float64) / 50.0
    positions = np.zeros((5, 3, 3), dtype=np.float64)
    positions[:, 0, 2] = 1.0  # pelvis
    positions[:, 1, 2] = 0.0  # stationary left foot on the ground
    positions[:, 2, 0] = np.arange(5) * 0.1  # fast right foot
    positions[:, 2, 2] = 0.20  # and well above the clearance
    protocol = FootContactProtocol(
        speed_threshold_m_s=0.20,
        height_clearance_m=0.05,
        speed_softness_m_s=0.02,
        height_softness_m=0.01,
    )

    contacts = derive_foot_contact_probabilities(
        body_positions=positions,
        body_names=("pelvis", "left_foot", "right_foot"),
        timestamps=timestamps,
        protocol=protocol,
    )
    assert contacts.shape == (5, 2)
    assert np.all((contacts >= 0.0) & (contacts <= 1.0))
    assert np.all(contacts[:, 0] > 0.99)
    assert np.all(contacts[:, 1] < 1e-6)

    validity = np.ones(positions.shape[:2], dtype=np.bool_)
    validity[2, 1] = False
    with pytest.raises(ValueError, match="cannot derive contacts.*invalid at frames"):
        derive_foot_contact_probabilities(
            body_positions=positions,
            body_names=("pelvis", "left_foot", "right_foot"),
            timestamps=timestamps,
            protocol=protocol,
            validity_mask=validity,
        )


def test_from_source_derives_contacts_when_none_and_records_protocol():
    arrays = _source_motion(frames=5)
    supplied_contacts = arrays.pop("contacts")
    protocol = FootContactProtocol(
        speed_threshold_m_s=1.5,
        height_clearance_m=0.2,
        speed_softness_m_s=0.1,
        height_softness_m=0.02,
        ground_height_m=0.0,
    )
    expected_source = derive_foot_contact_probabilities(
        body_positions=arrays["body_positions"],
        body_names=("pelvis", "left_foot", "right_foot"),
        timestamps=np.arange(5) / 25.0,
        protocol=protocol,
    )
    spec = HumanMotionSpec.from_source(
        source=MotionSource(dataset="unit-test", sequence_id="derived", sha256="67" * 32),
        provenance=_provenance(5, "derived"),
        source_fps=25.0,
        body_names=("pelvis", "left_foot", "right_foot"),
        segment_scales=_scales(),
        contact_protocol=protocol,
        contacts=None,
        **arrays,
    )
    assert spec.contact_origin == "derived"
    assert spec.contact_protocol == protocol
    assert spec.timebase.validity_interpolation == "fail_closed_all_valid"
    assert np.all(spec.validity_mask)
    np.testing.assert_allclose(spec.contacts[0], expected_source[0], atol=1e-12)
    np.testing.assert_allclose(spec.contacts[-1], expected_source[-1], atol=1e-12)
    assert not np.array_equal(spec.contacts, supplied_contacts)


def test_short_sequence_declares_fallback_and_truncates_non_grid_source_tail():
    arrays = _source_motion(frames=3)
    spec = HumanMotionSpec.from_source(
        source=MotionSource(dataset="unit-test", sequence_id="fallback", sha256="45" * 32),
        provenance=_provenance(3, "fallback"),
        source_fps=30.0,
        body_names=("pelvis", "left_foot", "right_foot"),
        segment_scales=_scales(),
        **arrays,
    )

    expected_timestamps = np.asarray([0.0, 0.02, 0.04, 0.06])
    np.testing.assert_allclose(spec.timebase.target_timestamps, expected_timestamps, atol=1e-12)
    assert spec.timebase.source_timestamps[-1] == pytest.approx(2.0 / 30.0)
    assert spec.timebase.target_timestamps[-1] < spec.timebase.source_timestamps[-1]
    np.testing.assert_allclose(np.diff(spec.timebase.target_timestamps), 1.0 / 50.0)
    with pytest.raises(ValueError, match="strictly uniform 50 Hz grid"):
        MotionTimebase(
            source_fps=30.0,
            source_timestamps=np.arange(3) / 30.0,
            target_fps=50.0,
            target_timestamps=np.asarray([0.0, 0.02, 0.04, 0.06, 2.0 / 30.0]),
            position_interpolation=POSITION_INTERPOLATION,
            position_interpolation_applied=SHORT_SEQUENCE_POSITION_FALLBACK,
            quaternion_interpolation="slerp",
        )
    assert spec.timebase.position_interpolation_applied == SHORT_SEQUENCE_POSITION_FALLBACK
    np.testing.assert_allclose(spec.root_position[0], arrays["root_position"][0], atol=1e-12)
    expected_last_root = np.asarray(
        [
            np.interp(
                spec.timebase.target_timestamps[-1],
                spec.timebase.source_timestamps,
                arrays["root_position"][:, axis],
            )
            for axis in range(3)
        ]
    )
    np.testing.assert_allclose(spec.root_position[-1], expected_last_root, atol=1e-12)
    assert not np.allclose(spec.root_position[-1], arrays["root_position"][-1])
    np.testing.assert_allclose(
        np.linalg.norm(spec.root_orientation_wxyz, axis=-1), 1.0, atol=1e-12
    )


def test_exact_30_to_50_hz_endpoint_is_not_misclassified_as_extrapolation():
    arrays = _source_motion(frames=100)
    spec = HumanMotionSpec.from_source(
        source=MotionSource(
            dataset="unit-test", sequence_id="30-to-50", sha256="46" * 32
        ),
        provenance=_provenance(100, "30-to-50"),
        source_fps=30.0,
        body_names=("pelvis", "left_foot", "right_foot"),
        segment_scales=_scales(),
        **arrays,
    )

    assert spec.timebase.source_timestamps[-1] == spec.timebase.target_timestamps[-1]
    assert spec.timebase.target_timestamps[-1] == 3.3
    assert spec.timebase.target_timestamps.size == 166
    np.testing.assert_array_equal(spec.root_position[-1], arrays["root_position"][-1])
