from dataclasses import replace
import inspect

import numpy as np
import pytest

from snmr.motion_spec import (
    BodySegmentScales,
    FootContactProtocol,
    HumanFrameConvention,
    HumanMotionSpec,
    MotionProvenance,
    MotionSource,
)
from snmr.robot_spec import (
    ControlSpec,
    FrameConvention,
    LinkSpec,
    RobotSpec,
    RuntimeSpec,
    SemanticManifest,
)
from snmr.semantic_metrics import (
    REQUIRED_SEMANTIC_ROLES,
    SEMANTIC_METRIC_SCHEMA_SHA256,
    RobotContactProtocol,
    RobotFKTrajectory,
    SemanticAnchor,
    SemanticCorrespondence,
    SemanticThresholds,
    evaluate_semantic_retargeting,
)


HUMAN_NAMES = REQUIRED_SEMANTIC_ROLES
ROLE_TO_LINK = {
    "pelvis": "base",
    "torso": "trunk",
    "head": "head_link",
    "left_hand": "left_gripper",
    "right_hand": "right_gripper",
    "left_foot": "left_sole",
    "right_foot": "right_sole",
}
ROBOT_LINK_NAMES = (
    "right_sole",
    "head_link",
    "base",
    "left_gripper",
    "trunk",
    "left_sole",
    "right_gripper",
)


def _axis_quaternion(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    axis_array /= np.linalg.norm(axis_array)
    return np.concatenate(([np.cos(angle / 2.0)], axis_array * np.sin(angle / 2.0)))


def _yaw_quaternion(angle: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle, dtype=np.float64)
    result = np.zeros(angle.shape + (4,), dtype=np.float64)
    result[..., 0] = np.cos(angle / 2.0)
    result[..., 3] = np.sin(angle / 2.0)
    return result


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.broadcast_arrays(np.asarray(left), np.asarray(right))
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quat_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result[..., 1:] *= -1.0
    return result


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    zeros = np.zeros(v.shape[:-1] + (1,), dtype=np.float64)
    return _quat_mul(_quat_mul(q, np.concatenate((zeros, v), axis=-1)), _quat_conjugate(q))[..., 1:]


def _rotate_xy(vectors: np.ndarray, angle: np.ndarray) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    output = np.empty_like(vectors)
    output[..., 0] = cosine * vectors[..., 0] - sine * vectors[..., 1]
    output[..., 1] = sine * vectors[..., 0] + cosine * vectors[..., 1]
    return output


def _world_trajectory(
    local_normalized: np.ndarray,
    relative_heading: np.ndarray,
    root_displacement_normalized: np.ndarray,
    root_heading_relative: np.ndarray,
    *,
    scale_m: float,
    initial_heading: float,
    origin_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames, bodies, _ = local_normalized.shape
    initial = np.full(frames, initial_heading, dtype=np.float64)
    root_xy = np.asarray(origin_xy, dtype=np.float64) + scale_m * _rotate_xy(
        root_displacement_normalized, initial
    )
    root_position = np.column_stack((root_xy, scale_m * local_normalized[:, 0, 2]))
    root_heading = initial_heading + root_heading_relative
    root_orientation = _yaw_quaternion(root_heading)
    positions = np.empty((frames, bodies, 3), dtype=np.float64)
    for body in range(bodies):
        positions[:, body, :2] = root_xy + scale_m * _rotate_xy(
            local_normalized[:, body, :2], root_heading
        )
        positions[:, body, 2] = scale_m * local_normalized[:, body, 2]
    orientations = _yaw_quaternion(root_heading[:, None] + relative_heading)
    return root_position, root_orientation, positions, orientations


def _local_motion(frames: int) -> tuple[np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, np.pi, frames)
    local = np.zeros((frames, len(HUMAN_NAMES), 3), dtype=np.float64)
    local[:, 0] = (0.0, 0.0, 0.50)
    local[:, 1, 0] = 0.04 + 0.01 * np.sin(phase)
    local[:, 1, 2] = 0.70 + 0.02 * np.sin(phase)
    local[:, 2, 0] = 0.03 + 0.015 * np.sin(phase)
    local[:, 2, 2] = 0.95 + 0.01 * np.cos(phase)
    local[:, 3, 0] = 0.18 + 0.08 * np.sin(phase)
    local[:, 3, 1] = 0.30 + 0.04 * np.cos(phase)
    local[:, 3, 2] = 0.75 + 0.03 * np.sin(phase)
    local[:, 4, 0] = 0.18 - 0.06 * np.sin(phase)
    local[:, 4, 1] = -0.30 - 0.04 * np.cos(phase)
    local[:, 4, 2] = 0.75 - 0.02 * np.sin(phase)
    local[:, 5, 0] = 0.04 + 0.09 * np.sin(phase)
    local[:, 5, 1] = 0.10
    local[:, 5, 2] = 0.03 + 0.02 * np.sin(phase)
    local[:, 6, 0] = 0.04 - 0.07 * np.sin(phase)
    local[:, 6, 1] = -0.10
    local[:, 6, 2] = 0.03 + 0.015 * np.sin(phase)
    relative_heading = np.zeros((frames, len(HUMAN_NAMES)), dtype=np.float64)
    relative_heading[:, 3] = 0.15 * np.sin(phase)
    relative_heading[:, 4] = -0.12 * np.sin(phase)
    relative_heading[:, 5] = 0.06 * np.sin(phase)
    relative_heading[:, 6] = -0.05 * np.sin(phase)
    return local, relative_heading


def _link(name: str, parent: str | None) -> LinkSpec:
    return LinkSpec(
        name=name,
        parent=parent,
        local_position=(0.0, 0.0, 0.0),
        local_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        mass=1.0,
        center_of_mass=(0.0, 0.0, 0.0),
        inertia_diagonal=(0.01, 0.01, 0.01),
        inertia_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def _robot_spec() -> RobotSpec:
    spec = RobotSpec(
        schema_version="snmr.robot.v0.1",
        asset_sha256="ab" * 32,
        frames=FrameConvention(),
        semantics=SemanticManifest(
            root_link="base",
            torso_link="trunk",
            head_link="head_link",
            left_hand_link="left_gripper",
            right_hand_link="right_gripper",
            left_foot_link="left_sole",
            right_foot_link="right_sole",
            allowed_contact_links=("left_sole", "right_sole"),
            symmetry_pairs=(("left_gripper", "right_gripper"), ("left_sole", "right_sole")),
        ),
        control=ControlSpec(),
        runtime=RuntimeSpec(simulation_dt=0.002),
        total_mass=7.0,
        standing_height=1.0,
        arm_span=0.6,
        links=tuple(
            _link(name, None if name == "base" else "base")
            for name in ("base", "trunk", "head_link", "left_gripper", "right_gripper", "left_sole", "right_sole")
        ),
        joints=(),
    )
    spec.validate()
    return spec


def _calibration(calibrated: bool) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    zero = np.zeros(3, dtype=np.float64)
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    human = {role: (zero.copy(), identity.copy()) for role in REQUIRED_SEMANTIC_ROLES}
    robot = {role: (zero.copy(), identity.copy()) for role in REQUIRED_SEMANTIC_ROLES}
    if calibrated:
        human["left_hand"] = (
            np.asarray([0.035, -0.012, 0.018]),
            _axis_quaternion((0.0, 1.0, 0.0), 0.31),
        )
        robot["left_hand"] = (
            np.asarray([-0.052, 0.021, 0.014]),
            _axis_quaternion((1.0, 0.0, 0.0), -0.43),
        )
    return human, robot


def _apply_inverse_calibration(
    positions: np.ndarray,
    orientations: np.ndarray,
    calibration: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    raw_positions = positions.copy()
    raw_orientations = orientations.copy()
    for index, role in enumerate(REQUIRED_SEMANTIC_ROLES):
        point, orientation_offset = calibration[role]
        raw_q = _quat_mul(orientations[:, index], _quat_conjugate(orientation_offset))
        raw_orientations[:, index] = raw_q
        raw_positions[:, index] = positions[:, index] - _quat_rotate(raw_q, point)
    return raw_positions, raw_orientations


def _correspondence(*, calibrated: bool = False, thresholds: SemanticThresholds | None = None) -> SemanticCorrespondence:
    human_calibration, robot_calibration = _calibration(calibrated)
    return SemanticCorrespondence(
        mapping_version="unit-test-map-v1",
        anchors=tuple(
            SemanticAnchor(
                role=role,
                human_body=role,
                robot_link=ROLE_TO_LINK[role],
                human_point_local_m=tuple(human_calibration[role][0]),
                robot_point_local_m=tuple(robot_calibration[role][0]),
                human_orientation_offset_wxyz=tuple(human_calibration[role][1]),
                robot_orientation_offset_wxyz=tuple(robot_calibration[role][1]),
            )
            for role in REQUIRED_SEMANTIC_ROLES
        ),
        contact_protocol=RobotContactProtocol(),
        thresholds=thresholds or SemanticThresholds(),
    )


def _scene(*, calibrated: bool = False) -> tuple[HumanMotionSpec, RobotSpec, RobotFKTrajectory, SemanticCorrespondence]:
    frames = 8
    timestamps = np.arange(frames, dtype=np.float64) / 50.0
    root_displacement = np.column_stack(
        (np.linspace(0.0, 0.28, frames), 0.015 * np.sin(np.linspace(0.0, np.pi, frames)))
    )
    root_heading = np.linspace(0.0, 0.28, frames)
    local, relative_heading = _local_motion(frames)
    human_root, human_root_q, human_semantic_p, human_semantic_q = _world_trajectory(
        local,
        relative_heading,
        root_displacement,
        root_heading,
        scale_m=2.0,
        initial_heading=0.0,
        origin_xy=(0.2, -0.1),
    )
    robot_root, robot_root_q, robot_semantic_p, robot_semantic_q = _world_trajectory(
        local,
        relative_heading,
        root_displacement,
        root_heading,
        scale_m=1.0,
        initial_heading=0.7,
        origin_xy=(3.0, -2.0),
    )
    human_calibration, robot_calibration = _calibration(calibrated)
    human_raw_p, human_raw_q = _apply_inverse_calibration(
        human_semantic_p, human_semantic_q, human_calibration
    )
    robot_raw_p, robot_raw_q = _apply_inverse_calibration(
        robot_semantic_p, robot_semantic_q, robot_calibration
    )
    # Root pose is the physical pelvis/base FK pose.  Pelvis calibration stays identity.
    np.testing.assert_allclose(human_raw_p[:, 0], human_root)
    np.testing.assert_allclose(robot_raw_p[:, 0], robot_root)
    human = HumanMotionSpec.from_source(
        source=MotionSource(dataset="semantic-unit-test", sequence_id="subject/clip", sha256="12" * 32),
        provenance=MotionProvenance(
            motion_id="semantic-unit-test:subject/clip",
            source_subject="subject",
            clip_range=(0, frames),
            split_id="test",
            transformations=("canonical_world",),
            preprocessing_version="semantic-test-v2",
        ),
        source_fps=50.0,
        body_names=HUMAN_NAMES,
        segment_scales=BodySegmentScales.from_lengths(
            normalization_length_m=2.0,
            torso_m=0.6,
            thigh_m=0.45,
            shin_m=0.42,
            upper_arm_m=0.30,
            forearm_m=0.25,
        ),
        root_position=human_root,
        root_orientation_wxyz=human_root_q,
        body_positions=human_raw_p,
        body_orientations_wxyz=human_raw_q,
        contacts=None,
        contact_protocol=FootContactProtocol(
            left_foot_body="left_foot",
            right_foot_body="right_foot",
            speed_threshold_m_s=0.40,
            height_clearance_m=0.10,
            speed_softness_m_s=0.10,
            height_softness_m=0.02,
            ground_height_m=0.0,
        ),
        frames=HumanFrameConvention(),
        validity_mask=np.ones((frames, len(HUMAN_NAMES)), dtype=np.bool_),
    )
    spec = _robot_spec()
    body_for_link = {link: REQUIRED_SEMANTIC_ROLES.index(role) for role, link in ROLE_TO_LINK.items()}
    indices = [body_for_link[name] for name in ROBOT_LINK_NAMES]
    fk = RobotFKTrajectory(
        timestamps_s=timestamps,
        frames=HumanFrameConvention(),
        robot_spec_sha256=spec.spec_hash,
        robot_asset_sha256=spec.asset_sha256,
        root_link="base",
        link_names=ROBOT_LINK_NAMES,
        link_positions_m=robot_raw_p[:, indices],
        link_orientations_wxyz=robot_raw_q[:, indices],
    )
    return human, spec, fk, _correspondence(calibrated=calibrated)


def _replace_fk(fk: RobotFKTrajectory, **changes) -> RobotFKTrajectory:
    values = {
        "timestamps_s": fk.timestamps_s,
        "frames": fk.frames,
        "robot_spec_sha256": fk.robot_spec_sha256,
        "robot_asset_sha256": fk.robot_asset_sha256,
        "root_link": fk.root_link,
        "link_names": fk.link_names,
        "link_positions_m": fk.link_positions_m,
        "link_orientations_wxyz": fk.link_orientations_wxyz,
    }
    values.update(changes)
    return RobotFKTrajectory(**values)


def _replace_link_local_motion(
    fk: RobotFKTrajectory, role: str, transform
) -> RobotFKTrajectory:
    positions = np.array(fk.link_positions_m, copy=True)
    root_position = fk.root_position_m
    root_q = fk.root_orientation_wxyz
    root_yaw = 2.0 * np.arctan2(root_q[:, 3], root_q[:, 0])
    index = fk.link_names.index(ROLE_TO_LINK[role])
    local = positions[:, index].copy()
    local[:, :2] -= root_position[:, :2]
    local[:, :2] = _rotate_xy(local[:, :2], -root_yaw)
    changed = transform(local)
    positions[:, index, :2] = root_position[:, :2] + _rotate_xy(changed[:, :2], root_yaw)
    positions[:, index, 2] = changed[:, 2]
    return _replace_fk(fk, link_positions_m=positions)


def test_exact_scaled_semantics_report_meter_and_normalized_errors_without_gmr():
    human, spec, fk, correspondence = _scene()
    report = evaluate_semantic_retargeting(human, spec, fk, correspondence)

    assert report.frame_count == 8
    assert report.duration_s == pytest.approx(0.14)
    assert report.robot_normalization_length_m == spec.standing_height
    for value in (
        report.keypoint_error_m_mean,
        report.keypoint_error_m_p95,
        report.keypoint_error_m_max,
        report.keypoint_error_normalized_mean,
        report.end_effector_relative_position_error_m_mean,
        report.end_effector_relative_orientation_error_rad_mean,
        report.contact_probability_mae,
        report.root_displacement_error_m_mean,
        report.root_heading_error_rad_mean,
        report.root_height_error_m_mean,
    ):
        assert value == pytest.approx(0.0, abs=2e-12)
    assert report.motion_fidelity_gate_pass
    assert report.motion_fidelity_violations == ()


def test_nonidentity_point_and_noncommuting_orientation_offsets_use_right_multiplication():
    human, spec, fk, correspondence = _scene(calibrated=True)
    report = evaluate_semantic_retargeting(human, spec, fk, correspondence)
    assert report.keypoint_error_m_max == pytest.approx(0.0, abs=2e-12)
    assert report.end_effector_relative_orientation_error_rad_mean == pytest.approx(0.0, abs=2e-12)

    anchors = list(correspondence.anchors)
    left = REQUIRED_SEMANTIC_ROLES.index("left_hand")
    anchors[left] = replace(anchors[left], robot_point_local_m=(0.0, 0.0, 0.0))
    wrong = replace(correspondence, anchors=tuple(anchors))
    attacked = evaluate_semantic_retargeting(human, spec, fk, wrong)
    assert dict(attacked.per_anchor_keypoint_error_m_mean)["left_hand"] > 0.04


def test_scale_is_robot_spec_bound_and_cannot_be_overridden_by_correspondence():
    human, spec, fk, correspondence = _scene()
    assert "robot_normalization_length_m" not in inspect.signature(SemanticCorrespondence).parameters
    attacked_spec = replace(spec, standing_height=2.0)
    attacked_spec.validate()
    with pytest.raises(ValueError, match="robot_spec_sha256 does not match"):
        evaluate_semantic_retargeting(human, attacked_spec, fk, correspondence)


def test_root_is_the_declared_fk_link_and_has_no_independent_candidate_buffer():
    human, spec, fk, correspondence = _scene()
    parameters = inspect.signature(RobotFKTrajectory).parameters
    assert "root_position_m" not in parameters
    assert "root_orientation_wxyz" not in parameters
    attacked = _replace_fk(fk, root_link="left_gripper")
    with pytest.raises(ValueError, match="root_link must equal"):
        evaluate_semantic_retargeting(human, spec, attacked, correspondence)


def test_contacts_are_derived_from_hash_bound_sole_fk_not_candidate_channels():
    human, spec, fk, correspondence = _scene()
    assert "contact_probabilities" not in inspect.signature(RobotFKTrajectory).parameters
    baseline = evaluate_semantic_retargeting(human, spec, fk, correspondence)
    positions = np.array(fk.link_positions_m, copy=True)
    sole_index = fk.link_names.index("left_sole")
    # A forged contact channel cannot be supplied.  The only way to change the detector is to
    # change the exact sole FK itself; pinning it at the ground makes that dependency visible.
    positions[:, sole_index] = positions[0, sole_index]
    positions[:, sole_index, 2] = 0.0
    attacked_fk = _replace_fk(fk, link_positions_m=positions)
    attacked = evaluate_semantic_retargeting(human, spec, attacked_fk, correspondence)
    assert attacked.contact_probability_mae > baseline.contact_probability_mae + 0.1
    assert attacked.robot_contact_buffer_sha256 != baseline.robot_contact_buffer_sha256
    assert attacked.robot_fk_buffer_sha256 != baseline.robot_fk_buffer_sha256


def test_per_anchor_two_sided_gates_detect_partial_collapse_and_over_amplitude():
    human, spec, fk, correspondence = _scene()

    collapsed = _replace_link_local_motion(
        fk, "left_hand", lambda local: np.repeat(np.mean(local, axis=0, keepdims=True), local.shape[0], axis=0)
    )
    collapse_report = evaluate_semantic_retargeting(human, spec, collapsed, correspondence)
    assert not dict(collapse_report.per_anchor_motion_gate_pass)["left_hand"]
    assert "left_hand:amplitude" in collapse_report.motion_fidelity_violations
    assert not collapse_report.motion_fidelity_gate_pass

    amplified = _replace_link_local_motion(
        fk, "right_hand", lambda local: np.mean(local, axis=0, keepdims=True) + 2.0 * (local - np.mean(local, axis=0, keepdims=True))
    )
    amplified_report = evaluate_semantic_retargeting(human, spec, amplified, correspondence)
    assert dict(amplified_report.per_anchor_amplitude_ratio)["right_hand"] > 1.5
    assert "right_hand:amplitude" in amplified_report.motion_fidelity_violations


def test_per_anchor_energy_and_acceleration_gates_detect_local_jitter():
    human, spec, fk, correspondence = _scene()

    def add_jitter(local: np.ndarray) -> np.ndarray:
        result = local.copy()
        result[:, 1] += 0.012 * np.where(np.arange(local.shape[0]) % 2 == 0, -1.0, 1.0)
        return result

    jittered = _replace_link_local_motion(fk, "left_hand", add_jitter)
    report = evaluate_semantic_retargeting(human, spec, jittered, correspondence)
    assert not dict(report.per_anchor_motion_gate_pass)["left_hand"]
    assert (
        "left_hand:energy" in report.motion_fidelity_violations
        or "left_hand:jitter" in report.motion_fidelity_violations
    )


def test_role_set_is_complete_ordered_and_mapping_is_robot_spec_bound():
    correspondence = _correspondence()
    with pytest.raises(ValueError, match="complete canonical role set"):
        replace(correspondence, anchors=correspondence.anchors[:-1])
    with pytest.raises(ValueError, match="complete canonical role set"):
        replace(correspondence, anchors=tuple(reversed(correspondence.anchors)))

    human, spec, fk, correspondence = _scene()
    anchors = list(correspondence.anchors)
    anchors[3] = replace(anchors[3], robot_link="right_gripper")
    # Duplicate mappings fail before evaluation; a unique but semantically wrong swap is caught
    # by the RobotSpec binding.
    anchors[4] = replace(anchors[4], robot_link="left_gripper")
    swapped = replace(correspondence, anchors=tuple(anchors))
    with pytest.raises(ValueError, match="must use RobotSpec link"):
        evaluate_semantic_retargeting(human, spec, fk, swapped)


def test_all_protocol_and_exact_buffer_hashes_are_stable_and_attack_sensitive():
    human, spec, fk, correspondence = _scene()
    first = evaluate_semantic_retargeting(human, spec, fk, correspondence)
    second = evaluate_semantic_retargeting(human, spec, fk, correspondence)
    assert first == second
    assert first.report_sha256 == second.report_sha256
    hashes = (
        first.metric_schema_sha256,
        first.human_motion_spec_sha256,
        first.human_motion_buffer_sha256,
        first.robot_spec_sha256,
        first.robot_asset_sha256,
        first.robot_fk_buffer_sha256,
        first.robot_contact_protocol_sha256,
        first.robot_contact_buffer_sha256,
        first.correspondence_sha256,
        first.thresholds_sha256,
        first.report_sha256,
    )
    assert first.metric_schema_sha256 == SEMANTIC_METRIC_SCHEMA_SHA256
    assert all(len(value) == 64 for value in hashes)

    changed_thresholds = replace(correspondence.thresholds, energy_ratio_max=5.0)
    changed_correspondence = replace(correspondence, thresholds=changed_thresholds)
    changed = evaluate_semantic_retargeting(human, spec, fk, changed_correspondence)
    assert changed.thresholds_sha256 != first.thresholds_sha256
    assert changed.correspondence_sha256 != first.correspondence_sha256
    assert changed.metric_schema_sha256 == first.metric_schema_sha256


def test_human_contacts_must_be_source_derived_and_hash_bound():
    human, spec, fk, correspondence = _scene()
    provided = replace(human, contact_origin="provided")
    provided.validate()
    with pytest.raises(ValueError, match="contacts derived"):
        evaluate_semantic_retargeting(provided, spec, fk, correspondence)


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    [
        ("link_positions_m", lambda value: np.where(np.indices(value.shape)[0] == 1, np.nan, value), "finite"),
        ("link_orientations_wxyz", lambda value: value * 2.0, "unit quaternions"),
        ("link_positions_m", lambda value: value[:, :-1], "must have shape"),
        ("robot_spec_sha256", lambda value: "bad", "64-character"),
    ],
)
def test_fk_contract_fails_closed_on_corrupt_exact_buffers(field, mutate, message):
    _, _, fk, _ = _scene()
    with pytest.raises(ValueError, match=message):
        _replace_fk(fk, **{field: mutate(getattr(fk, field))})


def test_public_evaluator_has_no_teacher_or_gmr_target_argument():
    parameters = tuple(inspect.signature(evaluate_semantic_retargeting).parameters)
    assert parameters == ("human_motion", "robot_spec", "robot_fk", "correspondence")
    human, spec, fk, correspondence = _scene()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        evaluate_semantic_retargeting(
            human, spec, fk, correspondence, teacher_target=object()
        )
