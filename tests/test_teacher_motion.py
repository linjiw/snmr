from dataclasses import replace

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
    JointSpec,
    LinkSpec,
    RobotSpec,
    RuntimeSpec,
    SemanticManifest,
)
from snmr.teacher_motion import (
    JOINT_POSITION_INTERPOLATION,
    ROOT_ORIENTATION_INTERPOLATION,
    ROOT_POSITION_INTERPOLATION,
    SHORT_ROOT_POSITION_INTERPOLATION,
    resample_teacher_qpos,
)


def _human(frames: int = 4, fps: float = 25.0) -> HumanMotionSpec:
    t = np.arange(frames, dtype=np.float64) / fps
    position = np.zeros((frames, 2, 3), dtype=np.float64)
    position[:, 0, 0] = t**2
    position[:, 0, 2] = 1.0
    position[:, 1] = position[:, 0] + np.array([0.0, 0.0, -1.0])
    orientation = np.zeros((frames, 2, 4), dtype=np.float64)
    orientation[..., 0] = 1.0
    validity = np.ones((frames, 2), dtype=np.bool_)
    return HumanMotionSpec.from_source(
        source=MotionSource("dataset", "sequence", "1" * 64),
        provenance=MotionProvenance(
            motion_id="motion",
            source_subject="subject",
            clip_range=(10, 10 + frames),
            split_id="train",
            transformations=("world_kinematics",),
            preprocessing_version="v1",
        ),
        source_fps=fps,
        body_names=("root", "foot"),
        segment_scales=BodySegmentScales(1.0, 0.5, 0.25, 0.25, 0.2, 0.2),
        root_position=position[:, 0],
        root_orientation_wxyz=orientation[:, 0],
        body_positions=position,
        body_orientations_wxyz=orientation,
        validity_mask=validity,
        contacts=None,
        contact_protocol=FootContactProtocol(
            left_foot_body="foot", right_foot_body="root"
        ),
        frames=HumanFrameConvention(),
    )


def _link(name: str, parent: str | None, z: float) -> LinkSpec:
    return LinkSpec(
        name=name,
        parent=parent,
        local_position=(0.0, 0.0, z),
        local_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        mass=1.0,
        center_of_mass=(0.0, 0.0, 0.0),
        inertia_diagonal=(0.1, 0.1, 0.1),
        inertia_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def _robot() -> RobotSpec:
    spec = RobotSpec(
        schema_version="snmr.robot.v0.1",
        asset_sha256="2" * 64,
        frames=FrameConvention(),
        semantics=SemanticManifest(root_link="pelvis"),
        control=ControlSpec(),
        runtime=RuntimeSpec(simulation_dt=0.002),
        total_mass=2.0,
        standing_height=1.0,
        arm_span=None,
        links=(_link("pelvis", None, 0.0), _link("child", "pelvis", -1.0)),
        joints=(
            JointSpec(
                name="hinge",
                parent_link="pelvis",
                child_link="child",
                joint_type="revolute",
                axis=(0.0, 1.0, 0.0),
                lower_limit=-1.0,
                upper_limit=1.0,
                velocity_limit=4.0,
                torque_limit=10.0,
                armature=0.0,
                damping=0.0,
                friction_loss=0.0,
                nominal_position=0.0,
            ),
        ),
    )
    spec.validate()
    return spec


def _qpos(frames: int) -> np.ndarray:
    time = np.arange(frames, dtype=np.float64)
    qpos = np.zeros((frames, 8), dtype=np.float64)
    qpos[:, :3] = np.stack((time**2, 0.1 * time, 0.5 + 0.01 * time), axis=-1)
    angle = np.linspace(0.0, np.pi / 2.0, frames)
    qpos[:, 3] = np.cos(angle / 2.0)
    qpos[:, 6] = np.sin(angle / 2.0)
    qpos[:, 7] = np.linspace(-0.8, 0.8, frames)
    return qpos


def test_teacher_resampling_binds_exact_human_timeline_and_methods():
    human = _human()
    teacher = resample_teacher_qpos(
        _qpos(4),
        source_fps=25.0,
        human_motion=human,
        robot_spec=_robot(),
        pair_artifact_sha256="3" * 64,
    )

    np.testing.assert_array_equal(teacher.timestamps_s, human.timebase.target_timestamps)
    assert teacher.root_position_interpolation == ROOT_POSITION_INTERPOLATION
    assert teacher.root_position_interpolation_applied == ROOT_POSITION_INTERPOLATION
    assert teacher.root_orientation_interpolation == ROOT_ORIENTATION_INTERPOLATION
    assert teacher.joint_position_interpolation == JOINT_POSITION_INTERPOLATION
    assert teacher.human_motion_spec_sha256 == human.spec_sha256
    assert teacher.robot_spec_sha256 == _robot().spec_hash
    assert teacher.joint_names == ("hinge",)
    assert teacher.joint_positions_rad.min() >= -0.8
    assert teacher.joint_positions_rad.max() <= 0.8
    # Every 25 Hz source sample is also an exact 50 Hz tick.  The registered
    # interpolators must reproduce those knots, including both endpoints.
    source = _qpos(4)
    np.testing.assert_allclose(teacher.root_position_m[::2], source[:, :3], atol=1e-6)
    np.testing.assert_allclose(
        np.abs(np.sum(teacher.root_orientation_wxyz[::2] * source[:, 3:7], axis=-1)),
        1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(teacher.joint_positions_rad[::2], source[:, 7:], atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(teacher.root_orientation_wxyz, axis=-1), 1.0, atol=1e-6
    )
    assert np.isfinite(teacher.root_position_m).all()
    assert np.isfinite(teacher.root_orientation_wxyz).all()
    assert np.isfinite(teacher.joint_positions_rad).all()
    assert len(teacher.buffer_sha256) == 64
    assert not teacher.timestamps_s.flags.writeable
    assert not teacher.joint_positions_rad.flags.writeable


def test_teacher_hash_changes_with_pair_or_robot_binding():
    human = _human()
    robot = _robot()
    first = resample_teacher_qpos(
        _qpos(4),
        source_fps=25.0,
        human_motion=human,
        robot_spec=robot,
        pair_artifact_sha256="3" * 64,
    )
    second = resample_teacher_qpos(
        _qpos(4),
        source_fps=25.0,
        human_motion=human,
        robot_spec=replace(robot, asset_sha256="4" * 64),
        pair_artifact_sha256="5" * 64,
    )
    assert first.buffer_sha256 != second.buffer_sha256


def test_short_sequence_uses_declared_root_fallback_and_linear_joints():
    human = _human(frames=3)
    teacher = resample_teacher_qpos(
        _qpos(3),
        source_fps=25.0,
        human_motion=human,
        robot_spec=_robot(),
        pair_artifact_sha256="3" * 64,
    )
    assert teacher.root_position_interpolation_applied == SHORT_ROOT_POSITION_INTERPOLATION
    np.testing.assert_allclose(teacher.joint_positions_rad[1], [-0.4], atol=1e-6)


def test_teacher_resampling_fails_closed_on_contract_drift():
    human = _human()
    robot = _robot()
    with pytest.raises(ValueError, match="source_fps must match"):
        resample_teacher_qpos(
            _qpos(4),
            source_fps=30.0,
            human_motion=human,
            robot_spec=robot,
            pair_artifact_sha256="3" * 64,
        )
    with pytest.raises(ValueError, match="qpos must have shape"):
        resample_teacher_qpos(
            _qpos(4)[:, :-1],
            source_fps=25.0,
            human_motion=human,
            robot_spec=robot,
            pair_artifact_sha256="3" * 64,
        )
    out_of_limit = _qpos(4)
    out_of_limit[2, 7] = 1.2
    with pytest.raises(ValueError, match="exceed RobotSpec limits"):
        resample_teacher_qpos(
            out_of_limit,
            source_fps=25.0,
            human_motion=human,
            robot_spec=robot,
            pair_artifact_sha256="3" * 64,
        )
    with pytest.raises(ValueError, match="unit wxyz"):
        bad_quaternion = _qpos(4)
        bad_quaternion[:, 3:7] *= 2.0
        resample_teacher_qpos(
            bad_quaternion,
            source_fps=25.0,
            human_motion=human,
            robot_spec=robot,
            pair_artifact_sha256="3" * 64,
        )
