from dataclasses import replace

import numpy as np
import pytest
import torch

from snmr.robot_spec import (
    ControlSpec,
    FrameConvention,
    RobotSpec,
    SemanticManifest,
    _quat_wxyz_to_rot6d,
)
from snmr.rotation import quat_to_rot6d


def g1_semantics() -> SemanticManifest:
    return SemanticManifest(
        root_link="pelvis",
        torso_link="torso_link",
        left_hand_link="left_rubber_hand_link",
        right_hand_link="right_rubber_hand_link",
        left_foot_link="left_ankle_roll_link",
        right_foot_link="right_ankle_roll_link",
        allowed_contact_links=("left_ankle_roll_link", "right_ankle_roll_link"),
    )


def test_mjcf_robot_spec_round_trip_and_identity_free_features(g1_mjcf):
    spec = RobotSpec.from_mjcf(g1_mjcf, g1_semantics())
    assert len(spec.links) == 50
    assert len(spec.joints) == 29
    assert spec.total_mass > 30.0
    assert spec.standing_height > 0.5
    assert len(spec.asset_sha256) == len(spec.spec_hash) == 64

    restored = RobotSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.spec_hash == spec.spec_hash

    features = spec.model_features()
    assert features.node_features.shape == (50, len(features.node_feature_names))
    assert features.parent_index.shape == (50,)
    assert features.parent_index[0] == -1
    assert np.isfinite(features.node_features).all()
    assert all("name" not in field and "hash" not in field and "path" not in field
               for field in features.node_feature_names)
    # The MJCF supplies effort ranges but no velocity limits or controller gains.
    assert features.dynamics_available.sum(axis=0).tolist() == [29.0, 0.0, 0.0, 0.0]


def test_dynamics_twin_changes_only_dynamics_and_features(g1_mjcf):
    spec = RobotSpec.from_mjcf(
        g1_mjcf,
        g1_semantics(),
        control=ControlSpec(control_dt=0.02, latency_seconds=0.0),
    )
    weak = spec.dynamics_twin(torque_scale=0.5, mass_scale=1.2, latency_seconds=0.03)
    assert weak.kinematic_hash == spec.kinematic_hash
    assert weak.dynamics_hash != spec.dynamics_hash
    assert weak.spec_hash != spec.spec_hash
    assert weak.total_mass == pytest.approx(1.2 * spec.total_mass)
    for original, changed in zip(spec.links, weak.links):
        assert changed.mass == pytest.approx(1.2 * original.mass)
        assert changed.inertia_diagonal == pytest.approx(
            tuple(1.2 * value for value in original.inertia_diagonal)
        )
    original_features = spec.model_features().node_features
    weak_features = weak.model_features().node_features
    assert original_features.shape == weak_features.shape
    assert not np.array_equal(original_features, weak_features)


def test_robot_spec_rejects_unknown_semantic_link(g1_mjcf):
    with pytest.raises(ValueError, match="unknown links"):
        RobotSpec.from_mjcf(
            g1_mjcf,
            SemanticManifest(root_link="pelvis", left_foot_link="not_a_real_link"),
        )


def test_robot_spec_rejects_nonhexadecimal_asset_digest(g1_mjcf):
    spec = RobotSpec.from_mjcf(g1_mjcf, g1_semantics())
    payload = spec.to_dict()
    payload["asset_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="hexadecimal"):
        RobotSpec.from_dict(payload)


def test_dynamics_twin_rejects_nonpositive_scales(g1_mjcf):
    spec = RobotSpec.from_mjcf(g1_mjcf, g1_semantics())
    with pytest.raises(ValueError, match="torque_scale"):
        spec.dynamics_twin(torque_scale=0.0)


def test_robot_spec_rot6d_matches_package_convention():
    quaternions = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.70710678, 0.70710678, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
    ], dtype=torch.float64)
    expected = quat_to_rot6d(quaternions).numpy()
    actual = np.asarray([_quat_wxyz_to_rot6d(row) for row in quaternions.numpy()])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_robot_spec_rejects_noncanonical_frames_and_nonfinite_physics(g1_mjcf):
    spec = RobotSpec.from_mjcf(g1_mjcf, g1_semantics())

    with pytest.raises(ValueError, match="up_axis must be the canonical value"):
        FrameConvention(up_axis="y")
    with pytest.raises(ValueError, match="control.control_dt must be finite"):
        replace(spec, control=replace(spec.control, control_dt=float("nan"))).validate()

    bad_link = replace(spec.links[1], local_position=(float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError, match="local_position.*finite"):
        replace(spec, links=(spec.links[0], bad_link, *spec.links[2:])).validate()

    bad_quaternion = replace(spec.links[1], local_rotation_wxyz=(2.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="unit wxyz quaternion"):
        replace(spec, links=(spec.links[0], bad_quaternion, *spec.links[2:])).validate()


def test_robot_spec_rejects_disconnected_cycle_and_joint_tree_mismatch(g1_mjcf):
    spec = RobotSpec.from_mjcf(g1_mjcf, g1_semantics())
    first = spec.links[1]
    second = spec.links[2]
    cycled_links = list(spec.links)
    cycled_links[1] = replace(first, parent=second.name)
    cycled_links[2] = replace(second, parent=first.name)
    with pytest.raises(ValueError, match="connected acyclic tree"):
        replace(spec, links=tuple(cycled_links)).validate()

    joint = spec.joints[0]
    wrong_parent = next(
        link.name
        for link in spec.links
        if link.name not in {joint.parent_link, joint.child_link}
    )
    with pytest.raises(ValueError, match="inconsistent with the link tree"):
        replace(
            spec,
            joints=(replace(joint, parent_link=wrong_parent), *spec.joints[1:]),
        ).validate()


def test_legacy_joint_payload_defaults_joint_frame_without_silent_invalidity(g1_mjcf):
    spec = RobotSpec.from_mjcf(g1_mjcf, g1_semantics())
    payload = spec.to_dict()
    for joint in payload["joints"]:
        joint.pop("local_position")
        joint.pop("local_rotation_wxyz")

    restored = RobotSpec.from_dict(payload)
    assert all(joint.local_position == (0.0, 0.0, 0.0) for joint in restored.joints)
    assert all(
        joint.local_rotation_wxyz == (1.0, 0.0, 0.0, 0.0)
        for joint in restored.joints
    )
