from dataclasses import replace

import numpy as np
import pytest
import torch

from snmr.robot_spec import (
    CollisionProxy,
    ControlSpec,
    FrameConvention,
    JointSpec,
    LinkSpec,
    RobotSpec,
    RuntimeSpec,
    SemanticManifest,
)
from snmr.robot_tokens import RobotGraphTokenizer, bounded_joint_positions


def _link(name: str, parent: str | None, xyz: tuple[float, float, float]) -> LinkSpec:
    return LinkSpec(
        name=name,
        parent=parent,
        local_position=xyz,
        local_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        mass=1.0,
        center_of_mass=(0.0, 0.0, 0.0),
        inertia_diagonal=(0.01, 0.01, 0.01),
        inertia_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        collision_proxies=(CollisionProxy(
            geometry_type="sphere",
            local_position=(0.0, 0.0, 0.0),
            local_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            size=(0.05, 0.05, 0.05),
            friction=(1.0, 0.005, 0.0001),
            contact_type=1,
            contact_affinity=1,
        ),),
    )


def _joint(name: str, parent: str, child: str, axis: tuple[float, float, float]) -> JointSpec:
    return JointSpec(
        name=name,
        parent_link=parent,
        child_link=child,
        joint_type="revolute",
        axis=axis,
        lower_limit=-0.8,
        upper_limit=1.2,
        velocity_limit=4.0,
        torque_limit=20.0,
        armature=0.01,
        damping=0.1,
        friction_loss=0.0,
        nominal_position=0.0,
        kp=30.0,
        kd=1.0,
    )


def _spec() -> RobotSpec:
    spec = RobotSpec(
        schema_version="snmr.robot.v0.1",
        asset_sha256="0" * 64,
        frames=FrameConvention(),
        semantics=SemanticManifest(
            root_link="pelvis",
            left_foot_link="left_foot",
            right_foot_link="right_foot",
            allowed_contact_links=("left_foot", "right_foot"),
            symmetry_pairs=(("left_foot", "right_foot"),),
        ),
        control=ControlSpec(control_dt=0.02, latency_seconds=0.0),
        runtime=RuntimeSpec(simulation_dt=0.002),
        total_mass=3.0,
        standing_height=1.0,
        arm_span=None,
        links=(
            _link("pelvis", None, (0.0, 0.0, 0.0)),
            _link("left_foot", "pelvis", (0.0, 0.1, -0.5)),
            _link("right_foot", "pelvis", (0.0, -0.1, -0.5)),
        ),
        joints=(
            _joint("left_joint", "pelvis", "left_foot", (0.0, 1.0, 0.0)),
            _joint("right_joint", "pelvis", "right_foot", (0.0, 1.0, 0.0)),
        ),
    )
    spec.validate()
    return spec


def _permuted(spec: RobotSpec, order: tuple[int, ...]) -> RobotSpec:
    return replace(
        spec,
        links=tuple(spec.links[index] for index in order),
        joints=tuple(reversed(spec.joints)),
    )


def _renamed(spec: RobotSpec) -> RobotSpec:
    mapping = {link.name: f"opaque_{index}" for index, link in enumerate(spec.links)}
    semantics = spec.semantics
    return replace(
        spec,
        links=tuple(replace(
            link,
            name=mapping[link.name],
            parent=None if link.parent is None else mapping[link.parent],
        ) for link in spec.links),
        joints=tuple(replace(
            joint,
            name=f"opaque_joint_{index}",
            parent_link=mapping[joint.parent_link],
            child_link=mapping[joint.child_link],
        ) for index, joint in enumerate(spec.joints)),
        semantics=replace(
            semantics,
            root_link=mapping[semantics.root_link],
            left_foot_link=mapping[semantics.left_foot_link],
            right_foot_link=mapping[semantics.right_foot_link],
            allowed_contact_links=tuple(mapping[name] for name in semantics.allowed_contact_links),
            symmetry_pairs=tuple(
                (mapping[left], mapping[right]) for left, right in semantics.symmetry_pairs
            ),
        ),
    )


def test_joint_permutation_equivariance_of_graph_contract():
    spec = _spec()
    permuted = _permuted(spec, (2, 0, 1))
    tokenizer = RobotGraphTokenizer("kinematic")
    original = tokenizer([spec])
    changed = tokenizer([permuted])

    changed_index = {name: i for i, name in enumerate(changed.node_names[0])}
    inverse = torch.tensor([changed_index[name] for name in original.node_names[0]])
    torch.testing.assert_close(
        original.node_features[0], changed.node_features[0, inverse], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        original.topology_features[0],
        changed.topology_features[0, inverse],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        original.tree_distance[0],
        changed.tree_distance[0][inverse][:, inverse],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(original.joint_mask[0], changed.joint_mask[0, inverse])


def test_variable_dof_shapes_and_padding_are_explicit():
    full = _spec()
    short = replace(
        full,
        semantics=replace(
            full.semantics,
            right_foot_link=None,
            allowed_contact_links=("left_foot",),
            symmetry_pairs=(),
        ),
        total_mass=2.0,
        links=full.links[:2],
        joints=full.joints[:1],
    )
    batch = RobotGraphTokenizer("kinematic")([full, short])
    assert batch.node_features.shape[:2] == (2, 3)
    assert batch.joint_mask.sum(dim=1).tolist() == [2, 1]
    assert batch.node_mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.parent_index[1].tolist() == [-1, 0, -2]
    assert batch.tree_distance[1, 2].tolist() == [-1, -1, -1]
    assert torch.isneginf(batch.attention_bias(0.5)[1, 2]).all()


def test_no_identity_or_serialization_features():
    batch = RobotGraphTokenizer("full")([_spec()])
    assert batch.node_features.shape[-1] == 37
    assert batch.topology_features.shape[-1] == 5
    forbidden = ("name", "path", "hash", "asset", "robot_id", "parent_index")
    assert not any(any(token in field for token in forbidden) for field in batch.feature_names)
    assert batch.dynamics_available.shape[-1] == 4


def test_link_and_joint_renaming_is_not_a_model_feature():
    spec = _spec()
    renamed = _renamed(spec)
    batch = RobotGraphTokenizer("full")([spec, renamed])
    torch.testing.assert_close(batch.node_features[0], batch.node_features[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        batch.topology_features[0], batch.topology_features[1], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(batch.tree_distance[0], batch.tree_distance[1])
    assert batch.node_names[0] != batch.node_names[1]


def test_kinematic_tokens_do_not_change_for_dynamics_twin():
    spec = _spec()
    twin = spec.dynamics_twin(torque_scale=0.5, mass_scale=1.4, latency_seconds=0.03)
    batch = RobotGraphTokenizer("kinematic")([spec, twin])
    torch.testing.assert_close(batch.node_features[0], batch.node_features[1], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(batch.dynamics_available) == 0
    full = RobotGraphTokenizer("full")([spec, twin])
    assert not torch.equal(full.node_features[0], full.node_features[1])


def test_kinematic_batch_hides_dynamics_availability_metadata():
    spec = _spec()
    unavailable = replace(
        spec,
        joints=tuple(
            replace(
                joint,
                torque_limit=None,
                velocity_limit=None,
                kp=None,
                kd=None,
            )
            for joint in spec.joints
        ),
    )
    kinematic = RobotGraphTokenizer("kinematic")([spec, unavailable])
    torch.testing.assert_close(
        kinematic.node_features[0],
        kinematic.node_features[1],
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(kinematic.dynamics_available) == 0

    full = RobotGraphTokenizer("full")([spec, unavailable])
    assert not torch.equal(full.dynamics_available[0], full.dynamics_available[1])


def test_offset_joint_changes_kinematic_hash_and_parent_to_joint_token():
    spec = _spec()
    offset_joint = replace(spec.joints[0], local_position=(0.20, 0.0, 0.0))
    offset = replace(spec, joints=(offset_joint, spec.joints[1]))
    offset.validate()

    assert offset.kinematic_hash != spec.kinematic_hash
    batch = RobotGraphTokenizer("kinematic")([spec, offset])
    left_index = batch.node_names[0].index("left_foot")
    x_field = batch.feature_names.index("parent_to_joint_x_over_height")
    assert batch.node_features[0, left_index, x_field] == pytest.approx(0.0)
    assert batch.node_features[1, left_index, x_field] == pytest.approx(0.20)
    assert batch.node_features.shape[-1] == len(batch.feature_names)

    full = RobotGraphTokenizer("full")([offset])
    assert full.node_features.shape[-1] == 37


def test_robot_spec_rejects_nonunit_joint_duplicate_attachment_and_bad_nominal():
    spec = _spec()
    with pytest.raises(ValueError, match="axis must be unit length"):
        replace(
            spec,
            joints=(replace(spec.joints[0], axis=(0.0, 2.0, 0.0)), spec.joints[1]),
        ).validate()
    with pytest.raises(ValueError, match="more than one scalar joint attachment"):
        replace(
            spec,
            joints=(
                spec.joints[0],
                replace(
                    spec.joints[0],
                    name="duplicate_left_joint",
                ),
                spec.joints[1],
            ),
        ).validate()
    with pytest.raises(ValueError, match="nominal_position lies outside"):
        replace(
            spec,
            joints=(
                replace(spec.joints[0], nominal_position=2.0),
                spec.joints[1],
            ),
        ).validate()


def test_robot_spec_rejects_ambiguous_semantic_sides_and_duplicate_contacts():
    spec = _spec()
    with pytest.raises(ValueError, match="must be distinct"):
        replace(
            spec,
            semantics=replace(spec.semantics, right_foot_link="left_foot"),
        ).validate()
    with pytest.raises(ValueError, match="allowed_contact_links must be unique"):
        replace(
            spec,
            semantics=replace(
                spec.semantics,
                allowed_contact_links=("left_foot", "left_foot"),
            ),
        ).validate()


def test_joint_limit_parameterization_is_bounded_and_node_shared():
    batch = RobotGraphTokenizer("kinematic")([_spec()])
    logits = torch.tensor([[100.0, -100.0, 0.0]])
    values = bounded_joint_positions(
        logits,
        batch.joint_lower,
        batch.joint_upper,
        batch.joint_mask,
    )
    assert values[0, 0] == 0.0
    assert values[0, 1] == pytest.approx(-0.8)
    assert values[0, 2] == pytest.approx(0.2)
    assert torch.all(values[batch.joint_mask] >= batch.joint_lower[batch.joint_mask])
    assert torch.all(values[batch.joint_mask] <= batch.joint_upper[batch.joint_mask])


def test_tree_bias_is_distance_monotone():
    batch = RobotGraphTokenizer("kinematic")([_spec()])
    bias = batch.attention_bias(gamma=0.7)[0]
    assert bias[0, 0] == 0.0
    assert bias[0, 1] == pytest.approx(-0.7)
    assert bias[1, 2] == pytest.approx(-1.4)
    with pytest.raises(ValueError, match="nonnegative"):
        batch.attention_bias(-0.1)
