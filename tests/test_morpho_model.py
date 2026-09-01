from dataclasses import replace

import pytest
import torch

from snmr.morpho_model import KinematicMorphoRetargeter, MorphoRetargetConfig
from snmr.robot_spec import (
    ControlSpec,
    FrameConvention,
    JointSpec,
    LinkSpec,
    RobotSpec,
    RuntimeSpec,
    SemanticManifest,
)
from snmr.robot_tokens import RobotGraphTokenizer


def _link(name: str, parent: str | None, position: tuple[float, float, float]) -> LinkSpec:
    return LinkSpec(
        name=name,
        parent=parent,
        local_position=position,
        local_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        mass=1.0,
        center_of_mass=(0.0, 0.0, 0.0),
        inertia_diagonal=(0.01, 0.01, 0.01),
        inertia_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def _joint(name: str, child: str, axis: tuple[float, float, float]) -> JointSpec:
    return JointSpec(
        name=name,
        parent_link="pelvis",
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


def _full_spec() -> RobotSpec:
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
            _joint("left_joint", "left_foot", (0.0, 1.0, 0.0)),
            _joint("right_joint", "right_foot", (1.0, 0.0, 0.0)),
        ),
    )
    spec.validate()
    return spec


def _short_spec(spec: RobotSpec) -> RobotSpec:
    short = replace(
        spec,
        semantics=replace(
            spec.semantics,
            right_foot_link=None,
            allowed_contact_links=("left_foot",),
            symmetry_pairs=(),
        ),
        total_mass=2.0,
        links=spec.links[:2],
        joints=spec.joints[:1],
    )
    short.validate()
    return short


def _renamed(spec: RobotSpec) -> RobotSpec:
    mapping = {link.name: f"opaque_{index}" for index, link in enumerate(spec.links)}
    semantics = spec.semantics
    renamed = replace(
        spec,
        links=tuple(
            replace(
                link,
                name=mapping[link.name],
                parent=None if link.parent is None else mapping[link.parent],
            )
            for link in spec.links
        ),
        joints=tuple(
            replace(
                joint,
                name=f"opaque_joint_{index}",
                parent_link=mapping[joint.parent_link],
                child_link=mapping[joint.child_link],
            )
            for index, joint in enumerate(spec.joints)
        ),
        semantics=replace(
            semantics,
            root_link=mapping[semantics.root_link],
            left_foot_link=mapping[semantics.left_foot_link],
            right_foot_link=mapping[semantics.right_foot_link],
            allowed_contact_links=tuple(
                mapping[name] for name in semantics.allowed_contact_links
            ),
            symmetry_pairs=tuple(
                (mapping[left], mapping[right])
                for left, right in semantics.symmetry_pairs
            ),
        ),
    )
    renamed.validate()
    return renamed


def _model() -> KinematicMorphoRetargeter:
    return KinematicMorphoRetargeter(
        MorphoRetargetConfig(
            human_token_dim=12,
            hidden_dim=32,
            num_heads=4,
            graph_layers=2,
            feedforward_multiplier=2,
            tree_bias_gamma=0.4,
        )
    )


def test_shapes_gradients_variable_dof_padding_and_joint_limits():
    torch.manual_seed(7)
    full = _full_spec()
    robot_tokens = RobotGraphTokenizer("kinematic")([full, _short_spec(full)])
    human = torch.randn(2, 5, 12, requires_grad=True)
    human_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    model = _model()

    output = model(human, robot_tokens, human_mask)

    assert output.joint_positions.shape == (2, 5, 3)
    assert output.joint_logits.shape == (2, 5, 3)
    assert output.robot_embeddings.shape == (2, 3, 32)
    assert output.valid_joint_mask.shape == (2, 5, 3)
    assert output.valid_joint_mask.sum(dim=(1, 2)).tolist() == [10, 3]
    assert torch.count_nonzero(output.joint_positions[~output.valid_joint_mask]) == 0
    assert torch.count_nonzero(output.robot_embeddings[1, 2]) == 0
    assert torch.count_nonzero(output.joint_logits[1, 3:]) == 0

    lower = robot_tokens.joint_lower[:, None, :].expand_as(output.joint_positions)
    upper = robot_tokens.joint_upper[:, None, :].expand_as(output.joint_positions)
    valid_positions = output.joint_positions[output.valid_joint_mask]
    assert torch.all(valid_positions >= lower[output.valid_joint_mask])
    assert torch.all(valid_positions <= upper[output.valid_joint_mask])

    loss = valid_positions.square().mean()
    loss.backward()
    assert human.grad is not None
    assert torch.isfinite(human.grad).all()
    assert torch.count_nonzero(human.grad) > 0
    for parameter in (
        model.robot_input[0].weight,
        model.graph_layers[0].attention.qkv.weight,
        model.cross_attention.key.weight,
        model.joint_head[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_serialization_permutation_equivariance_after_inverse_permutation():
    torch.manual_seed(11)
    spec = _full_spec()
    permuted = replace(
        spec,
        links=(spec.links[2], spec.links[0], spec.links[1]),
        joints=tuple(reversed(spec.joints)),
    )
    original_tokens = RobotGraphTokenizer("kinematic")([spec])
    permuted_tokens = RobotGraphTokenizer("kinematic")([permuted])
    changed_index = {
        name: index for index, name in enumerate(permuted_tokens.node_names[0])
    }
    inverse = torch.tensor(
        [changed_index[name] for name in original_tokens.node_names[0]],
        dtype=torch.long,
    )
    human = torch.randn(1, 4, 12)
    model = _model().eval()

    with torch.no_grad():
        original = model(human, original_tokens)
        changed = model(human, permuted_tokens)

    torch.testing.assert_close(
        original.joint_positions,
        changed.joint_positions[:, :, inverse],
        rtol=1e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        original.joint_logits,
        changed.joint_logits[:, :, inverse],
        rtol=1e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        original.robot_embeddings,
        changed.robot_embeddings[:, inverse],
        rtol=1e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        original.valid_joint_mask,
        changed.valid_joint_mask[:, :, inverse],
    )


def test_learned_output_is_invariant_to_opaque_link_and_joint_renaming():
    torch.manual_seed(13)
    spec = _full_spec()
    tokens = RobotGraphTokenizer("kinematic")([spec, _renamed(spec)])
    human_single = torch.randn(1, 4, 12)
    human = human_single.expand(2, -1, -1).clone()
    model = _model().eval()
    with torch.no_grad():
        output = model(human, tokens)
    torch.testing.assert_close(
        output.joint_positions[0], output.joint_positions[1], rtol=1.0e-6, atol=2.0e-7
    )
    torch.testing.assert_close(
        output.robot_embeddings[0], output.robot_embeddings[1], rtol=1.0e-6, atol=2.0e-7
    )


def test_counterfactual_kinematic_change_reaches_learned_output():
    torch.manual_seed(17)
    spec = _full_spec()
    changed = replace(
        spec,
        links=(
            spec.links[0],
            replace(spec.links[1], local_position=(0.2, 0.1, -0.8)),
            spec.links[2],
        ),
    )
    changed.validate()
    tokens = RobotGraphTokenizer("kinematic")([spec, changed])
    human_single = torch.randn(1, 5, 12)
    human = human_single.expand(2, -1, -1).clone()
    model = _model().eval()
    with torch.no_grad():
        output = model(human, tokens)
    assert not torch.allclose(output.robot_embeddings[0], output.robot_embeddings[1])
    assert not torch.allclose(output.joint_positions[0], output.joint_positions[1])


def test_kinematic_model_ignores_dynamics_interventions_and_availability_metadata():
    torch.manual_seed(19)
    spec = _full_spec()
    twin = spec.dynamics_twin(
        torque_scale=0.5,
        mass_scale=1.4,
        latency_seconds=0.03,
    )
    tokens = RobotGraphTokenizer("kinematic")([spec, twin])
    human_single = torch.randn(1, 3, 12)
    human = human_single.expand(2, -1, -1).clone()
    model = _model().eval()

    with torch.no_grad():
        output = model(human, tokens)
        tampered = model(
            human,
            replace(
                tokens,
                dynamics_available=torch.randn_like(tokens.dynamics_available),
            ),
        )

    torch.testing.assert_close(
        output.joint_positions[0], output.joint_positions[1], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        output.joint_logits[0], output.joint_logits[1], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        output.joint_positions, tampered.joint_positions, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        output.joint_logits, tampered.joint_logits, rtol=0.0, atol=0.0
    )


def test_full_robot_tokens_are_rejected_in_kinematic_experiment():
    model = _model()
    human = torch.randn(1, 2, 12)
    with pytest.raises(ValueError, match="feature_set='kinematic'"):
        model(human, RobotGraphTokenizer("full")([_full_spec()]))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda tokens: replace(
                tokens,
                joint_lower=tokens.joint_lower.masked_fill(
                    tokens.joint_mask, float("nan")
                ),
            ),
            "finite",
        ),
        (
            lambda tokens: replace(
                tokens,
                joint_upper=tokens.joint_upper.masked_fill(
                    tokens.joint_mask, float("inf")
                ),
            ),
            "finite",
        ),
        (
            lambda tokens: replace(
                tokens,
                tree_distance=tokens.tree_distance.clone().index_put_(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([1])),
                    torch.tensor([7]),
                ),
            ),
            "exactly match parent_index",
        ),
        (
            lambda tokens: replace(
                tokens,
                root_mask=torch.zeros_like(tokens.root_mask),
            ),
            "exactly one root",
        ),
        (
            lambda tokens: replace(
                tokens,
                joint_mask=torch.zeros_like(tokens.joint_mask),
            ),
            "at least one scalar joint",
        ),
    ],
)
def test_model_fails_closed_on_tampered_robot_token_contract(mutate, message):
    tokens = RobotGraphTokenizer("kinematic")([_full_spec()])
    with pytest.raises(ValueError, match=message):
        _model()(torch.randn(1, 2, 12), mutate(tokens))


def test_same_robot_output_is_invariant_to_padded_cobatch_context():
    torch.manual_seed(23)
    spec = _full_spec()
    short = _short_spec(spec)
    model = _model().eval()
    human = torch.randn(1, 4, 12)
    with torch.no_grad():
        alone = model(human, RobotGraphTokenizer("kinematic")([spec]))
        together = model(
            human.expand(2, -1, -1).clone(),
            RobotGraphTokenizer("kinematic")([spec, short]),
        )
    torch.testing.assert_close(
        alone.joint_positions[0], together.joint_positions[0], rtol=1e-6, atol=2e-6
    )
    torch.testing.assert_close(
        alone.robot_embeddings[0], together.robot_embeddings[0], rtol=1e-6, atol=2e-6
    )
