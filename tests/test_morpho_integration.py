from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from snmr.human import lafan1_skeleton, load_pair_npz
from snmr.model import MotionEncoder, SNMRConfig
from snmr.morpho_integration import FixedTargetMorphoRetargeter
from snmr.morpho_model import MorphoRetargetConfig
from snmr.motion_spec import (
    BodySegmentScales,
    FootContactProtocol,
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
from snmr.robot_tokens import RobotGraphTokenizer
from snmr.skeleton import SkeletonGraph


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


def _robot_spec() -> RobotSpec:
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


def _renamed(spec: RobotSpec) -> RobotSpec:
    mapping = {link.name: f"opaque_link_{i}" for i, link in enumerate(spec.links)}
    semantics = spec.semantics
    result = replace(
        spec,
        asset_sha256="f" * 64,
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
                name=f"opaque_joint_{i}",
                parent_link=mapping[joint.parent_link],
                child_link=mapping[joint.child_link],
            )
            for i, joint in enumerate(spec.joints)
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
    result.validate()
    return result


def _human_skeleton() -> SkeletonGraph:
    return SkeletonGraph(
        names=["pelvis", "left_foot", "right_foot"],
        parent_index=torch.tensor([-1, 0, 0], dtype=torch.long),
        rest_offset=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.1, -0.8], [0.0, -0.1, -0.8]]
        ),
        is_end_effector=torch.tensor([False, True, True]),
    )


def _human_tensors(batch: int = 1, time: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    positions = torch.randn(batch, time, 3, 3, generator=generator) * 0.05
    positions[..., 0, 2] += 1.0
    positions[..., 1, 1] += 0.1
    positions[..., 2, 1] -= 0.1
    orientations = torch.zeros(batch, time, 3, 4)
    orientations[..., 0] = 1.0
    return positions, orientations


def _model(tokens, *, temporal: bool = False) -> FixedTargetMorphoRetargeter:
    encoder = SNMRConfig(
        latent_dim=16,
        enc_hidden=16,
        enc_layers=1,
        heads=4,
        use_temporal=temporal,
        temporal_layers=1,
        temporal_heads=4,
    )
    decoder = MorphoRetargetConfig(
        human_token_dim=16,
        hidden_dim=16,
        num_heads=4,
        graph_layers=1,
        feedforward_multiplier=2,
        tree_bias_gamma=0.3,
    )
    return FixedTargetMorphoRetargeter(
        tokens,
        human_encoder_config=encoder,
        morpho_config=decoder,
    )


def _copy_learned_weights(
    source: FixedTargetMorphoRetargeter,
    target: FixedTargetMorphoRetargeter,
) -> None:
    target.human_encoder.load_state_dict(source.human_encoder.state_dict())
    target.joint_decoder.load_state_dict(source.joint_decoder.state_dict())
    target.root_head.load_state_dict(source.root_head.state_dict())


def test_masked_shapes_root_contract_and_gradients() -> None:
    torch.manual_seed(4)
    tokens = RobotGraphTokenizer("kinematic")([_robot_spec()])
    model = _model(tokens)
    positions, orientations = _human_tensors(batch=2, time=5)
    positions[1, 3:] = float("nan")
    orientations[1, 3:] = float("nan")
    positions.requires_grad_()
    time_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    node_mask = torch.ones(2, 3, dtype=torch.bool)

    output = model(
        positions,
        orientations,
        human_skeleton=_human_skeleton(),
        human_time_mask=time_mask,
        human_node_mask=node_mask,
    )

    assert isinstance(model.human_encoder, MotionEncoder)
    assert output.human_tokens.shape == (2, 5, 16)
    assert output.joint_output.joint_positions.shape == (2, 5, 3)
    assert output.root_position_local.shape == (2, 5, 3)
    assert output.root_orientation_local_wxyz.shape == (2, 5, 4)
    assert output.joint_output.valid_joint_mask.sum(dim=(1, 2)).tolist() == [10, 6]
    assert torch.count_nonzero(output.human_tokens[1, 3:]) == 0
    assert torch.count_nonzero(output.root_position_local[1, 3:]) == 0
    torch.testing.assert_close(
        output.root_orientation_local_wxyz[1, 3:],
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(output.root_orientation_local_wxyz, dim=-1),
        torch.ones(2, 5),
        rtol=0.0,
        atol=2.0e-6,
    )
    assert output.robot_token_audit_manifest == tokens.audit_manifest()

    valid_joints = output.joint_output.valid_joint_mask
    loss = output.joint_output.joint_positions[valid_joints].square().mean()
    loss = loss + output.root_position_local[time_mask].square().mean()
    loss = loss + output.root_orientation_local_wxyz[time_mask, 1].mean()
    loss.backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad[time_mask]).all()
    assert torch.count_nonzero(positions.grad[time_mask]) > 0
    for parameter in (
        model.human_encoder.input_proj.weight,
        model.joint_decoder.joint_head[-1].weight,
        model.root_head.network[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    root_gradient = model.root_head.network[-1].weight.grad
    assert torch.count_nonzero(root_gradient[:3]) > 0
    assert torch.count_nonzero(root_gradient[3:]) > 0

    with pytest.raises(NotImplementedError, match="derive them from predicted robot FK"):
        model(
            positions.detach(),
            orientations,
            human_skeleton=_human_skeleton(),
            human_time_mask=time_mask,
            human_node_mask=node_mask,
            require_contacts=True,
        )


def test_canonical_motion_spec_path_is_hash_bound_and_matches_tensor_path() -> None:
    skeleton = _human_skeleton()
    positions, orientations = _human_tensors(time=4)
    positions_np = positions[0].numpy()
    orientations_np = orientations[0].numpy()
    motion = HumanMotionSpec.from_source(
        source=MotionSource("synthetic", "clip-1", "a" * 64),
        provenance=MotionProvenance(
            motion_id="synthetic/clip-1",
            source_subject=None,
            clip_range=(0, 4),
            split_id="unit-test",
            transformations=("synthetic_fixture",),
            preprocessing_version="test.v1",
        ),
        source_fps=50.0,
        body_names=skeleton.names,
        segment_scales=BodySegmentScales(
            normalization_length_m=1.7,
            torso=0.3,
            thigh=0.25,
            shin=0.25,
            upper_arm=0.18,
            forearm=0.16,
        ),
        root_position=positions_np[:, 0],
        root_orientation_wxyz=orientations_np[:, 0],
        body_positions=positions_np,
        body_orientations_wxyz=orientations_np,
        validity_mask=np.ones((4, 3), dtype=np.bool_),
        contacts=np.zeros((4, 2)),
        contact_protocol=FootContactProtocol(
            left_foot_body="left_foot", right_foot_body="right_foot"
        ),
    )
    tokens = RobotGraphTokenizer("kinematic")([_robot_spec()])
    model = _model(tokens).eval()
    with torch.no_grad():
        canonical = model.forward_motion_spec(motion, human_skeleton=skeleton)
        direct = model(
            positions,
            orientations,
            human_skeleton=skeleton,
            human_time_mask=torch.ones(1, 4, dtype=torch.bool),
            human_node_mask=torch.ones(1, 3, dtype=torch.bool),
        )
    assert canonical.human_motion_spec_sha256 == motion.spec_sha256
    assert canonical.human_motion_buffer_sha256 == motion.buffer_sha256
    torch.testing.assert_close(
        canonical.joint_output.joint_positions,
        direct.joint_output.joint_positions,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        canonical.root_position_local, direct.root_position_local, rtol=0.0, atol=0.0
    )


def test_serialization_rename_and_dynamics_do_not_change_learned_outputs() -> None:
    spec = _robot_spec()
    permuted = replace(
        spec,
        links=(spec.links[2], spec.links[0], spec.links[1]),
        joints=tuple(reversed(spec.joints)),
    )
    renamed = _renamed(spec)
    dynamics = spec.dynamics_twin(torque_scale=0.5, latency_seconds=0.01)
    token_sets = [
        RobotGraphTokenizer("kinematic")([candidate])
        for candidate in (spec, permuted, renamed, dynamics)
    ]
    assert torch.count_nonzero(token_sets[0].dynamics_available) == 0
    torch.testing.assert_close(
        token_sets[0].node_features,
        token_sets[3].node_features,
        rtol=0.0,
        atol=0.0,
    )

    torch.manual_seed(12)
    baseline_model = _model(token_sets[0]).eval()
    variants = [_model(tokens).eval() for tokens in token_sets[1:]]
    for variant in variants:
        _copy_learned_weights(baseline_model, variant)
    positions, orientations = _human_tensors(time=4)
    kwargs = {
        "human_skeleton": _human_skeleton(),
        "human_time_mask": torch.ones(1, 4, dtype=torch.bool),
        "human_node_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    with torch.no_grad():
        baseline = baseline_model(positions, orientations, **kwargs)
        outputs = [model(positions, orientations, **kwargs) for model in variants]

    permuted_names = token_sets[1].node_names[0]
    inverse = torch.tensor(
        [permuted_names.index(name) for name in token_sets[0].node_names[0]]
    )
    torch.testing.assert_close(
        baseline.joint_output.joint_positions,
        outputs[0].joint_output.joint_positions[:, :, inverse],
        rtol=1.0e-6,
        atol=2.0e-6,
    )
    for output in outputs:
        torch.testing.assert_close(
            baseline.root_position_local, output.root_position_local, rtol=1e-6, atol=2e-6
        )
        torch.testing.assert_close(
            baseline.root_orientation_local_wxyz,
            output.root_orientation_local_wxyz,
            rtol=1e-6,
            atol=2e-6,
        )
    for output in outputs[1:]:
        torch.testing.assert_close(
            baseline.joint_output.joint_positions,
            output.joint_output.joint_positions,
            rtol=1e-6,
            atol=2e-6,
        )
    assert (
        baseline.robot_token_audit_manifest["manifest_sha256"]
        != outputs[1].robot_token_audit_manifest["manifest_sha256"]
    )
    assert all(
        forbidden not in name.lower()
        for name, _ in baseline_model.named_parameters()
        for forbidden in ("robot_id", "identity", "prompt", "name_embedding")
    )


def test_checkpoint_state_dict_round_trip_is_exact_and_audit_bound() -> None:
    tokens = RobotGraphTokenizer("kinematic")([_robot_spec()])
    torch.manual_seed(23)
    original = _model(tokens).eval()
    positions, orientations = _human_tensors(time=4)
    kwargs = {
        "human_skeleton": _human_skeleton(),
        "human_time_mask": torch.ones(1, 4, dtype=torch.bool),
        "human_node_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    with torch.no_grad():
        expected = original(positions, orientations, **kwargs)
    checkpoint = copy.deepcopy(original.state_dict())
    assert "_extra_state" in checkpoint
    assert "target_binding._extra_state" in checkpoint

    torch.manual_seed(999)
    restored = _model(tokens).eval()
    restored.load_state_dict(checkpoint, strict=True)
    with torch.no_grad():
        actual = restored(positions, orientations, **kwargs)
    torch.testing.assert_close(
        actual.joint_output.joint_positions,
        expected.joint_output.joint_positions,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual.root_position_local, expected.root_position_local, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actual.root_orientation_local_wxyz,
        expected.root_orientation_local_wxyz,
        rtol=0.0,
        atol=0.0,
    )
    assert actual.robot_token_audit_manifest == expected.robot_token_audit_manifest

    mismatched_config = SNMRConfig(
        latent_dim=16,
        enc_hidden=16,
        enc_layers=1,
        heads=4,
        use_temporal=False,
        temporal_layers=1,
        temporal_heads=4,
        temporal_positional=True,
    )
    mismatch = FixedTargetMorphoRetargeter(
        tokens,
        human_encoder_config=mismatched_config,
        morpho_config=MorphoRetargetConfig(
            human_token_dim=16,
            hidden_dim=16,
            num_heads=4,
            graph_layers=1,
            feedforward_multiplier=2,
            tree_bias_gamma=0.3,
        ),
    )
    with pytest.raises(ValueError, match="config/output convention"):
        mismatch.load_state_dict(checkpoint, strict=True)

    restored.target_binding.node_features[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="changed after construction/load"):
        restored(positions, orientations, **kwargs)


def _g1_semantics() -> SemanticManifest:
    return SemanticManifest(
        root_link="pelvis",
        torso_link="torso_link",
        left_hand_link="left_rubber_hand_link",
        right_hand_link="right_rubber_hand_link",
        left_foot_link="left_ankle_roll_link",
        right_foot_link="right_ankle_roll_link",
        allowed_contact_links=("left_ankle_roll_link", "right_ankle_roll_link"),
    )


def test_real_fixed_g1_forward_uses_29_joint_nodes(g1_mjcf, g1_pair_npz) -> None:
    spec = RobotSpec.from_mjcf(g1_mjcf, _g1_semantics())
    tokens = RobotGraphTokenizer("kinematic")([spec])
    pair = load_pair_npz(g1_pair_npz)
    frames = 4
    model = _model(tokens, temporal=True).eval()
    with torch.no_grad():
        output = model(
            pair["human_pos"][:frames],
            pair["human_quat"][:frames],
            human_skeleton=lafan1_skeleton(),
            human_time_mask=torch.ones(frames, dtype=torch.bool),
            human_node_mask=torch.ones(24, dtype=torch.bool),
        )
    assert output.joint_output.joint_positions.shape == (1, frames, 50)
    assert output.joint_output.valid_joint_mask.sum().item() == frames * 29
    assert output.root_position_local.shape == (1, frames, 3)
    torch.testing.assert_close(
        torch.linalg.vector_norm(output.root_orientation_local_wxyz, dim=-1),
        torch.ones(1, frames),
        rtol=0.0,
        atol=2.0e-6,
    )
    assert output.robot_token_audit_manifest == tokens.audit_manifest()
