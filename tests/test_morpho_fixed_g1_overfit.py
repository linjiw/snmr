from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.experiment_morpho_fixed_g1_overfit import (
    ARTIFACT_SCHEMA_VERSION,
    G1_HUMAN_ROOT_XY_SCALE,
    REGISTERED_PAIR_SHA256,
    REGISTERED_PARAMETER_COUNT,
    REGISTERED_RAW_SOURCE_SHA256,
    REGISTERED_MOTION_ID,
    REGISTERED_SOURCE_DATASET,
    REGISTERED_SOURCE_SEQUENCE_ID,
    REGISTERED_SOURCE_SUBJECT,
    REGISTERED_SPLIT_ID,
    FixedG1Protocol,
    _g1_semantics,
    _model_configs,
    run_overfit_experiment,
    validate_node_joint_mapping,
    write_artifacts_once,
)
from snmr import rotation as rot
from snmr.motion_spec import (
    BodySegmentScales,
    FootContactProtocol,
    HumanMotionSpec,
    MotionProvenance,
    MotionSource,
)
from snmr.morpho_integration import FixedTargetMorphoRetargeter
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
from snmr.teacher_motion import CanonicalTeacherMotion, ROOT_POSITION_INTERPOLATION


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
            _joint("right_joint", "right_foot", (0.0, 1.0, 0.0)),
        ),
    )
    spec.validate()
    return spec


def _human_skeleton() -> SkeletonGraph:
    return SkeletonGraph(
        names=["pelvis", "left_foot", "right_foot"],
        parent_index=torch.tensor([-1, 0, 0], dtype=torch.long),
        rest_offset=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.1, -0.8], [0.0, -0.1, -0.8]]
        ),
        is_end_effector=torch.tensor([False, True, True]),
    )


class _SyntheticKinematics:
    def __init__(self) -> None:
        self.num_dof = 2
        self.body_names = ["pelvis", "left_foot", "right_foot"]
        self.graph = SimpleNamespace(
            dof_body_index=torch.tensor([1, 2], dtype=torch.long)
        )

    def dof_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([-0.8, -0.8]), torch.tensor([1.2, 1.2])

    def forward_kinematics(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        dof_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left = torch.stack(
            (
                0.5 * torch.sin(dof_pos[..., 0]),
                torch.full_like(dof_pos[..., 0], 0.1),
                -0.5 * torch.cos(dof_pos[..., 0]),
            ),
            dim=-1,
        )
        right = torch.stack(
            (
                0.5 * torch.sin(dof_pos[..., 1]),
                torch.full_like(dof_pos[..., 1], -0.1),
                -0.5 * torch.cos(dof_pos[..., 1]),
            ),
            dim=-1,
        )
        offsets = torch.stack((torch.zeros_like(left), left, right), dim=-2)
        expanded_quat = root_quat.unsqueeze(-2).expand(offsets.shape[:-1] + (4,))
        positions = root_pos.unsqueeze(-2) + rot.quat_rotate(expanded_quat, offsets)
        orientations = expanded_quat.clone()
        return positions, orientations


def _motion_and_teacher() -> tuple[HumanMotionSpec, CanonicalTeacherMotion]:
    frames = 8
    t = np.arange(frames, dtype=np.float64) / 50.0
    root = np.stack((0.03 * t, -0.01 * t, 0.9 + 0.01 * t), axis=-1)
    offsets = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.1, -0.8], [0.0, -0.1, -0.8]],
        dtype=np.float64,
    )
    bodies = root[:, None, :] + offsets[None, :, :]
    quaternions = np.zeros((frames, 3, 4), dtype=np.float64)
    quaternions[..., 0] = 1.0
    motion = HumanMotionSpec.from_source(
        source=MotionSource("synthetic", "smoke", "1" * 64),
        provenance=MotionProvenance(
            motion_id="synthetic/smoke",
            source_subject=None,
            clip_range=(0, frames),
            split_id="smoke-only",
            transformations=("synthetic_fixture",),
            preprocessing_version="smoke.v1",
        ),
        source_fps=50.0,
        body_names=("pelvis", "left_foot", "right_foot"),
        segment_scales=BodySegmentScales(
            normalization_length_m=1.7,
            torso=0.3,
            thigh=0.25,
            shin=0.25,
            upper_arm=0.18,
            forearm=0.16,
        ),
        root_position=root,
        root_orientation_wxyz=quaternions[:, 0],
        body_positions=bodies,
        body_orientations_wxyz=quaternions,
        validity_mask=np.ones((frames, 3), dtype=np.bool_),
        contacts=np.zeros((frames, 2), dtype=np.float64),
        contact_protocol=FootContactProtocol(
            left_foot_body="left_foot", right_foot_body="right_foot"
        ),
    )
    joints = np.stack(
        (0.1 + 0.02 * np.sin(4.0 * t), -0.2 + 0.03 * np.cos(3.0 * t)),
        axis=-1,
    )
    teacher = CanonicalTeacherMotion(
        timestamps_s=motion.timebase.target_timestamps,
        root_position_m=root,
        root_orientation_wxyz=quaternions[:, 0],
        joint_positions_rad=joints,
        joint_names=("left_joint", "right_joint"),
        human_motion_spec_sha256=motion.spec_sha256,
        robot_spec_sha256=_robot_spec().spec_hash,
        robot_asset_sha256=_robot_spec().asset_sha256,
        pair_artifact_sha256="2" * 64,
        source_fps=50.0,
        root_position_interpolation_applied=ROOT_POSITION_INTERPOLATION,
    )
    return motion, teacher


def test_registered_protocol_is_frozen_and_smoke_is_never_eligible() -> None:
    registered = FixedG1Protocol()
    assert registered.steps == 1_000
    assert registered.window_frames == 128
    assert registered.target_start_frame == 0
    assert registered.human_root_xy_scale == G1_HUMAN_ROOT_XY_SCALE
    assert registered.registered_gate_eligible
    assert len(REGISTERED_PAIR_SHA256) == 64
    assert len(REGISTERED_RAW_SOURCE_SHA256) == 64
    assert (
        REGISTERED_SOURCE_DATASET,
        REGISTERED_SOURCE_SEQUENCE_ID,
        REGISTERED_MOTION_ID,
        REGISTERED_SOURCE_SUBJECT,
        REGISTERED_SPLIT_ID,
    ) == (
        "lafan1",
        "walk1_subject1",
        "lafan1/walk1_subject1",
        "subject1",
        "train",
    )
    human_config, morpho_config = _model_configs(False)
    model = FixedTargetMorphoRetargeter(
        RobotGraphTokenizer("kinematic")([_robot_spec()]),
        human_encoder_config=human_config,
        morpho_config=morpho_config,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_583_754
    assert REGISTERED_PARAMETER_COUNT == 1_583_754
    assert human_config.latent_dim == 128
    assert human_config.enc_hidden == 256
    assert human_config.enc_layers == 4
    assert human_config.temporal_positional is True
    assert morpho_config.hidden_dim == 128
    assert morpho_config.graph_layers == 3
    assert morpho_config.feedforward_multiplier == 4
    assert morpho_config.tree_bias_gamma == 0.5

    smoke = FixedG1Protocol.smoke_protocol()
    assert smoke.steps == 2 and smoke.window_frames == 8
    assert not smoke.registered_gate_eligible
    with pytest.raises(ValueError, match="frozen registered protocol"):
        FixedG1Protocol(steps=999)
    with pytest.raises(ValueError, match="at most 16 frames"):
        replace(smoke, window_frames=17)


def test_mapping_uses_joint_child_links_under_node_permutation() -> None:
    spec = _robot_spec()
    permuted = replace(
        spec,
        links=(spec.links[2], spec.links[0], spec.links[1]),
    )
    tokens = RobotGraphTokenizer("kinematic")([permuted])
    indices = validate_node_joint_mapping(
        permuted,
        tokens,
        ("left_joint", "right_joint"),
        _SyntheticKinematics(),
    )
    assert indices == (2, 0)

    with pytest.raises(ValueError, match="teacher joint order"):
        validate_node_joint_mapping(
            permuted,
            tokens,
            ("right_joint", "left_joint"),
            _SyntheticKinematics(),
        )


def test_real_g1_binding_matches_preregistered_hashes(g1_mjcf: str) -> None:
    spec = RobotSpec.from_mjcf(g1_mjcf, _g1_semantics())
    tokens = RobotGraphTokenizer("kinematic")([spec])
    assert spec.spec_hash == "4bf54c436b3338bd4f6a2ed72b4b11ba6680e63fe7b3fe8aa416c3f615d054cc"
    assert spec.kinematic_hash == "e4997d0e5d9b48bf82608273788f08e0f35b6aa8dcd5abac346e1aa6a2226a3c"
    assert tokens.model_buffer_sha256() == (
        "e0965c00ff5e57186d4ad72e5392acb68319447079708dba36da5987a8496486"
    )
    assert tokens.audit_manifest()["manifest_sha256"] == (
        "e33459b2d1ca122ec9bdbefe9d28097bca9b25d712c351ce2102c80430daa2f6"
    )


def test_synthetic_smoke_trains_learned_root_and_writes_once(tmp_path: Path) -> None:
    motion, teacher = _motion_and_teacher()
    artifacts = run_overfit_experiment(
        human_motion=motion,
        teacher_motion=teacher,
        robot_spec=_robot_spec(),
        robot_kinematics=_SyntheticKinematics(),
        human_skeleton=_human_skeleton(),
        protocol=FixedG1Protocol.smoke_protocol(),
        input_manifest={"fixture": "synthetic-memory-buffer"},
    )
    report = artifacts.report
    assert report["gate_decision"] == "SMOKE_NOT_GATE_ELIGIBLE"
    assert report["registered_gate_eligible"] is False
    assert report["canonical_contract"]["teacher_timestamps_exactly_equal_human"]
    assert report["canonical_contract"]["learned_root_never_teacher_forced"]
    assert report["canonical_contract"]["window_target_indices"] == [0, 8]
    assert len(report["canonical_contract"]["exact_window_buffer_sha256"]) == 64
    assert report["model"]["initial_state_dict_sha256"] != report["model"][
        "final_state_dict_sha256"
    ]
    assert len(report["optimization"]["loss_history"]) == 2
    expected_metrics = {
        "total_loss",
        "teacher_fk_mpjpe_m",
        "dof_mae_rad",
        "root_position_mae_m",
        "root_orientation_geodesic_rad",
        "joint_limit_violations",
    }
    assert expected_metrics.issubset(report["metrics_before"])
    assert expected_metrics.issubset(report["metrics_after"])
    assert set(report["gates"]) == {
        "last20_over_first20_loss_drop_ge_75pct",
        "teacher_fk_mpjpe_lt_5cm",
        "dof_mae_lt_0p10rad",
        "root_position_mae_lt_5cm",
        "root_geodesic_lt_0p15rad",
        "zero_joint_limit_violations",
        "serialization_equivariance_le_1e_5",
        "finite_loss_and_nonzero_gradients_all_named_blocks",
    }
    assert set(report["gradient_audit"]) == {
        "human_encoder",
        "robot_graph_encoder",
        "shared_joint_head",
        "root_head",
    }
    assert all(report["gradient_block_gates"].values())

    output = tmp_path / "smoke-result"
    manifest = write_artifacts_once(output, artifacts)
    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["write_once"] is True
    assert set(manifest["files"]) == {"report.json", "config.json", "checkpoint.pt"}
    saved = json.loads((output / "report.json").read_text())
    assert saved["config_sha256"] == report["config_sha256"]
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=False)
    assert checkpoint["final_state_dict_sha256"] == report["model"][
        "final_state_dict_sha256"
    ]
    with pytest.raises(FileExistsError, match="write-once"):
        write_artifacts_once(output, artifacts)
