#!/usr/bin/env python
"""Preregistered fixed-G1 MorphoRetarget amortisation experiment.

This is the smallest learned-model gate after the kinematic model/integration contracts:
one explicitly identified LAFAN1/GMR pair, one canonical 50 Hz window, and one G1.  It
is deliberately an overfit experiment.  Passing it only establishes that the new
variable-DoF decoder and its learned root head can amortise one teacher trajectory; it
does not establish cross-morphology generalisation.

Registered invocation properties are frozen in :class:`FixedG1Protocol`: the complete
``walk1_subject1`` source clip, target frames ``[0, 128)``, seed 0, and 1,000 AdamW updates.  ``--smoke``
uses eight target frames and two updates and is always labelled ineligible for the gate.
The 1,000-step run is intentionally not launched by this module's tests.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snmr import rotation as rot  # noqa: E402
from snmr.data import local_root_to_world, world_root_to_local  # noqa: E402
from snmr.experiment import (  # noqa: E402
    git_state,
    runtime_state,
    sha256_file,
    source_fingerprint,
    utc_now,
)
from snmr.human import (  # noqa: E402
    LAFAN1_BODY_NAMES,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)
from snmr.model import SNMRConfig  # noqa: E402
from snmr.morpho_integration import FixedTargetMorphoRetargeter  # noqa: E402
from snmr.morpho_model import MorphoRetargetConfig  # noqa: E402
from snmr.motion_adapter import adapt_lafan1_pair_npz  # noqa: E402
from snmr.motion_spec import (  # noqa: E402
    BilateralSegmentLandmarks,
    FootContactProtocol,
    HumanFrameConvention,
    HumanMotionSpec,
    MotionProvenance,
    MotionSource,
    TARGET_FPS,
)
from snmr.provenance import (  # noqa: E402
    ArtifactSnapshot,
    MjcfBundleSnapshot,
    source_revision_manifest,
)
from snmr.robot_model import RobotKinematics  # noqa: E402
from snmr.robot_spec import RobotSpec, SemanticManifest  # noqa: E402
from snmr.robot_tokens import RobotGraphTokenizer, RobotTokenBatch  # noqa: E402
from snmr.skeleton import SkeletonGraph  # noqa: E402
from snmr.teacher_motion import (  # noqa: E402
    CanonicalTeacherMotion,
    resample_teacher_qpos,
)


SCHEMA_VERSION = "snmr.morpho-fixed-g1-overfit.v0.1"
ARTIFACT_SCHEMA_VERSION = "snmr.morpho-fixed-g1-overfit-artifacts.v0.1"
REGISTERED_TARGET_START = 0
REGISTERED_WINDOW_FRAMES = 128
REGISTERED_STEPS = 1_000
REGISTERED_SEED = 0
REGISTERED_LEARNING_RATE = 2.0e-3
REGISTERED_WEIGHT_DECAY = 1.0e-4
REGISTERED_GRADIENT_CLIP = 1.0
REGISTERED_PARAMETER_COUNT = 1_583_754

# This is the train-side GMR G1 scale, frozen before the experiment.  It is used only
# to define the heading-local learned-root target/recomposition; it is not a held-out
# statistic and is not a model identity input.
G1_HUMAN_ROOT_XY_SCALE = 0.8749322702593619
REGISTERED_PAIR_SHA256 = "79565402a381d54122c28b9f1f88bc43a46dfdca9f5aeee0ff0afbaee6c8c845"
REGISTERED_RAW_SOURCE_SHA256 = "4c9d591f323ffa660d9de6ef20c5d7077e66b2102a009feedf01afa22a6fbc73"
REGISTERED_SOURCE_DATASET = "lafan1"
REGISTERED_SOURCE_SEQUENCE_ID = "walk1_subject1"
REGISTERED_MOTION_ID = "lafan1/walk1_subject1"
REGISTERED_SOURCE_SUBJECT = "subject1"
REGISTERED_SPLIT_ID = "train"
DOCUMENTED_GMR_PAIR_GENERATION_COMMIT = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
REGISTERED_G1_MJCF_ENTRYPOINT_SHA256 = (
    "8c586e4747da85804180fe44d8692e0fd8231356728b6327e256dca498087a78"
)
REGISTERED_G1_MJCF_BUNDLE_SHA256 = (
    "fe1b0118a9433b2631fd9b84a4dba88641db6756698714e59b841c53a320dc14"
)
PREREGISTRATION_PATH = (
    ROOT / "autoresearch/iterate-260901-0341/fixed_g1_amortization_prereg.json"
)

GATE_THRESHOLDS: Mapping[str, float | int] = {
    "minimum_total_loss_drop_fraction": 0.75,
    "maximum_teacher_fk_mpjpe_m": 0.05,
    "maximum_dof_mae_rad": 0.10,
    "maximum_root_position_mae_m": 0.05,
    "maximum_root_orientation_geodesic_rad": 0.15,
    "maximum_joint_limit_violations": 0,
    "maximum_serialization_equivariance_abs": 1.0e-5,
}

LOSS_WEIGHTS: Mapping[str, float] = {
    "distill": 1.0,
    "joint_limits": 0.1,
    "smoothness": 0.01,
}


@dataclass(frozen=True)
class FixedG1Protocol:
    """Frozen optimizer/window protocol, with an explicitly ineligible smoke mode."""

    target_start_frame: int = REGISTERED_TARGET_START
    window_frames: int = REGISTERED_WINDOW_FRAMES
    steps: int = REGISTERED_STEPS
    seed: int = REGISTERED_SEED
    optimizer: str = "AdamW"
    learning_rate: float = REGISTERED_LEARNING_RATE
    weight_decay: float = REGISTERED_WEIGHT_DECAY
    gradient_clip_norm: float = REGISTERED_GRADIENT_CLIP
    human_root_xy_scale: float = G1_HUMAN_ROOT_XY_SCALE
    device: str = "cpu"
    smoke: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "target_start_frame",
            "window_frames",
            "steps",
            "seed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.target_start_frame < 0 or self.window_frames <= 0 or self.steps <= 0:
            raise ValueError("target start, window size, and step count must be valid")
        if self.optimizer != "AdamW":
            raise ValueError("the fixed-G1 experiment requires AdamW")
        for name in (
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "human_root_xy_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty torch device string")

        registered = (
            self.target_start_frame == REGISTERED_TARGET_START
            and self.window_frames == REGISTERED_WINDOW_FRAMES
            and self.steps == REGISTERED_STEPS
            and self.seed == REGISTERED_SEED
            and self.learning_rate == REGISTERED_LEARNING_RATE
            and self.weight_decay == REGISTERED_WEIGHT_DECAY
            and self.gradient_clip_norm == REGISTERED_GRADIENT_CLIP
            and self.human_root_xy_scale == G1_HUMAN_ROOT_XY_SCALE
        )
        if not self.smoke and not registered:
            raise ValueError("non-smoke runs must use the frozen registered protocol")
        if self.smoke and (self.window_frames > 16 or self.steps > 10):
            raise ValueError("smoke mode is limited to at most 16 frames and 10 steps")

    @classmethod
    def smoke_protocol(cls, *, device: str = "cpu") -> "FixedG1Protocol":
        return cls(
            target_start_frame=0,
            window_frames=8,
            steps=2,
            device=device,
            smoke=True,
        )

    @property
    def registered_gate_eligible(self) -> bool:
        return not self.smoke


@dataclass(frozen=True)
class WindowTensors:
    """Exact tensors consumed by the model/loss for the one registered window."""

    human_positions_w: torch.Tensor
    human_orientations_wxyz: torch.Tensor
    human_static: torch.Tensor
    anchor_position_w: torch.Tensor
    anchor_orientation_wxyz: torch.Tensor
    teacher_root_position_w: torch.Tensor
    teacher_root_orientation_wxyz: torch.Tensor
    teacher_root_position_local: torch.Tensor
    teacher_root_orientation_local_wxyz: torch.Tensor
    teacher_joint_positions_rad: torch.Tensor
    teacher_body_positions_w: torch.Tensor
    joint_node_indices: torch.Tensor
    timestamps_s: torch.Tensor
    exact_window_buffer_sha256: str


@dataclass(frozen=True)
class ExperimentArtifacts:
    report: dict[str, Any]
    checkpoint: dict[str, Any]


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_array_hash(digest: Any, name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"cannot hash non-finite array {name}")
    header = json.dumps(
        {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = memoryview(array).cast("B")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"snmr.state-dict-labelled-buffers.v1\0")
    for name in sorted(state):
        value = state[name]
        if isinstance(value, torch.Tensor):
            _update_array_hash(digest, name, value)
        else:
            encoded = json.dumps(
                {"name": name, "value": value},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _landmarks() -> BilateralSegmentLandmarks:
    return BilateralSegmentLandmarks(
        torso=(("LeftUpLeg", "LeftShoulder"), ("RightUpLeg", "RightShoulder")),
        thigh=(("LeftUpLeg", "LeftLeg"), ("RightUpLeg", "RightLeg")),
        shin=(("LeftLeg", "LeftFoot"), ("RightLeg", "RightFoot")),
        upper_arm=(("LeftArm", "LeftForeArm"), ("RightArm", "RightForeArm")),
        forearm=(("LeftForeArm", "LeftHand"), ("RightForeArm", "RightHand")),
    )


def _contact_protocol() -> FootContactProtocol:
    return FootContactProtocol(
        left_foot_body="LeftToe",
        right_foot_body="RightToe",
        speed_threshold_m_s=0.2,
        height_clearance_m=0.05,
        speed_softness_m_s=0.05,
        height_softness_m=0.01,
        ground_height_m=0.0,
    )


def _g1_semantics() -> SemanticManifest:
    return SemanticManifest(
        root_link="pelvis",
        torso_link="torso_link",
        left_hand_link="left_rubber_hand_link",
        right_hand_link="right_rubber_hand_link",
        left_foot_link="left_ankle_roll_link",
        right_foot_link="right_ankle_roll_link",
        allowed_contact_links=("left_ankle_roll_link", "right_ankle_roll_link"),
        symmetry_pairs=(
            ("left_ankle_roll_link", "right_ankle_roll_link"),
            ("left_rubber_hand_link", "right_rubber_hand_link"),
        ),
    )


def _model_configs(smoke: bool) -> tuple[SNMRConfig, MorphoRetargetConfig]:
    if smoke:
        human = SNMRConfig(
            latent_dim=16,
            enc_hidden=16,
            enc_layers=1,
            heads=4,
            use_temporal=False,
            temporal_layers=1,
            temporal_heads=4,
            temporal_positional=False,
        )
        morpho = MorphoRetargetConfig(
            human_token_dim=16,
            hidden_dim=16,
            num_heads=4,
            graph_layers=1,
            feedforward_multiplier=2,
            tree_bias_gamma=0.3,
        )
        return human, morpho
    human = SNMRConfig(
        latent_dim=128,
        enc_hidden=256,
        enc_layers=4,
        heads=4,
        use_temporal=True,
        temporal_layers=2,
        temporal_heads=4,
        temporal_positional=True,
    )
    morpho = MorphoRetargetConfig(
        human_token_dim=128,
        hidden_dim=128,
        num_heads=4,
        graph_layers=3,
        feedforward_multiplier=4,
        tree_bias_gamma=0.5,
    )
    return human, morpho


def validate_node_joint_mapping(
    robot_spec: RobotSpec,
    robot_tokens: RobotTokenBatch,
    teacher_joint_names: Sequence[str],
    robot_kinematics: Any,
) -> tuple[int, ...]:
    """Bind teacher qpos columns to joint child-link nodes, never node indices."""

    robot_spec.validate()
    if robot_tokens.node_features.shape[0] != 1:
        raise ValueError("fixed-G1 mapping requires exactly one tokenized robot")
    expected_joint_names = tuple(joint.name for joint in robot_spec.joints)
    if tuple(teacher_joint_names) != expected_joint_names:
        raise ValueError("teacher joint order does not exactly match RobotSpec joint order")
    if len(expected_joint_names) != int(robot_tokens.joint_mask.sum().item()):
        raise ValueError("RobotToken joint mask count disagrees with RobotSpec")

    node_names = tuple(robot_tokens.node_names[0])
    if len(set(node_names)) != len(node_names):
        raise ValueError("RobotToken node names must be unique audit metadata")
    indices: list[int] = []
    for joint in robot_spec.joints:
        try:
            node_index = node_names.index(joint.child_link)
        except ValueError as exc:
            raise ValueError(
                f"joint child link {joint.child_link!r} is absent from RobotToken nodes"
            ) from exc
        if not bool(robot_tokens.joint_mask[0, node_index]):
            raise ValueError(f"joint child link {joint.child_link!r} is not a joint token")
        lower = float(robot_tokens.joint_lower[0, node_index])
        upper = float(robot_tokens.joint_upper[0, node_index])
        if not math.isclose(
            lower, float(joint.lower_limit), rel_tol=0.0, abs_tol=1.0e-6
        ) or not math.isclose(
            upper, float(joint.upper_limit), rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise ValueError(f"joint limits changed while mapping {joint.name!r}")
        indices.append(node_index)
    if len(set(indices)) != len(indices):
        raise ValueError("multiple RobotSpec joints map to one output node")

    if int(robot_kinematics.num_dof) != len(robot_spec.joints):
        raise ValueError("RobotKinematics DoF count disagrees with RobotSpec")
    graph = robot_kinematics.graph
    dof_child_links = tuple(
        robot_kinematics.body_names[int(body_index)]
        for body_index in graph.dof_body_index.detach().cpu().tolist()
    )
    expected_child_links = tuple(joint.child_link for joint in robot_spec.joints)
    if dof_child_links != expected_child_links:
        raise ValueError(
            "RobotKinematics qpos order does not match RobotSpec child-link order"
        )
    return tuple(indices)


def _window_buffer_sha256(
    *,
    timestamps: Any,
    human_positions: Any,
    human_orientations: Any,
    anchor_position: Any,
    anchor_orientation: Any,
    teacher_root_position: Any,
    teacher_root_orientation: Any,
    teacher_joints: Any,
    joint_node_indices: Any,
) -> str:
    digest = hashlib.sha256(b"snmr.morpho-fixed-g1-window-buffers.v1\0")
    for name, value in (
        ("timestamps_s", timestamps),
        ("human_positions_w", human_positions),
        ("human_orientations_wxyz", human_orientations),
        ("anchor_position_w", anchor_position),
        ("anchor_orientation_wxyz", anchor_orientation),
        ("teacher_root_position_w", teacher_root_position),
        ("teacher_root_orientation_wxyz", teacher_root_orientation),
        ("teacher_joint_positions_rad", teacher_joints),
        ("joint_node_indices", joint_node_indices),
    ):
        _update_array_hash(digest, name, value)
    return digest.hexdigest()


def prepare_window(
    *,
    human_motion: HumanMotionSpec,
    teacher_motion: CanonicalTeacherMotion,
    robot_spec: RobotSpec,
    robot_tokens: RobotTokenBatch,
    robot_kinematics: Any,
    human_skeleton: SkeletonGraph,
    protocol: FixedG1Protocol,
) -> WindowTensors:
    """Validate the two canonical timelines and materialize exactly one target window."""

    human_motion.validate()
    robot_spec.validate()
    if human_motion.timebase.target_fps != TARGET_FPS:
        raise ValueError("HumanMotionSpec target timeline must be exactly 50 Hz")
    if teacher_motion.human_motion_spec_sha256 != human_motion.spec_sha256:
        raise ValueError("teacher is not bound to this HumanMotionSpec")
    if teacher_motion.robot_spec_sha256 != robot_spec.spec_hash:
        raise ValueError("teacher is not bound to this RobotSpec")
    if teacher_motion.robot_asset_sha256 != robot_spec.asset_sha256:
        raise ValueError("teacher is not bound to this robot asset")
    if not np.array_equal(
        teacher_motion.timestamps_s, human_motion.timebase.target_timestamps
    ):
        raise ValueError("teacher timestamps must exactly equal HumanMotionSpec timestamps")
    if tuple(human_skeleton.names) != tuple(human_motion.body_names):
        raise ValueError("human skeleton order does not match HumanMotionSpec")

    start = protocol.target_start_frame
    stop = start + protocol.window_frames
    if stop > len(human_motion.timebase.target_timestamps):
        raise ValueError(
            f"canonical motion has too few target frames for [{start}, {stop})"
        )
    joint_node_indices = validate_node_joint_mapping(
        robot_spec,
        robot_tokens,
        teacher_motion.joint_names,
        robot_kinematics,
    )
    device = torch.device(protocol.device)
    dtype = robot_tokens.node_features.dtype

    def tensor(value: Any) -> torch.Tensor:
        return torch.as_tensor(np.array(value, copy=True), device=device, dtype=dtype)

    human_positions = tensor(human_motion.body_positions[start:stop])
    human_orientations = tensor(
        human_motion.body_orientations_wxyz[start:stop]
    )
    anchor_position = tensor(human_motion.root_position[start:stop]).clone()
    anchor_position[:, :2] *= protocol.human_root_xy_scale
    anchor_orientation = tensor(human_motion.root_orientation_wxyz[start:stop])
    teacher_root_position = tensor(teacher_motion.root_position_m[start:stop])
    teacher_root_orientation = tensor(
        teacher_motion.root_orientation_wxyz[start:stop]
    )
    teacher_joints = tensor(teacher_motion.joint_positions_rad[start:stop])
    timestamps = torch.as_tensor(
        np.array(teacher_motion.timestamps_s[start:stop], copy=True),
        device=device,
        dtype=torch.float64,
    )
    local_position, local_orientation = world_root_to_local(
        anchor_position,
        anchor_orientation,
        teacher_root_position,
        teacher_root_orientation,
    )
    teacher_body_positions, _ = robot_kinematics.forward_kinematics(
        teacher_root_position, teacher_root_orientation, teacher_joints
    )
    moved_skeleton = human_skeleton.to(device)
    static = human_static_features(
        moved_skeleton, body_pos_sample=human_positions
    )
    node_index_tensor = torch.tensor(
        joint_node_indices, dtype=torch.long, device=device
    )
    exact_hash = _window_buffer_sha256(
        timestamps=timestamps,
        human_positions=human_positions,
        human_orientations=human_orientations,
        anchor_position=anchor_position,
        anchor_orientation=anchor_orientation,
        teacher_root_position=teacher_root_position,
        teacher_root_orientation=teacher_root_orientation,
        teacher_joints=teacher_joints,
        joint_node_indices=node_index_tensor,
    )
    return WindowTensors(
        human_positions_w=human_positions,
        human_orientations_wxyz=human_orientations,
        human_static=static,
        anchor_position_w=anchor_position,
        anchor_orientation_wxyz=anchor_orientation,
        teacher_root_position_w=teacher_root_position,
        teacher_root_orientation_wxyz=teacher_root_orientation,
        teacher_root_position_local=local_position,
        teacher_root_orientation_local_wxyz=local_orientation,
        teacher_joint_positions_rad=teacher_joints,
        teacher_body_positions_w=teacher_body_positions.detach(),
        joint_node_indices=node_index_tensor,
        timestamps_s=timestamps,
        exact_window_buffer_sha256=exact_hash,
    )


def _predict(
    model: FixedTargetMorphoRetargeter,
    window: WindowTensors,
    human_skeleton: SkeletonGraph,
    *,
    joint_node_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    time = window.human_positions_w.shape[0]
    output = model(
        window.human_positions_w,
        window.human_orientations_wxyz,
        human_skeleton=human_skeleton,
        human_time_mask=torch.ones(time, dtype=torch.bool, device=window.human_positions_w.device),
        human_node_mask=torch.ones(
            window.human_positions_w.shape[1],
            dtype=torch.bool,
            device=window.human_positions_w.device,
        ),
        human_static=window.human_static,
    )
    indices = window.joint_node_indices if joint_node_indices is None else joint_node_indices
    local_position = output.root_position_local[0]
    local_orientation = output.root_orientation_local_wxyz[0]
    world_position, world_orientation = local_root_to_world(
        window.anchor_position_w,
        window.anchor_orientation_wxyz,
        local_position,
        local_orientation,
    )
    return {
        "joint_positions_rad": output.joint_output.joint_positions[0].index_select(
            -1, indices
        ),
        "root_position_local": local_position,
        "root_orientation_local_wxyz": local_orientation,
        "root_position_w": world_position,
        "root_orientation_wxyz": world_orientation,
    }


def _loss_components(
    prediction: Mapping[str, torch.Tensor],
    window: WindowTensors,
    robot_kinematics: Any,
) -> dict[str, torch.Tensor]:
    joints = prediction["joint_positions_rad"]
    local_position = prediction["root_position_local"]
    distill = (
        torch.mean((joints - window.teacher_joint_positions_rad) ** 2)
        + torch.mean((local_position - window.teacher_root_position_local) ** 2)
        + torch.mean(
            (
                rot.quat_to_matrix(prediction["root_orientation_local_wxyz"])
                - rot.quat_to_matrix(window.teacher_root_orientation_local_wxyz)
            )
            ** 2
        )
    )
    lower, upper = robot_kinematics.dof_limits()
    lower = lower.to(device=joints.device, dtype=joints.dtype)
    upper = upper.to(device=joints.device, dtype=joints.dtype)
    limits = torch.mean(
        torch.relu(joints - upper) ** 2 + torch.relu(lower - joints) ** 2
    )
    if joints.shape[0] < 3:
        smoothness = joints.sum() * 0.0
    else:
        joint_acceleration = joints[2:] - 2.0 * joints[1:-1] + joints[:-2]
        root_acceleration = (
            local_position[2:]
            - 2.0 * local_position[1:-1]
            + local_position[:-2]
        )
        smoothness = torch.mean(joint_acceleration**2) + torch.mean(
            root_acceleration**2
        )
        if joints.shape[0] >= 4:
            joint_jerk = (
                joints[3:]
                - 3.0 * joints[2:-1]
                + 3.0 * joints[1:-2]
                - joints[:-3]
            )
            smoothness = smoothness + 0.1 * torch.mean(joint_jerk**2)
    components = {
        "distill": distill,
        "joint_limits": limits,
        "smoothness": smoothness,
    }
    components["total_loss"] = sum(
        LOSS_WEIGHTS[name] * value for name, value in components.items()
    )
    return components


def _metrics(
    prediction: Mapping[str, torch.Tensor],
    window: WindowTensors,
    robot_spec: RobotSpec,
    robot_kinematics: Any,
) -> dict[str, float | int]:
    predicted_body, _ = robot_kinematics.forward_kinematics(
        prediction["root_position_w"],
        prediction["root_orientation_wxyz"],
        prediction["joint_positions_rad"],
    )
    losses = _loss_components(prediction, window, robot_kinematics)
    lower = torch.tensor(
        [joint.lower_limit for joint in robot_spec.joints],
        dtype=prediction["joint_positions_rad"].dtype,
        device=prediction["joint_positions_rad"].device,
    )
    upper = torch.tensor(
        [joint.upper_limit for joint in robot_spec.joints],
        dtype=prediction["joint_positions_rad"].dtype,
        device=prediction["joint_positions_rad"].device,
    )
    lower_excess = lower - prediction["joint_positions_rad"]
    upper_excess = prediction["joint_positions_rad"] - upper
    max_excess = torch.maximum(lower_excess, upper_excess).clamp_min(0.0)
    violations = int(torch.count_nonzero(max_excess > 1.0e-6).item())
    return {
        **{name: float(value.detach().cpu()) for name, value in losses.items()},
        "teacher_fk_mpjpe_m": float(
            torch.linalg.vector_norm(
                predicted_body - window.teacher_body_positions_w, dim=-1
            ).mean().detach().cpu()
        ),
        "dof_mae_rad": float(
            torch.mean(
                torch.abs(
                    prediction["joint_positions_rad"]
                    - window.teacher_joint_positions_rad
                )
            ).detach().cpu()
        ),
        # The preregistered root ruler is heading-local.  This Euclidean error is
        # numerically equal in world coordinates because both trajectories use the
        # same anchor and heading, but we compute it in the declared local frame.
        "root_position_mae_m": float(
            torch.linalg.vector_norm(
                prediction["root_position_local"]
                - window.teacher_root_position_local,
                dim=-1,
            ).mean().detach().cpu()
        ),
        "root_orientation_geodesic_rad": float(
            rot.quat_geodesic_angle(
                prediction["root_orientation_wxyz"],
                window.teacher_root_orientation_wxyz,
            ).mean().detach().cpu()
        ),
        "joint_limit_violations": violations,
        "joint_limit_violation_fraction": float(
            violations / prediction["joint_positions_rad"].numel()
        ),
        "maximum_joint_limit_excess_rad": float(max_excess.max().detach().cpu()),
    }


def _serialization_equivariance_error(
    *,
    trained_model: FixedTargetMorphoRetargeter,
    human_config: SNMRConfig,
    morpho_config: MorphoRetargetConfig,
    robot_spec: RobotSpec,
    window: WindowTensors,
    human_skeleton: SkeletonGraph,
) -> float:
    permuted_spec = replace(
        robot_spec,
        links=tuple(reversed(robot_spec.links)),
        joints=tuple(reversed(robot_spec.joints)),
    )
    permuted_spec.validate()
    permuted_tokens = RobotGraphTokenizer("kinematic")([permuted_spec]).to(
        window.human_positions_w.device
    )
    permuted_model = FixedTargetMorphoRetargeter(
        permuted_tokens,
        human_encoder_config=human_config,
        morpho_config=morpho_config,
    ).to(window.human_positions_w.device)
    permuted_model.human_encoder.load_state_dict(trained_model.human_encoder.state_dict())
    permuted_model.joint_decoder.load_state_dict(trained_model.joint_decoder.state_dict())
    permuted_model.root_head.load_state_dict(trained_model.root_head.state_dict())
    permuted_model.eval()

    names = tuple(permuted_tokens.node_names[0])
    original_joint_order = tuple(joint.child_link for joint in robot_spec.joints)
    indices = torch.tensor(
        [names.index(child) for child in original_joint_order],
        dtype=torch.long,
        device=window.human_positions_w.device,
    )
    with torch.no_grad():
        original = _predict(trained_model, window, human_skeleton)
        permuted = _predict(
            permuted_model,
            window,
            human_skeleton,
            joint_node_indices=indices,
        )
    errors = [
        torch.max(torch.abs(original["joint_positions_rad"] - permuted["joint_positions_rad"])),
        torch.max(torch.abs(original["root_position_local"] - permuted["root_position_local"])),
    ]
    q0 = original["root_orientation_local_wxyz"]
    q1 = permuted["root_orientation_local_wxyz"]
    quat_error = torch.minimum(
        torch.max(torch.abs(q0 - q1)), torch.max(torch.abs(q0 + q1))
    )
    errors.append(quat_error)
    return float(torch.stack(errors).max().cpu())


def _set_determinism(protocol: FixedG1Protocol) -> None:
    random.seed(protocol.seed)
    np.random.seed(protocol.seed)
    torch.manual_seed(protocol.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol.seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _gradient_block_parameters(
    model: FixedTargetMorphoRetargeter,
) -> dict[str, tuple[tuple[str, torch.nn.Parameter], ...]]:
    named = tuple(model.named_parameters())
    prefixes = {
        "human_encoder": ("human_encoder.",),
        "robot_graph_encoder": (
            "joint_decoder.robot_input.",
            "joint_decoder.graph_layers.",
            "joint_decoder.human_input.",
            "joint_decoder.cross_attention.",
        ),
        "shared_joint_head": ("joint_decoder.joint_head.",),
        "root_head": ("root_head.",),
    }
    result = {
        block: tuple(
            (name, parameter)
            for name, parameter in named
            if any(name.startswith(prefix) for prefix in block_prefixes)
        )
        for block, block_prefixes in prefixes.items()
    }
    empty = [name for name, parameters in result.items() if not parameters]
    if empty:
        raise AssertionError(f"registered gradient blocks have no parameters: {empty}")
    return result


def _new_gradient_audit(
    blocks: Mapping[str, Sequence[tuple[str, torch.nn.Parameter]]],
) -> dict[str, dict[str, Any]]:
    return {
        block: {
            "parameter_count": int(sum(parameter.numel() for _, parameter in parameters)),
            "steps_observed": 0,
            "all_parameters_had_gradients_every_step": True,
            "all_gradients_finite_every_step": True,
            "nonzero_gradient_step_count": 0,
            "minimum_l2_norm": None,
            "maximum_l2_norm": 0.0,
            "parameters_missing_gradient": [],
        }
        for block, parameters in blocks.items()
    }


def _observe_gradients(
    blocks: Mapping[str, Sequence[tuple[str, torch.nn.Parameter]]],
    audit: dict[str, dict[str, Any]],
) -> None:
    for block, parameters in blocks.items():
        record = audit[block]
        record["steps_observed"] += 1
        missing = [name for name, parameter in parameters if parameter.grad is None]
        if missing:
            record["all_parameters_had_gradients_every_step"] = False
            record["parameters_missing_gradient"] = sorted(
                set(record["parameters_missing_gradient"]).union(missing)
            )
        gradients = [
            parameter.grad.detach() for _, parameter in parameters if parameter.grad is not None
        ]
        finite = not missing and all(torch.isfinite(gradient).all() for gradient in gradients)
        if not finite:
            record["all_gradients_finite_every_step"] = False
        if gradients and finite:
            squared_norm = sum(
                float(torch.sum(gradient.double() ** 2).cpu()) for gradient in gradients
            )
            norm = math.sqrt(squared_norm)
            if norm > 0.0:
                record["nonzero_gradient_step_count"] += 1
            previous_minimum = record["minimum_l2_norm"]
            record["minimum_l2_norm"] = (
                norm if previous_minimum is None else min(previous_minimum, norm)
            )
            record["maximum_l2_norm"] = max(record["maximum_l2_norm"], norm)


def run_overfit_experiment(
    *,
    human_motion: HumanMotionSpec,
    teacher_motion: CanonicalTeacherMotion,
    robot_spec: RobotSpec,
    robot_kinematics: Any,
    human_skeleton: SkeletonGraph,
    protocol: FixedG1Protocol,
    input_manifest: Mapping[str, Any],
) -> ExperimentArtifacts:
    """Run the one-window fit.  Tests call this only with the smoke protocol."""

    _set_determinism(protocol)
    device = torch.device(protocol.device)
    tokens = RobotGraphTokenizer("kinematic")([robot_spec]).to(device)
    human_config, morpho_config = _model_configs(protocol.smoke)
    config_payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "human_encoder": asdict(human_config),
        "human_temporal_transformer_dropout": 0.1 if human_config.use_temporal else 0.0,
        "morpho_decoder": asdict(morpho_config),
        "loss_weights": dict(LOSS_WEIGHTS),
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "optimization_schedule": "constant",
        "initialization": "from_scratch",
        "batch_windows_per_step": 1,
        "pre_run_amendments": [
            {
                "field": "human_encoder.temporal_positional",
                "preregistered_value": False,
                "run_value": True,
                "parameter_count_change": 0,
                "rationale": (
                    "the existing content-only temporal transformer has no order signal; "
                    "the amendment was declared before any 1000-step result"
                ),
            }
        ]
        if not protocol.smoke
        else [],
        "root_target_convention": (
            "human-heading-local-xy_absolute-z_with_frozen_train-side-g1-xy-scale"
        ),
        "contact_output": "absent_contacts_must_be_derived_from_predicted_robot_fk",
    }
    config_sha256 = _canonical_json_sha256(config_payload)
    window = prepare_window(
        human_motion=human_motion,
        teacher_motion=teacher_motion,
        robot_spec=robot_spec,
        robot_tokens=tokens,
        robot_kinematics=robot_kinematics,
        human_skeleton=human_skeleton,
        protocol=protocol,
    )
    model = FixedTargetMorphoRetargeter(
        tokens,
        human_encoder_config=human_config,
        morpho_config=morpho_config,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not protocol.smoke and parameter_count != REGISTERED_PARAMETER_COUNT:
        raise RuntimeError(
            "model parameter count differs from preregistration "
            f"({parameter_count} != {REGISTERED_PARAMETER_COUNT})"
        )
    # Before/after metrics are deterministic inference endpoints.  Optimization below
    # uses normal train mode (including the encoder's registered default dropout).
    model.eval()
    initial_state_sha256 = _state_dict_sha256(model.state_dict())
    with torch.no_grad():
        before_prediction = _predict(model, window, human_skeleton)
        before = _metrics(before_prediction, window, robot_spec, robot_kinematics)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    gradient_blocks = _gradient_block_parameters(model)
    gradient_audit = _new_gradient_audit(gradient_blocks)
    loss_history: list[float] = []
    model.train()
    for _ in range(protocol.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(model, window, human_skeleton)
        components = _loss_components(prediction, window, robot_kinematics)
        total = components["total_loss"]
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite fixed-G1 overfit loss")
        total.backward()
        _observe_gradients(gradient_blocks, gradient_audit)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), protocol.gradient_clip_norm
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError("non-finite fixed-G1 gradient norm")
        optimizer.step()
        loss_history.append(float(total.detach().cpu()))

    model.eval()
    with torch.no_grad():
        after_prediction = _predict(model, window, human_skeleton)
        after = _metrics(after_prediction, window, robot_spec, robot_kinematics)
    loss_drop = 1.0 - after["total_loss"] / max(before["total_loss"], 1.0e-30)
    span = min(20, len(loss_history))
    first_mean = float(np.mean(loss_history[:span]))
    last_mean = float(np.mean(loss_history[-span:]))
    smoothed_drop = 1.0 - last_mean / max(first_mean, 1.0e-30)
    equivariance_error = _serialization_equivariance_error(
        trained_model=model,
        human_config=human_config,
        morpho_config=morpho_config,
        robot_spec=robot_spec,
        window=window,
        human_skeleton=human_skeleton,
    )

    gradient_block_gates = {
        block: bool(
            record["steps_observed"] == protocol.steps
            and record["all_parameters_had_gradients_every_step"]
            and record["all_gradients_finite_every_step"]
            and record["nonzero_gradient_step_count"] > 0
        )
        for block, record in gradient_audit.items()
    }
    finite_loss_metrics = all(
        math.isfinite(float(value))
        for metrics in (before, after)
        for value in metrics.values()
    ) and all(math.isfinite(value) for value in loss_history)
    gates = {
        "last20_over_first20_loss_drop_ge_75pct": smoothed_drop
        >= float(GATE_THRESHOLDS["minimum_total_loss_drop_fraction"]),
        "teacher_fk_mpjpe_lt_5cm": after["teacher_fk_mpjpe_m"]
        < float(GATE_THRESHOLDS["maximum_teacher_fk_mpjpe_m"]),
        "dof_mae_lt_0p10rad": after["dof_mae_rad"]
        < float(GATE_THRESHOLDS["maximum_dof_mae_rad"]),
        "root_position_mae_lt_5cm": after["root_position_mae_m"]
        < float(GATE_THRESHOLDS["maximum_root_position_mae_m"]),
        "root_geodesic_lt_0p15rad": after["root_orientation_geodesic_rad"]
        < float(GATE_THRESHOLDS["maximum_root_orientation_geodesic_rad"]),
        "zero_joint_limit_violations": after["joint_limit_violations"]
        <= int(GATE_THRESHOLDS["maximum_joint_limit_violations"]),
        "serialization_equivariance_le_1e_5": equivariance_error
        <= float(GATE_THRESHOLDS["maximum_serialization_equivariance_abs"]),
        "finite_loss_and_nonzero_gradients_all_named_blocks": (
            finite_loss_metrics and all(gradient_block_gates.values())
        ),
    }
    scientific_pass = all(gates.values())
    gate_decision = (
        "SMOKE_NOT_GATE_ELIGIBLE"
        if protocol.smoke
        else ("PASS" if scientific_pass else "FAIL")
    )
    final_state_sha256 = _state_dict_sha256(model.state_dict())
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_utc": utc_now(),
        "scope": "fixed-G1 single-window teacher amortization only",
        "claims_excluded": [
            "held-out morphology generalization",
            "physics feasibility",
            "contact prediction",
            "dynamics conditioning",
        ],
        "registered_gate_eligible": protocol.registered_gate_eligible,
        "gate_decision": gate_decision,
        "scientific_gate_pass": scientific_pass,
        "gates": gates,
        "config": config_payload,
        "config_sha256": config_sha256,
        "inputs": dict(input_manifest),
        "canonical_contract": {
            "human_motion_spec_sha256": human_motion.spec_sha256,
            "human_motion_buffer_sha256": human_motion.buffer_sha256,
            "human_target_fps": human_motion.timebase.target_fps,
            "human_source_fps": human_motion.timebase.source_fps,
            "human_source_frame_count": int(
                human_motion.timebase.source_timestamps.size
            ),
            "human_canonical_frame_count": int(
                human_motion.timebase.target_timestamps.size
            ),
            "human_resampling": {
                "positions": human_motion.timebase.position_interpolation,
                "positions_applied": (
                    human_motion.timebase.position_interpolation_applied
                ),
                "orientations": human_motion.timebase.quaternion_interpolation,
                "contacts": human_motion.timebase.contact_interpolation,
            },
            "teacher_motion_buffer_sha256": teacher_motion.buffer_sha256,
            "teacher_timestamps_exactly_equal_human": True,
            "teacher_resampling": {
                "root_position": teacher_motion.root_position_interpolation,
                "root_position_applied": (
                    teacher_motion.root_position_interpolation_applied
                ),
                "root_orientation": teacher_motion.root_orientation_interpolation,
                "joint_position": teacher_motion.joint_position_interpolation,
            },
            "teacher_joint_names": list(teacher_motion.joint_names),
            "robot_spec_sha256": robot_spec.spec_hash,
            "robot_kinematic_sha256": robot_spec.kinematic_hash,
            "robot_asset_sha256": robot_spec.asset_sha256,
            "robot_token_audit": tokens.audit_manifest(),
            "exact_window_buffer_sha256": window.exact_window_buffer_sha256,
            "window_target_indices": [
                protocol.target_start_frame,
                protocol.target_start_frame + protocol.window_frames,
            ],
            "window_timestamps_s": [
                float(window.timestamps_s[0].cpu()),
                float(window.timestamps_s[-1].cpu()),
            ],
            "joint_node_indices_audit_only": window.joint_node_indices.cpu().tolist(),
            "joint_mapping": "RobotSpec joint name/order -> child-link token",
            "learned_root_never_teacher_forced": True,
        },
        "model": {
            "parameter_count": parameter_count,
            "registered_parameter_count": REGISTERED_PARAMETER_COUNT,
            "optimization_mode": "train_mode_with_default_transformer_dropout",
            "initial_state_dict_sha256": initial_state_sha256,
            "final_state_dict_sha256": final_state_sha256,
        },
        "metrics_before": before,
        "metrics_after": after,
        "optimization": {
            "total_loss_drop_fraction_before_to_after": float(loss_drop),
            "registered_last20_over_first20_loss_drop_fraction": float(smoothed_drop),
            "first_up_to_20_step_loss_mean": first_mean,
            "last_up_to_20_step_loss_mean": last_mean,
            "loss_history": loss_history,
        },
        "gradient_audit": gradient_audit,
        "gradient_block_gates": gradient_block_gates,
        "serialization_equivariance_max_abs": equivariance_error,
    }
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "config": config_payload,
        "config_sha256": config_sha256,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "initial_state_dict_sha256": initial_state_sha256,
        "final_state_dict_sha256": final_state_sha256,
        "human_motion_spec_sha256": human_motion.spec_sha256,
        "teacher_motion_buffer_sha256": teacher_motion.buffer_sha256,
        "exact_window_buffer_sha256": window.exact_window_buffer_sha256,
    }
    return ExperimentArtifacts(report=report, checkpoint=checkpoint)


def write_artifacts_once(
    output_dir: str | Path,
    artifacts: ExperimentArtifacts,
) -> dict[str, Any]:
    """Publish a new result directory without overwriting any prior evidence."""

    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"write-once output directory already exists: {destination}"
        ) from exc

    report_path = destination / "report.json"
    config_path = destination / "config.json"
    checkpoint_path = destination / "checkpoint.pt"
    report_path.write_text(
        json.dumps(artifacts.report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            artifacts.report["config"], indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(artifacts.checkpoint, checkpoint_path)
    files = {
        name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in (
            ("report.json", report_path),
            ("config.json", config_path),
            ("checkpoint.pt", checkpoint_path),
        )
    }
    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "write_once": True,
        "files": files,
    }
    manifest["manifest_sha256"] = _canonical_json_sha256(manifest)
    (destination / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment(protocol: FixedG1Protocol) -> dict[str, Any]:
    result = runtime_state(protocol.device)
    result.update(
        {
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "mujoco": _package_version("mujoco"),
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        }
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-npz", type=Path, required=True)
    parser.add_argument("--g1-mjcf", type=Path, required=True)
    parser.add_argument(
        "--raw-human-source",
        type=Path,
        required=True,
        help="Exact raw BVH/source file; hashed separately from the derived pair NPZ.",
    )
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-sequence-id", required=True)
    parser.add_argument("--motion-id", required=True)
    parser.add_argument("--source-subject", default=None)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--newton-root",
        type=Path,
        default=None,
        help="Newton checkout whose exact revision is recorded (required for registered runs).",
    )
    parser.add_argument(
        "--isaac-lab-root",
        type=Path,
        default=None,
        help="Isaac Lab checkout whose exact revision is recorded (required for registered runs).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 8 frames/2 steps; the report is never eligible for the registered gate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    protocol = (
        FixedG1Protocol.smoke_protocol(device=args.device)
        if args.smoke
        else FixedG1Protocol(device=args.device)
    )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"write-once output directory already exists: {output}")

    # Capture mutable inputs once.  Every path-only consumer below reads only private
    # materializations of these same immutable bytes.
    pair_snapshot = ArtifactSnapshot.capture(args.pair_npz)
    raw_source_snapshot = ArtifactSnapshot.capture(args.raw_human_source)
    asset_bundle = MjcfBundleSnapshot.capture(args.g1_mjcf)
    preregistration_snapshot = ArtifactSnapshot.capture(PREREGISTRATION_PATH)
    source_code = source_fingerprint(
        ROOT, "scripts/experiment_morpho_fixed_g1_overfit.py"
    )
    revisions = source_revision_manifest(
        snmr_path=ROOT,
        newton_path=args.newton_root,
        isaac_lab_path=args.isaac_lab_root,
    )
    repository = git_state(ROOT)
    if not protocol.smoke and repository.get("dirty") is not False:
        raise RuntimeError(
            "registered fixed-G1 run requires a clean SNMR checkout; use --smoke for wiring"
        )
    if not protocol.smoke:
        for dependency in ("newton", "isaac_lab"):
            if (
                revisions.get(f"{dependency}_repo_status") != "available"
                or not isinstance(revisions.get(f"{dependency}_commit"), str)
                or not isinstance(revisions.get(f"{dependency}_dirty"), bool)
            ):
                raise RuntimeError(
                    f"registered run requires an exact {dependency} Git revision and dirty flag"
                )
        declared_identity = {
            "source_dataset": args.source_dataset,
            "source_sequence_id": args.source_sequence_id,
            "motion_id": args.motion_id,
            "source_subject": args.source_subject,
            "split_id": args.split_id,
        }
        registered_identity = {
            "source_dataset": REGISTERED_SOURCE_DATASET,
            "source_sequence_id": REGISTERED_SOURCE_SEQUENCE_ID,
            "motion_id": REGISTERED_MOTION_ID,
            "source_subject": REGISTERED_SOURCE_SUBJECT,
            "split_id": REGISTERED_SPLIT_ID,
        }
        if declared_identity != registered_identity:
            raise ValueError(
                "registered run metadata must exactly identify the frozen source clip: "
                f"{registered_identity}"
            )

    with tempfile.TemporaryDirectory(prefix="snmr-fixed-g1-overfit-") as directory:
        private_root = Path(directory)
        pair_path = pair_snapshot.materialize(private_root / "pair.npz")
        asset_path = asset_bundle.materialize(private_root / "g1_asset")
        pair = load_pair_npz(str(pair_path), dtype=torch.float64)
        if pair["robot"] != "unitree_g1":
            raise ValueError(
                f"fixed-G1 experiment requires pair robot 'unitree_g1', got {pair['robot']!r}"
            )
        if not protocol.smoke and pair_snapshot.sha256 != REGISTERED_PAIR_SHA256:
            raise ValueError("registered run requires the frozen walk1_subject1 pair bytes")
        if not protocol.smoke and raw_source_snapshot.sha256 != REGISTERED_RAW_SOURCE_SHA256:
            raise ValueError("registered run requires the frozen walk1_subject1 raw BVH bytes")
        if not protocol.smoke and (
            asset_bundle.entrypoint_snapshot.sha256
            != REGISTERED_G1_MJCF_ENTRYPOINT_SHA256
            or asset_bundle.sha256 != REGISTERED_G1_MJCF_BUNDLE_SHA256
        ):
            raise ValueError("registered run requires the frozen complete G1 MJCF bundle")
        source_stop_frame = int(pair["qpos"].shape[0])
        if source_stop_frame < 2:
            raise ValueError("pair must contain at least two source frames")

        source = MotionSource(
            dataset=args.source_dataset,
            sequence_id=args.source_sequence_id,
            sha256=raw_source_snapshot.sha256,
        )
        provenance = MotionProvenance(
            motion_id=args.motion_id,
            source_subject=args.source_subject,
            clip_range=(0, source_stop_frame),
            split_id=args.split_id,
            transformations=(
                "gmr_lafan1_world_kinematics",
                "pair_npz_export",
                "HumanMotionSpec_v1_canonical_50hz",
            ),
            preprocessing_version="morpho-fixed-g1-overfit.v0.1",
        )
        adaptation = adapt_lafan1_pair_npz(
            pair_path,
            pair_artifact_sha256=pair_snapshot.sha256,
            source=source,
            provenance=provenance,
            body_names=tuple(LAFAN1_BODY_NAMES),
            root_body_name="Hips",
            landmark_pairs=_landmarks(),
            contact_protocol=_contact_protocol(),
            frames=HumanFrameConvention(),
            validity_mask=np.ones(
                (
                    source_stop_frame,
                    len(LAFAN1_BODY_NAMES),
                ),
                dtype=np.bool_,
            ),
        )
        robot_spec = RobotSpec.from_mjcf(asset_path, _g1_semantics())
        if robot_spec.asset_sha256 != asset_bundle.entrypoint_snapshot.sha256:
            raise AssertionError("RobotSpec did not consume the captured MJCF entrypoint bytes")
        robot_kinematics = RobotKinematics(str(asset_path), device=protocol.device)
        selected_qpos = pair["qpos"].detach().cpu().numpy()
        teacher = resample_teacher_qpos(
            selected_qpos,
            source_fps=pair["fps"],
            human_motion=adaptation.motion_spec,
            robot_spec=robot_spec,
            pair_artifact_sha256=pair_snapshot.sha256,
        )
        input_manifest = {
            "raw_human_source": raw_source_snapshot.manifest(),
            "derived_pair_artifact": pair_snapshot.manifest(),
            "g1_mjcf_bundle": asset_bundle.manifest(),
            "preregistration": preregistration_snapshot.manifest(),
            "human_motion_adaptation": adaptation.to_manifest(),
            "source_code": source_code,
            "source_revisions": revisions,
            "snmr_git": repository,
            "holosoma_asset_git": git_state(args.g1_mjcf.expanduser().resolve().parent),
            "documented_gmr_pair_generation_commit": (
                DOCUMENTED_GMR_PAIR_GENERATION_COMMIT
            ),
            "runtime_environment": _environment(protocol),
            "invocation": {
                "argv": list(sys.argv if argv is None else argv),
                "command": shlex.join(sys.argv if argv is None else argv),
                "cwd": os.getcwd(),
                "started_utc": utc_now(),
            },
        }
        input_manifest["manifest_sha256"] = _canonical_json_sha256(input_manifest)
        artifacts = run_overfit_experiment(
            human_motion=adaptation.motion_spec,
            teacher_motion=teacher,
            robot_spec=robot_spec,
            robot_kinematics=robot_kinematics,
            human_skeleton=lafan1_skeleton(protocol.device),
            protocol=protocol,
            input_manifest=input_manifest,
        )

    artifact_manifest = write_artifacts_once(output, artifacts)
    print(json.dumps({
        "output_dir": str(output),
        "gate_decision": artifacts.report["gate_decision"],
        "artifact_manifest_sha256": artifact_manifest["manifest_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
