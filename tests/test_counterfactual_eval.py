from types import SimpleNamespace

import pytest
import torch

from snmr.integration.counterfactual_eval import (
    E71_FULL_STATE_TENSOR_NAMES,
    E71_EVALUATION_CONDITIONS,
    E71_RUNTIME_CONTRACT,
    FROZEN_ROOT_BODY_INDEX,
    _write_reference_state,
    assert_frozen_root_convention,
    audit_same_state_proprio,
    audit_same_state_tensors,
    branch_start_steps_for_grid,
    build_four_cell_grid,
    normalized_branch_coordinate,
    pooled_goal_scale,
    standardized_branch_squared_error,
    validate_start_steps,
)


def test_e71_full_state_contract_is_sorted_and_unique() -> None:
    assert tuple(sorted(E71_FULL_STATE_TENSOR_NAMES)) == E71_FULL_STATE_TENSOR_NAMES
    assert len(set(E71_FULL_STATE_TENSOR_NAMES)) == len(E71_FULL_STATE_TENSOR_NAMES)
    assert "integration_qacc_warmstart" in E71_FULL_STATE_TENSOR_NAMES
    assert "action_term.joint_control.prev_dof_vel" in E71_FULL_STATE_TENSOR_NAMES
    assert E71_EVALUATION_CONDITIONS["runtime_contract"] is E71_RUNTIME_CONTRACT
    assert E71_EVALUATION_CONDITIONS["future_only_branch_samples"] is True
    assert E71_RUNTIME_CONTRACT["max_episode_length_s"] == 20.0


def _pair_report() -> dict:
    return {
        "clips": ["walk_a", "walk_b"],
        "windows": [
            {"time_seconds_first": 0.2, "time_seconds_second": 0.3},
            {"time_seconds_first": 1.0, "time_seconds_second": 1.2},
        ],
    }


def test_build_four_cell_grid_decouples_state_and_command_frames() -> None:
    grid = build_four_cell_grid(
        _pair_report(),
        clip_names=["walk_a", "walk_b"],
        motion_starts=torch.tensor([0, 100]),
        motion_ends=torch.tensor([100, 200]),
        fps=10.0,
        horizon_steps=10,
    )

    assert grid.num_cells == 8
    assert grid.pair_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert grid.state_sides.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert grid.command_sides.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert grid.state_start_steps[:4].tolist() == [2, 2, 103, 103]
    assert grid.command_start_steps[:4].tolist() == [2, 103, 2, 103]
    first, second = branch_start_steps_for_grid(grid)
    assert first.tolist() == [2, 2, 2, 2, 10, 10, 10, 10]
    assert second.tolist() == [103, 103, 103, 103, 112, 112, 112, 112]


def test_build_four_cell_grid_rejects_wrong_clip_order_and_short_horizon() -> None:
    kwargs = {
        "motion_starts": torch.tensor([0, 100]),
        "motion_ends": torch.tensor([100, 122]),
        "fps": 10.0,
        "horizon_steps": 10,
    }
    with pytest.raises(ValueError, match="loaded motion order"):
        build_four_cell_grid(
            _pair_report(), clip_names=["walk_b", "walk_a"], **kwargs
        )
    with pytest.raises(ValueError, match="registered horizon"):
        build_four_cell_grid(
            _pair_report(), clip_names=["walk_a", "walk_b"], **kwargs
        )


def test_validate_start_steps_requires_preceding_frame() -> None:
    with pytest.raises(ValueError, match="preceding"):
        validate_start_steps(
            torch.tensor([0, 101]),
            torch.tensor([0, 100]),
            torch.tensor([100, 200]),
            horizon_steps=10,
            label="state",
        )


def test_audit_same_state_proprio_passes_exact_pairs_and_detects_command_leak() -> None:
    grid = build_four_cell_grid(
        _pair_report(),
        clip_names=["walk_a", "walk_b"],
        motion_starts=torch.tensor([0, 100]),
        motion_ends=torch.tensor([100, 200]),
        fps=10.0,
        horizon_steps=10,
    )
    proprio = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [7.0, 8.0],
        ]
    )
    report = audit_same_state_proprio(proprio, grid)
    assert report == {
        "num_state_comparisons": 4,
        "max_abs_difference": 0.0,
        "tolerance": 1.0e-6,
        "passed": True,
    }

    proprio[1, 0] += 2.0e-6
    report = audit_same_state_proprio(proprio, grid)
    assert report["passed"] is False
    assert report["max_abs_difference"] == pytest.approx(2.0e-6, abs=5.0e-8)


def test_audit_same_state_tensors_checks_hidden_state_and_empty_fields() -> None:
    grid = build_four_cell_grid(
        _pair_report(),
        clip_names=["walk_a", "walk_b"],
        motion_starts=torch.tensor([0, 100]),
        motion_ends=torch.tensor([100, 200]),
        fps=10.0,
        horizon_steps=10,
    )
    base = torch.tensor([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0], [4.0], [4.0]])
    report = audit_same_state_tensors(
        {
            "qacc_warmstart": base,
            "act": torch.empty(8, 0),
            "eq_active": torch.tensor([[True], [True], [False], [False]] * 2),
        },
        grid,
    )
    assert report["passed"] is True
    assert report["num_tensors"] == 3
    assert report["max_abs_difference"] == 0.0

    leaked = base.clone()
    leaked[5] += 1.0e-4
    report = audit_same_state_tensors({"qacc_warmstart": leaked}, grid)
    assert report["passed"] is False
    assert report["per_tensor"]["qacc_warmstart"]["passed"] is False

    discrete = torch.tensor([[1], [1], [2], [2], [3], [4], [5], [5]])
    report = audit_same_state_tensors({"sleep_state": discrete}, grid)
    assert report["passed"] is False
    assert report["per_tensor"]["sleep_state"]["max_abs_difference"] == 1.0


def test_write_reference_state_changes_physics_not_command_cursor() -> None:
    frames, envs, dofs = 8, 3, 2
    motion = SimpleNamespace(
        joint_pos=torch.arange(frames * dofs, dtype=torch.float32).reshape(frames, dofs),
        joint_vel=torch.arange(frames * dofs, dtype=torch.float32).reshape(frames, dofs) + 100,
        body_pos_w=torch.arange(frames * 3, dtype=torch.float32).reshape(frames, 1, 3),
        body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * frames),
        body_lin_vel_w=torch.ones(frames, 1, 3),
        body_ang_vel_w=torch.full((frames, 1, 3), 2.0),
        has_object=False,
    )
    sim = SimpleNamespace(
        dof_pos=torch.zeros(envs, dofs),
        dof_vel=torch.zeros(envs, dofs),
        robot_root_states=torch.zeros(envs, 13),
        scene=SimpleNamespace(
            env_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        ),
    )
    command = SimpleNamespace(
        motion=motion,
        motion_ids=torch.tensor([1, 0, 1]),
        time_steps=torch.tensor([5, 2, 6]),
        _env=SimpleNamespace(simulator=sim),
    )
    before_ids = command.motion_ids.clone()
    before_times = command.time_steps.clone()

    _write_reference_state(command, torch.tensor([0, 2]), torch.tensor([1, 4]))

    assert torch.equal(command.motion_ids, before_ids)
    assert torch.equal(command.time_steps, before_times)
    assert torch.equal(sim.dof_pos[0], motion.joint_pos[1])
    assert torch.equal(sim.dof_pos[2], motion.joint_pos[4])
    assert torch.equal(
        sim.robot_root_states[2, :3], motion.body_pos_w[4, 0] + sim.scene.env_origins[2]
    )


# ---------------------------------------------------------------------------------------
# Root-initialization convention.
#
# The frozen E70 reset (``wbt_bodyfix.reset``, lines 117-122) writes ``joint_pos``/
# ``joint_vel`` and the ``root_*_w`` family into the simulator.  Those Holosoma properties
# read motion body slot 0 (``wbt.py``:877-890) -- the pelvis, i.e. the free-joint body backing
# ``robot_root_states`` -- not ``ref_body_index`` (``torso_link``), which belongs to the
# unrelated ``ref_*`` family at ``wbt.py``:860-874 and drives tracking, never a reset.
#
# The mocks below use FOUR bodies with a per-body-distinguishable value and
# ``ref_body_index == 2``, so any confusion between the two families is fatal to the
# assertions.  The pre-existing one-body mock could not distinguish them.
# ---------------------------------------------------------------------------------------

_NUM_MOCK_BODIES = 4
_MOCK_REF_BODY_INDEX = 2


class _MockMotionCommand:
    """Mock exposing the Holosoma ``MotionCommand`` reset contract verbatim.

    ``joint_pos``/``joint_vel`` and ``root_*_w`` are transcribed from
    ``holosoma/managers/command/terms/wbt.py`` so a test against this mock is a test against
    the frozen reset's actual arithmetic.
    """

    root_body_index = 0

    def __init__(self, motion, sim, time_steps: torch.Tensor) -> None:
        self.motion = motion
        self._env = SimpleNamespace(simulator=sim)
        self.time_steps = time_steps
        self.motion_ids = torch.zeros_like(time_steps)
        self.ref_body_index = _MOCK_REF_BODY_INDEX

    # wbt.py:834-838
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    # wbt.py:876-890
    @property
    def root_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps, self.root_body_index]
            + self._env.simulator.scene.env_origins
        )

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.root_body_index]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.root_body_index]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.root_body_index]


class _RefBodyRootMockCommand(_MockMotionCommand):
    """Counterfactual upstream in which the ``root_*_w`` family moved to ``ref_body_index``."""

    root_body_index = _MOCK_REF_BODY_INDEX


def _multi_body_motion(frames: int = 8, dofs: int = 2):
    bodies = _NUM_MOCK_BODIES
    # Body slot b carries a value band of its own, so an index mistake cannot alias.
    body_bias = (torch.arange(bodies, dtype=torch.float32) * 1000.0).reshape(1, bodies, 1)
    frame_bias = (torch.arange(frames, dtype=torch.float32) * 10.0).reshape(frames, 1, 1)
    axis = torch.tensor([0.1, 0.2, 0.3]).reshape(1, 1, 3)
    base = body_bias + frame_bias + axis
    quat = torch.nn.functional.normalize(base.repeat(1, 1, 2)[..., :4], dim=-1)
    return SimpleNamespace(
        joint_pos=torch.arange(frames * dofs, dtype=torch.float32).reshape(frames, dofs),
        joint_vel=torch.arange(frames * dofs, dtype=torch.float32).reshape(frames, dofs) + 100,
        body_pos_w=base.clone(),
        body_quat_w=quat,
        body_lin_vel_w=base.clone() + 0.5,
        body_ang_vel_w=base.clone() - 0.5,
        has_object=False,
    )


def _multi_body_sim(envs: int = 3, dofs: int = 2):
    return SimpleNamespace(
        dof_pos=torch.zeros(envs, dofs),
        dof_vel=torch.zeros(envs, dofs),
        robot_root_states=torch.zeros(envs, 13),
        scene=SimpleNamespace(
            env_origins=torch.tensor(
                [[0.0, 0.0, 0.0], [10.0, -1.0, 0.5], [20.0, 2.0, -0.25]][:envs]
            )
        ),
    )


def test_write_reference_state_uses_the_frozen_root_body_on_a_multi_body_motion() -> None:
    """Regression: the written root must be motion slot 0, never ``ref_body_index``."""

    assert FROZEN_ROOT_BODY_INDEX == 0
    motion = _multi_body_motion()
    sim = _multi_body_sim()
    command = _MockMotionCommand(motion, sim, torch.tensor([5, 2, 6]))
    env_ids = torch.tensor([0, 2])
    state_steps = torch.tensor([1, 4])
    origins = sim.scene.env_origins[env_ids]

    _write_reference_state(command, env_ids, state_steps)

    root = sim.robot_root_states[env_ids]
    assert torch.equal(root[:, :3], motion.body_pos_w[state_steps, 0] + origins)
    assert torch.equal(root[:, 3:7], motion.body_quat_w[state_steps, 0])
    assert torch.equal(root[:, 7:10], motion.body_lin_vel_w[state_steps, 0])
    assert torch.equal(root[:, 10:13], motion.body_ang_vel_w[state_steps, 0])

    # The origin offset applies to position only -- fixing an index must not add one.
    assert not torch.equal(root[:, :3], motion.body_pos_w[state_steps, 0])
    for columns, source in (
        (slice(7, 10), motion.body_lin_vel_w),
        (slice(10, 13), motion.body_ang_vel_w),
    ):
        assert not torch.allclose(root[:, columns], source[state_steps, 0] + origins)

    # ... and the reference body is emphatically not what got written.
    ref = command.ref_body_index
    assert ref != FROZEN_ROOT_BODY_INDEX
    assert not torch.allclose(root[:, :3], motion.body_pos_w[state_steps, ref] + origins)
    assert not torch.allclose(root[:, 3:7], motion.body_quat_w[state_steps, ref])
    assert not torch.allclose(root[:, 7:10], motion.body_lin_vel_w[state_steps, ref])
    assert not torch.allclose(root[:, 10:13], motion.body_ang_vel_w[state_steps, ref])

    # Untouched environments and the command cursor stay untouched.
    assert torch.equal(sim.robot_root_states[1], torch.zeros(13))
    assert command.time_steps.tolist() == [5, 2, 6]


def _frozen_wbt_bodyfix_reset(command, sim, env_ids: torch.Tensor) -> None:
    """Transcription of ``wbt_bodyfix.reset`` lines 117-122 (the frozen E70 write)."""

    sim.dof_pos[env_ids] = command.joint_pos[env_ids]
    sim.dof_vel[env_ids] = command.joint_vel[env_ids]
    sim.robot_root_states[env_ids, :3] = command.root_pos_w[env_ids]
    sim.robot_root_states[env_ids, 3:7] = command.root_quat_w[env_ids]
    sim.robot_root_states[env_ids, 7:10] = command.root_lin_vel_w[env_ids]
    sim.robot_root_states[env_ids, 10:13] = command.root_ang_vel_w[env_ids]


def test_write_reference_state_is_byte_identical_to_frozen_reset_on_the_diagonal() -> None:
    """On a diagonal cell (state side == command side) E71 must reproduce E70 exactly."""

    motion = _multi_body_motion()
    env_ids = torch.tensor([0, 1, 2])
    # Diagonal cell: the state cursor and the command cursor coincide, so the frozen reset
    # and the counterfactual reset are evaluated on identical inputs.
    state_steps = torch.tensor([1, 4, 6])

    frozen_sim = _multi_body_sim()
    frozen_command = _MockMotionCommand(motion, frozen_sim, state_steps.clone())
    _frozen_wbt_bodyfix_reset(frozen_command, frozen_sim, env_ids)

    counterfactual_sim = _multi_body_sim()
    counterfactual_command = _MockMotionCommand(motion, counterfactual_sim, state_steps.clone())
    assert_frozen_root_convention(counterfactual_command, env_ids)
    _write_reference_state(counterfactual_command, env_ids, state_steps)

    for name in ("robot_root_states", "dof_pos", "dof_vel"):
        frozen = getattr(frozen_sim, name)
        counterfactual = getattr(counterfactual_sim, name)
        assert frozen.dtype == counterfactual.dtype
        assert torch.equal(frozen, counterfactual), name
        assert frozen.numpy().tobytes() == counterfactual.numpy().tobytes(), name

    # The frozen write must not be trivially zero, or the comparison proves nothing.
    assert frozen_sim.robot_root_states.abs().sum() > 0


def test_assert_frozen_root_convention_catches_a_root_family_redefinition() -> None:
    """The runtime guard is what makes ``FROZEN_ROOT_BODY_INDEX`` non-assumptive."""

    motion = _multi_body_motion()
    env_ids = torch.tensor([0, 2])
    time_steps = torch.tensor([5, 2, 6])

    faithful = _MockMotionCommand(motion, _multi_body_sim(), time_steps.clone())
    assert_frozen_root_convention(faithful, env_ids)

    drifted = _RefBodyRootMockCommand(motion, _multi_body_sim(), time_steps.clone())
    with pytest.raises(RuntimeError, match="frozen root convention broken"):
        assert_frozen_root_convention(drifted, env_ids)


def test_standardized_branch_error_uses_pooled_scale_and_handles_constant_dims() -> None:
    joint_pos = torch.tensor([[0.0, 2.0], [2.0, 2.0]])
    joint_vel = torch.tensor([[0.0, -1.0], [4.0, 1.0]])
    scale = pooled_goal_scale(joint_pos, joint_vel)
    assert scale.tolist() == pytest.approx([1.0, 1.0, 2.0, 1.0])

    observed = torch.tensor([[0.0, 2.0, 0.0, 0.0]])
    reference_a = torch.tensor([[0.0, 2.0, 0.0, -1.0]])
    reference_b = torch.tensor([[2.0, 2.0, 4.0, 1.0]])
    error_a, error_b = standardized_branch_squared_error(
        observed, reference_a, reference_b, scale
    )
    assert error_a.tolist() == pytest.approx([0.25])
    assert error_b.tolist() == pytest.approx([2.25])


def test_normalized_branch_coordinate_has_fixed_reference_endpoints() -> None:
    q_ab = torch.tensor([0.5, 2.0])
    at_a = normalized_branch_coordinate(torch.zeros(2), q_ab, q_ab)
    at_b = normalized_branch_coordinate(q_ab, torch.zeros(2), q_ab)
    assert at_a.tolist() == pytest.approx([-1.0, -1.0])
    assert at_b.tolist() == pytest.approx([1.0, 1.0])

    # Equal orthogonal error added to both branches cancels from the coordinate.
    shifted = normalized_branch_coordinate(q_ab + 7.0, torch.full((2,), 7.0), q_ab)
    assert shifted.tolist() == pytest.approx([1.0, 1.0])


def test_normalized_branch_coordinate_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="identical"):
        normalized_branch_coordinate(torch.zeros(1), torch.zeros(2), torch.zeros(1))
    with pytest.raises(ValueError, match="nonnegative"):
        normalized_branch_coordinate(torch.tensor([-1.0]), torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError, match="epsilon"):
        normalized_branch_coordinate(torch.zeros(1), torch.zeros(1), torch.ones(1), epsilon=0)
