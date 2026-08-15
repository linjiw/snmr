from types import SimpleNamespace

import pytest
import torch

from snmr.integration.counterfactual_eval import (
    E71_FULL_STATE_TENSOR_NAMES,
    E71_EVALUATION_CONDITIONS,
    E71_RUNTIME_CONTRACT,
    _write_reference_state,
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
