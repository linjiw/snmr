from __future__ import annotations

import pytest
import torch

from scripts.e70_video_runtime import (
    capture_start_grid,
    expected_simulator_envs,
    validate_capture_name,
)


def test_capture_start_grid_places_exact_start_in_env_zero() -> None:
    motion_starts = torch.tensor([0, 1000])
    motion_ends = torch.tensor([1000, 2200])
    grid = capture_start_grid(
        "1200", motion_starts, motion_ends, num_envs=8, horizon_steps=500
    )
    assert grid.shape == (8,)
    assert grid[0].item() == 1200
    assert torch.all((grid[1:] >= 1) & (grid[1:] <= 1698))


def test_capture_start_grid_rejects_boundary_crossing_and_noncanonical_input() -> None:
    starts, ends = torch.tensor([0, 1000]), torch.tensor([1000, 2200])
    with pytest.raises(ValueError, match="inside one motion"):
        capture_start_grid("800", starts, ends, num_envs=1, horizon_steps=500)
    with pytest.raises(ValueError, match="canonical"):
        capture_start_grid("+1200", starts, ends, num_envs=1, horizon_steps=500)


def test_marginal_intervention_requires_population_but_other_captures_do_not() -> None:
    assert expected_simulator_envs("marginal_random") == 1024
    assert expected_simulator_envs("none") == 1
    assert validate_capture_name("snmr_walk1_subject1") == "snmr_walk1_subject1"
    with pytest.raises(ValueError, match="invalid capture name"):
        validate_capture_name("../escape")
