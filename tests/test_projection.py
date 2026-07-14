import pytest
import torch

from snmr.metrics import FOOT_BODIES
from snmr.projection import (
    WindowedProjectionConfig,
    _norm_bounded_translation,
    build_stance_anchors,
    windowed_contact_projection,
)
from snmr.robot_model import RobotKinematics


def _standing_motion(rk, T=12, seed=7):
    generator = torch.Generator().manual_seed(seed)
    lo, hi = rk.dof_limits()
    dof = ((lo + hi) / 2).expand(T, -1).clone()
    dof += 0.03 * torch.randn(T, rk.num_dof, generator=generator)
    root_pos = torch.zeros(T, 3)
    root_pos[:, 2] = 0.75
    root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(T, 4).clone()
    return root_pos, root_quat, dof


def test_build_stance_anchors_keeps_per_foot_intervals_separate():
    feet = torch.zeros(8, 2, 3)
    feet[:, 0, 0] = torch.arange(8)
    feet[:, 1, 1] = torch.arange(8)
    mask = torch.zeros(8, 2, dtype=torch.bool)
    mask[1:4, 0] = True
    mask[5:8, 0] = True
    mask[2:7, 1] = True

    targets, active, intervals = build_stance_anchors(feet, mask)

    assert len(intervals) == 3
    assert active.equal(mask)
    assert torch.all(targets[1:4, 0, 0] == 2)
    assert torch.all(targets[5:8, 0, 0] == 6)
    assert torch.all(targets[2:7, 1, 1] == 4)
    assert active[2:4].all()  # both feet constrained during double support


def test_norm_bounded_root_translation_has_gradient_at_zero():
    raw = torch.zeros(4, 3, requires_grad=True)
    projected = _norm_bounded_translation(raw, 0.04)
    projected.sum().backward()

    assert torch.allclose(raw.grad, torch.full_like(raw, 0.04))


def test_windowed_projection_empty_mask_is_exact_identity(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    feet = FOOT_BODIES["unitree_g1"]
    root_pos, root_quat, dof = _standing_motion(rk)
    mask = torch.zeros(dof.shape[0], len(feet), dtype=torch.bool)

    result = windowed_contact_projection(
        rk,
        root_pos,
        root_quat,
        dof,
        feet,
        mask,
    )

    assert torch.equal(result.root_pos, root_pos)
    assert torch.equal(result.root_quat, root_quat)
    assert torch.equal(result.dof_pos, dof)
    assert result.diagnostics["reason"] == "empty_contact_support"


def test_windowed_projection_reduces_anchor_error_within_all_bounds(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    feet = FOOT_BODIES["unitree_g1"]
    root_pos, root_quat, dof = _standing_motion(rk)
    mask = torch.ones(dof.shape[0], len(feet), dtype=torch.bool)
    config = WindowedProjectionConfig(
        max_iterations=15,
        stance_weight=2000.0,
        deviation_weight=0.05,
        velocity_weight=0.2,
        acceleration_weight=0.5,
        root_translation_bound_m=0.02,
        root_yaw_bound_rad=0.05,
        joint_delta_bound_rad=0.2,
    )

    result = windowed_contact_projection(
        rk,
        root_pos,
        root_quat,
        dof,
        feet,
        mask,
        config=config,
    )

    diagnostics = result.diagnostics
    assert diagnostics["accepted"]
    assert diagnostics["final_anchor_rmse_m"] < diagnostics["initial_anchor_rmse_m"]
    assert diagnostics["max_root_translation_m"] <= config.root_translation_bound_m + 1e-6
    assert diagnostics["max_root_yaw_rad"] <= config.root_yaw_bound_rad + 1e-6
    assert diagnostics["max_joint_delta_rad"] <= config.joint_delta_bound_rad + 1e-6
    lo, hi = rk.dof_limits()
    assert (result.dof_pos >= lo - 1e-6).all()
    assert (result.dof_pos <= hi + 1e-6).all()
    assert torch.allclose(result.root_quat.norm(dim=-1), torch.ones(dof.shape[0]), atol=1e-6)
    assert len(diagnostics["stance_intervals"]) == len(feet)


def test_windowed_projection_rejects_wrong_mask_shape(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    feet = FOOT_BODIES["unitree_g1"]
    root_pos, root_quat, dof = _standing_motion(rk)

    with pytest.raises(ValueError, match="contact mask shape"):
        windowed_contact_projection(
            rk,
            root_pos,
            root_quat,
            dof,
            feet,
            torch.ones(dof.shape[0], 1, dtype=torch.bool),
        )
