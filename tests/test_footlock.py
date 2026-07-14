"""E26 foot-lock: mask dilation, correction smoothing, and the masked leg-IK lock."""

import pytest
import torch

from snmr.footlock import (
    _leg_dof_indices,
    _solve_foot_dls,
    dilate_contact_mask,
    foot_lock,
    foot_lock_masked,
    smooth_correction,
)
from snmr.metrics import FOOT_BODIES
from snmr.robot_model import RobotKinematics


def _standing_motion(rk, T=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo, hi = rk.dof_limits()
    mid = (lo + hi) / 2
    dof = mid.expand(T, -1).clone()
    dof += 0.02 * torch.randn(T, rk.num_dof, generator=g)  # mm-level per-frame noise -> skate
    root_pos = torch.zeros(T, 3)
    root_pos[:, 2] = 0.75
    root_quat = torch.tensor([1.0, 0, 0, 0]).expand(T, 4).clone()
    return root_pos, root_quat, dof


def test_dilate_merges_gaps_and_extends():
    mask = torch.zeros(20, 2, dtype=torch.bool)
    mask[2:6, 0] = True
    mask[8:12, 0] = True   # gap of 2 -> merged at merge_gap>=2
    mask[15:17, 1] = True
    out = dilate_contact_mask(mask, merge_gap=3, extend=1)
    assert out[1:13, 0].all()          # merged [2,6)+[8,12) -> [2,12), extended -> [1,13)
    assert not out[0, 0] and not out[13, 0]
    assert out[14:18, 1].all()         # extended by 1 both sides
    assert out.sum() > mask.sum()


def test_dilate_noop_on_empty_and_full():
    empty = torch.zeros(10, 2, dtype=torch.bool)
    assert dilate_contact_mask(empty).sum() == 0
    full = torch.ones(10, 2, dtype=torch.bool)
    assert dilate_contact_mask(full).all()


def test_smooth_correction_preserves_constant_offset():
    T, D = 30, 5
    raw = torch.randn(T, D)
    delta = torch.ones(T, D) * 0.3          # constant correction -> low-pass is identity
    sm = smooth_correction(raw, raw + delta, sigma=1.5)
    assert torch.allclose(sm, raw + delta, atol=1e-5)


def test_smooth_correction_damps_alternating_noise():
    T, D = 30, 3
    raw = torch.zeros(T, D)
    noise = 0.1 * (-1.0) ** torch.arange(T, dtype=torch.float32)
    locked = raw + noise.unsqueeze(-1)
    sm = smooth_correction(raw, locked, sigma=1.0)
    assert sm.abs().max() < 0.05             # Nyquist noise strongly attenuated


def test_foot_lock_masked_reduces_stance_velocity(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    feet = FOOT_BODIES["unitree_g1"]
    root_pos, root_quat, dof = _standing_motion(rk)
    T = dof.shape[0]
    mask = torch.ones(T, len(feet), dtype=torch.bool)   # whole window is stance

    foot_idx = [rk.body_index(n) for n in feet]

    def stance_speed(d):
        bp, _ = rk.forward_kinematics(root_pos, root_quat, d)
        f = bp[:, foot_idx, :2]
        return (f[1:] - f[:-1]).norm(dim=-1).mean()

    v_raw = stance_speed(dof)
    locked = foot_lock_masked(rk, root_pos, root_quat, dof, feet, mask, smooth_sigma=1.0)
    v_locked = stance_speed(locked)
    assert v_locked < 0.5 * v_raw            # oscillation substantially removed
    assert torch.isfinite(locked).all()
    lo, hi = rk.dof_limits()
    assert (locked >= lo).all() and (locked <= hi).all()


def test_batched_dls_matches_independent_frame_solves(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    foot = FOOT_BODIES["unitree_g1"][0]
    root_pos, root_quat, dof = _standing_motion(rk, T=4)
    body_pos, _ = rk.forward_kinematics(root_pos, root_quat, dof)
    foot_index = rk.body_index(foot)
    target = body_pos[:, foot_index].detach().clone()
    target[:, 0] += torch.tensor([0.004, -0.003, 0.002, -0.001])
    leg_dofs = torch.tensor(_leg_dof_indices(rk, foot), dtype=torch.long)

    batched = _solve_foot_dls(
        rk, root_pos, root_quat, dof, foot_index, leg_dofs, target, 3, 0.5, 1e-2
    )
    independent = torch.stack([
        _solve_foot_dls(
            rk,
            root_pos[frame],
            root_quat[frame],
            dof[frame],
            foot_index,
            leg_dofs,
            target[frame],
            3,
            0.5,
            1e-2,
        )
        for frame in range(dof.shape[0])
    ])
    assert torch.allclose(batched, independent, atol=1e-6, rtol=1e-6)


def test_foot_lock_masked_respects_empty_mask(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    feet = FOOT_BODIES["unitree_g1"]
    root_pos, root_quat, dof = _standing_motion(rk, T=8)
    mask = torch.zeros(8, len(feet), dtype=torch.bool)
    locked = foot_lock_masked(rk, root_pos, root_quat, dof, feet, mask)
    assert torch.allclose(locked, dof)       # no contact intervals -> identity


def test_foot_lock_masked_skips_isolated_contact(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    feet = FOOT_BODIES["unitree_g1"]
    root_pos, root_quat, dof = _standing_motion(rk, T=8)
    mask = torch.zeros(8, len(feet), dtype=torch.bool)
    mask[4, 0] = True
    locked = foot_lock_masked(
        rk,
        root_pos,
        root_quat,
        dof,
        feet,
        mask,
        smooth_sigma=1.0,
        merge_gap=0,
        extend=0,
    )
    assert torch.isfinite(locked).all()
    assert torch.allclose(locked, dof)


def test_foot_lock_masked_rejects_wrong_mask_shape(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    feet = FOOT_BODIES["unitree_g1"]
    root_pos, root_quat, dof = _standing_motion(rk, T=8)
    with pytest.raises(ValueError, match="contact mask shape"):
        foot_lock_masked(
            rk,
            root_pos,
            root_quat,
            dof,
            feet,
            torch.zeros(8, 1, dtype=torch.bool),
        )


def test_legacy_foot_lock_still_runs(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    feet = FOOT_BODIES["unitree_g1"]
    root_pos, root_quat, dof = _standing_motion(rk, T=8)
    out = foot_lock(rk, root_pos, root_quat, dof, feet, fps=30.0, iters=2)
    assert out.shape == dof.shape and torch.isfinite(out).all()
