"""Metric correctness on synthetic trajectories with known ground-truth values."""

import torch

from snmr.metrics import (
    FOOT_BODIES,
    MotionMetrics,
    compute_metrics,
    contact_motion_metrics,
    detect_contact,
    detect_contact_height_hysteresis,
)
from snmr.robot_model import RobotKinematics


def _standing(rk, T=40, z=0.75):
    root_pos = torch.zeros(T, 3)
    root_pos[:, 2] = z
    root_quat = torch.tensor([1.0, 0, 0, 0]).expand(T, 4).clone()
    dof = torch.zeros(T, rk.num_dof)
    return root_pos, root_quat, dof


def test_identical_trajectories_zero_mpjpe(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    traj = _standing(rk)
    m = compute_metrics(rk, *traj, fps=50.0, foot_body_names=FOOT_BODIES["unitree_g1"],
                        reference=traj)
    assert m.mpjpe_m == 0.0
    assert m.dof_err_rad == 0.0
    assert m.limit_violation_fraction == 0.0
    assert m.dof_jerk == 0.0


def test_constant_offset_mpjpe(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    ref = _standing(rk)
    shifted = (ref[0] + torch.tensor([0.05, 0.0, 0.0]), ref[1], ref[2])
    m = compute_metrics(rk, *shifted, fps=50.0, foot_body_names=FOOT_BODIES["unitree_g1"],
                        reference=ref)
    # a rigid 5 cm translation shifts every body exactly 5 cm
    assert abs(m.mpjpe_m - 0.05) < 1e-6


def test_foot_skate_detected_on_sliding_stand(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    T, fps = 40, 50.0
    ref = _standing(rk, T)
    # candidate slides horizontally at 0.5 m/s while the reference stands still (feet in contact)
    slide = ref[0].clone()
    slide[:, 0] = torch.arange(T) * (0.5 / fps)
    m = compute_metrics(rk, slide, ref[1], ref[2], fps=fps,
                        foot_body_names=FOOT_BODIES["unitree_g1"], reference=ref)
    assert 0.4 < m.foot_skate_speed_ms < 0.6, m.foot_skate_speed_ms
    assert m.foot_slide_fraction > 0.9


def test_no_skate_when_reference_says_airborne(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    T, fps = 40, 50.0
    ref_pos, ref_quat, ref_dof = _standing(rk, T)
    # reference feet move fast (airborne/running) -> no contact -> candidate sliding is not penalised
    ref_moving = ref_pos.clone()
    ref_moving[:, 0] = torch.arange(T) * (1.0 / fps)
    cand = ref_pos.clone()
    cand[:, 0] = torch.arange(T) * (0.5 / fps)
    m = compute_metrics(rk, cand, ref_quat, ref_dof, fps=fps,
                        foot_body_names=FOOT_BODIES["unitree_g1"],
                        reference=(ref_moving, ref_quat, ref_dof))
    assert m.foot_skate_speed_ms == 0.0 or m.foot_slide_fraction == 0.0


def test_penetration_measured(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    T = 20
    root_pos, root_quat, dof = _standing(rk, T, z=0.75)
    # sink the whole robot so feet go below the ground by ~5 cm
    body_pos, _ = rk.forward_kinematics(root_pos[0], root_quat[0], dof[0])
    foot_idx = rk.body_index(FOOT_BODIES["unitree_g1"][0])
    foot_z = float(body_pos[foot_idx, 2])
    sunk = root_pos.clone()
    sunk[:, 2] -= foot_z + 0.05
    m = compute_metrics(rk, sunk, root_quat, dof, fps=50.0,
                        foot_body_names=FOOT_BODIES["unitree_g1"])
    assert m.penetration_max_m > 0.04
    assert m.penetration_fraction == 1.0


def test_limit_violations_counted(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    root_pos, root_quat, dof = _standing(rk, 10)
    lo, hi = rk.dof_limits()
    dof = dof.clone()
    dof[:, 0] = hi[0] + 0.5  # first dof out of range every frame
    m = compute_metrics(rk, root_pos, root_quat, dof, fps=50.0,
                        foot_body_names=FOOT_BODIES["unitree_g1"])
    expected = 1.0 / rk.num_dof
    assert abs(m.limit_violation_fraction - expected) < 1e-6


def test_mann_skate_zero_when_still_or_airborne():
    from snmr.metrics import mann_foot_skate

    T = 30
    still = torch.zeros(T, 2, 3)
    assert mann_foot_skate(still, fps=30.0) == 0.0
    airborne = still.clone()
    airborne[:, :, 2] = 0.5  # relative height stays 0 (clip minimum) — raise only later frames
    airborne[15:, :, 2] = 0.8
    airborne[15:, :, 0] = torch.arange(15).float()[:, None] * 0.02  # slides while high
    s = mann_foot_skate(airborne, fps=30.0)
    assert s == 0.0, s  # sliding above H must not count


def test_mann_skate_counts_ground_slide():
    from snmr.metrics import mann_foot_skate

    T = 30
    f = torch.zeros(T, 1, 3)
    f[:, 0, 0] = torch.arange(T).float() * 0.01  # 1 cm/frame slide at ground level
    s = mann_foot_skate(f, fps=30.0)
    # weight at h=0 is (2 - 2^0) = 1 -> skate ~ 1 cm/frame (first frame has no displacement)
    assert 0.9 < s < 1.1, s


def test_joint_jump_and_limit_proximity(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    root_pos, root_quat, dof = _standing(rk, 10)
    dof = dof.clone()
    dof[5, 0] = 0.6  # one 0.6-rad step up then 0.6 down = 2 jump frames of 9 steps
    m = compute_metrics(rk, root_pos, root_quat, dof, fps=50.0,
                        foot_body_names=FOOT_BODIES["unitree_g1"])
    assert abs(m.joint_jump_fraction - 2.0 / 9.0) < 1e-6
    lo, hi = rk.dof_limits()
    dof2 = torch.zeros(10, rk.num_dof)
    dof2[:, 3] = hi[3] - 0.01  # within 0.05 rad of the knee limit every frame
    m2 = compute_metrics(rk, root_pos, root_quat, dof2, fps=50.0,
                         foot_body_names=FOOT_BODIES["unitree_g1"])
    assert m2.limit_proximity_fraction == 1.0
    assert m2.limit_violation_fraction == 0.0


def test_contact_detection_heights():
    # synthetic single foot: contact for first half (low, still), airborne second half
    T = 20
    foot = torch.zeros(T, 1, 3)
    foot[10:, 0, 2] = 0.3  # lifts up
    c = detect_contact(foot, fps=50.0)
    assert c[:10, 0].all()
    assert not c[10:, 0].any()


def test_height_contact_hysteresis_requires_reentry_threshold():
    foot = torch.zeros(6, 1, 3)
    foot[:, 0, 2] = torch.tensor([0.00, 0.02, 0.04, 0.06, 0.04, 0.02])
    contact = detect_contact_height_hysteresis(
        foot, enter_height=0.03, exit_height=0.05, ground_z=0.0
    )
    assert contact[:, 0].tolist() == [True, True, True, False, False, True]


def test_explicit_contact_mask_removes_speed_gate_circularity(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    frames, fps = 20, 50.0
    moving = _standing(rk, frames)
    moving[0][:, 0] = torch.arange(frames) * 0.5 / fps
    explicit = torch.ones(frames, 2, dtype=torch.bool)
    metrics = compute_metrics(
        rk,
        *moving,
        fps=fps,
        foot_body_names=FOOT_BODIES["unitree_g1"],
        contact_mask=explicit,
    )
    assert 0.4 < metrics.foot_skate_speed_ms < 0.6
    assert metrics.foot_slide_fraction > 0.9


def test_floating_is_reference_relative_during_stance(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    reference = _standing(rk, 20)
    raised_root = reference[0].clone()
    raised_root[:, 2] += 0.05
    mask = torch.ones(20, 2, dtype=torch.bool)
    metrics = compute_metrics(
        rk,
        raised_root,
        reference[1],
        reference[2],
        fps=50.0,
        foot_body_names=FOOT_BODIES["unitree_g1"],
        reference=reference,
        contact_mask=mask,
    )
    assert abs(metrics.foot_floating_mean_m - 0.05) < 1e-6
    assert metrics.foot_floating_fraction == 1.0


def test_contact_motion_metrics_reports_per_foot_values():
    frames, fps = 10, 50.0
    feet = torch.zeros(frames, 2, 3)
    feet[:, 0, 0] = torch.arange(frames) * 0.2 / fps
    feet[:, 1, 0] = torch.arange(frames) * 0.4 / fps
    mask = torch.ones(frames, 2, dtype=torch.bool)
    metrics = contact_motion_metrics(feet, fps, mask, slide_speed_threshold=0.3)
    assert abs(metrics["stance_speed_ms"] - 0.3) < 1e-6
    assert abs(metrics["per_foot_stance_speed_ms"][0] - 0.2) < 1e-6
    assert abs(metrics["per_foot_stance_speed_ms"][1] - 0.4) < 1e-6
    assert abs(metrics["slide_fraction"] - 0.5) < 1e-6

    reference = feet.clone()
    raised = feet.clone()
    raised[:, 0, 2] += 0.05
    floating = contact_motion_metrics(
        raised, fps, mask, reference_foot_pos=reference
    )
    assert abs(floating["floating_mean_m"] - 0.025) < 1e-6
    assert abs(floating["floating_fraction"] - 0.5) < 1e-6
    assert floating["per_foot_floating_fraction"] == [1.0, 0.0]


def test_contact_motion_metrics_empty_mask_is_undefined():
    feet = torch.zeros(4, 2, 3)
    metrics = contact_motion_metrics(
        feet,
        fps=30.0,
        contact_mask=torch.zeros(4, 2, dtype=torch.bool),
    )
    assert metrics["contact_samples"] == 0
    assert metrics["stance_speed_ms"] is None
    assert metrics["slide_fraction"] is None
    assert metrics["floating_mean_m"] is None
    assert metrics["floating_fraction"] is None
    assert metrics["per_foot_stance_speed_ms"] == [None, None]


def test_all_foot_bodies_exist():
    """Every FOOT_BODIES entry must name real bodies in its robot's MJCF."""
    import pytest

    from snmr import paths as snmr_paths

    checked = 0
    for robot, feet in FOOT_BODIES.items():
        try:
            p = snmr_paths.robot_mjcf(robot)
        except (KeyError, FileNotFoundError):
            continue
        if not p.exists():
            continue
        rk = RobotKinematics(str(p))
        for f in feet:
            assert f in rk.body_names, f"{robot}: foot body '{f}' not in MJCF"
        checked += 1
    if checked == 0:
        pytest.skip("no robot MJCFs found")
