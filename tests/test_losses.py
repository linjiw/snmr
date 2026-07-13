import torch

from snmr import losses
from snmr.metrics import FOOT_BODIES
from snmr.robot_model import RobotKinematics


def _translated_standing(rk, speed: float, fps: float, frames: int = 10):
    root_pos = torch.zeros(frames, 3)
    root_pos[:, 0] = torch.arange(frames) * speed / fps
    root_pos[:, 2] = 0.75
    root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(frames, 4).clone()
    dof = torch.zeros(frames, rk.num_dof)
    return {"root_pos": root_pos, "root_quat": root_quat, "dof_pos": dof}


def test_teacher_stance_velocity_is_mps_and_normalized_by_active_samples(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    fps = 50.0
    speed = 0.4
    pred = _translated_standing(rk, speed, fps)
    feet = [rk.body_index(name) for name in FOOT_BODIES["unitree_g1"]]
    mask_one = torch.zeros(10, 2)
    mask_one[1:, 0] = 1.0
    mask_two = torch.ones(10, 2)

    one = losses.teacher_stance_velocity_loss(pred, rk, feet, mask_one, fps)
    two = losses.teacher_stance_velocity_loss(pred, rk, feet, mask_two, fps)
    expected = speed**2 / 2.0  # x velocity contributes; y velocity is zero
    assert abs(float(one) - expected) < 1e-6
    assert abs(float(two) - expected) < 1e-6


def test_teacher_stance_velocity_empty_mask_is_differentiable_zero(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    pred = _translated_standing(rk, 0.4, 50.0)
    pred["root_pos"].requires_grad_(True)
    feet = [rk.body_index(name) for name in FOOT_BODIES["unitree_g1"]]
    loss = losses.teacher_stance_velocity_loss(
        pred, rk, feet, torch.zeros(10, 2), fps=50.0
    )
    assert float(loss) == 0.0
    loss.backward()
    assert pred["root_pos"].grad is not None
    assert float(pred["root_pos"].grad.abs().sum()) == 0.0


def test_phase_balanced_teacher_velocity_does_not_weight_by_prevalence(g1_mjcf):
    rk = RobotKinematics(g1_mjcf)
    fps = 50.0
    pred = _translated_standing(rk, 0.4, fps)
    teacher = _translated_standing(rk, 0.0, fps)
    feet = [rk.body_index(name) for name in FOOT_BODIES["unitree_g1"]]
    contact = torch.zeros(10, 2)
    contact[1:3] = 1.0
    loss = losses.teacher_foot_velocity_loss(
        pred,
        rk,
        teacher["root_pos"],
        teacher["root_quat"],
        teacher["dof_pos"],
        feet,
        fps,
        contact_mask=contact,
        phase_balanced=True,
    )
    assert abs(float(loss) - 0.4**2 / 3.0) < 1e-6
