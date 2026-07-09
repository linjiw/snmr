"""Retargeting quality metrics (design doc §6), computed identically for any qpos trajectory.

All metrics take a robot motion as (root_pos (T,3), root_quat wxyz (T,4), dof_pos (T,D)) plus the
robot's :class:`RobotKinematics`, so SNMR predictions and the GMR teacher are scored by exactly the
same code path. Definitions follow the conventions used in the literature we benchmark against:

  * **MPJPE** — mean per-body position error (m) between two trajectories' FK, whole-body.
  * **Foot skating** — mean XY drift speed (m/s) of foot bodies during contact, where contact is
    detected on the *reference* trajectory (height+velocity heuristic, matching holosoma's
    ``extract_foot_sticking_sequence_velocity`` thresholds: low = within 3 cm of that foot's clip
    minimum height; slow = XY speed < 0.3 m/s). Reported with the fraction of contact frames that
    slide (> 1 cm/s, holosoma's ``sliding_threshold`` scaled by fps).
  * **Ground penetration** — mean and max depth (m) any foot body dips below z=0, plus fraction of
    frames with penetration > 1 cm (OmniRetarget/holosoma tolerance).
  * **Jerk** — mean magnitude of the third finite difference of dof positions (rad/s^3) and of body
    positions (m/s^3); lower = smoother, the RL-consumability proxy.
  * **Joint-limit violations** — fraction of (frame, dof) samples outside limits (should be 0 for
    SNMR by construction; nonzero values flag model/data contract bugs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .robot_model import RobotKinematics


@dataclass
class MotionMetrics:
    mpjpe_m: float | None = None                 # vs a reference trajectory (None if no reference)
    per_body_mpjpe: dict = field(default_factory=dict)
    foot_skate_speed_ms: float = 0.0             # mean XY speed of feet during reference contact
    foot_slide_fraction: float = 0.0             # fraction of contact frames sliding > threshold
    foot_skate_mann_cm: float = 0.0              # MANN formula: v·(2−2^(h/H)), H=2.5 cm [cm/frame]
    penetration_mean_m: float = 0.0
    penetration_max_m: float = 0.0
    penetration_fraction: float = 0.0            # frames with any foot below -1 cm
    dof_jerk: float = 0.0                        # rad/s^3
    body_jerk: float = 0.0                       # m/s^3
    joint_jump_fraction: float = 0.0             # NMR: frames with any |Δdof| > 0.5 rad/step
    limit_violation_fraction: float = 0.0
    limit_proximity_fraction: float = 0.0        # NMR: frames with any dof within 0.05 rad of limit
    dof_err_rad: float | None = None             # vs reference dof

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "per_body_mpjpe"}
        return d


def mann_foot_skate(foot_pos: torch.Tensor, fps: float, H: float = 0.025) -> float:
    """Foot-skate metric from MANN (Zhang et al., SIGGRAPH 2018), the animation-community standard.

    s = v_xy · (2 − 2^(h/H)) accumulated over frames where foot height h ≤ H (2.5 cm), where v_xy is
    the per-frame horizontal displacement. Heights are measured relative to each foot's clip minimum.
    Returns the mean skate per frame in **cm** (Motion-VAEs convention: ground-truth mocap ≈ 0.1).
    """
    T = foot_pos.shape[0]
    if T < 2:
        return 0.0
    h = foot_pos[..., 2] - foot_pos[..., 2].min(dim=0, keepdim=True).values  # (T, F)
    disp = torch.zeros_like(h)
    disp[1:] = (foot_pos[1:, :, :2] - foot_pos[:-1, :, :2]).norm(dim=-1)     # m/frame
    w = (2.0 - torch.pow(2.0, (h / H).clamp(max=1.0)))
    s = disp * w * (h <= H)
    return float(s.sum() / T * 100.0)  # cm per frame, averaged over frames (summed over feet)


def detect_contact(
    foot_pos: torch.Tensor, fps: float,
    height_threshold: float = 0.03, speed_threshold: float = 0.3,
) -> torch.Tensor:
    """(T, F, 3) foot positions -> (T, F) bool contact mask (height + XY-speed heuristic).

    Heights are measured relative to each foot's own minimum over the clip (robust to ground offset
    and to feet whose body origin sits above the sole).
    """
    T = foot_pos.shape[0]
    height = foot_pos[..., 2] - foot_pos[..., 2].min(dim=0, keepdim=True).values
    low = height < height_threshold
    speed = torch.zeros_like(height)
    if T > 1:
        speed[1:] = (foot_pos[1:, :, :2] - foot_pos[:-1, :, :2]).norm(dim=-1) * fps
        speed[0] = speed[1]
    return low & (speed < speed_threshold)


def compute_metrics(
    kin: RobotKinematics,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    dof_pos: torch.Tensor,
    fps: float,
    foot_body_names: list[str],
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    slide_speed_threshold: float = 0.01 * 30,  # holosoma: 0.01 m/frame at 30 fps -> 0.3 m/s
) -> MotionMetrics:
    """Score one trajectory; if ``reference`` (root_pos, root_quat, dof_pos) is given, also MPJPE.

    Contact for the foot-skate metric is detected on the reference when provided (so the candidate
    is penalised for sliding when the *teacher* says the foot should be planted), else on the
    candidate itself.
    """
    m = MotionMetrics()
    T = root_pos.shape[0]
    body_pos, _ = kin.forward_kinematics(root_pos, root_quat, dof_pos)
    foot_idx = [kin.body_index(n) for n in foot_body_names]
    feet = body_pos[:, foot_idx, :]

    # --- reference-dependent metrics -------------------------------------------------------
    if reference is not None:
        ref_body_pos, _ = kin.forward_kinematics(*reference)
        err = (body_pos - ref_body_pos).norm(dim=-1)  # (T, B)
        m.mpjpe_m = float(err.mean())
        m.per_body_mpjpe = {
            name: float(err[:, i].mean()) for i, name in enumerate(kin.body_names)
        }
        m.dof_err_rad = float((dof_pos - reference[2]).abs().mean())
        contact = detect_contact(ref_body_pos[:, foot_idx, :], fps)
    else:
        contact = detect_contact(feet, fps)

    # --- foot skating ----------------------------------------------------------------------
    if T > 1:
        xy_speed = torch.zeros(T, len(foot_idx), device=feet.device)
        xy_speed[1:] = (feet[1:, :, :2] - feet[:-1, :, :2]).norm(dim=-1) * fps
        xy_speed[0] = xy_speed[1]
        c = contact.to(feet.dtype)
        denom = float(c.sum().clamp_min(1))
        m.foot_skate_speed_ms = float((xy_speed * c).sum() / denom)
        m.foot_slide_fraction = float(((xy_speed > slide_speed_threshold) & contact).sum() / denom)
        m.foot_skate_mann_cm = mann_foot_skate(feet, fps)

    # --- ground penetration ----------------------------------------------------------------
    depth = (-feet[..., 2]).clamp_min(0.0)
    m.penetration_mean_m = float(depth.mean())
    m.penetration_max_m = float(depth.max())
    m.penetration_fraction = float((depth.max(dim=1).values > 0.01).float().mean())

    # --- smoothness (jerk) -----------------------------------------------------------------
    if T >= 4:
        d3 = dof_pos[3:] - 3 * dof_pos[2:-1] + 3 * dof_pos[1:-2] - dof_pos[:-3]
        m.dof_jerk = float(d3.abs().mean() * fps**3)
        b3 = body_pos[3:] - 3 * body_pos[2:-1] + 3 * body_pos[1:-2] - body_pos[:-3]
        m.body_jerk = float(b3.norm(dim=-1).mean() * fps**3)

    # --- joint jumps (NMR: any dof step > 0.5 rad) ------------------------------------------
    if T > 1:
        step_size = (dof_pos[1:] - dof_pos[:-1]).abs().max(dim=1).values
        m.joint_jump_fraction = float((step_size > 0.5).float().mean())

    # --- joint limits ----------------------------------------------------------------------
    lo, hi = kin.dof_limits()
    lo = lo.to(dof_pos.dtype)
    hi = hi.to(dof_pos.dtype)
    viol = (dof_pos < lo - 1e-6) | (dof_pos > hi + 1e-6)
    m.limit_violation_fraction = float(viol.float().mean())
    near = (dof_pos < lo + 0.05) | (dof_pos > hi - 0.05)  # NMR proximity threshold
    m.limit_proximity_fraction = float(near.any(dim=1).float().mean())

    return m


# Foot bodies per robot (lowest/most-distal foot link on each leg, verified against each MJCF).
FOOT_BODIES = {
    "unitree_g1": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "booster_t1_29dof": ["left_toe_link", "right_toe_link"],
    "fourier_n1": ["left_foot_pitch_link", "right_foot_pitch_link"],
    "engineai_pm01": ["LINK_FOOT_L", "LINK_FOOT_R"],
    "stanford_toddy": ["left_toe_link", "right_toe_link"],
}
