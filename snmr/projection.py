"""Windowed constrained contact projection for Gate 1 C6.

Unlike :mod:`snmr.footlock`, this solver optimizes one temporal window jointly. It builds one
world-space anchor per contiguous stance interval and uses differentiable L-BFGS over both leg
chains plus bounded root translation/yaw corrections. Joint and root constraints are enforced by
the variable parameterization, while correction magnitude and first/second temporal differences
regularize the solution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from . import rotation as rot
from .footlock import _intervals, _leg_dof_indices, dilate_contact_mask
from .robot_model import RobotKinematics


@dataclass(frozen=True)
class WindowedProjectionConfig:
    max_iterations: int = 30
    history_size: int = 10
    learning_rate: float = 1.0
    stance_weight: float = 1000.0
    deviation_weight: float = 0.1
    velocity_weight: float = 0.5
    acceleration_weight: float = 1.0
    root_translation_bound_m: float = 0.04
    root_yaw_bound_rad: float = 0.12
    joint_delta_bound_rad: float = 0.35
    merge_gap: int = 0
    extend: int = 0
    min_stance_frames: int = 2
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9


@dataclass
class WindowedProjectionResult:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    dof_pos: torch.Tensor
    diagnostics: dict[str, Any]


def _validate_inputs(
    kin: RobotKinematics,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    dof_pos: torch.Tensor,
    foot_body_names: list[str],
    contact_mask: torch.Tensor,
    config: WindowedProjectionConfig,
) -> None:
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"expected root_pos (T,3), got {tuple(root_pos.shape)}")
    if root_quat.shape != (root_pos.shape[0], 4):
        raise ValueError(f"expected root_quat (T,4), got {tuple(root_quat.shape)}")
    if dof_pos.shape != (root_pos.shape[0], kin.num_dof):
        raise ValueError(
            f"expected dof_pos ({root_pos.shape[0]},{kin.num_dof}), got {tuple(dof_pos.shape)}"
        )
    expected_mask = (root_pos.shape[0], len(foot_body_names))
    if contact_mask.shape != expected_mask:
        raise ValueError(
            f"contact mask shape {tuple(contact_mask.shape)} != expected {expected_mask}"
        )
    if not (
        root_pos.is_floating_point()
        and root_quat.is_floating_point()
        and dof_pos.is_floating_point()
    ):
        raise ValueError("projection inputs must use floating-point tensors")
    devices = {root_pos.device, root_quat.device, dof_pos.device, contact_mask.device}
    if len(devices) != 1:
        raise ValueError(
            f"all projection inputs must share one device, got {sorted(map(str, devices))}"
        )

    positive = {
        "max_iterations": config.max_iterations,
        "history_size": config.history_size,
        "learning_rate": config.learning_rate,
        "root_translation_bound_m": config.root_translation_bound_m,
        "root_yaw_bound_rad": config.root_yaw_bound_rad,
        "joint_delta_bound_rad": config.joint_delta_bound_rad,
        "min_stance_frames": config.min_stance_frames,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"projection config values must be positive: {invalid}")
    weights = (
        config.stance_weight,
        config.deviation_weight,
        config.velocity_weight,
        config.acceleration_weight,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("projection objective weights must be nonnegative")
    if config.merge_gap < 0 or config.extend < 0:
        raise ValueError("merge_gap and extend must be nonnegative")

    lo, hi = kin.dof_limits()
    lo = lo.to(device=dof_pos.device, dtype=dof_pos.dtype)
    hi = hi.to(device=dof_pos.device, dtype=dof_pos.dtype)
    if bool(((dof_pos < lo - 1e-5) | (dof_pos > hi + 1e-5)).any()):
        raise ValueError("input dof_pos violates robot joint limits")


def build_stance_anchors(
    foot_pos: torch.Tensor,
    contact_mask: torch.Tensor,
    *,
    min_stance_frames: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Build one fixed anchor for each contiguous per-foot stance interval."""
    if foot_pos.ndim != 3 or foot_pos.shape[-1] != 3:
        raise ValueError(f"expected foot_pos (T,F,3), got {tuple(foot_pos.shape)}")
    if contact_mask.shape != foot_pos.shape[:2]:
        raise ValueError(
            f"contact mask shape {tuple(contact_mask.shape)} != feet {tuple(foot_pos.shape[:2])}"
        )
    if min_stance_frames <= 0:
        raise ValueError("min_stance_frames must be positive")

    targets = torch.zeros_like(foot_pos)
    active = torch.zeros_like(contact_mask, dtype=torch.bool)
    intervals: list[dict[str, Any]] = []
    with torch.no_grad():
        for foot_index in range(foot_pos.shape[1]):
            for lo, hi in _intervals(contact_mask[:, foot_index].bool()):
                if hi - lo < min_stance_frames:
                    continue
                segment = foot_pos[lo:hi, foot_index]
                anchor = torch.empty(3, device=foot_pos.device, dtype=foot_pos.dtype)
                anchor[:2] = segment[:, :2].median(dim=0).values
                anchor[2] = segment[:, 2].min()
                targets[lo:hi, foot_index] = anchor
                active[lo:hi, foot_index] = True
                intervals.append({
                    "foot_index": foot_index,
                    "start": lo,
                    "end": hi,
                    "anchor": [float(value) for value in anchor.cpu()],
                })
    return targets, active, intervals


def _norm_bounded_translation(raw: torch.Tensor, bound: float) -> torch.Tensor:
    norm = raw.norm(dim=-1, keepdim=True)
    epsilon = torch.finfo(raw.dtype).eps
    ratio = torch.tanh(norm) / norm.clamp_min(epsilon)
    scale = torch.where(norm > epsilon, ratio, torch.ones_like(norm))
    return bound * raw * scale


def _joint_raw_initial(lower_delta: torch.Tensor, upper_delta: torch.Tensor) -> torch.Tensor:
    span = (upper_delta - lower_delta).clamp_min(torch.finfo(lower_delta.dtype).eps)
    fraction = (-lower_delta / span).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(fraction) - torch.log1p(-fraction)


def _joint_delta_from_raw(
    raw: torch.Tensor,
    lower_delta: torch.Tensor,
    upper_delta: torch.Tensor,
) -> torch.Tensor:
    return lower_delta + (upper_delta - lower_delta) * torch.sigmoid(raw)


def windowed_contact_projection(
    kin: RobotKinematics,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    dof_pos: torch.Tensor,
    foot_body_names: list[str],
    contact_mask: torch.Tensor,
    *,
    config: WindowedProjectionConfig | None = None,
) -> WindowedProjectionResult:
    """Jointly project one motion window onto fixed stance anchors.

    The caller supplies a non-circular contact mask, normally inferred from source motion or a
    teacher signal. The optimizer never thresholds the candidate output being corrected.
    """
    cfg = config or WindowedProjectionConfig()
    _validate_inputs(
        kin,
        root_pos,
        root_quat,
        dof_pos,
        foot_body_names,
        contact_mask,
        cfg,
    )
    mask = dilate_contact_mask(
        contact_mask.bool(),
        merge_gap=cfg.merge_gap,
        extend=cfg.extend,
    )
    foot_indices = [kin.body_index(name) for name in foot_body_names]
    with torch.no_grad():
        body_pos, _ = kin.forward_kinematics(root_pos, root_quat, dof_pos)
        original_feet = body_pos[:, foot_indices]
        targets, active, intervals = build_stance_anchors(
            original_feet,
            mask,
            min_stance_frames=cfg.min_stance_frames,
        )

    base_diagnostics: dict[str, Any] = {
        "method": "windowed_lbfgs",
        "config": asdict(cfg),
        "stance_intervals": intervals,
        "stance_samples": int(active.sum()),
    }
    if not bool(active.any()):
        base_diagnostics.update({
            "accepted": True,
            "reason": "empty_contact_support",
            "optimizer_iterations": 0,
            "function_evaluations": 0,
        })
        return WindowedProjectionResult(
            root_pos=root_pos.clone(),
            root_quat=root_quat.clone(),
            dof_pos=dof_pos.clone(),
            diagnostics=base_diagnostics,
        )

    leg_dofs = sorted(
        {
            dof_index
            for foot_name in foot_body_names
            for dof_index in _leg_dof_indices(kin, foot_name)
        }
    )
    if not leg_dofs:
        raise ValueError("no leg degrees of freedom found for the requested feet")
    leg_index = torch.tensor(leg_dofs, dtype=torch.long, device=dof_pos.device)
    base_leg = dof_pos[:, leg_index]
    lo, hi = kin.dof_limits()
    lo = lo.to(device=dof_pos.device, dtype=dof_pos.dtype)[leg_index]
    hi = hi.to(device=dof_pos.device, dtype=dof_pos.dtype)[leg_index]
    edit_bound = torch.as_tensor(
        cfg.joint_delta_bound_rad,
        device=dof_pos.device,
        dtype=dof_pos.dtype,
    )
    lower_delta = torch.maximum(lo - base_leg, -edit_bound)
    upper_delta = torch.minimum(hi - base_leg, edit_bound)

    joint_raw = torch.nn.Parameter(_joint_raw_initial(lower_delta, upper_delta))
    root_translation_raw = torch.nn.Parameter(torch.zeros_like(root_pos))
    root_yaw_raw = torch.nn.Parameter(root_pos.new_zeros(root_pos.shape[0]))
    parameters = [joint_raw, root_translation_raw, root_yaw_raw]
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=cfg.learning_rate,
        max_iter=cfg.max_iterations,
        history_size=cfg.history_size,
        tolerance_grad=cfg.tolerance_grad,
        tolerance_change=cfg.tolerance_change,
        line_search_fn="strong_wolfe",
    )
    yaw_axis = root_pos.new_tensor([0.0, 0.0, 1.0]).expand(root_pos.shape[0], -1)

    def decode() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        joint_delta = _joint_delta_from_raw(joint_raw, lower_delta, upper_delta)
        projected_dof = dof_pos.index_copy(1, leg_index, base_leg + joint_delta)
        translation_delta = _norm_bounded_translation(
            root_translation_raw,
            cfg.root_translation_bound_m,
        )
        yaw_delta = cfg.root_yaw_bound_rad * torch.tanh(root_yaw_raw)
        yaw_quat = rot.axis_angle_to_quat(yaw_axis, yaw_delta)
        projected_quat = rot.quat_normalize(rot.quat_mul(yaw_quat, root_quat))
        projected_root = root_pos + translation_delta
        normalized_correction = torch.cat(
            (
                translation_delta / cfg.root_translation_bound_m,
                (yaw_delta / cfg.root_yaw_bound_rad).unsqueeze(-1),
                joint_delta / cfg.joint_delta_bound_rad,
            ),
            dim=-1,
        )
        return projected_root, projected_quat, projected_dof, normalized_correction

    def objective() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected_root, projected_quat, projected_dof, correction = decode()
        projected_body, _ = kin.forward_kinematics(
            projected_root,
            projected_quat,
            projected_dof,
        )
        projected_feet = projected_body[:, foot_indices]
        stance = (projected_feet[active] - targets[active]).square().sum(dim=-1).mean()
        deviation = correction.square().mean()
        velocity = (
            (correction[1:] - correction[:-1]).square().mean()
            if correction.shape[0] > 1
            else correction.new_zeros(())
        )
        acceleration = (
            (correction[2:] - 2 * correction[1:-1] + correction[:-2]).square().mean()
            if correction.shape[0] > 2
            else correction.new_zeros(())
        )
        terms = {
            "stance": stance,
            "deviation": deviation,
            "velocity": velocity,
            "acceleration": acceleration,
        }
        total = (
            cfg.stance_weight * stance
            + cfg.deviation_weight * deviation
            + cfg.velocity_weight * velocity
            + cfg.acceleration_weight * acceleration
        )
        return total, terms

    with torch.no_grad():
        initial_total, initial_terms = objective()
    function_evaluations = 0

    def closure() -> torch.Tensor:
        nonlocal function_evaluations
        optimizer.zero_grad()
        total, _ = objective()
        if not bool(torch.isfinite(total)):
            raise RuntimeError("windowed projection objective became nonfinite")
        total.backward()
        function_evaluations += 1
        return total

    optimizer.step(closure)
    with torch.no_grad():
        final_root, final_quat, final_dof, _ = decode()
        final_total, final_terms = objective()
        final_body, _ = kin.forward_kinematics(final_root, final_quat, final_dof)
        final_feet = final_body[:, foot_indices]
        initial_rmse = torch.sqrt(
            (original_feet[active] - targets[active]).square().sum(dim=-1).mean()
        )
        final_rmse = torch.sqrt(
            (final_feet[active] - targets[active]).square().sum(dim=-1).mean()
        )
        translation_delta = final_root - root_pos
        yaw_delta = cfg.root_yaw_bound_rad * torch.tanh(root_yaw_raw)
        joint_delta = final_dof[:, leg_index] - base_leg
        state = optimizer.state.get(joint_raw, {})
        accepted = bool(torch.isfinite(final_total) and final_total <= initial_total + 1e-9)

    diagnostics = {
        **base_diagnostics,
        "accepted": accepted,
        "reason": "objective_nonincreasing" if accepted else "objective_increased",
        "optimizer_iterations": int(state.get("n_iter", 0)),
        "function_evaluations": int(state.get("func_evals", function_evaluations)),
        "initial_total_loss": float(initial_total),
        "final_total_loss": float(final_total),
        "initial_terms": {name: float(value) for name, value in initial_terms.items()},
        "final_terms": {name: float(value) for name, value in final_terms.items()},
        "initial_anchor_rmse_m": float(initial_rmse),
        "final_anchor_rmse_m": float(final_rmse),
        "max_root_translation_m": float(translation_delta.norm(dim=-1).max()),
        "max_root_yaw_rad": float(yaw_delta.abs().max()),
        "max_joint_delta_rad": float(joint_delta.abs().max()),
        "optimized_dof_indices": leg_dofs,
    }
    if not accepted:
        return WindowedProjectionResult(
            root_pos=root_pos.clone(),
            root_quat=root_quat.clone(),
            dof_pos=dof_pos.clone(),
            diagnostics=diagnostics,
        )
    return WindowedProjectionResult(
        root_pos=final_root.detach(),
        root_quat=final_quat.detach(),
        dof_pos=final_dof.detach(),
        diagnostics=diagnostics,
    )
