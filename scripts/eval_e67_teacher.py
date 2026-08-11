#!/usr/bin/env python
"""Terminal-aware specialist-teacher evaluation for the E67 protocol.

The current Holosoma checkout no longer ships the historical ``wbt_metrics`` callback.
This evaluator keeps the checkpoint's actor/normalizer frozen, realizes an exact
phase-stratified grid through :mod:`snmr.integration.wbt_bodyfix`, and records the three
teacher gates directly from simulator state.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import torch
import tyro

from snmr.integration import wbt_bodyfix

wbt_bodyfix.patch()

from holosoma.agents.modules.module_utils import setup_ppo_actor_module  # noqa: E402
from holosoma.agents.ppo.ppo import EmpiricalNormalization  # noqa: E402
from holosoma.train_agent import AnnotatedExperimentConfig  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402

HORIZON_STEPS = 500


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checkpoint_path = pathlib.Path(os.environ["E67_TEACHER_CKPT"])
    report_path = pathlib.Path(os.environ["E67_TEACHER_REPORT"])
    horizon_steps = int(os.environ.get("E67_EVAL_HORIZON_STEPS", str(HORIZON_STEPS)))
    if horizon_steps < 1:
        raise ValueError("E67_EVAL_HORIZON_STEPS must be positive")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    env, device, sim_app = setup_simulation_environment(cfg)
    obs_dims = env.observation_manager.get_obs_dims()
    actor_dim = int(obs_dims["actor_obs"])
    history_length = {
        "actor_obs": env.observation_manager.cfg.groups["actor_obs"].history_length,
        "critic_obs": env.observation_manager.cfg.groups["critic_obs"].history_length,
    }
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor = setup_ppo_actor_module(
        obs_dim_dict=obs_dims,
        module_config=cfg.algo.config.module_dict.actor,
        num_actions=env.robot_config.actions_dim,
        init_noise_std=cfg.algo.config.init_noise_std,
        device=device,
        history_length=history_length,
    )
    actor.load_state_dict(checkpoint["actor_model_state_dict"])
    actor.eval()
    actor_norm = EmpiricalNormalization(shape=actor_dim, device=device)
    actor_norm.load_state_dict(checkpoint["actor_obs_normalizer_state_dict"])
    actor_norm.eval()

    env.set_is_evaluating()
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command.motion.num_motions != 1:
        raise ValueError("specialist evaluation requires exactly one motion")
    first = int(motion_command.motion.motion_start_idx[0]) + 1
    last = int(motion_command.motion.motion_end_idx[0]) - horizon_steps - 2
    if last < first:
        raise ValueError("motion is shorter than the 10-second evaluation horizon")
    starts = torch.linspace(first, last, env.num_envs, device=device).long()
    motion_command.set_evaluation_start_steps(starts)
    obs_dict = env.reset_all()
    if not torch.equal(motion_command.time_steps, starts):
        raise RuntimeError("evaluation reset did not realize the requested start grid")

    active = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    completed = torch.zeros_like(active)
    survival = torch.zeros(env.num_envs, device=device)
    rmse_sum = torch.zeros(env.num_envs, device=device)
    with torch.no_grad():
        for step in range(horizon_steps):
            normalized = actor_norm(obs_dict["actor_obs"], update=False)
            actions = actor.act_inference({"actor_obs": normalized})
            obs_dict, _, dones, _ = env.step({"actions": actions})
            joint_rmse = (
                motion_command.robot_joint_pos - motion_command.joint_pos
            ).square().mean(dim=-1).sqrt()
            rmse_sum[active] += joint_rmse[active]
            survival[active] += 1
            active &= ~dones.bool()
            if step == horizon_steps - 1:
                completed = active.clone()

    per_rollout_rmse = rmse_sum / survival.clamp_min(1)
    report = {
        "protocol": "E67 specialist teacher gate v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "num_rollouts": env.num_envs,
        "horizon_steps": horizon_steps,
        "dt": float(env.dt),
        "completion_rate": float(completed.float().mean()),
        "mean_survival_s": float(survival.mean() * env.dt),
        "joint_rmse_rad": float(per_rollout_rmse.mean()),
        "finite": bool(
            torch.isfinite(per_rollout_rmse).all()
            and torch.isfinite(survival).all()
        ),
        "start_steps": starts.cpu().tolist(),
    }
    report["passes_gate"] = bool(
        report["completion_rate"] >= 0.80
        and report["mean_survival_s"] >= 9.0
        and report["finite"]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report), flush=True)
    close_simulation_app(sim_app)
    if not report["passes_gate"]:
        raise SystemExit("specialist teacher failed the E67 gate")


if __name__ == "__main__":
    main()
