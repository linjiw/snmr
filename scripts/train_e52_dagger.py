#!/usr/bin/env python
"""E52 Stage-1: act-through-latent CVAE command, trained by DAgger from an explicit teacher.

Design: docs/E52_STAGE3_CVAE_DESIGN.md (constraints C1-C12). Student = prior/posterior/decoder
CVAE whose latent z_cmd is the actor's ONLY motion command (C1); trained by pure DAgger from
the frozen arm-A explicit-command policy (C4); residual-to-prior posterior mean (C2); learned
conditional prior over proprio (+ SNMR latent window in arm A') (C3); tied fixed sigma -> the
ControlVAE closed-form KL ||mu_res||^2/(2 sigma^2) (D2-A); episodic reparameterization noise
(C8); PULSE latent-smoothness regularizer (C11).

Environment obs config is UNCHANGED (teacher needs the explicit motion_command term), so:
  teacher input   = full actor_obs (motion_command first, per g1 observation.py order)
  student proprio = actor_obs[MOTION_CMD_DIM:]  (normalized with the teacher's frozen stats)
  student z_snmr  = frozen SNMR latents at offsets (0,+5) (C6), fetched via
                    snmr.integration.wbt_latent from the z-augmented motion npz
  posterior extra = critic_obs (privileged, incl. explicit reference), teacher's critic stats

Arms (E52_ARM): a_prior_snmr  — prior sees [proprio, proj(z_snmr window)]  (H2 transfer arm)
                b_prior_proprio — prior sees proprio only                   (PULSE-style control)
A' - B isolates whether frozen-SNMR retargeting knowledge in the prior helps the command
interface. (Co-training the SNMR encoder itself is Stage-3b, deferred.)

Deployment/eval path: z_cmd = mu_prior (deterministic; UniTracker v1), decoder(proprio, z_cmd).
Run in .venv-wbt with PYTHONPATH=<snmr repo>. Standard holosoma tyro CLI args; E52 params via
env vars: E52_ARM, E52_TEACHER_CKPT, E52_OUT, E52_ROUNDS, E52_BETA_KL, E52_ALPHA_SMOOTH.
"""

import dataclasses  # noqa: F401  (parity with sibling drivers; tyro configs are dataclasses)
import json
import os
import pathlib

import numpy as np
import torch
import tyro

from snmr.integration import wbt_bodyfix, wbt_latent

wbt_bodyfix.patch()

from holosoma.agents.modules.module_utils import setup_ppo_actor_module  # noqa: E402
from holosoma.agents.ppo.ppo import EmpiricalNormalization  # noqa: E402
from holosoma.train_agent import AnnotatedExperimentConfig  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402

MOTION_CMD_DIM = 58          # ref joint_pos(29) + joint_vel(29), first term of actor_obs
Z_SNMR_DIM = 128
Z_OFFSETS = (0, 5)           # C6: current + 0.1 s at 50 Hz
Z_CMD_DIM = 64               # D4
SIGMA = 0.3                  # tied fixed sigma (ControlVAE); KL = ||mu_res||^2/(2*sigma^2)
HORIZON_STEPS = 500
EVAL_ROLLOUTS = 100


def mlp(sizes, out_dim):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.ELU()]
    layers += [torch.nn.Linear(sizes[-1], out_dim)]
    return torch.nn.Sequential(*layers)


class Student(torch.nn.Module):
    def __init__(self, proprio_dim, priv_dim, num_act, use_snmr_prior: bool):
        super().__init__()
        self.use_snmr_prior = use_snmr_prior
        self.z_proj = mlp([Z_SNMR_DIM * len(Z_OFFSETS), 256], Z_CMD_DIM)
        prior_in = proprio_dim + (Z_CMD_DIM if use_snmr_prior else 0)
        self.prior = mlp([prior_in, 512, 256], Z_CMD_DIM)
        self.posterior = mlp([prior_in + priv_dim, 512, 256], Z_CMD_DIM)  # residual head (C2)
        self.decoder = mlp([proprio_dim + Z_CMD_DIM, 512, 256, 128], num_act)

    def prior_input(self, proprio, z_snmr_window):
        if self.use_snmr_prior:
            return torch.cat([proprio, self.z_proj(z_snmr_window)], -1)
        return proprio

    def mu_prior(self, proprio, z_snmr_window):
        return self.prior(self.prior_input(proprio, z_snmr_window))

    def mu_residual(self, proprio, z_snmr_window, priv):
        return self.posterior(torch.cat([self.prior_input(proprio, z_snmr_window), priv], -1))

    def act(self, proprio, z_cmd):
        return self.decoder(torch.cat([proprio, z_cmd], -1))


def main() -> None:
    arm = os.environ["E52_ARM"]
    assert arm in ("a_prior_snmr", "b_prior_proprio"), arm
    teacher_ckpt = pathlib.Path(os.environ["E52_TEACHER_CKPT"])
    out = pathlib.Path(os.environ["E52_OUT"])
    rounds = int(os.environ.get("E52_ROUNDS", "2000"))
    beta_kl = float(os.environ.get("E52_BETA_KL", "0.1"))
    alpha_smooth = float(os.environ.get("E52_ALPHA_SMOOTH", "0.005"))
    eval_only = os.environ.get("E52_EVAL_ONLY", "") == "1"
    steps_per_round, epochs, minibatch = 24, 5, 4096
    out.mkdir(parents=True, exist_ok=True)

    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    env, device, sim_app = setup_simulation_environment(tyro_cfg)
    obs_dims = env.observation_manager.get_obs_dims()
    actor_dim, critic_dim = int(obs_dims["actor_obs"]), int(obs_dims["critic_obs"])
    num_act = env.robot_config.actions_dim
    proprio_dim = actor_dim - MOTION_CMD_DIM

    # --- frozen teacher (arm-A explicit policy) + its obs normalizers -------------------
    ckpt = torch.load(teacher_ckpt, map_location=device)
    history_length = {
        "actor_obs": env.observation_manager.cfg.groups["actor_obs"].history_length,
        "critic_obs": env.observation_manager.cfg.groups["critic_obs"].history_length,
    }
    teacher = setup_ppo_actor_module(
        obs_dim_dict=obs_dims,
        module_config=tyro_cfg.algo.config.module_dict.actor,
        num_actions=num_act,
        init_noise_std=tyro_cfg.algo.config.init_noise_std,
        device=device,
        history_length=history_length,
    )
    teacher.load_state_dict(ckpt["actor_model_state_dict"])
    teacher.eval()
    actor_norm = EmpiricalNormalization(shape=actor_dim, device=device)
    actor_norm.load_state_dict(ckpt["actor_obs_normalizer_state_dict"])
    actor_norm.eval()
    critic_norm = EmpiricalNormalization(shape=critic_dim, device=device)
    critic_norm.load_state_dict(ckpt["critic_obs_normalizer_state_dict"])
    critic_norm.eval()
    for module in (teacher, actor_norm, critic_norm):
        for p in module.parameters():
            p.requires_grad_(False)

    # --- SNMR latent access + per-clip standardization ----------------------------------
    motion_command = env.command_manager.get_state("motion_command")
    latents = wbt_latent._ensure_latent_loaded(motion_command)
    z_mean, z_std = latents.mean(0, keepdim=True), latents.std(0, keepdim=True) + 1e-6

    def z_window() -> torch.Tensor:
        zs = wbt_latent._latent_at_offsets(motion_command, Z_OFFSETS)
        return torch.cat([(z - z_mean) / z_std for z in zs], -1)

    student = Student(proprio_dim, critic_dim, num_act, arm == "a_prior_snmr").to(device)
    opt = torch.optim.Adam(student.parameters(), lr=3e-4)

    if eval_only:
        saved = torch.load(out / f"{arm}_student.pt", map_location=device)
        student.load_state_dict(saved["student"])
        rounds = 0

    def split_obs(obs_dict):
        full = actor_norm(obs_dict["actor_obs"], update=False)
        priv = critic_norm(obs_dict["critic_obs"], update=False)
        return full, full[:, MOTION_CMD_DIM:], priv

    n_envs = env.num_envs
    eps = torch.randn(n_envs, Z_CMD_DIM, device=device)  # episodic noise (C8)
    obs_dict = env.reset_all()
    prev_mu_post = torch.zeros(n_envs, Z_CMD_DIM, device=device)
    prev_valid = torch.zeros(n_envs, device=device)
    log_path = out / f"{arm}_train_log.jsonl"

    for rnd in range(rounds):
        buf = {k: [] for k in ("proprio", "zwin", "priv", "a_teacher", "prev_mu", "prev_valid")}
        p_teacher = max(0.0, 1.0 - rnd / 200.0)  # short Ross-style mixing schedule
        with torch.no_grad():
            for _ in range(steps_per_round):
                full, proprio, priv = split_obs(obs_dict)
                zwin = z_window()
                a_teacher = teacher.act_inference({"actor_obs": full})
                mu_p = student.mu_prior(proprio, zwin)
                mu_q = mu_p + student.mu_residual(proprio, zwin, priv)
                # v2: collect along the DEPLOYMENT path (prior z + episodic noise) so the
                # teacher labels the states a prior-driven student actually visits. v1
                # collected with posterior z; the prior path was out-of-distribution and
                # collapsed at eval (0.1% completion) while the posterior path hit 0.84.
                a_student = student.act(proprio, mu_p + SIGMA * eps)
                mix = (torch.rand(n_envs, 1, device=device) < p_teacher).float()
                actions = mix * a_teacher + (1 - mix) * a_student
                for k, v in (("proprio", proprio), ("zwin", zwin), ("priv", priv),
                             ("a_teacher", a_teacher), ("prev_mu", prev_mu_post),
                             ("prev_valid", prev_valid)):
                    buf[k].append(v.clone())
                prev_mu_post = mu_q
                prev_valid = torch.ones(n_envs, device=device)
                obs, rew, dones, extras = env.step({"actions": actions})
                obs_dict = obs
                done_idx = dones.nonzero(as_tuple=True)[0]
                if len(done_idx):
                    eps[done_idx] = torch.randn(len(done_idx), Z_CMD_DIM, device=device)
                    prev_valid[done_idx] = 0.0
        data = {k: torch.cat(v) for k, v in buf.items()}
        n = data["proprio"].shape[0]
        losses = []
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, minibatch):
                idx = perm[i:i + minibatch]
                proprio, zwin, priv = data["proprio"][idx], data["zwin"][idx], data["priv"][idx]
                mu_p = student.mu_prior(proprio, zwin)
                mu_res = student.mu_residual(proprio, zwin, priv)
                mu_q = mu_p + mu_res
                # v2: half the samples use PRIOR z in the action loss (ControlVAE trains on
                # ~40% prior-sampled latents) — forces the prior itself to carry the command
                # so the deployment path (z = mu_p) is in-distribution for the decoder.
                use_prior = (torch.rand(mu_q.shape[0], 1, device=device) < 0.5).float()
                mu_z = use_prior * mu_p + (1 - use_prior) * mu_q
                z = mu_z + SIGMA * torch.randn_like(mu_z)
                a = student.act(proprio, z)
                l_act = (a - data["a_teacher"][idx]).square().sum(-1).mean()
                l_kl = mu_res.square().sum(-1).mean() / (2 * SIGMA**2)
                l_sm = (data["prev_valid"][idx]
                        * (mu_q - data["prev_mu"][idx]).square().sum(-1)).mean()
                loss = l_act + beta_kl * l_kl + alpha_smooth * l_sm
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                losses.append((l_act.item(), l_kl.item(), l_sm.item()))
        if rnd % 50 == 0 or rnd == rounds - 1:
            la, lk, ls = np.mean(losses, 0)
            rec = {"round": rnd, "p_teacher": p_teacher, "l_action": la, "l_kl": lk, "l_smooth": ls}
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    torch.save({"student": student.state_dict(), "arm": arm,
                "z_mean": z_mean.cpu(), "z_std": z_std.cpu(),
                "config": {"beta_kl": beta_kl, "alpha_smooth": alpha_smooth, "sigma": SIGMA,
                           "z_cmd_dim": Z_CMD_DIM, "offsets": Z_OFFSETS, "rounds": rounds}},
               out / f"{arm}_student.pt")

    # --- deployment-path eval: z = mu_prior (or posterior, diagnostic), phase-stratified ---
    eval_z = os.environ.get("E52_EVAL_Z", "prior")  # prior | posterior
    student.eval()
    env.set_is_evaluating()
    motion_steps = int(motion_command.motion.time_step_total)
    starts = torch.linspace(1, motion_steps - HORIZON_STEPS - 2, n_envs).long().to(device)
    motion_command.set_evaluation_start_steps(starts)
    obs_dict = env.reset_all()
    active = torch.ones(n_envs, dtype=torch.bool, device=device)
    completed = torch.zeros_like(active)
    survival = torch.zeros(n_envs, device=device)
    rmse_sum = torch.zeros(n_envs, device=device)
    with torch.no_grad():
        for step in range(HORIZON_STEPS):
            _, proprio, priv = split_obs(obs_dict)
            zwin = z_window()
            z_cmd = student.mu_prior(proprio, zwin)
            if eval_z == "posterior":
                z_cmd = z_cmd + student.mu_residual(proprio, zwin, priv)
            actions = student.act(proprio, z_cmd)
            obs_dict, _, dones, _ = env.step({"actions": actions})
            rmse = env.log_dict["eval/error_joint_pos_rmse"]
            rmse_sum[active] += torch.as_tensor(rmse, device=device)[active]
            survival[active] += 1
            failed_now = active & dones.bool()
            active &= ~failed_now
            if step == HORIZON_STEPS - 1:
                completed = active.clone()
    report = {
        "arm": arm,
        "eval_z": eval_z,
        "num_rollouts": n_envs,
        "completion_rate": float(completed.float().mean()),
        "mean_survival_s": float(survival.mean() * env.dt),
        "joint_rmse_rad": float((rmse_sum / survival.clamp_min(1)).mean()),
        "teacher_ckpt": str(teacher_ckpt),
        "rounds": rounds, "beta_kl": beta_kl,
    }
    suffix = "" if eval_z == "prior" else f"_{eval_z}"
    (out / f"{arm}_eval{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    close_simulation_app(sim_app)


if __name__ == "__main__":
    main()
