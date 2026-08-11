#!/usr/bin/env python
"""Act-through-latent command student, trained by stabilized online distillation.

Design: docs/E52_STAGE3_CVAE_DESIGN.md (constraints C1-C12). Student = prior/posterior/decoder
CVAE whose latent z_cmd is the action decoder's ONLY motion command (C1); trained from
the frozen arm-A explicit-command policy (C4); residual-to-prior posterior mean (C2); learned
conditional prior over proprio (+ SNMR latent window in arm A') (C3); tied fixed sigma -> the
ControlVAE closed-form KL ||mu_res||^2/(2 sigma^2) (D2-A); episodic reparameterization noise
(C8); PULSE latent-smoothness regularizer (C11).

Environment obs config is UNCHANGED (teacher needs the explicit motion_command term), so:
  teacher input   = full actor_obs (terms concatenated ALPHABETICALLY — see DEFECT-2 note)
  student proprio = actor_obs[PROPRIO_SLICE] (actions/base_ang_vel/dof_pos/dof_vel only,
                    normalized with the teacher's frozen stats; NO reference-derived dims)
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

Post-E66 stabilization is enabled by default: current-parameter paired smoothness,
round replay, a nonzero teacher floor, periodic/best-validation checkpoints, and
pre-specified divergence aborts.  Legacy E52 artifacts remain untouched.
"""

import dataclasses  # noqa: F401  (parity with sibling drivers; tyro configs are dataclasses)
import glob
import json
import os
import pathlib

import numpy as np
import torch
import tyro

from snmr.integration import wbt_bodyfix, wbt_latent
from snmr.integration.distillation import (
    CommandStudent,
    DivergenceGate,
    RoundReplayBuffer,
    destroy_command_code,
    paired_temporal_smoothness,
    route_teacher_actions,
    same_phase_shuffled_latents,
    shared_time_index_latents,
    teacher_mix_probability,
)

wbt_bodyfix.patch()

from holosoma.agents.modules.module_utils import setup_ppo_actor_module  # noqa: E402
from holosoma.agents.ppo.ppo import EmpiricalNormalization  # noqa: E402
from holosoma.train_agent import AnnotatedExperimentConfig  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402

# DEFECT-2 FIX (2026-08-02): holosoma's ObservationManager concatenates terms
# ALPHABETICALLY (manager.py "sorted(obs_tensors.keys())"), NOT in config order. The g1
# WBT actor_obs layout at pinned rev 9fb2b57, verified by runtime probe (max-abs-diff 0.0):
#   actions[0:29] base_ang_vel[29:32] dof_pos[32:61] dof_vel[61:90]
#   motion_command[90:148] motion_ref_ori_b[148:154]
# v1-v3 sliced cmd = full[:, :58] (= actions+bav+dof_pos[:26]) and proprio = full[:, 58:],
# so the DECODER saw the true reference inside "proprio" (C1 violated in every run).
# v4: proprio = the four proprioceptive terms; goal = motion_command + ref_ori (all
# reference-derived dims). Old checkpoints (proprio 96 / cmd 58) cannot load into v4
# shapes (proprio 90 / goal 64) — intentional, prevents silently mixing regimes.
PROPRIO_SLICE = slice(0, 90)
GOAL_SLICE = slice(90, 154)
MOTION_CMD_DIM = 64          # prior goal dim: motion_command(58) + motion_ref_ori_b(6)
Z_SNMR_DIM = 128
Z_OFFSETS = (0, 5)           # C6: current + 0.1 s at 50 Hz
Z_CMD_DIM = 64               # D4
SIGMA = 0.3                  # tied fixed sigma (ControlVAE); KL = ||mu_res||^2/(2*sigma^2)
HORIZON_STEPS = 500
EVAL_ROLLOUTS = 100


def main() -> None:
    arm = os.environ["E52_ARM"]
    ARM_GOALS = {"a_prior_snmr": "snmr", "b_prior_proprio": "none",
                 "c_prior_explicit": "explicit", "d_prior_explicit_snmr": "explicit+snmr"}
    assert arm in ARM_GOALS, arm
    teacher_manifest_path = os.environ.get("E52_TEACHER_MANIFEST", "")
    if teacher_manifest_path:
        teacher_manifest = json.loads(pathlib.Path(teacher_manifest_path).read_text())
        teacher_entries = teacher_manifest.get("motions", [])
        if not teacher_entries:
            raise ValueError("E52_TEACHER_MANIFEST has no motions")
        teacher_ckpts = [pathlib.Path(entry["checkpoint"]) for entry in teacher_entries]
    else:
        teacher_entries = None
        teacher_ckpts = [pathlib.Path(os.environ["E52_TEACHER_CKPT"])]
    for teacher_ckpt in teacher_ckpts:
        if not teacher_ckpt.is_file():
            raise FileNotFoundError(f"teacher checkpoint not found: {teacher_ckpt}")
    out = pathlib.Path(os.environ["E52_OUT"])
    rounds = int(os.environ.get("E52_ROUNDS", "2000"))
    beta_kl = float(os.environ.get("E52_BETA_KL", "0.1"))
    alpha_smooth = float(os.environ.get("E52_ALPHA_SMOOTH", "0.005"))
    eval_only = os.environ.get("E52_EVAL_ONLY", "") == "1"
    prior_mix = float(os.environ.get("E52_PRIOR_MIX", "0"))  # E60 factorial knob: fraction of
    # action-loss samples decoding PRIOR z instead of posterior z (v2 used 0.5, v3 uses 0)
    deterministic = os.environ.get("E52_DET", "") == "1"  # E62: deterministic 64-d goal
    # encoder baseline — no posterior/KL/noise; z = mu_prior everywhere; action loss only.
    # Isolates whether the CVAE machinery matters or any learned 64-d readout suffices.
    phase_only = os.environ.get("E52_PHASE_ONLY", "") == "1"  # E63/E67: time-index control
    # z_ret-only arm — replace the SNMR latents with a FIXED random projection of a
    # sinusoidal frame-index embedding (same fetch path, same dims, zero motion content).
    # Multi-motion codes reset at every clip boundary, so global concatenation offsets do
    # not leak a motion ID.
    shuffle_latent = os.environ.get("E52_SHUFFLE_LATENT", "") == "1"
    replay_rounds = int(os.environ.get("E52_REPLAY_ROUNDS", "4"))
    teacher_floor = float(os.environ.get("E52_TEACHER_FLOOR", "0.1"))
    teacher_anneal_rounds = int(os.environ.get("E52_TEACHER_ANNEAL_ROUNDS", "200"))
    checkpoint_every = int(os.environ.get("E52_CKPT_EVERY", "50"))
    validation_every = int(os.environ.get("E52_VAL_EVERY", "50"))
    validation_samples = int(os.environ.get("E52_VAL_SAMPLES", "4096"))
    best_after = int(os.environ.get("E52_BEST_AFTER", "50"))
    max_kl = float(os.environ.get("E52_MAX_KL", "10000"))
    max_latent_norm = float(os.environ.get("E52_MAX_LATENT_NORM", "100"))
    if phase_only and shuffle_latent:
        raise ValueError("time-index and shuffled-latent controls are mutually exclusive")
    if checkpoint_every < 1 or validation_every < 1 or validation_samples < 1:
        raise ValueError("checkpoint/validation intervals and sample count must be positive")
    steps_per_round, epochs, minibatch = 24, 5, 4096
    out.mkdir(parents=True, exist_ok=True)

    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    env, device, sim_app = setup_simulation_environment(tyro_cfg)
    obs_dims = env.observation_manager.get_obs_dims()
    actor_dim, critic_dim = int(obs_dims["actor_obs"]), int(obs_dims["critic_obs"])
    assert actor_dim == 154, f"obs layout changed ({actor_dim}); re-verify slices"
    num_act = env.robot_config.actions_dim
    proprio_dim = PROPRIO_SLICE.stop - PROPRIO_SLICE.start

    # --- frozen specialist teacher(s) + their observation normalizers ------------------
    history_length = {
        "actor_obs": env.observation_manager.cfg.groups["actor_obs"].history_length,
        "critic_obs": env.observation_manager.cfg.groups["critic_obs"].history_length,
    }
    teachers = []
    teacher_actor_norms = []
    checkpoints = []
    for teacher_ckpt in teacher_ckpts:
        checkpoint = torch.load(teacher_ckpt, map_location=device)
        teacher = setup_ppo_actor_module(
            obs_dim_dict=obs_dims,
            module_config=tyro_cfg.algo.config.module_dict.actor,
            num_actions=num_act,
            init_noise_std=tyro_cfg.algo.config.init_noise_std,
            device=device,
            history_length=history_length,
        )
        teacher.load_state_dict(checkpoint["actor_model_state_dict"])
        teacher.eval()
        teacher_norm = EmpiricalNormalization(shape=actor_dim, device=device)
        teacher_norm.load_state_dict(checkpoint["actor_obs_normalizer_state_dict"])
        teacher_norm.eval()
        teachers.append(teacher)
        teacher_actor_norms.append(teacher_norm)
        checkpoints.append(checkpoint)

    # Student inputs use one shared normalization (the first specialist), so normalizer
    # statistics cannot serve as a per-motion identifier. Each teacher uses its own actor
    # normalizer only while producing the routed label.
    actor_norm = teacher_actor_norms[0]
    critic_norm = EmpiricalNormalization(shape=critic_dim, device=device)
    critic_norm.load_state_dict(checkpoints[0]["critic_obs_normalizer_state_dict"])
    critic_norm.eval()
    for module in (*teachers, *teacher_actor_norms, critic_norm):
        for p in module.parameters():
            p.requires_grad_(False)

    # --- SNMR latent access + one global training-pool standardization ------------------
    motion_command = env.command_manager.get_state("motion_command")
    motion_teacher_ids = torch.arange(
        motion_command.motion.num_motions, device=device, dtype=torch.long
    )
    if teacher_entries is not None:
        motion_files = []
        for directory in str(motion_command.motion_cfg.motion_dir).split(","):
            motion_files.extend(
                sorted(glob.glob(str(pathlib.Path(directory.strip()).expanduser() / "*.npz")))
            )
        if eval_only and motion_command.motion.num_motions == 1:
            # Per-clip deployment evaluation uses the shared student and the first
            # specialist's frozen normalization stored by the training recipe.  Teacher
            # routing is never invoked when rounds == 0, so the two-entry manifest need
            # not match the intentionally one-file evaluation environment.
            matches = [
                index
                for index, entry in enumerate(teacher_entries)
                if pathlib.Path(motion_files[0]).stem.startswith(str(entry["clip"]))
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"single evaluation motion {motion_files[0]} does not uniquely match "
                    "the teacher manifest"
                )
            motion_teacher_ids = torch.tensor(matches, device=device, dtype=torch.long)
        elif len(motion_files) != len(teacher_entries):
            raise ValueError(
                f"teacher manifest has {len(teacher_entries)} motions but motion_dir has "
                f"{len(motion_files)} files"
            )
        else:
            for index, (motion_file, entry) in enumerate(
                zip(motion_files, teacher_entries)
            ):
                clip = str(entry["clip"])
                if not pathlib.Path(motion_file).stem.startswith(clip):
                    raise ValueError(
                        f"teacher manifest motion {index}={clip!r} does not match "
                        f"sorted motion file {motion_file}"
                    )
    elif motion_command.motion.num_motions != 1:
        raise ValueError("multi-motion distillation requires E52_TEACHER_MANIFEST")

    def routed_teacher_action(raw_actor_obs: torch.Tensor) -> torch.Tensor:
        actions = []
        for teacher, normalizer in zip(teachers, teacher_actor_norms):
            normalized = normalizer(raw_actor_obs, update=False)
            actions.append(teacher.act_inference({"actor_obs": normalized}))
        stacked = torch.stack(actions, dim=0)
        routed_ids = motion_teacher_ids[motion_command.motion_ids]
        return route_teacher_actions(stacked, routed_ids)

    latents = wbt_latent._ensure_latent_loaded(motion_command)
    if phase_only:
        latents = shared_time_index_latents(
            motion_command.motion.motion_start_idx,
            motion_command.motion.motion_end_idx,
            output_dim=Z_SNMR_DIM,
        )
        motion_command.motion.latent_z = latents  # _ensure_latent_loaded returns this
    elif shuffle_latent:
        latents = same_phase_shuffled_latents(
            latents,
            motion_command.motion.motion_start_idx,
            motion_command.motion.motion_end_idx,
        )
        motion_command.motion.latent_z = latents
    z_mean, z_std = latents.mean(0, keepdim=True), latents.std(0, keepdim=True) + 1e-6

    def z_window() -> torch.Tensor:
        zs = wbt_latent._latent_at_offsets(motion_command, Z_OFFSETS)
        return torch.cat([(z - z_mean) / z_std for z in zs], -1)

    student = CommandStudent(
        proprio_dim,
        critic_dim,
        num_act,
        ARM_GOALS[arm],
        MOTION_CMD_DIM,
        z_window_dim=Z_SNMR_DIM * len(Z_OFFSETS),
        z_cmd_dim=Z_CMD_DIM,
    ).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=3e-4)

    if eval_only:
        saved = torch.load(out / f"{arm}_student.pt", map_location=device)
        student.load_state_dict(saved["student"])
        z_mean = saved["z_mean"].to(device)
        z_std = saved["z_std"].to(device)
        rounds = 0

    def split_obs(obs_dict):
        full = actor_norm(obs_dict["actor_obs"], update=False)
        priv = critic_norm(obs_dict["critic_obs"], update=False)
        return full, full[:, PROPRIO_SLICE], full[:, GOAL_SLICE], priv

    n_envs = env.num_envs
    eps = torch.randn(n_envs, Z_CMD_DIM, device=device)  # episodic noise (C8)
    obs_dict = env.reset_all()
    previous = {
        "proprio": torch.zeros(n_envs, proprio_dim, device=device),
        "zwin": torch.zeros(n_envs, Z_SNMR_DIM * len(Z_OFFSETS), device=device),
        "cmd": torch.zeros(n_envs, MOTION_CMD_DIM, device=device),
        "priv": torch.zeros(n_envs, critic_dim, device=device),
    }
    prev_valid = torch.zeros(n_envs, device=device)
    log_path = out / f"{arm}_train_log.jsonl"
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if rounds and log_path.exists() and os.environ.get("E52_ALLOW_EXISTING", "") != "1":
        raise FileExistsError(
            f"refusing to append to recorded run {log_path}; choose a new E52_OUT"
        )

    replay = RoundReplayBuffer(replay_rounds)
    divergence = DivergenceGate(max_kl=max_kl, max_latent_norm=max_latent_norm)
    validation_data = None
    best_validation = float("inf")
    best_round = None
    best_path = checkpoint_dir / f"{arm}_best.pt"

    def checkpoint_payload(round_index, *, status, validation_loss=None):
        return {
            "student": student.state_dict(),
            "optimizer": opt.state_dict(),
            "arm": arm,
            "round": round_index,
            "status": status,
            "validation_action_loss": validation_loss,
            "z_mean": z_mean.cpu(),
            "z_std": z_std.cpu(),
            "config": {
                "beta_kl": beta_kl,
                "alpha_smooth": alpha_smooth,
                "sigma": SIGMA,
                "z_cmd_dim": Z_CMD_DIM,
                "offsets": Z_OFFSETS,
                "rounds": rounds,
                "deterministic": deterministic,
                "time_index": phase_only,
                "shuffle_latent": shuffle_latent,
                "replay_rounds": replay_rounds,
                "teacher_floor": teacher_floor,
                "teacher_anneal_rounds": teacher_anneal_rounds,
                "checkpoint_every": checkpoint_every,
                "validation_every": validation_every,
                "validation_samples": validation_samples,
                "max_kl": max_kl,
                "max_latent_norm": max_latent_norm,
                "teacher_ckpts": [str(path) for path in teacher_ckpts],
                "teacher_manifest": teacher_manifest_path or None,
            },
        }

    def save_checkpoint(path, round_index, *, status, validation_loss=None):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            checkpoint_payload(
                round_index, status=status, validation_loss=validation_loss
            ),
            tmp,
        )
        tmp.replace(path)

    def validation_action_loss():
        if validation_data is None:
            return float("nan")
        was_training = student.training
        student.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            size = validation_data["proprio"].shape[0]
            for start in range(0, size, minibatch):
                stop = min(start + minibatch, size)
                proprio = validation_data["proprio"][start:stop]
                mu_prior = student.mu_prior(
                    proprio,
                    validation_data["zwin"][start:stop],
                    validation_data["cmd"][start:stop],
                )
                action = student.act(proprio, mu_prior)
                squared = (
                    action - validation_data["a_teacher"][start:stop]
                ).square().sum(dim=-1)
                total += float(squared.sum())
                count += squared.numel()
        student.train(was_training)
        return total / max(count, 1)

    def abort_run(reason, round_index, metrics):
        record = {"round": round_index, "status": "aborted", "reason": reason, **metrics}
        (out / "ABORTED.json").write_text(json.dumps(record, indent=2) + "\n")
        save_checkpoint(
            checkpoint_dir / f"{arm}_aborted_round_{round_index + 1:05d}.pt",
            round_index,
            status=f"aborted:{reason}",
        )
        close_simulation_app(sim_app)
        raise RuntimeError(f"distillation aborted at round {round_index}: {reason}")

    for rnd in range(rounds):
        fields = (
            "proprio", "zwin", "cmd", "priv", "a_teacher",
            "prev_proprio", "prev_zwin", "prev_cmd", "prev_priv", "prev_valid",
        )
        buf = {key: [] for key in fields}
        p_teacher = teacher_mix_probability(
            rnd, anneal_rounds=teacher_anneal_rounds, floor=teacher_floor
        )
        with torch.no_grad():
            for _ in range(steps_per_round):
                full, proprio, cmd, priv = split_obs(obs_dict)
                zwin = z_window()
                a_teacher = routed_teacher_action(obs_dict["actor_obs"])
                mu_p = student.mu_prior(proprio, zwin, cmd)
                # Collect along the deployment path. Teacher labels every visited state.
                a_student = student.act(proprio, mu_p if deterministic
                                        else mu_p + SIGMA * eps)
                mix = (torch.rand(n_envs, 1, device=device) < p_teacher).float()
                actions = mix * a_teacher + (1 - mix) * a_student
                for k, v in (("proprio", proprio), ("zwin", zwin), ("cmd", cmd),
                             ("priv", priv), ("a_teacher", a_teacher),
                             ("prev_proprio", previous["proprio"]),
                             ("prev_zwin", previous["zwin"]),
                             ("prev_cmd", previous["cmd"]),
                             ("prev_priv", previous["priv"]),
                             ("prev_valid", prev_valid)):
                    buf[k].append(v.clone())
                previous = {
                    "proprio": proprio.clone(),
                    "zwin": zwin.clone(),
                    "cmd": cmd.clone(),
                    "priv": priv.clone(),
                }
                prev_valid = torch.ones(n_envs, device=device)
                obs, rew, dones, extras = env.step({"actions": actions})
                obs_dict = obs
                done_idx = dones.nonzero(as_tuple=True)[0]
                if len(done_idx):
                    eps[done_idx] = torch.randn(len(done_idx), Z_CMD_DIM, device=device)
                    prev_valid[done_idx] = 0.0
        round_data = {k: torch.cat(v) for k, v in buf.items()}
        if validation_data is None:
            sample_count = min(validation_samples, round_data["proprio"].shape[0] - 1)
            permutation = torch.randperm(round_data["proprio"].shape[0], device=device)
            val_idx, train_idx = permutation[:sample_count], permutation[sample_count:]
            validation_data = {
                key: round_data[key][val_idx].clone()
                for key in ("proprio", "zwin", "cmd", "a_teacher")
            }
            round_data = {key: value[train_idx] for key, value in round_data.items()}
        replay.append(round_data)
        data = replay.merged()
        n = data["proprio"].shape[0]
        losses = []
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, minibatch):
                idx = perm[i:i + minibatch]
                proprio, zwin, cmd, priv = (data["proprio"][idx], data["zwin"][idx],
                                            data["cmd"][idx], data["priv"][idx])
                mu_p = student.mu_prior(proprio, zwin, cmd)
                if deterministic:
                    mu_res = torch.zeros_like(mu_p)
                    mu_q = mu_p
                    mu_z = mu_p
                else:
                    mu_res = student.mu_residual(proprio, zwin, cmd, priv)
                    mu_q = mu_p + mu_res
                    # E60 knob: fraction of action-loss samples that decode prior z.
                    if prior_mix > 0:
                        use_prior = (
                            torch.rand(mu_q.shape[0], 1, device=device) < prior_mix
                        ).float()
                        mu_z = use_prior * mu_p + (1 - use_prior) * mu_q
                    else:
                        mu_z = mu_q
                z = mu_z if deterministic else mu_z + SIGMA * torch.randn_like(mu_z)
                a = student.act(proprio, z)
                l_act = (a - data["a_teacher"][idx]).square().sum(-1).mean()
                if deterministic:
                    l_kl = mu_q.sum() * 0.0
                    l_sm = mu_q.sum() * 0.0
                else:
                    l_kl = mu_res.square().sum(-1).mean() / (2 * SIGMA**2)
                    prev_mu_p = student.mu_prior(
                        data["prev_proprio"][idx],
                        data["prev_zwin"][idx],
                        data["prev_cmd"][idx],
                    )
                    prev_mu_q = prev_mu_p + student.mu_residual(
                        data["prev_proprio"][idx],
                        data["prev_zwin"][idx],
                        data["prev_cmd"][idx],
                        data["prev_priv"][idx],
                    )
                    l_sm = paired_temporal_smoothness(
                        mu_q, prev_mu_q, data["prev_valid"][idx]
                    )
                loss = l_act if deterministic else l_act + beta_kl * l_kl + alpha_smooth * l_sm
                if not torch.isfinite(loss):
                    abort_run("nonfinite_minibatch_loss", rnd, {"p_teacher": p_teacher})
                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    abort_run("nonfinite_gradient_norm", rnd, {"p_teacher": p_teacher})
                opt.step()
                losses.append((l_act.item(), l_kl.item(), l_sm.item(), mu_q.norm(dim=-1).mean().item()))
        la, lk, ls, latent_norm = np.mean(losses, 0)
        abort_reason = divergence.check(
            rnd, action=la, kl=lk, smooth=ls, latent_norm=latent_norm
        )
        if abort_reason is not None:
            abort_run(
                abort_reason,
                rnd,
                {"p_teacher": p_teacher, "l_action": la, "l_kl": lk,
                 "l_smooth": ls, "latent_norm": latent_norm},
            )

        completed_rounds = rnd + 1
        should_validate = completed_rounds % validation_every == 0 or rnd == rounds - 1
        val_loss = validation_action_loss() if should_validate else None
        if should_validate and (
            completed_rounds >= best_after or best_round is None and rnd == rounds - 1
        ) and val_loss < best_validation:
            best_validation = val_loss
            best_round = rnd
            save_checkpoint(
                best_path, rnd, status="best_validation", validation_loss=val_loss
            )
        if completed_rounds % checkpoint_every == 0 or rnd == rounds - 1:
            save_checkpoint(
                checkpoint_dir / f"{arm}_round_{completed_rounds:05d}.pt",
                rnd,
                status="periodic",
                validation_loss=val_loss,
            )
        if should_validate or rnd % 50 == 0 or rnd == rounds - 1:
            rec = {
                "round": rnd,
                "p_teacher": p_teacher,
                "replay_rounds": len(replay),
                "samples": n,
                "l_action": la,
                "l_kl": lk,
                "l_smooth": ls,
                "latent_norm": latent_norm,
                "validation_action_loss": val_loss,
                "best_validation_action_loss": (
                    best_validation if np.isfinite(best_validation) else None
                ),
                "best_round": best_round,
            }
            print(json.dumps(rec), flush=True)
            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    if not eval_only:
        final_path = checkpoint_dir / f"{arm}_final.pt"
        save_checkpoint(
            final_path,
            rounds - 1,
            status="final_unselected",
            validation_loss=validation_action_loss(),
        )
        selected = torch.load(best_path if best_path.exists() else final_path, map_location=device)
        student.load_state_dict(selected["student"])
        selected["status"] = "selected_best_validation"
        selected["selected_round"] = selected["round"]
        selected["final_round"] = rounds - 1
        canonical_tmp = out / f"{arm}_student.pt.tmp"
        torch.save(selected, canonical_tmp)
        canonical_tmp.replace(out / f"{arm}_student.pt")
        if os.environ.get("E52_SKIP_INLINE_EVAL", "") == "1":
            print("training complete; inline evaluation disabled by E52_SKIP_INLINE_EVAL=1")
            close_simulation_app(sim_app)
            return

    # --- deployment-path eval: z = mu_prior (or posterior, diagnostic), phase-stratified ---
    dump_z = os.environ.get("E52_DUMP_Z", "")  # E64: save (z_cmd, cmd, proprio) during
    # eval for capacity analysis (Var(mu_prior) vs sigma^2, effective rank, linear R^2)
    dump_buf = {"z": [], "cmd": [], "proprio": []} if dump_z else None
    eval_z = os.environ.get("E52_EVAL_Z", "prior")  # prior | posterior
    noise_cmd = float(os.environ.get("E52_EVAL_NOISE_CMD", "0"))      # sigma, normalized units
    noise_z = float(os.environ.get("E52_EVAL_NOISE_ZRET", "0"))       # sigma, standardized z units
    noise_prop = float(os.environ.get("E52_EVAL_NOISE_PROPRIO", "0"))
    destroy_zcmd = os.environ.get("E52_EVAL_DESTROY_ZCMD", "none")
    if destroy_zcmd not in {"none", "zero", "shuffle", "marginal_random"}:
        raise ValueError(
            "E52_EVAL_DESTROY_ZCMD must be none, zero, shuffle, or marginal_random"
        )
    hold_z = int(os.environ.get("E52_EVAL_HOLD_Z", "0"))  # E65: refresh z_cmd only every
    # k control ticks (zero-order hold) — latency/dropout robustness readout. k=1 = off.
    ambiguity_precheck = os.environ.get("E52_EVAL_STARTS_JSON", "")
    student.eval()
    env.set_is_evaluating()
    motion_starts = motion_command.motion.motion_start_idx
    motion_ends = motion_command.motion.motion_end_idx
    num_motions = motion_command.motion.num_motions
    clip_names = (
        [str(teacher_entries[int(index)]["clip"]) for index in motion_teacher_ids]
        if teacher_entries is not None
        else [f"motion_{index}" for index in range(num_motions)]
    )
    ambiguity_pair_ids = None
    ambiguity_sides = None
    if ambiguity_precheck:
        precheck = json.loads(pathlib.Path(ambiguity_precheck).read_text())
        preferred_pair = precheck.get("preferred_pair")
        if preferred_pair not in precheck.get("pairs", {}):
            raise ValueError("ambiguity precheck has no valid preferred pair")
        pair_report = precheck["pairs"][preferred_pair]
        if pair_report["clips"] != clip_names:
            raise ValueError(
                f"ambiguity clips {pair_report['clips']} do not match loaded clips "
                f"{clip_names}"
            )
        if num_motions != 2:
            raise ValueError("ambiguity evaluation requires exactly two loaded motions")
        fps = float(motion_command.motion.fps)
        base_starts = []
        base_pair_ids = []
        base_sides = []
        for pair_id, window in enumerate(pair_report["windows"]):
            for motion_id, side in enumerate(("first", "second")):
                local_frame = int(round(float(window[f"time_seconds_{side}"]) * fps))
                global_frame = int(motion_starts[motion_id]) + local_frame
                if global_frame + HORIZON_STEPS >= int(motion_ends[motion_id]):
                    raise ValueError(
                        f"ambiguity pair {pair_id} side {side} lacks a full rollout"
                    )
                base_starts.append(global_frame)
                base_pair_ids.append(pair_id)
                base_sides.append(motion_id)
        if not base_starts:
            raise ValueError("ambiguity precheck selected zero windows")
        repeated = torch.arange(n_envs, device=device) % len(base_starts)
        starts = torch.tensor(base_starts, device=device)[repeated]
        ambiguity_pair_ids = torch.tensor(base_pair_ids, device=device)[repeated]
        ambiguity_sides = torch.tensor(base_sides, device=device)[repeated]
    else:
        # Equal-count per-clip phase grids prevent a longer trajectory from dominating
        # the macro evaluation.  Starts stay one frame inside each boundary because the
        # patched reset initializes the preceding frame before BaseTask's warmup step.
        counts = [n_envs // num_motions] * num_motions
        for index in range(n_envs % num_motions):
            counts[index] += 1
        grids = []
        for motion_id, count in enumerate(counts):
            first = int(motion_starts[motion_id]) + 1
            last = int(motion_ends[motion_id]) - HORIZON_STEPS - 2
            if last < first:
                raise ValueError(f"motion {motion_id} is shorter than the eval horizon")
            grids.append(torch.linspace(first, last, count, device=device).long())
        starts = torch.cat(grids)
    motion_command.set_evaluation_start_steps(starts)
    obs_dict = env.reset_all()
    if not torch.equal(motion_command.time_steps, starts):
        raise RuntimeError("evaluation reset did not realize the requested start grid")
    eval_motion_ids = motion_command.motion_ids.clone()
    active = torch.ones(n_envs, dtype=torch.bool, device=device)
    completed = torch.zeros_like(active)
    survival = torch.zeros(n_envs, device=device)
    rmse_sum = torch.zeros(n_envs, device=device)
    teacher_error_sum = torch.zeros(n_envs, device=device)
    with torch.no_grad():
        for step in range(HORIZON_STEPS):
            _, proprio, cmd, priv = split_obs(obs_dict)
            zwin = z_window()
            if noise_cmd > 0:
                cmd = cmd + noise_cmd * torch.randn_like(cmd)
            if noise_z > 0:
                zwin = zwin + noise_z * torch.randn_like(zwin)
            if noise_prop > 0:
                proprio = proprio + noise_prop * torch.randn_like(proprio)
            if hold_z > 1 and step % hold_z != 0:
                z_cmd = held_z  # zero-order hold between refreshes
            else:
                z_cmd = student.mu_prior(proprio, zwin, cmd)
                if eval_z == "posterior":
                    z_cmd = z_cmd + student.mu_residual(proprio, zwin, cmd, priv)
                held_z = z_cmd
            z_cmd = destroy_command_code(z_cmd, destroy_zcmd)
            if dump_buf is not None and step % 5 == 0:
                dump_buf["z"].append(z_cmd.cpu()); dump_buf["cmd"].append(cmd.cpu())
                dump_buf["proprio"].append(proprio.cpu())
            actions = student.act(proprio, z_cmd)
            teacher_actions = routed_teacher_action(obs_dict["actor_obs"])
            action_error = (actions - teacher_actions).square().mean(dim=-1).sqrt()
            teacher_error_sum[active] += action_error[active]
            obs_dict, _, dones, _ = env.step({"actions": actions})
            rmse = (
                motion_command.robot_joint_pos - motion_command.joint_pos
            ).square().mean(dim=-1).sqrt()
            rmse_sum[active] += rmse[active]
            survival[active] += 1
            failed_now = active & dones.bool()
            active &= ~failed_now
            if step == HORIZON_STEPS - 1:
                completed = active.clone()
    per_env_rmse = rmse_sum / survival.clamp_min(1)
    per_env_teacher_error = teacher_error_sum / survival.clamp_min(1)
    per_clip = {}
    for motion_id, clip in enumerate(clip_names):
        mask = eval_motion_ids == motion_id
        per_clip[clip] = {
            "num_rollouts": int(mask.sum()),
            "completion_rate": float(completed[mask].float().mean()),
            "mean_survival_s": float(survival[mask].mean() * env.dt),
            "joint_rmse_rad": float(per_env_rmse[mask].mean()),
            "teacher_action_rmse": float(per_env_teacher_error[mask].mean()),
        }
    report = {
        "arm": arm,
        "eval_z": eval_z,
        "destroy_zcmd": destroy_zcmd,
        "noise_cmd": noise_cmd, "noise_zret": noise_z, "noise_proprio": noise_prop,
        "num_rollouts": n_envs,
        "evaluation_seed": int(tyro_cfg.training.seed),
        "completion_rate": float(completed.float().mean()),
        "mean_survival_s": float(survival.mean() * env.dt),
        "joint_rmse_rad": float(per_env_rmse.mean()),
        "teacher_action_rmse": float(per_env_teacher_error.mean()),
        "per_clip": per_clip,
        "start_steps": starts.cpu().tolist(),
        "motion_ids": eval_motion_ids.cpu().tolist(),
        "completed": completed.cpu().tolist(),
        "survival_s": (survival * env.dt).cpu().tolist(),
        "joint_rmse_by_rollout": per_env_rmse.cpu().tolist(),
        "teacher_action_rmse_by_rollout": per_env_teacher_error.cpu().tolist(),
        "teacher_ckpts": [str(path) for path in teacher_ckpts],
        "rounds": rounds, "beta_kl": beta_kl, "prior_mix": prior_mix,
        "deterministic": deterministic,
    }
    if ambiguity_precheck:
        report["ambiguity_precheck"] = str(pathlib.Path(ambiguity_precheck).resolve())
        report["ambiguity_pair_ids"] = ambiguity_pair_ids.cpu().tolist()
        report["ambiguity_sides"] = ambiguity_sides.cpu().tolist()
    if dump_buf is not None:
        np.savez_compressed(dump_z, **{k: torch.cat(v).numpy() for k, v in dump_buf.items()})
    suffix_parts = []
    if ambiguity_precheck:
        suffix_parts.append("ambiguity")
    if eval_z != "prior":
        suffix_parts.append(eval_z)
    if destroy_zcmd != "none":
        suffix_parts.append(f"destroy_{destroy_zcmd}")
    suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)
    (out / f"{arm}_eval{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    close_simulation_app(sim_app)


if __name__ == "__main__":
    main()
