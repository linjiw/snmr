#!/usr/bin/env python
"""Render one exact-start E70 rollout without modifying the frozen trainer.

The seed-confirmation queue hash-locks ``train_e52_dagger.py`` and its distillation
utilities.  This dedicated evaluator reproduces their deployment path while adding
only capture-specific start, lifecycle, and provenance handling.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import tyro

from scripts.e70_video_runtime import (
    capture_start_grid,
    expected_simulator_envs,
    fit_offscreen_framebuffer,
    validate_capture_name,
)
from snmr.integration import wbt_bodyfix, wbt_latent
from snmr.integration.distillation import (
    CommandStudent,
    destroy_command_code,
    route_teacher_actions,
    same_phase_shuffled_latents,
    shared_time_index_latents,
)

wbt_bodyfix.patch()

from holosoma.agents.modules.module_utils import setup_ppo_actor_module  # noqa: E402
from holosoma.agents.ppo.ppo import EmpiricalNormalization  # noqa: E402
from holosoma.train_agent import AnnotatedExperimentConfig  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


PROPRIO_SLICE = slice(0, 90)
GOAL_SLICE = slice(90, 154)
MOTION_CMD_DIM = 64
Z_SNMR_DIM = 128
Z_OFFSETS = (0, 5)
Z_CMD_DIM = 64
HORIZON_STEPS = 500


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    arm = os.environ["E52_ARM"]
    goals = {
        "a_prior_snmr": "snmr",
        "b_prior_proprio": "none",
        "c_prior_explicit": "explicit",
        "d_prior_explicit_snmr": "explicit+snmr",
    }
    if arm not in goals:
        raise ValueError(f"unsupported capture arm {arm!r}")
    out = pathlib.Path(os.environ["E52_OUT"])
    manifest_path = pathlib.Path(os.environ["E52_TEACHER_MANIFEST"])
    capture_name = validate_capture_name(os.environ["E70_VIDEO_CAPTURE_NAME"])
    exact_start = os.environ["E52_EVAL_EXACT_START"]
    report_path = pathlib.Path(os.environ["E70_VIDEO_REPORT_PATH"])
    phase_only = os.environ.get("E52_PHASE_ONLY", "") == "1"
    shuffle_latent = os.environ.get("E52_SHUFFLE_LATENT", "") == "1"
    destroy_zcmd = os.environ.get("E52_EVAL_DESTROY_ZCMD", "none")
    if phase_only and shuffle_latent:
        raise ValueError("time-index and shuffled-latent controls are mutually exclusive")
    if destroy_zcmd not in {"none", "zero", "shuffle", "marginal_random"}:
        raise ValueError(f"unsupported command destruction {destroy_zcmd!r}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite capture report {report_path}")

    teacher_manifest = json.loads(manifest_path.read_text())
    teacher_entries = teacher_manifest.get("motions", [])
    if not teacher_entries:
        raise ValueError("teacher manifest has no motions")
    teacher_paths = [pathlib.Path(entry["checkpoint"]) for entry in teacher_entries]
    if not all(path.is_file() for path in teacher_paths):
        raise FileNotFoundError("a teacher checkpoint is missing")
    teacher_hashes = [sha256_file(path) for path in teacher_paths]
    if teacher_hashes != [str(entry["checkpoint_sha256"]) for entry in teacher_entries]:
        raise ValueError("a teacher checkpoint differs from the frozen manifest")
    student_path = out / f"{arm}_student.pt"
    if not student_path.is_file():
        raise FileNotFoundError(student_path)

    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    env, device, sim_app = setup_simulation_environment(tyro_cfg)
    try:
        fit_offscreen_framebuffer(env)
        n_envs = env.num_envs
        expected_envs = expected_simulator_envs(destroy_zcmd)
        if n_envs != expected_envs:
            raise ValueError(
                f"capture {capture_name} requires {expected_envs} simulator envs, got {n_envs}"
            )
        obs_dims = env.observation_manager.get_obs_dims()
        actor_dim, critic_dim = int(obs_dims["actor_obs"]), int(obs_dims["critic_obs"])
        if actor_dim != 154:
            raise ValueError(f"WBT observation layout changed: {actor_dim}")
        num_actions = env.robot_config.actions_dim
        history_length = {
            key: env.observation_manager.cfg.groups[key].history_length
            for key in ("actor_obs", "critic_obs")
        }

        teachers = []
        teacher_norms = []
        teacher_checkpoints = []
        for teacher_path in teacher_paths:
            checkpoint = torch.load(teacher_path, map_location=device, weights_only=False)
            teacher = setup_ppo_actor_module(
                obs_dim_dict=obs_dims,
                module_config=tyro_cfg.algo.config.module_dict.actor,
                num_actions=num_actions,
                init_noise_std=tyro_cfg.algo.config.init_noise_std,
                device=device,
                history_length=history_length,
            )
            teacher.load_state_dict(checkpoint["actor_model_state_dict"])
            teacher.eval()
            normalizer = EmpiricalNormalization(shape=actor_dim, device=device)
            normalizer.load_state_dict(checkpoint["actor_obs_normalizer_state_dict"])
            normalizer.eval()
            teachers.append(teacher)
            teacher_norms.append(normalizer)
            teacher_checkpoints.append(checkpoint)
        actor_norm = teacher_norms[0]
        critic_norm = EmpiricalNormalization(shape=critic_dim, device=device)
        critic_norm.load_state_dict(
            teacher_checkpoints[0]["critic_obs_normalizer_state_dict"]
        )
        critic_norm.eval()
        for module in (*teachers, *teacher_norms, critic_norm):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        motion_command = env.command_manager.get_state("motion_command")
        motion_files = []
        for directory in str(motion_command.motion_cfg.motion_dir).split(","):
            motion_files.extend(
                sorted(glob.glob(str(pathlib.Path(directory.strip()).expanduser() / "*.npz")))
            )
        if len(motion_files) != len(teacher_entries):
            raise ValueError("capture motion count does not match the teacher manifest")
        for index, (motion_file, entry) in enumerate(zip(motion_files, teacher_entries)):
            if not pathlib.Path(motion_file).stem.startswith(str(entry["clip"])):
                raise ValueError(f"motion {index} does not match the teacher manifest")
        motion_paths = [pathlib.Path(path).resolve() for path in motion_files]
        motion_hashes = [sha256_file(path) for path in motion_paths]
        motion_teacher_ids = torch.arange(
            motion_command.motion.num_motions, device=device, dtype=torch.long
        )
        clip_names = [str(entry["clip"]) for entry in teacher_entries]

        latents = wbt_latent._ensure_latent_loaded(motion_command)
        if phase_only:
            latents = shared_time_index_latents(
                motion_command.motion.motion_start_idx,
                motion_command.motion.motion_end_idx,
                output_dim=Z_SNMR_DIM,
            )
            motion_command.motion.latent_z = latents
        elif shuffle_latent:
            latents = same_phase_shuffled_latents(
                latents,
                motion_command.motion.motion_start_idx,
                motion_command.motion.motion_end_idx,
            )
            motion_command.motion.latent_z = latents

        saved = torch.load(student_path, map_location=device, weights_only=False)
        config = saved.get("config", {})
        if saved.get("arm") != arm or config.get("deterministic") is not True:
            raise ValueError("capture requires the matching deterministic student")
        if bool(config.get("time_index")) != phase_only:
            raise ValueError("capture time-index flag differs from the student checkpoint")
        if bool(config.get("shuffle_latent")) != shuffle_latent:
            raise ValueError("capture shuffle flag differs from the student checkpoint")
        student = CommandStudent(
            PROPRIO_SLICE.stop,
            critic_dim,
            num_actions,
            goals[arm],
            MOTION_CMD_DIM,
            z_window_dim=Z_SNMR_DIM * len(Z_OFFSETS),
            z_cmd_dim=Z_CMD_DIM,
        ).to(device)
        student.load_state_dict(saved["student"])
        student.eval()
        z_mean = saved["z_mean"].to(device)
        z_std = saved["z_std"].to(device)

        def split_obs(obs_dict):
            full = actor_norm(obs_dict["actor_obs"], update=False)
            return full[:, PROPRIO_SLICE], full[:, GOAL_SLICE]

        def z_window() -> torch.Tensor:
            values = wbt_latent._latent_at_offsets(motion_command, Z_OFFSETS)
            return torch.cat([(value - z_mean) / z_std for value in values], dim=-1)

        def teacher_action(raw_actor_obs: torch.Tensor) -> torch.Tensor:
            actions = [
                teacher.act_inference({"actor_obs": normalizer(raw_actor_obs, update=False)})
                for teacher, normalizer in zip(teachers, teacher_norms)
            ]
            return route_teacher_actions(
                torch.stack(actions, dim=0), motion_teacher_ids[motion_command.motion_ids]
            )

        env.set_is_evaluating()
        starts = capture_start_grid(
            exact_start,
            motion_command.motion.motion_start_idx,
            motion_command.motion.motion_end_idx,
            num_envs=n_envs,
            horizon_steps=HORIZON_STEPS,
        )
        motion_command.set_evaluation_start_steps(starts)
        obs_dict = env.reset_all()
        if not torch.equal(motion_command.time_steps, starts):
            raise RuntimeError("capture reset did not realize the requested start grid")
        selected_motion_id = int(motion_command.motion_ids[0])

        completed = False
        survival_steps = 0
        rmse_sum = 0.0
        teacher_error_sum = 0.0
        with torch.no_grad():
            for step in range(HORIZON_STEPS):
                proprio, command = split_obs(obs_dict)
                zwin = z_window()
                z_cmd = student.mu_prior(proprio, zwin, command)
                z_cmd = destroy_command_code(z_cmd, destroy_zcmd)
                if not torch.isfinite(z_cmd).all():
                    raise RuntimeError("capture intervention produced a non-finite command code")
                actions = student.act(proprio, z_cmd)
                labels = teacher_action(obs_dict["actor_obs"])
                teacher_error_sum += float(
                    (actions[0] - labels[0]).square().mean().sqrt()
                )
                obs_dict, _, dones, _ = env.step({"actions": actions})
                survival_steps = step + 1
                rmse_sum += float(
                    (
                        motion_command.robot_joint_pos[0]
                        - motion_command.joint_pos[0]
                    ).square().mean().sqrt()
                )
                if bool(dones[0]):
                    break
            else:
                completed = True

        recorder = getattr(env.simulator, "video_recorder", None)
        if recorder is None or not recorder.enabled:
            raise RuntimeError("simulator video recorder is disabled")
        video_config = dataclasses.asdict(recorder.config)
        recorder.stop_recording()

        report = {
            "protocol": "E70 exact simulation capture report v1",
            "capture_name": capture_name,
            "arm": arm,
            "phase_only": phase_only,
            "shuffle_latent": shuffle_latent,
            "destroy_zcmd": destroy_zcmd,
            "evaluation_seed": int(tyro_cfg.training.seed),
            "num_rollouts": 1,
            "simulator_num_envs": n_envs,
            "intervention_pool_size": n_envs if destroy_zcmd == "marginal_random" else 1,
            "exact_start": int(starts[0]),
            "start_steps": [int(starts[0])],
            "motion_ids": [selected_motion_id],
            "clip": clip_names[selected_motion_id],
            "completed": [completed],
            "completion_rate": float(completed),
            "steps_executed": survival_steps,
            "survival_s": [survival_steps * env.dt],
            "mean_survival_s": survival_steps * env.dt,
            "joint_rmse_rad": rmse_sum / max(survival_steps, 1),
            "teacher_action_rmse": teacher_error_sum / max(survival_steps, 1),
            "student_checkpoint": str(student_path.resolve()),
            "student_checkpoint_sha256": sha256_file(student_path),
            "teacher_ckpts": [str(path.resolve()) for path in teacher_paths],
            "teacher_checkpoint_sha256": teacher_hashes,
            "motion_files": [str(path) for path in motion_paths],
            "motion_sha256": motion_hashes,
            "video_capture": True,
            "video_config": video_config,
            "runtime": str(pathlib.Path(__file__).resolve()),
            "runtime_sha256": sha256_file(pathlib.Path(__file__)),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)
        print(json.dumps(report), flush=True)
    finally:
        close_simulation_app(sim_app)


if __name__ == "__main__":
    main()
