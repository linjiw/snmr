"""E50 physics-repaired teacher: record simulated rollout states from a holosoma WBT eval.

Design (docs/E50_PHYSICS_REPAIRED_TEACHER_PROTOCOL.md §3): run ``eval_agent.py`` with BOTH the
stock ``wbt_metrics`` callback (phase-stratified starts over the clip, per-env completion /
survival accounting — the segmentation ground truth) and this recorder, which dumps per-step
simulated state for ALL envs. The export script then stitches completed segments into a
physics-repaired reference (`scripts/export_e50_repaired_pairs.py`).

The pinned holosoma clone is never edited. ``patch()`` swaps the module attribute
``holosoma.agents.callbacks.recording.EvalRecordingCallback`` for :class:`RepairRecordingCallback`
before config instantiation; holosoma's ``instantiate()`` resolves the ``_target_`` string through
the module attribute, so the stock ``--recording.config.enabled`` / ``--recording.config.output-path``
CLI flags then construct this class (``env_id`` is ignored — all envs are recorded). Launch through
``scripts/eval_agent_repair.py``, which applies the patch and then defers to holosoma's eval main.

This module imports holosoma at import time — use it only inside the WBT env (.venv-wbt).

Channels (per policy step, 50 Hz), each ``(T, num_envs, ...)``:
  dof_pos, dof_vel               (T, N, J)   simulated joints
  root_pos, root_quat_xyzw       (T, N, 3/4) simulated floating base
  root_lin_vel, root_ang_vel     (T, N, 3)
  time_steps                     (T, N)      commanded reference frame index (MotionCommand)
  dones                          (T, N)      env terminated at this step (post-step)
  feet_contact_force             (T, N, F)   contact-force norm per foot body [N]
Metadata: dt, foot body names, dof/body names, motion_file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from holosoma.agents.callbacks.base_callback import RLEvalCallback

FOOT_BODY_KEYWORD = "ankle_roll_link"  # g1; override via SNMR_E50_FOOT_KEYWORD


class RepairRecordingCallback(RLEvalCallback):
    """Multi-env rollout state recorder for physics-repair export (E50)."""

    def __init__(self, config, training_loop=None):
        super().__init__(config, training_loop)
        output_path = str(config.output_path)
        if not output_path.endswith(".npz"):
            output_path += ".npz"
        self.output_path = output_path
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._metadata: dict = {}
        self._foot_indices: list[int] = []
        self._step_count = 0

    def _get_env(self):
        return self.training_loop._unwrap_env()

    @staticmethod
    def _np(tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().copy()

    def on_pre_evaluate_policy(self) -> None:
        env = self._get_env()
        sim = env.simulator
        keyword = os.environ.get("SNMR_E50_FOOT_KEYWORD", FOOT_BODY_KEYWORD)
        body_names = list(getattr(sim, "body_names", []))
        # contact registers on the collision sub-bodies (ankle_roll_sphere_*), so group ALL
        # bodies containing the keyword stem by side and sum per side; report canonical names
        stem = keyword.replace("_link", "")
        self._foot_groups = []
        foot_canonical = []
        for side in ("left", "right"):
            group = [i for i, n in enumerate(body_names) if stem in n and n.startswith(side)]
            if group:
                self._foot_groups.append(group)
                foot_canonical.append(f"{side}_{keyword}")
        if not self._foot_groups:
            raise RuntimeError(
                f"RepairRecordingCallback: no body matching {stem!r} in {body_names}"
            )
        motion_command = env.command_manager.get_state("motion_command")
        self._metadata = {
            "dt": float(env.dt),
            "num_envs": int(env.num_envs),
            "dof_names": list(getattr(sim, "dof_names", [])),
            "body_names": body_names,
            "foot_body_names": foot_canonical,
            "foot_body_groups": [[body_names[i] for i in g] for g in self._foot_groups],
            "motion_file": str(motion_command.motion_cfg.motion_file),
            "motion_steps": int(motion_command.motion.time_step_total),
        }
        for name in (
            "dof_pos",
            "dof_vel",
            "root_pos",
            "root_quat_xyzw",
            "root_lin_vel",
            "root_ang_vel",
            "time_steps",
            "dones",
            "feet_contact_force",
        ):
            self._buffers[name] = []

    def _foot_contact_force(self, sim) -> "torch.Tensor":
        """True per-foot-side contact-force norm, (num_envs, n_sides).

        On the Warp backend ``cfrc_ext`` is only populated when a sensor requires
        ``rne_postconstraint`` (mujoco_warp sensor.py gates it), and holosoma's
        ``create_force_view`` slices ``[..., :3]`` — the *torque* half of the
        (torque, force) spatial vector. Run the kernel explicitly, take the force half,
        and sum vectors over each side's collision sub-bodies before taking the norm.
        Falls back to ``sim.contact_forces`` on other backends.
        """
        backend = getattr(sim, "backend", None)
        if backend is not None and hasattr(backend, "mjw_model"):
            import mujoco_warp as mjw

            mjw.rne_postconstraint(backend.mjw_model, backend.mjw_data)
            per_body = backend.cfrc_t[..., 3:6]
        else:
            per_body = sim.contact_forces
        return torch.stack(
            [per_body[:, g, :].sum(dim=1).norm(dim=-1) for g in self._foot_groups], dim=-1
        )

    def on_post_eval_env_step(self, actor_state: dict) -> dict:
        env = self._get_env()
        sim = env.simulator
        motion_command = env.command_manager.get_state("motion_command")

        self._buffers["dof_pos"].append(self._np(sim.dof_pos))
        self._buffers["dof_vel"].append(self._np(sim.dof_vel))
        root = sim.robot_root_states
        self._buffers["root_pos"].append(self._np(root[:, :3]))
        self._buffers["root_quat_xyzw"].append(self._np(root[:, 3:7]))
        self._buffers["root_lin_vel"].append(self._np(root[:, 7:10]))
        self._buffers["root_ang_vel"].append(self._np(root[:, 10:13]))
        self._buffers["time_steps"].append(self._np(motion_command.time_steps))
        dones = actor_state.get("dones")
        if dones is None:
            raise RuntimeError("RepairRecordingCallback requires dones in actor_state")
        self._buffers["dones"].append(self._np(torch.as_tensor(dones).bool()))
        self._buffers["feet_contact_force"].append(self._np(self._foot_contact_force(sim)))
        self._step_count += 1
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        if self._step_count == 0:
            raise RuntimeError("RepairRecordingCallback recorded zero steps")
        arrays = {name: np.stack(values, axis=0) for name, values in self._buffers.items()}
        arrays["_metadata_json"] = np.array(json.dumps(self._metadata))
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **arrays)
        summary = ", ".join(f"{k}{list(v.shape)}" for k, v in arrays.items() if k[0] != "_")
        print(f"RepairRecordingCallback: saved {self._step_count} steps -> {path}\n  {summary}")


def patch() -> None:
    """Swap holosoma's EvalRecordingCallback for the repair recorder. Idempotent."""
    from holosoma.agents.callbacks import recording as recording_mod

    if getattr(recording_mod, "_snmr_repair_patched", False):
        return
    recording_mod.EvalRecordingCallback = RepairRecordingCallback
    recording_mod._snmr_repair_patched = True
