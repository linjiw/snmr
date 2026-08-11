"""DEFECT-1 fix: world-body off-by-one in holosoma's MuJoCo(-Warp) WBT rigid-body indexing.

The bug (pinned clone 9fb2b57, verified 2026-07-27):
  - ``simulator/mujoco/mujoco.py:436-446``: ``num_bodies = model.nbody`` (world INCLUDED)
    but ``_body_list = body_names`` EXCLUDES world → 33-wide state tensors, 32-entry name list.
  - ``_rigid_body_pos/rot/vel/ang_vel`` are filled from raw ``mjw_data.xpos``/... where
    index 0 = world (``backends/warp_backend.py``).
  - ``managers/command/terms/wbt.py:582-585`` builds ``tracked_body_indexes``/``ref_body_index``
    by name against the 32-entry list and applies them to BOTH the motion npz arrays
    (name-mapped, correct) and the 33-wide sim tensors (lines 971-1015, OFF BY ONE).
  Consequence: every sim-side body read is shifted one body toward the root; tracked slot 0
  (pelvis) reads the WORLD body (always at the origin) → the relative-body-position reward
  underflows exp() to exactly 0.0 and carried no gradient in any MuJoCo-Warp WBT run.
  IsaacGym/IsaacSim backends build both lists from robot-only assets and are unaffected —
  which is why upstream (IsaacSim-pinned WBT) never saw it.

The fix: after ``MotionCommand.setup``, compute ``offset = sim_tensor_width - len(_body_list)``
(1 on MuJoCo, 0 on Isaac*) and override the eight sim-side properties to use offset-corrected
indices. Reward, termination, and critic-observation paths all read through these properties,
so one patch point covers them. Motion-side properties (motion npz) are untouched.

Known remaining mjwarp issue NOT fixed here (does not affect tracking gradients):
``UndesiredContacts`` reads ``contact_forces`` which on Warp is (a) the torque half of
cfrc_ext and (b) also world-indexed — see snmr/integration/wbt_repair.py. The term is a small
penalty (-0.1) and effectively inert either way; fixing it would change training conditions
vs all prior runs, so it is left as-is and documented.

Usage: call ``patch()`` before environment construction (wired into
``scripts/train_agent_joint_reward.py`` and ``scripts/eval_agent_repair.py``). Idempotent.
"""

from __future__ import annotations


def patch() -> None:
    from holosoma.managers.command.terms import wbt as cmd_wbt

    MotionCommand = cmd_wbt.MotionCommand
    if getattr(MotionCommand, "_snmr_bodyfix_patched", False):
        return

    orig_setup = MotionCommand.setup
    orig_reset = MotionCommand.reset

    def setup(self):
        orig_setup(self)
        sim = self._env.simulator
        width = sim._rigid_body_pos.shape[1]
        names = len(sim._body_list)
        offset = width - names
        if offset not in (0, 1):
            raise RuntimeError(
                f"wbt_bodyfix: unexpected body-tensor/name mismatch ({width} vs {names})"
            )
        self._snmr_sim_body_offset = offset
        self._snmr_sim_tracked_body_indexes = self.tracked_body_indexes + offset
        self._snmr_sim_ref_body_index = self.ref_body_index + offset
        print(
            f"wbt_bodyfix: sim body tensors {width}-wide, name list {names} -> "
            f"offset {offset} applied to sim-side body indices"
        )

    MotionCommand.setup = setup

    def set_evaluation_start_steps(self, starts):
        """Freeze the next evaluation resets to explicit global motion frames.

        Current Holosoma evaluation resets otherwise place every environment at the
        beginning of a randomly selected clip.  The SNMR studies require paired,
        phase-stratified starts.  Store the requested global frame for each environment;
        the patched reset below applies it before BaseTask flushes state to the simulator.
        """
        import torch

        starts = torch.as_tensor(starts, device=self.device, dtype=torch.long)
        if starts.shape != (self.num_envs,):
            raise ValueError(
                f"evaluation starts must have shape ({self.num_envs},), got "
                f"{tuple(starts.shape)}"
            )
        if torch.any(starts < 1) or torch.any(starts >= self.motion.time_step_total - 1):
            raise ValueError("evaluation starts fall outside the loaded motion frames")
        self._snmr_evaluation_start_steps = starts.clone()

    def reset(self, env_ids):
        import torch

        orig_reset(self, env_ids)
        requested = getattr(self, "_snmr_evaluation_start_steps", None)
        if requested is None or not self._env.is_evaluating:
            return

        env_ids = self._ensure_index_tensor(env_ids)
        requested_starts = requested[env_ids]
        motion_ids = torch.searchsorted(
            self.motion.motion_end_idx, requested_starts, right=True
        )
        if torch.any(motion_ids >= self.motion.num_motions):
            raise ValueError("evaluation start maps beyond the final motion")
        lower = self.motion.motion_start_idx[motion_ids]
        upper = self.motion.motion_end_idx[motion_ids]
        if torch.any((requested_starts <= lower) | (requested_starts >= upper - 1)):
            raise ValueError("evaluation start does not map inside a valid motion")

        # BaseTask.reset_all performs one zero-action transition before returning the
        # first observation. Initialize at the preceding reference frame so that the
        # observation delivered to the evaluator is exactly at ``requested_starts``.
        starts = requested_starts - 1

        self.motion_ids[env_ids] = motion_ids
        self.time_steps[env_ids] = starts

        # Reinitialize exactly at the requested reference state.  Phase diversity comes
        # from the frozen start grid itself; avoiding reset noise makes paired policy
        # comparisons share identical physical initial conditions.
        sim = self._env.simulator
        sim.dof_pos[env_ids] = self.joint_pos[env_ids]
        sim.dof_vel[env_ids] = self.joint_vel[env_ids]
        sim.robot_root_states[env_ids, :3] = self.root_pos_w[env_ids]
        sim.robot_root_states[env_ids, 3:7] = self.root_quat_w[env_ids]
        sim.robot_root_states[env_ids, 7:10] = self.root_lin_vel_w[env_ids]
        sim.robot_root_states[env_ids, 10:13] = self.root_ang_vel_w[env_ids]

        if self.motion.has_object:
            object_pos = self.object_pos_w[env_ids]
            object_quat = self.object_quat_w[env_ids]
            object_vel = self.object_lin_vel_w[env_ids]
            object_states = torch.cat(
                (object_pos, object_quat, object_vel, torch.zeros_like(object_vel)), dim=-1
            )
            sim.set_actor_states([self.object_name], env_ids, object_states)

    MotionCommand.set_evaluation_start_steps = set_evaluation_start_steps
    MotionCommand.reset = reset

    def robot_body_pos_w(self):
        return self._env.simulator._rigid_body_pos[:, self._snmr_sim_tracked_body_indexes, :]

    def robot_body_quat_w(self):
        return self._env.simulator._rigid_body_rot[:, self._snmr_sim_tracked_body_indexes, :]

    def robot_body_lin_vel_w(self):
        return self._env.simulator._rigid_body_vel[:, self._snmr_sim_tracked_body_indexes, :]

    def robot_body_ang_vel_w(self):
        return self._env.simulator._rigid_body_ang_vel[:, self._snmr_sim_tracked_body_indexes, :]

    def robot_ref_pos_w(self):
        return self._env.simulator._rigid_body_pos[:, self._snmr_sim_ref_body_index, :]

    def robot_ref_quat_w(self):
        return self._env.simulator._rigid_body_rot[:, self._snmr_sim_ref_body_index, :]

    def robot_ref_lin_vel_w(self):
        return self._env.simulator._rigid_body_vel[:, self._snmr_sim_ref_body_index, :]

    def robot_ref_ang_vel_w(self):
        return self._env.simulator._rigid_body_ang_vel[:, self._snmr_sim_ref_body_index, :]

    for name, fn in (
        ("robot_body_pos_w", robot_body_pos_w),
        ("robot_body_quat_w", robot_body_quat_w),
        ("robot_body_lin_vel_w", robot_body_lin_vel_w),
        ("robot_body_ang_vel_w", robot_body_ang_vel_w),
        ("robot_ref_pos_w", robot_ref_pos_w),
        ("robot_ref_quat_w", robot_ref_quat_w),
        ("robot_ref_lin_vel_w", robot_ref_lin_vel_w),
        ("robot_ref_ang_vel_w", robot_ref_ang_vel_w),
    ):
        if not isinstance(getattr(MotionCommand, name), property):
            raise RuntimeError(f"wbt_bodyfix: MotionCommand.{name} is not a property")
        setattr(MotionCommand, name, property(fn))

    MotionCommand._snmr_bodyfix_patched = True
