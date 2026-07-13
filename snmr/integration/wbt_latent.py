"""Make the SNMR latent (`latent_z` in the motion NPZ) observable by a holosoma WBT policy — via
monkeypatch, so the pinned holosoma clone is never edited (repo policy).

Importing this module (before building the WBT env) does two things to holosoma in memory:
  1. `MotionLoader._load_data_from_motion_npz` is wrapped to also load a `latent_z` array (T, d)
     when present, exposing `loader.latent_z`. Absent -> a zero column of width `SNMR_LATENT_DIM`
     (default 128), so a z-command run on a vanilla NPZ degrades to zeros rather than crashing.
  2. A new observation term `snmr_latent` is registered on the WBT observation module, returning the
     current-frame latent `latent_z[time_steps]` — a drop-in for holosoma's `motion_command` term.

Usage on the training machine (holosoma env):
    import snmr.integration.wbt_latent  # noqa: F401  (before train_agent builds the env)
    # then add an obs term  func="holosoma.managers.observation.terms.wbt:snmr_latent"
    # (or the fully-qualified patched name printed by patch()).

This is exercised locally only up to the loader-patch level (holosoma's full RL stack needs the
mujoco_warp env). The design doc N10 arm consumes this: actor obs = [z_t window ⊕ embodiment code ⊕
raw current frame]; the reward path is unchanged (still on decoded q̂).
"""

from __future__ import annotations

import os

import numpy as np

SNMR_LATENT_DIM = int(os.environ.get("SNMR_LATENT_DIM", "128"))


def patch() -> None:
    """Apply the monkeypatches. Idempotent; safe to call multiple times."""
    import torch
    from holosoma.managers.command.terms import wbt as cmd_wbt
    from holosoma.managers.observation.terms import wbt as obs_wbt

    MotionLoader = cmd_wbt.MotionLoader
    if getattr(MotionLoader, "_snmr_latent_patched", False):
        return

    orig_load = MotionLoader._load_data_from_motion_npz

    def load_with_latent(self, motion_file, device):
        ret = orig_load(self, motion_file, device)
        # peek the npz again only for the optional extra (cheap; small array)
        latent = None
        try:
            with np.load(motion_file) as data:
                if "latent_z" in data.files:
                    latent = torch.tensor(np.asarray(data["latent_z"]), dtype=torch.float32, device=device)
        except Exception:
            latent = None
        if latent is None:
            latent = torch.zeros(self.time_step_total, SNMR_LATENT_DIM, dtype=torch.float32, device=device)
        assert latent.shape[0] == self.time_step_total, (
            f"latent_z frames {latent.shape[0]} != motion frames {self.time_step_total}"
        )
        self.latent_z = latent
        return ret

    MotionLoader._load_data_from_motion_npz = load_with_latent
    MotionLoader._snmr_latent_patched = True

    # MultiMotionLoader concatenates loaders along time; concat their latents too if present.
    MultiMotionLoader = getattr(cmd_wbt, "MultiMotionLoader", None)
    if MultiMotionLoader is not None and not getattr(MultiMotionLoader, "_snmr_latent_patched", False):
        orig_init = MultiMotionLoader.__init__

        def init_with_latent(self, *a, **k):
            orig_init(self, *a, **k)
            loaders = getattr(self, "_loaders", None) or getattr(self, "loaders", None)
            if loaders and all(hasattr(l, "latent_z") for l in loaders):
                self.latent_z = torch.cat([l.latent_z for l in loaders], dim=0)
        MultiMotionLoader.__init__ = init_with_latent
        MultiMotionLoader._snmr_latent_patched = True

    def snmr_latent(env):
        mc = obs_wbt._get_motion_command_and_assert_type(env)
        return mc.motion.latent_z[mc.time_steps]

    obs_wbt.snmr_latent = snmr_latent


def latent_dim() -> int:
    return SNMR_LATENT_DIM
