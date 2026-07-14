"""Expose an SNMR latent stored in a motion NPZ to a holosoma WBT policy.

The observation functions in this module can be referenced directly from a holosoma observation
configuration, so the pinned holosoma clone is never edited. They lazily load ``latent_z`` from the
same WBT NPZ used by ``MotionCommand`` and cache it on the motion loader.

Three policy interfaces are provided:

``motion_command_with_current_latent``
    Explicit GMR joint command concatenated with the current SNMR latent.

``motion_command_with_latent_preview``
    Explicit GMR joint command concatenated with ``[z_t, z_{t+0.2s}-z_t,
    z_{t+0.5s}-z_t]`` at the 50 Hz WBT rate. Future indices are clipped at the current clip end.

``snmr_latent``
    The current latent alone, retained for latent-only experiments.

The older :func:`patch` API remains available. Calling it before environment construction:
  1. `MotionLoader._load_data_from_motion_npz` is wrapped to also load a `latent_z` array (T, d)
     when present, exposing `loader.latent_z`. Absent -> a zero column of width `SNMR_LATENT_DIM`
     (default 128), so a z-command run on a vanilla NPZ degrades to zeros rather than crashing.
  2. A new observation term `snmr_latent` is registered on the WBT observation module, returning the
     current-frame latent `latent_z[time_steps]` — a drop-in for holosoma's `motion_command` term.

For the direct path, set an existing observation term's ``func`` to, for example,
``snmr.integration.wbt_latent:motion_command_with_current_latent``. The reward and termination
paths remain on the explicit robot-space GMR reference.
"""

from __future__ import annotations

import os

import numpy as np

SNMR_LATENT_DIM = int(os.environ.get("SNMR_LATENT_DIM", "128"))
PREVIEW_OFFSETS = (0, 10, 25)  # current, +0.2 s, +0.5 s at the 50 Hz WBT policy rate


def _motion_command(env):
    from holosoma.managers.observation.terms.wbt import (
        _get_motion_command_and_assert_type,
    )

    return _get_motion_command_and_assert_type(env)


def _ensure_latent_loaded(motion_command):
    """Load and cache ``latent_z`` for the single-motion WBT path."""
    import torch

    motion = motion_command.motion
    if hasattr(motion, "latent_z"):
        return motion.latent_z
    if motion_command.motion_cfg.motion_dir:
        raise ValueError(
            "lazy latent loading currently supports motion_file only; "
            "call patch() before environment setup for MultiMotionLoader"
        )

    motion_file = motion_command.motion_cfg.motion_file
    with np.load(motion_file, allow_pickle=False) as data:
        if "latent_z" not in data.files:
            raise ValueError(f"WBT motion file has no latent_z field: {motion_file}")
        latent_np = np.asarray(data["latent_z"], dtype=np.float32)
    if latent_np.ndim != 2:
        raise ValueError(f"latent_z must have shape (T,d), got {latent_np.shape}")
    if latent_np.shape[0] != motion.time_step_total:
        raise ValueError(
            f"latent_z frames {latent_np.shape[0]} != motion frames "
            f"{motion.time_step_total}"
        )
    if not np.isfinite(latent_np).all():
        raise ValueError("latent_z contains nonfinite values")
    motion.latent_z = torch.as_tensor(
        latent_np, dtype=torch.float32, device=motion_command.device
    )
    return motion.latent_z


def _latent_at_offsets(motion_command, offsets: tuple[int, ...]):
    """Gather per-environment latents without crossing a clip boundary."""
    import torch

    latent = _ensure_latent_loaded(motion_command)
    current = motion_command.time_steps
    end = (
        motion_command.motion.motion_end_idx[motion_command.motion_ids] - 1
    )
    gathered = []
    for offset in offsets:
        index = torch.minimum(current + offset, end)
        gathered.append(latent[index])
    return gathered


def snmr_latent(env):
    """Current-frame SNMR latent ``z_t``."""
    motion_command = _motion_command(env)
    return _latent_at_offsets(motion_command, (0,))[0]


def snmr_latent_tangent_preview(env):
    """Return ``[z_t, z_t+0.2s-z_t, z_t+0.5s-z_t]`` for motion anticipation."""
    z0, z_short, z_long = _latent_at_offsets(
        _motion_command(env), PREVIEW_OFFSETS
    )
    return _cat((z0, z_short - z0, z_long - z0))


def motion_command_with_current_latent(env):
    """Explicit GMR joint command augmented with current SNMR latent."""
    motion_command = _motion_command(env)
    z0 = _latent_at_offsets(motion_command, (0,))[0]
    return _cat((motion_command.command, z0))


def motion_command_with_latent_preview(env):
    """Explicit GMR command augmented with current and future-delta latents."""
    motion_command = _motion_command(env)
    z0, z_short, z_long = _latent_at_offsets(motion_command, PREVIEW_OFFSETS)
    return _cat(
        (motion_command.command, z0, z_short - z0, z_long - z0)
    )


def _cat(values):
    """Keep torch imported lazily so the module remains dependency-light at import."""
    import torch

    return torch.cat(tuple(values), dim=-1)


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

    obs_wbt.snmr_latent = snmr_latent
    obs_wbt.snmr_latent_tangent_preview = snmr_latent_tangent_preview
    obs_wbt.motion_command_with_current_latent = (
        motion_command_with_current_latent
    )
    obs_wbt.motion_command_with_latent_preview = (
        motion_command_with_latent_preview
    )


def latent_dim() -> int:
    return SNMR_LATENT_DIM
