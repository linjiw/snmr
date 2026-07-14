from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from snmr.integration import wbt_latent


def _fake_command(tmp_path):
    latent = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
    path = tmp_path / "motion.npz"
    np.savez(path, latent_z=latent)
    motion = SimpleNamespace(
        time_step_total=7,
        motion_end_idx=torch.tensor([4, 7]),
    )
    return SimpleNamespace(
        motion=motion,
        motion_cfg=SimpleNamespace(motion_file=str(path), motion_dir=""),
        device="cpu",
        time_steps=torch.tensor([2, 5]),
        motion_ids=torch.tensor([0, 1]),
        command=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    ), torch.from_numpy(latent)


def test_current_latent_is_lazily_loaded_and_appended(tmp_path, monkeypatch):
    command, latent = _fake_command(tmp_path)
    monkeypatch.setattr(wbt_latent, "_motion_command", lambda env: command)

    observed = wbt_latent.motion_command_with_current_latent(object())

    assert observed.shape == (2, 5)
    assert torch.equal(observed[:, :2], command.command)
    assert torch.equal(observed[:, 2:], latent[command.time_steps])
    cached_ptr = command.motion.latent_z.data_ptr()
    observed_again = wbt_latent.motion_command_with_current_latent(object())
    assert command.motion.latent_z.data_ptr() == cached_ptr
    assert torch.equal(observed_again, observed)


def test_preview_is_clipped_per_motion_and_uses_latent_deltas(
    tmp_path, monkeypatch
):
    command, latent = _fake_command(tmp_path)
    monkeypatch.setattr(wbt_latent, "_motion_command", lambda env: command)

    observed = wbt_latent.motion_command_with_latent_preview(object())

    z0 = latent[[2, 5]]
    z_end = latent[[3, 6]]
    expected = torch.cat(
        [command.command, z0, z_end - z0, z_end - z0], dim=-1
    )
    assert observed.shape == (2, 11)
    assert torch.equal(observed, expected)


def test_attach_latent_preserves_reference_fields(tmp_path):
    from scripts.attach_latent_to_wbt import build_augmented_reference

    reference_path = tmp_path / "reference.npz"
    latent_path = tmp_path / "latent.npz"
    joint_pos = np.arange(20, dtype=np.float64).reshape(4, 5)
    body_names = np.array(["pelvis", "foot"])
    np.savez(
        reference_path,
        fps=np.array([50]),
        joint_pos=joint_pos,
        body_names=body_names,
    )
    latent = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.savez(
        latent_path,
        fps=np.array([50]),
        joint_pos=np.zeros_like(joint_pos),
        latent_z=latent,
    )

    out = build_augmented_reference(reference_path, latent_path)

    assert np.array_equal(out["joint_pos"], joint_pos)
    assert np.array_equal(out["body_names"], body_names)
    assert np.array_equal(out["latent_z"], latent)
