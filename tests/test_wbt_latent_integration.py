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


def test_latent_window_gathers_absolute_latents_clipped(tmp_path, monkeypatch):
    command, latent = _fake_command(tmp_path)
    monkeypatch.setattr(wbt_latent, "_motion_command", lambda env: command)

    observed = wbt_latent.snmr_latent_window(object())

    # WINDOW_OFFSETS=(0,5,10,25); env0 t=2 end 3, env1 t=5 end 6 -> all future offsets clip to end.
    idx0 = [min(2 + o, 3) for o in wbt_latent.WINDOW_OFFSETS]
    idx1 = [min(5 + o, 6) for o in wbt_latent.WINDOW_OFFSETS]
    expected = torch.stack(
        [
            torch.cat([latent[i] for i in idx0]),
            torch.cat([latent[i] for i in idx1]),
        ]
    )
    width = len(wbt_latent.WINDOW_OFFSETS) * latent.shape[1]
    assert observed.shape == (2, width)
    assert torch.equal(observed, expected)


def test_explicit_preview_gathers_future_joint_pos_clipped(tmp_path, monkeypatch):
    command, _ = _fake_command(tmp_path)
    joint_pos = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 2)
    command.motion.joint_pos = joint_pos
    monkeypatch.setattr(wbt_latent, "_motion_command", lambda env: command)

    observed = wbt_latent.motion_command_with_explicit_preview(object())

    # env 0: t=2 in clip [0,4) -> offsets +10/+25 clip to frame 3
    # env 1: t=5 in clip [4,7) -> offsets clip to frame 6
    expected = torch.cat(
        [command.command, joint_pos[[3, 6]], joint_pos[[3, 6]]], dim=-1
    )
    assert observed.shape == (2, 6)
    assert torch.equal(observed, expected)


def test_latent_preview_command_has_no_explicit_command(tmp_path, monkeypatch):
    command, latent = _fake_command(tmp_path)
    monkeypatch.setattr(wbt_latent, "_motion_command", lambda env: command)

    observed = wbt_latent.latent_preview_command(object())

    z0 = latent[[2, 5]]
    z_end = latent[[3, 6]]
    expected = torch.cat([z0, z_end - z0, z_end - z0], dim=-1)
    # 3 latent blocks only — must NOT contain the 2-wide explicit command.
    assert observed.shape == (2, 9)
    assert torch.equal(observed, expected)


def test_multi_motion_latent_concatenates_in_glob_order(tmp_path, monkeypatch):
    z_a = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    z_b = np.arange(100, 100 + 3 * 3, dtype=np.float32).reshape(3, 3)
    np.savez(tmp_path / "a_clip.npz", latent_z=z_a)
    np.savez(tmp_path / "b_clip.npz", latent_z=z_b)
    motion = SimpleNamespace(
        time_step_total=7,
        motion_start_idx=torch.tensor([0, 4]),
        motion_end_idx=torch.tensor([4, 7]),
    )
    command = SimpleNamespace(
        motion=motion,
        motion_cfg=SimpleNamespace(motion_file="", motion_dir=str(tmp_path)),
        device="cpu",
        time_steps=torch.tensor([2, 5]),
        motion_ids=torch.tensor([0, 1]),
        command=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    monkeypatch.setattr(wbt_latent, "_motion_command", lambda env: command)

    observed = wbt_latent.snmr_latent(object())

    expected_full = torch.from_numpy(np.concatenate([z_a, z_b]))
    assert torch.equal(motion.latent_z, expected_full)
    assert torch.equal(observed, expected_full[[2, 5]])


def test_multi_motion_latent_rejects_length_mismatch(tmp_path, monkeypatch):
    import pytest

    np.savez(tmp_path / "a_clip.npz", latent_z=np.zeros((5, 3), dtype=np.float32))
    motion = SimpleNamespace(
        time_step_total=4,
        motion_start_idx=torch.tensor([0]),
        motion_end_idx=torch.tensor([4]),
    )
    command = SimpleNamespace(
        motion=motion,
        motion_cfg=SimpleNamespace(motion_file="", motion_dir=str(tmp_path)),
        device="cpu",
        time_steps=torch.tensor([0]),
        motion_ids=torch.tensor([0]),
        command=torch.tensor([[1.0]]),
    )
    monkeypatch.setattr(wbt_latent, "_motion_command", lambda env: command)

    with pytest.raises(ValueError, match="latent_z frames"):
        wbt_latent.snmr_latent(object())


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
