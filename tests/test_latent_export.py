"""The z-augmented WBT export must (a) stay standard-schema valid and (b) time-align latent_z."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from export_wbt_with_latent import resample_latent  # noqa: E402


def test_resample_latent_endpoints_and_length():
    # a ramp latent: endpoints preserved, output length ~ dst/src * input
    T, d = 30, 8
    z = np.linspace(0, 1, T)[:, None] * np.ones((1, d))
    out = resample_latent(z, src_fps=30.0, dst_fps=50.0)
    assert np.allclose(out[0], z[0], atol=1e-6)
    assert np.allclose(out[-1], z[-1], atol=1e-3)
    # 30 frames @30fps = ~0.967s -> ~49 frames @50fps
    assert 47 <= out.shape[0] <= 50
    # monotonic ramp stays monotonic (no overshoot)
    assert np.all(np.diff(out[:, 0]) >= -1e-6)


def test_latent_export_is_standard_schema_plus_latent():
    """If the export artifact exists, it must carry all standard WBT keys (so vanilla training
    loads it) PLUS latent_z time-aligned to joint_pos."""
    npz = pathlib.Path(__file__).resolve().parents[1] / "runs" / "wbt_latent" / "walk1_z.npz"
    if not npz.exists():
        pytest.skip("run scripts/export_wbt_with_latent.py first")
    data = np.load(npz)
    required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w", "body_names", "joint_names"}
    assert required <= set(data.files), f"missing standard keys: {required - set(data.files)}"
    assert "latent_z" in data.files
    # z frame count matches the motion frame count (time-aligned for per-frame conditioning)
    assert data["latent_z"].shape[0] == data["joint_pos"].shape[0]
    assert data["latent_z"].shape[1] in (64, 128)  # our latent dims
    assert np.isfinite(data["latent_z"]).all()
