import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_wbt_reference_quality.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_wbt_reference_quality", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def _write_reference(path, *, moving_feet=False, body_names=None):
    frames = 4
    bodies = body_names or list(quality.DEFAULT_FEET)
    body_pos = np.zeros((frames, 2, 3), dtype=np.float64)
    if moving_feet:
        body_pos[:, :, 0] = np.arange(frames)[:, None] * 0.01
    identity = np.zeros((frames, 2, 4), dtype=np.float64)
    identity[..., 0] = 1.0
    joint_pos = np.zeros((frames, 9), dtype=np.float64)
    joint_pos[:, 3] = 1.0
    np.savez(
        path,
        fps=np.asarray([50]),
        joint_pos=joint_pos,
        joint_vel=np.zeros((frames, 8)),
        body_pos_w=body_pos,
        body_quat_w=identity,
        body_lin_vel_w=np.zeros((frames, 2, 3)),
        body_ang_vel_w=np.zeros((frames, 2, 3)),
        joint_names=np.asarray(["joint_a", "joint_b"]),
        body_names=np.asarray(bodies),
    )


def test_reference_quality_reports_pose_and_stance_context(tmp_path):
    gmr = tmp_path / "gmr.npz"
    snmr = tmp_path / "snmr.npz"
    _write_reference(gmr)
    _write_reference(snmr, moving_feet=True)

    result = quality.analyze(gmr, snmr)

    assert result["passed"], result["protocol_errors"]
    assert result["aligned_errors"]["body_mpjpe_m"]["mean"] == 0.015
    contact = result["gmr_height_mask_context"]
    assert contact["gmr"]["stance_speed_ms"] == 0.0
    assert contact["snmr"]["stance_speed_ms"] == 0.5
    assert result["decoded_height_mask_context"]["agreement"] == 1.0


def test_reference_quality_rejects_misaligned_body_names(tmp_path):
    gmr = tmp_path / "gmr.npz"
    snmr = tmp_path / "snmr.npz"
    _write_reference(gmr)
    _write_reference(snmr, body_names=["wrong_left", "wrong_right"])

    result = quality.analyze(gmr, snmr)

    assert not result["passed"]
    assert "body_names differ" in result["protocol_errors"]
