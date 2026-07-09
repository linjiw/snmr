"""End-to-end overfit-a-batch: the model must learn to reproduce a real motion.

This is the decisive Phase-1 smoke test. We take a real 60-frame window from the holosoma G1 tracking
NPZ, run the full SNMR pipeline (encode robot pose features -> latent -> decode to qpos), and fit for a
few hundred steps. Success = the reconstruction error drops substantially and reaches a small absolute
value, proving the encode/latent/decode path + FK-consistent losses can represent genuine motion.

Kept deliberately small so it runs on CPU in well under a minute.
"""

import numpy as np
import torch

from snmr import rotation as rot
from snmr.data import RobotMotion, load_holosoma_wbt_npz
from snmr.model import SNMR, SNMRConfig
from snmr.robot_model import RobotKinematics
from snmr.train import fit_motion


def test_overfit_real_motion(g1_mjcf, g1_train_npz):
    torch.manual_seed(0)
    rk = RobotKinematics(g1_mjcf)
    full = load_holosoma_wbt_npz(g1_train_npz)
    # a 60-frame window
    n = min(60, full.num_frames)
    motion = RobotMotion(full.root_pos[:n], full.root_quat[:n], full.dof_pos[:n], full.fps)

    model = SNMR(SNMRConfig(latent_dim=64, enc_hidden=128, dec_hidden=128))
    res = fit_motion(model, rk, motion, steps=400, lr=2e-3, verbose=False)

    first = np.mean(res.losses[:5])
    last = np.mean(res.losses[-5:])
    assert last < first * 0.25, f"loss did not drop enough: {first:.4f} -> {last:.4f}"

    # absolute reconstruction quality on dof + root
    pred = res.final_pred
    dof_err = (pred["dof_pos"] - motion.dof_pos).abs().mean().item()
    pos_err = (pred["root_pos"] - motion.root_pos).abs().mean().item()
    ang_err = rot.quat_geodesic_angle(pred["root_quat"], motion.root_quat).mean().item()
    assert dof_err < 0.15, f"dof error too high: {dof_err}"
    assert pos_err < 0.10, f"root pos error too high: {pos_err}"
    assert ang_err < 0.30, f"root angle error too high: {ang_err}"


def test_fk_reconstruction_of_fit(g1_mjcf, g1_train_npz):
    """After fitting, the FK body positions of the prediction should be close to the target's FK."""
    torch.manual_seed(1)
    rk = RobotKinematics(g1_mjcf)
    full = load_holosoma_wbt_npz(g1_train_npz)
    n = min(40, full.num_frames)
    motion = RobotMotion(full.root_pos[:n], full.root_quat[:n], full.dof_pos[:n], full.fps)
    model = SNMR(SNMRConfig(latent_dim=64, enc_hidden=128, dec_hidden=128))
    res = fit_motion(model, rk, motion, steps=700, lr=3e-3)
    pred = res.final_pred
    bp_pred, _ = rk.forward_kinematics(pred["root_pos"], pred["root_quat"], pred["dof_pos"])
    bp_tgt, _ = rk.forward_kinematics(motion.root_pos, motion.root_quat, motion.dof_pos)
    mpjpe = (bp_pred - bp_tgt).norm(dim=-1).mean().item()
    # Whole-body mean per-joint position error after overfitting one clip; converges toward ~5cm
    # at 800 steps (see scripts/overfit_batch.py), so 10cm at 700 steps is a safe smoke threshold.
    assert mpjpe < 0.10, f"MPJPE too high after fit: {mpjpe}"
