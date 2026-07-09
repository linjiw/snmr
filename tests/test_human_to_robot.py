"""End-to-end human→latent→robot retargeting against the GMR teacher (miniature of Phase-1 N3).

Takes a real (LAFAN1 human, GMR-teacher qpos) pair, fits the full retarget_human_to_robot path with
the distillation loss, and requires substantial convergence. This is the production data flow —
different skeleton on the encoder side (24 human bodies) than the decoder side (G1 robot graph).
"""

import pathlib

import numpy as np
import pytest
import torch

from snmr.human import human_static_features, lafan1_skeleton, load_pair_npz
from snmr.losses import total_loss
from snmr.model import SNMR, SNMRConfig
from snmr.robot_model import RobotKinematics

def test_human_to_robot_training_converges(g1_mjcf, g1_pair_npz):
    torch.manual_seed(0)
    pair = load_pair_npz(g1_pair_npz)
    n = 48
    human_pos = pair["human_pos"][:n]
    human_quat = pair["human_quat"][:n]
    qpos = pair["qpos"][:n]
    teacher = {
        "root_pos": qpos[:, 0:3],
        "root_quat": qpos[:, 3:7],
        "dof_pos": qpos[:, 7:],
    }

    rk = RobotKinematics(g1_mjcf)
    skel = lafan1_skeleton()
    static = human_static_features(skel, body_pos_sample=pair["human_pos"])

    model = SNMR(SNMRConfig(latent_dim=64, enc_hidden=128, dec_hidden=128))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    history = []
    for step in range(220):
        opt.zero_grad()
        pred = model.retarget_human_to_robot(human_pos, human_quat, skel, static, rk)
        loss, _ = total_loss(pred, rk, teacher=teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach()))

    first = np.mean(history[:5])
    last = np.mean(history[-5:])
    assert last < first * 0.3, f"human->robot training did not converge: {first:.4f} -> {last:.4f}"

    # teacher qpos root height is near the G1 standing height; prediction should be in that regime
    pred = model.retarget_human_to_robot(human_pos, human_quat, skel, static, rk)
    assert (pred["root_pos"][:, 2] - teacher["root_pos"][:, 2]).abs().mean() < 0.15
