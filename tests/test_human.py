"""Human-side (LAFAN1) skeleton, features, and contact flags — validated on a real pair NPZ."""

import pathlib

import pytest
import torch

from snmr import rotation as rot
from snmr.human import (
    LAFAN1_BODY_NAMES,
    LAFAN1_PARENTS,
    foot_contact_flags,
    human_pose_features,
    human_static_features,
    lafan1_skeleton,
    load_pair_npz,
)

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "pair_g1_aiming1_subject1.npz"


def _real_pair():
    if FIXTURE.exists():
        return load_pair_npz(str(FIXTURE))
    from snmr import paths

    try:
        files = sorted(paths.pairs_dir("unitree_g1").glob("*.npz"))
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if not files:
        pytest.skip("no pair NPZs found (run scripts/make_pairs_lafan1.py)")
    return load_pair_npz(str(files[0]))


def test_lafan1_skeleton_topology():
    sk = lafan1_skeleton()
    assert sk.num_nodes == 24
    assert len(LAFAN1_PARENTS) == len(LAFAN1_BODY_NAMES) == 24
    # end effectors include toes, hands, head, footmods
    for name in ["LeftToe", "RightToe", "LeftHand", "RightHand", "Head", "LeftFootMod"]:
        assert sk.is_end_effector[sk.names.index(name)], name
    # parent-before-child ordering enforced by SkeletonGraph.__post_init__ (constructing = passing)


def test_pair_npz_loads_and_matches_names():
    pair = _real_pair()
    assert pair["human_names"] == LAFAN1_BODY_NAMES
    T = pair["human_pos"].shape[0]
    assert pair["human_pos"].shape == (T, 24, 3)
    assert pair["human_quat"].shape == (T, 24, 4)
    assert pair["qpos"].shape[1] == 7 + 29  # G1
    assert pair["fps"] == 30.0
    # quats ~unit
    norms = pair["human_quat"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_human_features_heading_invariance():
    pair = _real_pair()
    pos = pair["human_pos"][:20]
    quat = pair["human_quat"][:20]
    feats_a = human_pose_features(pos, quat)

    # rotate the whole world by a yaw + translate; features must not change
    T, J, _ = pos.shape
    zaxis = torch.zeros(T, J, 3)
    zaxis[..., 2] = 1.0
    yaw = torch.full((T, J), 0.9)
    dq = rot.axis_angle_to_quat(zaxis, yaw)
    pos_b = rot.quat_rotate(dq, pos) + torch.tensor([2.0, -1.0, 0.0])
    quat_b = rot.quat_mul(dq, quat)
    feats_b = human_pose_features(pos_b, quat_b)
    assert torch.allclose(feats_a, feats_b, atol=1e-4), (feats_a - feats_b).abs().max().item()


def test_human_feature_dim_matches_robot_side():
    pair = _real_pair()
    feats = human_pose_features(pair["human_pos"][:5], pair["human_quat"][:5])
    assert feats.shape == (5, 24, 12)  # 3 pos + 6 rot6d + 3 vel — same as robot_pose_features


def test_static_features_schema():
    pair = _real_pair()
    sk = lafan1_skeleton()
    sf = human_static_features(sk, body_pos_sample=pair["human_pos"])
    assert sf.shape == (24, 8)
    # root has has_dof=0, others 1
    assert sf[0, 6] == 0.0 and sf[1:, 6].min() == 1.0
    # data-derived offsets: thigh bone should have nonzero length
    thigh = sf[LAFAN1_BODY_NAMES.index("LeftLeg"), :3].norm()
    assert 0.2 < thigh < 0.8, f"implausible thigh length {thigh}"


def test_contact_flags_plausible_on_real_clip():
    pair = _real_pair()
    flags = foot_contact_flags(pair["human_pos"], pair["human_names"])
    T = flags.shape[0]
    assert flags.shape == (T, 2)
    frac = flags.mean().item()
    # In natural motion (walk/aiming), feet are in contact a meaningful but not total fraction.
    assert 0.05 < frac < 0.99, f"implausible contact fraction {frac}"


def test_encoder_consumes_human_motion():
    """The skeleton-agnostic encoder must accept the human graph end-to-end (24 nodes, not robot)."""
    from snmr.model import SNMR, SNMRConfig, _adjacency

    pair = _real_pair()
    sk = lafan1_skeleton()
    feats = human_pose_features(pair["human_pos"][:16], pair["human_quat"][:16])
    static = human_static_features(sk, body_pos_sample=pair["human_pos"])
    adj = _adjacency(sk)
    model = SNMR(SNMRConfig(latent_dim=32, enc_hidden=64, dec_hidden=64))
    z = model.encode(feats, static, adj)
    assert z.shape == (16, 32)
    assert torch.isfinite(z).all()
