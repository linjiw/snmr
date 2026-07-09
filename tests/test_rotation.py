"""Correctness of rotation utilities, cross-checked against scipy."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciR

from snmr import rotation as rot


def _random_quats_wxyz(n, seed=0):
    rng = np.random.default_rng(seed)
    q_xyzw = SciR.random(n, rng).as_quat()  # scipy is xyzw
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
    return torch.tensor(q_wxyz, dtype=torch.float64), q_xyzw


def test_quat_to_matrix_matches_scipy():
    q_wxyz, q_xyzw = _random_quats_wxyz(64)
    m = rot.quat_to_matrix(q_wxyz).numpy()
    m_ref = SciR.from_quat(q_xyzw).as_matrix()
    assert np.allclose(m, m_ref, atol=1e-10)


def test_quat_mul_matches_scipy():
    a_w, a_x = _random_quats_wxyz(32, seed=1)
    b_w, b_x = _random_quats_wxyz(32, seed=2)
    prod = rot.quat_mul(a_w, b_w).numpy()
    prod_ref_xyzw = (SciR.from_quat(a_x) * SciR.from_quat(b_x)).as_quat()
    prod_ref = np.concatenate([prod_ref_xyzw[:, 3:4], prod_ref_xyzw[:, :3]], axis=1)
    # quaternion double cover: compare up to sign
    same = np.allclose(prod, prod_ref, atol=1e-8)
    flipped = np.allclose(prod, -prod_ref, atol=1e-8)
    assert same or np.all(np.isclose(np.abs((prod * prod_ref).sum(1)), 1.0, atol=1e-6))


def test_quat_rotate_matches_matrix():
    q_w, q_x = _random_quats_wxyz(32, seed=3)
    v = torch.randn(32, 3, dtype=torch.float64)
    out = rot.quat_rotate(q_w, v).numpy()
    ref = np.einsum("nij,nj->ni", SciR.from_quat(q_x).as_matrix(), v.numpy())
    assert np.allclose(out, ref, atol=1e-9)


def test_rot6d_roundtrip_is_identity_on_valid_matrices():
    q_w, _ = _random_quats_wxyz(128, seed=4)
    m = rot.quat_to_matrix(q_w)
    r6 = rot.matrix_to_rot6d(m)
    m2 = rot.rot6d_to_matrix(r6)
    assert torch.allclose(m, m2, atol=1e-6)


def test_rot6d_to_matrix_is_orthonormal_on_garbage_input():
    # The Gram-Schmidt decode must always produce a valid rotation, even from arbitrary 6-vectors
    # (this is what makes it a safe network output head).
    r6 = torch.randn(256, 6, dtype=torch.float64)
    m = rot.rot6d_to_matrix(r6)
    eye = m.transpose(-1, -2) @ m
    assert torch.allclose(eye, torch.eye(3, dtype=torch.float64).expand_as(eye), atol=1e-6)
    det = torch.linalg.det(m)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-6)


def test_quat_matrix_roundtrip():
    q_w, _ = _random_quats_wxyz(128, seed=5)
    m = rot.quat_to_matrix(q_w)
    q2 = rot.matrix_to_quat(m)
    # compare as rotations (sign-agnostic)
    ang = rot.quat_geodesic_angle(q_w, q2)
    assert torch.all(ang < 1e-5)


def test_geodesic_angle_matches_scipy():
    q0_w, q0_x = _random_quats_wxyz(64, seed=6)
    q1_w, q1_x = _random_quats_wxyz(64, seed=7)
    ang = rot.quat_geodesic_angle(q0_w, q1_w).numpy()
    rel = SciR.from_quat(q0_x).inv() * SciR.from_quat(q1_x)
    ang_ref = rel.magnitude()
    assert np.allclose(ang, ang_ref, atol=1e-5)


def test_convention_helpers_roundtrip():
    q_w, _ = _random_quats_wxyz(16, seed=8)
    assert torch.allclose(rot.xyzw_to_wxyz(rot.wxyz_to_xyzw(q_w)), q_w)
