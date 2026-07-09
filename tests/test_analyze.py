"""Unit tests for the latent-analysis primitives (probe/CKA/retrieval) on synthetic data."""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from analyze_latent import linear_cka, linear_probe, retrieval  # noqa: E402


def test_cka_identity_and_orthogonal_invariance():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 16))
    assert abs(linear_cka(a, a) - 1.0) < 1e-6           # identical -> 1
    q, _ = np.linalg.qr(rng.standard_normal((16, 16)))  # orthogonal transform
    assert abs(linear_cka(a, a @ q) - 1.0) < 1e-5       # CKA is rotation-invariant
    b = rng.standard_normal((200, 16))
    assert linear_cka(a, b) < 0.3                        # independent -> low


def test_retrieval_perfect_and_chance():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((50, 8))
    perfect = retrieval(z, z.copy())
    assert perfect["R@1"] == 1.0 and perfect["MRR"] == 1.0 and perfect["median_rank"] == 1.0
    # a permuted gallery (no correspondence) should be near chance R@1 = 1/50
    scrambled = retrieval(z, rng.standard_normal((50, 8)))
    assert scrambled["R@1"] < 0.2


def test_linear_probe_separable_vs_random():
    rng = np.random.default_rng(2)
    n_per, n_cls = 60, 3
    # separable: each class centered far apart -> high accuracy
    feat, label, groups = [], [], []
    for c in range(n_cls):
        center = np.zeros(8)
        center[c] = 10.0
        feat.append(center + rng.standard_normal((n_per, 8)))
        label += [c] * n_per
        groups += list(rng.integers(0, 6, n_per))  # random clip groups
    feat = np.concatenate(feat)
    acc, chance = linear_probe(feat, np.array(label), np.array(groups))
    assert chance == pytest.approx(1 / 3, abs=1e-6)
    assert acc > 0.9

    # random labels -> near chance
    acc_r, _ = linear_probe(feat, rng.integers(0, n_cls, len(label)), np.array(groups))
    assert acc_r < 0.6


def test_motion_category_parsing():
    from analyze_latent import _motion_category

    assert _motion_category("walk1_subject5") == "walk"
    assert _motion_category("dance2_subject4") == "dance"
    assert _motion_category("sprint1_subject4") == "sprint"
