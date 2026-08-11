import numpy as np

from scripts.analyze_e67_results import (
    clustered_paired_interval,
    hierarchical_paired_interval,
)


def test_clustered_paired_interval_preserves_pairing_and_direction():
    differences = np.array([1, 1, 0, 0, 1, 1], dtype=float)
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    interval = clustered_paired_interval(
        differences, clusters, seed=7, replicates=2_000
    )
    assert interval["difference"] == 2 / 3
    assert interval["ci95_low"] >= 0
    assert interval["ci95_high"] == 1
    assert interval["clusters"] == 3


def test_clustered_paired_interval_rejects_unpaired_shapes():
    try:
        clustered_paired_interval(np.ones(3), np.ones(2))
    except ValueError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("mismatched paired inputs were accepted")


def test_hierarchical_interval_preserves_seed_identity_and_pair_weighting():
    differences = [
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([-1.0, -1.0, 0.0, 0.0]),
    ]
    clusters = [np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]
    interval = hierarchical_paired_interval(
        differences, clusters, seed=7, replicates=4_000
    )
    assert interval["difference"] == 0.0
    assert interval["ci95_low"] < 0.0 < interval["ci95_high"]
    assert interval["clusters"] == 2
    assert interval["training_seeds"] == 2
    assert interval["per_seed_difference"] == [0.5, -0.5]


def test_hierarchical_single_seed_preserves_registered_cluster_interval():
    differences = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    registered = clustered_paired_interval(
        differences, clusters, seed=11, replicates=2_000
    )
    hierarchical = hierarchical_paired_interval(
        [differences], [clusters], seed=11, replicates=2_000
    )
    assert {key: hierarchical[key] for key in registered} == registered
    assert hierarchical["training_seeds"] == 1
    assert hierarchical["per_seed_difference"] == [2 / 3]


def test_hierarchical_interval_rejects_mismatched_pair_grids():
    try:
        hierarchical_paired_interval(
            [np.ones(2), np.ones(2)],
            [np.array([0, 1]), np.array([0, 2])],
        )
    except ValueError as error:
        assert "same ambiguity-pair clusters" in str(error)
    else:
        raise AssertionError("mismatched cross-seed pair grids were accepted")
