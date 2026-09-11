"""Count-first contact-audit metrics: edge cases and the pooled-F1 counterexample."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from snmr.contact_audit import (
    ConfusionCounts,
    WindowMasks,
    audit_windows,
    f1_from_counts,
    frame_level_counts,
    iou_from_counts,
    legacy_ratio_mean,
    macro_scores,
    micro_scores,
    per_column_counts,
    precision_from_counts,
    recall_from_counts,
    scores_from_counts,
)


# --- counts -------------------------------------------------------------------------------


def test_counts_from_masks_match_hand_count():
    cand = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=bool)
    orac = np.array([[1, 0], [0, 1], [0, 1], [0, 0]], dtype=bool)
    c = ConfusionCounts.from_masks(cand, orac)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 2, 1, 3)
    assert c.total == 8
    per_foot = per_column_counts(cand, orac)
    assert ConfusionCounts.sum(per_foot) == c


def test_counts_accept_torch_and_reject_shape_mismatch():
    cand = torch.tensor([[True, False], [False, True]])
    orac = torch.tensor([[True, True], [False, False]])
    c = ConfusionCounts.from_masks(cand, orac)
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    with pytest.raises(ValueError):
        ConfusionCounts.from_masks(cand, orac[:1])
    with pytest.raises(ValueError):
        ConfusionCounts(1, -1, 0, 0)
    with pytest.raises(TypeError):
        ConfusionCounts(1.0, 0, 0, 0)  # type: ignore[arg-type]


# --- the edge cases the frozen auditor got wrong -----------------------------------------


def test_zero_overlap_with_errors_is_f1_zero_not_nan():
    """TP=0, FP>0, FN>0: the old code returned NaN (precision+recall == 0) and dropped it."""
    c = ConfusionCounts(tp=0, fp=5, fn=3, tn=100)
    assert precision_from_counts(c) == 0.0
    assert recall_from_counts(c) == 0.0
    assert f1_from_counts(c) == 0.0
    assert iou_from_counts(c) == 0.0
    legacy = legacy_ratio_mean([c])
    assert legacy["f1"] is None  # the frozen rule discarded this window entirely


def test_oracle_empty_candidate_positive_is_f1_zero():
    """Oracle has no positives, candidate predicts some: F1 = 0/(FP) = 0, recall undefined."""
    c = ConfusionCounts(tp=0, fp=7, fn=0, tn=50)
    assert precision_from_counts(c) == 0.0
    assert recall_from_counts(c) is None
    assert f1_from_counts(c) == 0.0
    scores = scores_from_counts(c)
    assert scores["recall"] is None and scores["f1"] == 0.0
    assert legacy_ratio_mean([c])["f1"] is None


def test_all_empty_window_policy_is_explicit():
    c = ConfusionCounts(tp=0, fp=0, fn=0, tn=40)
    assert c.is_all_empty
    assert f1_from_counts(c, "undefined") is None
    assert f1_from_counts(c, "one") == 1.0
    assert f1_from_counts(c, "zero") == 0.0
    with pytest.raises(ValueError):
        f1_from_counts(c, "nan")  # type: ignore[arg-type]
    scores = scores_from_counts(c, "one")
    assert scores["all_empty"] is True and scores["all_empty_policy"] == "one"


def test_two_window_counterexample_pooled_f1_one_versus_count_f1():
    """The reviewer's constructed case.

    Window 1: one correctly detected contact frame.  Window 2: 100 false-positive frames only.
    The frozen pooled rule averages the surviving F1 values -> 1.0.  Summed counts give
    2*1 / (2*1 + 100 + 0) = 2/102.
    """
    w1 = ConfusionCounts(tp=1, fp=0, fn=0, tn=99)
    w2 = ConfusionCounts(tp=0, fp=100, fn=0, tn=0)
    legacy = legacy_ratio_mean([w1, w2])
    assert legacy["f1"] == pytest.approx(1.0)
    assert legacy["windows_contributing"]["f1"] == 1  # w2 silently dropped
    micro = micro_scores([w1, w2])
    assert micro["f1"] == pytest.approx(2 / 102)
    assert micro["precision"] == pytest.approx(1 / 101)
    assert micro["recall"] == pytest.approx(1.0)
    assert micro["windows"] == 2


def test_legacy_populations_differ_across_metrics():
    """The frozen per-clip triple (P 0.057, R 0.890, F1 0.493) shape: F1 from fewer windows."""
    rows = [
        ConfusionCounts(tp=0, fp=30, fn=0, tn=100),  # oracle empty -> legacy drops recall+F1
        ConfusionCounts(tp=5, fp=50, fn=1, tn=100),
        ConfusionCounts(tp=0, fp=40, fn=2, tn=100),  # zero overlap -> legacy drops F1
    ]
    legacy = legacy_ratio_mean(rows)
    pops = legacy["windows_contributing"]
    assert pops["precision"] == 3 and pops["recall"] == 2 and pops["f1"] == 1
    micro = micro_scores(rows)
    # One shared confusion matrix: P, R and F1 must be mutually consistent.
    p, r, f1 = micro["precision"], micro["recall"], micro["f1"]
    assert f1 == pytest.approx(2 * p * r / (p + r))
    # The legacy triple is not consistent with any single confusion matrix.
    lp, lr, lf1 = legacy["precision"], legacy["recall"], legacy["f1"]
    assert lf1 != pytest.approx(2 * lp * lr / (lp + lr))


# --- micro vs macro labelling -------------------------------------------------------------


def test_micro_scores_are_ratios_of_summed_counts():
    rows = [ConfusionCounts(3, 1, 2, 10), ConfusionCounts(0, 4, 0, 20), ConfusionCounts(0, 0, 0, 5)]
    micro = micro_scores(rows)
    total = ConfusionCounts.sum(rows)
    assert micro["counts"] == total.to_dict()
    assert micro["f1"] == pytest.approx(2 * 3 / (2 * 3 + 5 + 2))
    assert micro["windows_all_empty"] == 1
    assert micro["windows_with_oracle_positives"] == 1
    assert "window-weighted" in micro["estimand"]


def test_macro_scores_state_unit_weights_and_undefined_policy():
    rows = [ConfusionCounts(1, 0, 0, 9), ConfusionCounts(0, 10, 0, 0), ConfusionCounts(0, 0, 0, 10)]
    excl = macro_scores(rows, "f1", unit="window", weights="uniform", undefined="exclude")
    assert excl["value"] == pytest.approx((1.0 + 0.0) / 2)
    assert excl["n_eligible"] == 2 and excl["n_undefined"] == 1 and excl["n_total"] == 3
    zero = macro_scores(rows, "f1", weights="uniform", undefined="zero")
    assert zero["value"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    weighted = macro_scores(rows, "f1", weights="samples", undefined="exclude")
    assert weighted["value"] == pytest.approx((1.0 * 10 + 0.0 * 10) / 20)
    assert "Not a confusion recomposition" in excl["estimand"]
    with pytest.raises(ValueError):
        macro_scores(rows, "f1", weights="median")  # type: ignore[arg-type]


def test_audit_windows_keeps_every_window_in_the_denominator():
    rows = [
        {"clip": "a", "counts": ConfusionCounts(0, 20, 5, 100)},
        {"clip": "a", "counts": {"tp": 4, "fp": 1, "fn": 1, "tn": 100}},
    ]
    out = audit_windows(rows)
    assert out["windows"] == 2
    assert out["micro"]["windows"] == 2
    assert out["legacy_ratio_mean"]["windows_contributing"]["f1"] == 1
    assert out["micro"]["f1"] == pytest.approx(8 / (8 + 21 + 6))


# --- overlapping windows ------------------------------------------------------------------


def test_frame_level_counts_deduplicate_overlap():
    cand = np.zeros((10, 1), dtype=bool)
    orac = np.zeros((10, 1), dtype=bool)
    cand[2:6, 0] = True
    orac[3:7, 0] = True
    w1 = WindowMasks(0, cand[0:6], orac[0:6])
    w2 = WindowMasks(4, cand[4:10], orac[4:10])  # frames 4,5 covered twice
    summed = ConfusionCounts.sum(
        [ConfusionCounts.from_masks(w.candidate, w.oracle) for w in (w1, w2)]
    )
    frame = frame_level_counts([w1, w2])
    assert frame["frames_union"] == 10
    assert frame["frames_multiply_covered"] == 2
    assert frame["overlap_disagreements"] == 0
    # Frames 4 and 5 are TP in both windows: the summed count double-counts them.
    assert summed.tp == frame["counts"]["tp"] + 2
    assert frame["counts"] == ConfusionCounts.from_masks(cand, orac).to_dict()


def test_frame_level_counts_report_overlap_disagreements_instead_of_hiding_them():
    a = np.ones((4, 1), dtype=bool)
    b = np.zeros((4, 1), dtype=bool)
    w1 = WindowMasks(0, a, a)
    w2 = WindowMasks(2, b, b)  # disagrees with w1 on frames 2,3 for both masks
    frame = frame_level_counts([w1, w2])
    assert frame["overlap_disagreements"] == 4
    assert frame["frames_multiply_covered"] == 2


def test_no_nan_leaves_the_module():
    rows = [ConfusionCounts(0, 0, 0, 5), ConfusionCounts(0, 3, 0, 5)]
    for block in (micro_scores(rows), scores_from_counts(rows[0])):
        for value in block.values():
            if isinstance(value, float):
                assert math.isfinite(value)
