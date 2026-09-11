"""Count-first agreement metrics for binary contact masks.

Why this module exists
----------------------
The Gate 1b pre-study auditor (``scripts/audit_contact_masks.py``, frozen 2026-07-14)
computed F1 per window *through* precision and recall, returned ``NaN`` whenever
``precision + recall == 0`` or whenever either ratio was undefined, and then averaged the
surviving per-window ratios, dropping ``NaN`` separately for each metric.  That has three
consequences:

1. A window with zero overlap but real errors (TP = 0, FP > 0, FN > 0) has a well-defined
   F1 of 0, yet the old code discarded it.  Discarding failed windows inflates F1.
2. A window where the oracle has no positives but the candidate predicts positives
   (TP = FN = 0, FP > 0) also has F1 = 0 (the count denominator ``2TP + FP + FN`` is
   positive), yet it was discarded as well.
3. Precision, recall and F1 were averaged over *different* window populations, so the
   reported triple could not describe any single confusion matrix.

Everything here is computed from raw confusion counts.  F1 is ``2TP / (2TP + FP + FN)``
whenever that denominator is positive; the only genuinely undefined case is the all-empty
window (TP = FP = FN = 0), whose convention is an explicit, recorded parameter rather than a
silent exclusion.

Two estimands are provided and named:

* **micro** — scores of the *summed* confusion counts.  This is the confusion-recomposition
  the old docstring claimed.  When evaluation windows overlap, summed-window counts weight
  overlapped frames more than once; that is a *window-weighted* estimand, and
  :func:`frame_level_counts` gives the frame-level alternative in which every frame of the
  union is counted once.
* **macro** — an average of per-window ratios.  It is reported only with its unit, weights,
  eligible denominator and undefined-window policy spelled out, and it is never labelled
  as confusion recomposition.

The legacy aggregate is reproduced by :func:`legacy_ratio_mean` so that old and new numbers
can be reconciled side by side; it must not be used for new claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np

AllEmptyPolicy = Literal["undefined", "one", "zero"]
UndefinedPolicy = Literal["exclude", "zero"]
MacroWeights = Literal["uniform", "samples"]

_ALL_EMPTY_POLICIES = ("undefined", "one", "zero")
_UNDEFINED_POLICIES = ("exclude", "zero")
_MACRO_WEIGHTS = ("uniform", "samples")


def _as_bool_array(mask: Any, name: str) -> np.ndarray:
    """Accept torch tensors, numpy arrays or nested sequences; return a bool ndarray."""
    if hasattr(mask, "detach"):  # torch.Tensor without importing torch here
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        if not np.all(np.isin(array, (0, 1))):
            raise ValueError(f"{name} must be binary (bool or 0/1), got other values")
        array = array.astype(bool)
    return array


@dataclass(frozen=True)
class ConfusionCounts:
    """Raw confusion counts of a candidate mask scored against an oracle mask."""

    tp: int
    fp: int
    fn: int
    tn: int

    def __post_init__(self) -> None:
        for field in ("tp", "fp", "fn", "tn"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{field} must be an integer count, got {value!r}")
            if value < 0:
                raise ValueError(f"{field} must be nonnegative, got {value}")
            object.__setattr__(self, field, int(value))

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def oracle_positives(self) -> int:
        return self.tp + self.fn

    @property
    def candidate_positives(self) -> int:
        return self.tp + self.fp

    @property
    def is_all_empty(self) -> bool:
        """No positive in either mask: the only case where F1 is genuinely undefined."""
        return self.tp == 0 and self.fp == 0 and self.fn == 0

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        if not isinstance(other, ConfusionCounts):
            return NotImplemented
        return ConfusionCounts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    @classmethod
    def zero(cls) -> "ConfusionCounts":
        return cls(0, 0, 0, 0)

    @classmethod
    def from_masks(cls, candidate: Any, oracle: Any) -> "ConfusionCounts":
        """Count agreement between two equally shaped binary masks (any shape)."""
        cand = _as_bool_array(candidate, "candidate")
        orac = _as_bool_array(oracle, "oracle")
        if cand.shape != orac.shape:
            raise ValueError(
                f"candidate and oracle masks must share a shape, got {cand.shape} vs {orac.shape}"
            )
        return cls(
            tp=int(np.count_nonzero(cand & orac)),
            fp=int(np.count_nonzero(cand & ~orac)),
            fn=int(np.count_nonzero(~cand & orac)),
            tn=int(np.count_nonzero(~cand & ~orac)),
        )

    @classmethod
    def sum(cls, counts: Iterable["ConfusionCounts"]) -> "ConfusionCounts":
        total = cls.zero()
        for item in counts:
            if not isinstance(item, ConfusionCounts):
                raise TypeError("sum() expects ConfusionCounts instances")
            total = total + item
        return total

    def to_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfusionCounts":
        return cls(int(data["tp"]), int(data["fp"]), int(data["fn"]), int(data["tn"]))


def per_column_counts(candidate: Any, oracle: Any) -> list[ConfusionCounts]:
    """Counts for each column of ``(T, F)`` masks, e.g. one entry per foot."""
    cand = _as_bool_array(candidate, "candidate")
    orac = _as_bool_array(oracle, "oracle")
    if cand.ndim != 2 or cand.shape != orac.shape:
        raise ValueError("per_column_counts expects matching (T, F) masks")
    return [ConfusionCounts.from_masks(cand[:, f], orac[:, f]) for f in range(cand.shape[1])]


def _check_all_empty_policy(policy: str) -> None:
    if policy not in _ALL_EMPTY_POLICIES:
        raise ValueError(f"all_empty must be one of {_ALL_EMPTY_POLICIES}, got {policy!r}")


def precision_from_counts(counts: ConfusionCounts) -> float | None:
    """TP / (TP + FP); ``None`` when the candidate predicts no positives."""
    denominator = counts.tp + counts.fp
    return counts.tp / denominator if denominator > 0 else None


def recall_from_counts(counts: ConfusionCounts) -> float | None:
    """TP / (TP + FN); ``None`` when the oracle has no positives."""
    denominator = counts.tp + counts.fn
    return counts.tp / denominator if denominator > 0 else None


def f1_from_counts(counts: ConfusionCounts, all_empty: AllEmptyPolicy = "undefined") -> float | None:
    """``2TP / (2TP + FP + FN)`` whenever the denominator is positive.

    Zero overlap with real errors gives 0, not NaN.  The all-empty window is governed by the
    explicit ``all_empty`` convention: ``"undefined"`` (returns ``None``), ``"one"`` (perfect
    agreement on an empty positive class), or ``"zero"``.
    """
    _check_all_empty_policy(all_empty)
    denominator = 2 * counts.tp + counts.fp + counts.fn
    if denominator > 0:
        return 2 * counts.tp / denominator
    if all_empty == "one":
        return 1.0
    if all_empty == "zero":
        return 0.0
    return None


def iou_from_counts(counts: ConfusionCounts, all_empty: AllEmptyPolicy = "undefined") -> float | None:
    """TP / (TP + FP + FN), with the same all-empty convention as F1."""
    _check_all_empty_policy(all_empty)
    denominator = counts.tp + counts.fp + counts.fn
    if denominator > 0:
        return counts.tp / denominator
    if all_empty == "one":
        return 1.0
    if all_empty == "zero":
        return 0.0
    return None


def scores_from_counts(
    counts: ConfusionCounts, all_empty: AllEmptyPolicy = "undefined"
) -> dict[str, Any]:
    """All ratio metrics of one confusion matrix.  Undefined ratios are ``None``, never NaN."""
    _check_all_empty_policy(all_empty)
    n = counts.total
    return {
        "precision": precision_from_counts(counts),
        "recall": recall_from_counts(counts),
        "f1": f1_from_counts(counts, all_empty),
        "iou": iou_from_counts(counts, all_empty),
        "candidate_prevalence": counts.candidate_positives / n if n else None,
        "oracle_prevalence": counts.oracle_positives / n if n else None,
        "agreement": (counts.tp + counts.tn) / n if n else None,
        "samples": n,
        "counts": counts.to_dict(),
        "all_empty_policy": all_empty,
        "all_empty": counts.is_all_empty,
    }


def micro_scores(
    counts: Iterable[ConfusionCounts], all_empty: AllEmptyPolicy = "undefined"
) -> dict[str, Any]:
    """Scores of the summed confusion counts (true confusion recomposition).

    If the summed windows overlap in frame coordinates this is a *window-weighted*
    estimand; see :func:`frame_level_counts` for the frame-level one.
    """
    rows = list(counts)
    total = ConfusionCounts.sum(rows)
    out = scores_from_counts(total, all_empty)
    out["estimand"] = (
        "micro: ratios of summed TP/FP/FN/TN over all contributing windows "
        "(window-weighted if windows overlap)"
    )
    out["windows"] = len(rows)
    out["windows_all_empty"] = sum(r.is_all_empty for r in rows)
    out["windows_with_oracle_positives"] = sum(r.oracle_positives > 0 for r in rows)
    out["windows_with_candidate_positives"] = sum(r.candidate_positives > 0 for r in rows)
    return out


def macro_scores(
    counts: Sequence[ConfusionCounts],
    statistic: str,
    *,
    unit: str = "window",
    weights: MacroWeights = "uniform",
    all_empty: AllEmptyPolicy = "undefined",
    undefined: UndefinedPolicy = "exclude",
) -> dict[str, Any]:
    """Average of per-unit ratios, with every convention stated in the result.

    ``statistic`` is one of ``precision``, ``recall``, ``f1``, ``iou``.  Units whose ratio
    is undefined are either excluded from the denominator (``undefined="exclude"``) or
    scored as 0 (``undefined="zero"``); both choices are recorded.  This is *not* confusion
    recomposition and must not be labelled as such.
    """
    if weights not in _MACRO_WEIGHTS:
        raise ValueError(f"weights must be one of {_MACRO_WEIGHTS}, got {weights!r}")
    if undefined not in _UNDEFINED_POLICIES:
        raise ValueError(f"undefined must be one of {_UNDEFINED_POLICIES}, got {undefined!r}")
    functions = {
        "precision": lambda c: precision_from_counts(c),
        "recall": lambda c: recall_from_counts(c),
        "f1": lambda c: f1_from_counts(c, all_empty),
        "iou": lambda c: iou_from_counts(c, all_empty),
    }
    if statistic not in functions:
        raise ValueError(f"unknown statistic {statistic!r}")
    values: list[float] = []
    unit_weights: list[float] = []
    n_undefined = 0
    for item in counts:
        value = functions[statistic](item)
        weight = 1.0 if weights == "uniform" else float(item.total)
        if value is None:
            n_undefined += 1
            if undefined == "exclude":
                continue
            value = 0.0
        values.append(float(value))
        unit_weights.append(weight)
    weight_total = float(sum(unit_weights))
    mean = (
        float(sum(v * w for v, w in zip(values, unit_weights)) / weight_total)
        if weight_total > 0
        else None
    )
    return {
        "estimand": (
            f"macro: {weights}-weighted mean of per-{unit} {statistic} ratios; "
            f"undefined units {'excluded from the denominator' if undefined == 'exclude' else 'scored as 0'}; "
            f"all-empty convention {all_empty!r}. Not a confusion recomposition."
        ),
        "statistic": statistic,
        "unit": unit,
        "weights": weights,
        "all_empty_policy": all_empty,
        "undefined_policy": undefined,
        "value": mean,
        "n_total": len(counts),
        "n_eligible": len(values),
        "n_undefined": n_undefined,
    }


def legacy_ratio_mean(counts: Sequence[ConfusionCounts]) -> dict[str, Any]:
    """Reproduce the frozen 2026-07-14 ``pooled`` aggregate from raw counts, for reconciliation.

    Per window: precision/recall are NaN when their denominators are zero; F1 is NaN when
    either ratio is NaN *or* when ``precision + recall == 0``; IoU is NaN when TP+FP+FN == 0.
    Then each metric is averaged over the windows where it is not NaN, weighted by window
    sample count.  Labelled ``legacy`` because it silently drops failed windows and averages
    ratios over metric-specific populations.  Never use for new claims.
    """
    per_window: list[dict[str, float]] = []
    for c in counts:
        n = c.total
        precision = c.tp / (c.tp + c.fp) if c.tp + c.fp > 0 else float("nan")
        recall = c.tp / (c.tp + c.fn) if c.tp + c.fn > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision == precision and recall == recall and precision + recall > 0
            else float("nan")
        )
        iou = c.tp / (c.tp + c.fp + c.fn) if c.tp + c.fp + c.fn > 0 else float("nan")
        per_window.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "candidate_prevalence": (c.tp + c.fp) / n if n else float("nan"),
                "oracle_prevalence": (c.tp + c.fn) / n if n else float("nan"),
                "samples": float(n),
            }
        )
    out: dict[str, Any] = {
        "estimand": (
            "legacy: sample-weighted mean of per-window ratios with NaN windows dropped "
            "independently per metric (the frozen 2026-07-14 auditor's `pooled`). "
            "Inflates F1; reported only for reconciliation."
        )
    }
    populations: dict[str, int] = {}
    for key in ("precision", "recall", "f1", "iou", "candidate_prevalence", "oracle_prevalence"):
        vals = [(r[key], r["samples"]) for r in per_window if r[key] == r[key]]
        w = sum(s for _, s in vals)
        out[key] = float(sum(v * s for v, s in vals) / w) if w else None
        populations[key] = len(vals)
    out["windows_contributing"] = populations
    out["samples"] = float(sum(r["samples"] for r in per_window))
    out["windows"] = len(per_window)
    return out


@dataclass(frozen=True)
class WindowMasks:
    """One evaluation window: clip-frame span plus the candidate/oracle ``(T, F)`` masks."""

    start: int
    candidate: np.ndarray
    oracle: np.ndarray

    def __post_init__(self) -> None:
        cand = _as_bool_array(self.candidate, "candidate")
        orac = _as_bool_array(self.oracle, "oracle")
        if cand.ndim != 2 or cand.shape != orac.shape:
            raise ValueError("WindowMasks expects matching (T, F) candidate/oracle masks")
        if isinstance(self.start, bool) or int(self.start) < 0:
            raise ValueError("start must be a nonnegative frame index")
        object.__setattr__(self, "candidate", cand)
        object.__setattr__(self, "oracle", orac)
        object.__setattr__(self, "start", int(self.start))

    @property
    def end(self) -> int:  # exclusive
        return self.start + int(self.candidate.shape[0])


def frame_level_counts(windows: Sequence[WindowMasks]) -> dict[str, Any]:
    """Confusion counts over the *union* of window frames, each frame counted once.

    Where windows overlap, the masks must agree on the shared frames (they do when both are
    clip-local); any disagreement is counted and reported rather than silently resolved,
    and the first window's value is used.  The result records how many frames were covered
    by more than one window so that the window-weighted micro estimand can be compared.
    """
    if not windows:
        return {
            "counts": ConfusionCounts.zero().to_dict(),
            "frames_union": 0,
            "frames_multiply_covered": 0,
            "overlap_disagreements": 0,
        }
    feet = windows[0].candidate.shape[1]
    length = max(w.end for w in windows)
    cand = np.zeros((length, feet), dtype=bool)
    orac = np.zeros((length, feet), dtype=bool)
    covered = np.zeros(length, dtype=np.int64)
    disagreements = 0
    for w in windows:
        if w.candidate.shape[1] != feet:
            raise ValueError("all windows must have the same number of mask columns")
        span = slice(w.start, w.end)
        already = covered[span] > 0
        if already.any():
            disagreements += int(np.count_nonzero(cand[span][already] != w.candidate[already]))
            disagreements += int(np.count_nonzero(orac[span][already] != w.oracle[already]))
        fresh = ~already
        cand[span][fresh] = w.candidate[fresh]
        orac[span][fresh] = w.oracle[fresh]
        covered[span] += 1
    keep = covered > 0
    counts = ConfusionCounts.from_masks(cand[keep], orac[keep])
    return {
        "counts": counts.to_dict(),
        "frames_union": int(np.count_nonzero(keep)),
        "frames_multiply_covered": int(np.count_nonzero(covered > 1)),
        "overlap_disagreements": disagreements,
    }


def audit_windows(
    rows: Sequence[dict[str, Any]],
    *,
    all_empty: AllEmptyPolicy = "undefined",
) -> dict[str, Any]:
    """Aggregate a list of per-window rows ``{"counts": ConfusionCounts, ...}``.

    Returns the micro scores, labelled macro scores for F1/precision/recall, the legacy
    aggregate for reconciliation, and the per-window population bookkeeping.  It never
    drops a window: every row is in ``windows`` and in the micro sum.
    """
    counts = [r["counts"] if isinstance(r["counts"], ConfusionCounts) else ConfusionCounts.from_dict(r["counts"]) for r in rows]
    return {
        "micro": micro_scores(counts, all_empty),
        "macro_window_uniform": {
            stat: macro_scores(counts, stat, unit="window", weights="uniform", all_empty=all_empty)
            for stat in ("precision", "recall", "f1", "iou")
        },
        "legacy_ratio_mean": legacy_ratio_mean(counts),
        "windows": len(counts),
    }


def finite_or_none(value: Any) -> Any:
    """JSON helper: turn NaN/inf into ``None`` so undefined never masquerades as a number."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
