#!/usr/bin/env python
"""Paired dropout-severity analysis for E78 (docs/LATENT_BENEFIT_PROGRAM_2026-08-15.md §E4).

Reads the per-rollout ``completed`` / ``start_steps`` / ``motion_ids`` arrays that every
E52-family evaluation report writes, and for a treatment arm vs a reference arm reports,
per severity:

* clean completion of each arm and their paired difference (clean-regression check);
* the E77-addendum matched-subset contrast: completion under dropout restricted to the
  rollouts BOTH arms complete cleanly, with McNemar discordant counts;
* the primary endpoint, the paired all-rollout completion difference under dropout with a
  cluster bootstrap over start windows (rollouts sharing a start step move together);
* survival-time difference, which is completion's lower-variance sibling.

Marginal retention ratios (degraded / own clean) are deliberately NOT reported: they
launder a clean-condition gap (E77 addendum).

usage:
  analyze_e78_dropout.py --treatment DIR:ARM --reference DIR:ARM [--treatment ...]
                         [--pattern 'maskall_hold_f{frac}_s{seg}'] [--out report.json]

Each DIR must contain ``<ARM>_eval.json`` (clean) and the sweep reports
``<ARM>_eval_<pattern>.json``.  Multiple ``--treatment DIR:ARM`` values (seeds) are
pooled by concatenating rollouts, matched pairwise to the reference in the same order.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

SEVERITY_RE = re.compile(r"_eval_(mask\w+?_hold_f(?P<frac>[0-9.]+)_s(?P<lo>\d+)-(?P<hi>\d+))\.json$")


def load_report(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def discover_severities(directory: pathlib.Path, arm: str) -> dict[str, pathlib.Path]:
    found = {}
    for path in sorted(directory.glob(f"{arm}_eval_mask*.json")):
        m = SEVERITY_RE.search(path.name)
        if m:
            found[m.group(1)] = path
    return found


def paired_arrays(treatment: list[dict], reference: list[dict]) -> tuple[np.ndarray, ...]:
    """Concatenate seeds and check the pairing invariant (same starts, same motions)."""
    t_done, r_done, t_surv, r_surv, starts, motions = [], [], [], [], [], []
    for t, r in zip(treatment, reference):
        if t["start_steps"] != r["start_steps"] or t["motion_ids"] != r["motion_ids"]:
            raise ValueError("treatment and reference reports are not paired (start_steps/motion_ids differ)")
        t_done.append(np.asarray(t["completed"], dtype=bool))
        r_done.append(np.asarray(r["completed"], dtype=bool))
        t_surv.append(np.asarray(t["survival_s"], dtype=float))
        r_surv.append(np.asarray(r["survival_s"], dtype=float))
        starts.append(np.asarray(t["start_steps"]))
        motions.append(np.asarray(t["motion_ids"]))
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    return cat(t_done), cat(r_done), cat(t_surv), cat(r_surv), cat(starts), cat(motions)


def cluster_bootstrap_diff(
    x: np.ndarray, y: np.ndarray, clusters: np.ndarray, *, n_boot: int = 4000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean paired difference with a percentile CI from resampling clusters with replacement."""
    ids, inverse = np.unique(clusters, return_inverse=True)
    diff = x.astype(float) - y.astype(float)
    per_cluster_sum = np.bincount(inverse, weights=diff, minlength=len(ids))
    per_cluster_n = np.bincount(inverse, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    boot = per_cluster_sum[draws].sum(1) / per_cluster_n[draws].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def analyze(treatment: list[dict], reference: list[dict],
            treatment_clean: list[dict], reference_clean: list[dict]) -> dict:
    t_done, r_done, t_surv, r_surv, starts, _ = paired_arrays(treatment, reference)
    tc, rc, *_ = paired_arrays(treatment_clean, reference_clean)
    both_clean = tc & rc
    d, lo, hi = cluster_bootstrap_diff(t_done, r_done, starts)
    sd, slo, shi = cluster_bootstrap_diff(t_surv, r_surv, starts)
    out = {
        "n_rollouts": int(t_done.size),
        "treatment_completion": float(t_done.mean()),
        "reference_completion": float(r_done.mean()),
        "paired_diff": d, "paired_diff_ci95": [lo, hi],
        "treatment_survival_s": float(t_surv.mean()),
        "reference_survival_s": float(r_surv.mean()),
        "survival_diff_s": sd, "survival_diff_ci95": [slo, shi],
        "matched_subset": {
            "n": int(both_clean.sum()),
            "treatment": float(t_done[both_clean].mean()) if both_clean.any() else None,
            "reference": float(r_done[both_clean].mean()) if both_clean.any() else None,
            "treatment_only": int((t_done & ~r_done & both_clean).sum()),
            "reference_only": int((r_done & ~t_done & both_clean).sum()),
        },
    }
    return out


def parse_pairs(values: list[str]) -> list[tuple[pathlib.Path, str]]:
    pairs = []
    for value in values:
        directory, _, arm = value.rpartition(":")
        if not directory or not arm:
            raise SystemExit(f"expected DIR:ARM, got {value!r}")
        pairs.append((pathlib.Path(directory), arm))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--treatment", action="append", required=True, help="DIR:ARM (repeat for seeds)")
    ap.add_argument("--reference", action="append", required=True, help="DIR:ARM (repeat for seeds)")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    treat = parse_pairs(args.treatment)
    ref = parse_pairs(args.reference)
    if len(treat) != len(ref):
        raise SystemExit("need one --reference per --treatment (seed-paired)")

    t_clean = [load_report(d / f"{a}_eval.json") for d, a in treat]
    r_clean = [load_report(d / f"{a}_eval.json") for d, a in ref]
    result = {"clean": analyze(t_clean, r_clean, t_clean, r_clean), "severities": {}}
    severities = discover_severities(*treat[0])
    for label, _ in severities.items():
        try:
            t = [load_report(d / f"{a}_eval_{label}.json") for d, a in treat]
            r = [load_report(d / f"{a}_eval_{label}.json") for d, a in ref]
        except FileNotFoundError as exc:
            result["severities"][label] = {"missing": str(exc)}
            continue
        result["severities"][label] = analyze(t, r, t_clean, r_clean)

    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
