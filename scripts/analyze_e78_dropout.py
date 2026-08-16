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
* survival-time difference, which is completion's lower-variance sibling;
* **floor-relative retention** when a goal-blind floor arm is supplied (``--floor DIR:ARM``):

      R = (C_degraded - C_floor) / (C_clean - C_floor)

  the fraction of an arm's *channel-derived advantage* that survives the corruption.
  ``R = 1`` no loss; ``R = 0`` fell exactly to the goal-blind floor; **``R < 0`` means the arm
  ended up worse than having no command at all — its own channel actively harmed it**, a
  failure mode that completion, retention ratios and cross-arm differences all hide
  (E78-F: the explicit arm reaches R = -0.64 while every other arm stays positive).
  The floor arm is measured under the same dropout, which also absorbs any survivorship
  artifact, since dropout is a structural no-op for a goal-blind policy.

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

SEVERITY_RE = re.compile(r"_eval_(?:ambiguity_)?(mask\w+?_(?:hold|zero|extrapolate|cycle)_f(?P<frac>[0-9.]+)_s(?P<lo>\d+)-(?P<hi>\d+))\.json$")
GRID_PREFIX = {"general": "", "ambiguity": "ambiguity_"}


def load_report(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def discover_severities(directory: pathlib.Path, arm: str, grid: str = "general") -> dict[str, pathlib.Path]:
    found = {}
    for path in sorted(directory.glob(f"{arm}_eval_{GRID_PREFIX[grid]}mask*.json")):
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
        if ("ambiguity_pair_ids" in t) != ("ambiguity_pair_ids" in r) or (
            "ambiguity_pair_ids" in t and t["ambiguity_pair_ids"] != r["ambiguity_pair_ids"]
        ):
            raise ValueError("treatment and reference reports are not paired (ambiguity pair ids differ)")
        t_done.append(np.asarray(t["completed"], dtype=bool))
        r_done.append(np.asarray(r["completed"], dtype=bool))
        t_surv.append(np.asarray(t["survival_s"], dtype=float))
        r_surv.append(np.asarray(r["survival_s"], dtype=float))
        # Cluster key: the registered E70 unit — frame pair on the ambiguity grid, start step
        # on the general grid (rollouts sharing a start move together).
        key = t["ambiguity_pair_ids"] if "ambiguity_pair_ids" in t else t["start_steps"]
        starts.append(np.asarray(key) + 10 ** 7 * len(starts))  # keep seeds as separate clusters
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


def floor_relative_retention(
    degraded: float, clean: float, floor_degraded: float, floor_clean: float
) -> float | None:
    """Fraction of an arm's advantage over a goal-blind policy that survives the corruption."""
    advantage = clean - floor_clean
    if advantage <= 0:
        return None
    return (degraded - floor_degraded) / advantage


def analyze(treatment: list[dict], reference: list[dict],
            treatment_clean: list[dict], reference_clean: list[dict],
            floor: list[dict] | None = None, floor_clean: list[dict] | None = None) -> dict:
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
    if floor is not None and floor_clean is not None:
        fd = np.concatenate([np.asarray(f["completed"], dtype=bool) for f in floor]).mean()
        fc = np.concatenate([np.asarray(f["completed"], dtype=bool) for f in floor_clean]).mean()
        tc_mean = tc.mean()
        rc_mean = rc.mean()
        out["floor"] = {"floor_degraded": float(fd), "floor_clean": float(fc)}
        out["floor_relative_retention"] = {
            "treatment": floor_relative_retention(float(t_done.mean()), float(tc_mean), float(fd), float(fc)),
            "reference": floor_relative_retention(float(r_done.mean()), float(rc_mean), float(fd), float(fc)),
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
    ap.add_argument("--grid", choices=("general", "ambiguity"), default="general",
                    help="general start grid (primary) or the frozen 69-pair ambiguity grid (co-secondary)")
    ap.add_argument("--floor", action="append",
                    help="DIR:ARM of the goal-blind arm under the same dropout (enables floor-relative retention)")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    prefix = GRID_PREFIX[args.grid]
    treat = parse_pairs(args.treatment)
    ref = parse_pairs(args.reference)
    if len(treat) != len(ref):
        raise SystemExit("need one --reference per --treatment (seed-paired)")

    clean_name = f"_eval{'_ambiguity' if args.grid == 'ambiguity' else ''}.json"
    t_clean = [load_report(d / f"{a}{clean_name}") for d, a in treat]
    r_clean = [load_report(d / f"{a}{clean_name}") for d, a in ref]
    floor_pairs = parse_pairs(args.floor) if args.floor else None
    f_clean = [load_report(d / f"{a}{clean_name}") for d, a in floor_pairs] if floor_pairs else None
    result = {"grid": args.grid,
              "clean": analyze(t_clean, r_clean, t_clean, r_clean, f_clean, f_clean),
              "severities": {}}
    severities = discover_severities(*treat[0], grid=args.grid)
    for label, _ in severities.items():
        try:
            t = [load_report(d / f"{a}_eval_{prefix}{label}.json") for d, a in treat]
            r = [load_report(d / f"{a}_eval_{prefix}{label}.json") for d, a in ref]
        except FileNotFoundError as exc:
            result["severities"][label] = {"missing": str(exc)}
            continue
        f = None
        if floor_pairs:
            try:
                f = [load_report(d / f"{a}_eval_{prefix}{label}.json") for d, a in floor_pairs]
            except FileNotFoundError:
                f = None
        result["severities"][label] = analyze(t, r, t_clean, r_clean, f, f_clean)

    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
