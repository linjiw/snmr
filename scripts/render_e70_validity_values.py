#!/usr/bin/env python
"""Render the manuscript's validity/self-audit numbers as hash-stamped LaTeX macros.

These are the numbers the paper uses to declare its own limits rather than its results:

* the median future distance over the 69 SELECTED ambiguity windows -- the manuscript
  previously printed 1.113, which is the median over the 675 threshold-ELIGIBLE windows;
* the seed-level sensitivity interval on A-T, computed with the training run as the sampling
  unit, which is what the paper's own estimand implies;
* the E76 evaluation-replication floor and the primary interval widened by it.

Every value is computed here from a hash-stamped input so none is hand-transcribed.

Sources: docs/E76_EVALUATION_REPLICATION.md (method), the frozen ambiguity precheck, and the
frozen three-seed analyzer.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import pathlib
import statistics


PRECHECK_PROTOCOL = "E70 reference-only ambiguity precheck v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_median_future(precheck: dict) -> tuple[float, int, float, int]:
    """Median future distance over the SELECTED windows, and over the eligible pool."""
    pair = precheck.get("preferred_pair")
    report = precheck["pairs"][pair]
    windows = report["windows"]
    selected = [float(w["future_distance"]) for w in windows]
    if not selected:
        raise ValueError("the precheck records no selected windows")
    pool_median = float(report["eligible_future_distance"]["median"])
    pool_n = int(report["num_threshold_eligible"])
    return statistics.median(selected), len(selected), pool_median, pool_n


def seed_level_interval(analysis: dict, key: str) -> tuple[float, float, float, list[float]]:
    """Two-sided 95% t-interval treating the training run as the sampling unit."""
    per_seed = analysis["per_seed_differences"]
    values = [float(per_seed[str(s)][key]) for s in sorted(int(k) for k in per_seed)]
    n = len(values)
    if n < 3:
        raise ValueError(f"a seed-level interval needs at least 3 training seeds, got {n}")
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(n)
    # t_{0.975} for n-1 = 2 degrees of freedom.
    t_crit = 4.302652729911275
    return mean, mean - t_crit * sem, mean + t_crit * sem, values


def replication_floor(replication_root: pathlib.Path) -> tuple[float, float]:
    """Per-arm completion sd and the fraction of rollout outcomes that differ between repeats."""
    sds = []
    flips = []
    for arm in sorted(p.name for p in replication_root.iterdir() if p.is_dir()):
        reports = sorted(glob.glob(str(replication_root / arm / "seed0" / "repeat*" / "*_eval_ambiguity.json")))
        if len(reports) < 2:
            continue
        loaded = [json.loads(pathlib.Path(p).read_text()) for p in reports]
        sds.append(statistics.stdev([float(r["completion_rate"]) for r in loaded]))
        base = loaded[0]["completed"]
        for other in loaded[1:]:
            flips.append(sum(1 for a, b in zip(base, other["completed"]) if a != b) / len(base))
    if not sds:
        raise ValueError(f"no replicated arms found under {replication_root}")
    return statistics.fmean(sds), statistics.fmean(flips)


def latex_macros(
    *, precheck_path: pathlib.Path, analysis_path: pathlib.Path, replication_root: pathlib.Path
) -> str:
    precheck = json.loads(precheck_path.read_text())
    if precheck.get("protocol") != PRECHECK_PROTOCOL:
        raise ValueError(f"unexpected precheck protocol: {precheck.get('protocol')!r}")
    analysis = json.loads(analysis_path.read_text())
    if sorted(analysis.get("seeds", [])) != [0, 1, 2]:
        raise ValueError("the validity macros require the frozen three-seed analyzer")

    sel_median, sel_n, pool_median, pool_n = selected_median_future(precheck)
    if sel_n != 69:
        raise ValueError(f"expected the 69 frozen windows, found {sel_n}")

    at_mean, at_lo, at_hi = seed_level_interval(analysis, "snmr_minus_time")[:3]
    rep_sd, rep_flip = replication_floor(replication_root)

    # Propagate the replication component into the registered primary interval.
    at = analysis["snmr_minus_time"]
    point = float(at["difference"])
    boot_sd = (float(at["ci95_high"]) - float(at["ci95_low"])) / (2 * 1.959963984540054)
    eval_sd = rep_sd * math.sqrt(2) / math.sqrt(len(analysis["seeds"]))
    combined = math.hypot(boot_sd, eval_sd)
    wide_lo = point - 1.959963984540054 * combined
    wide_hi = point + 1.959963984540054 * combined
    widen_pct = 100.0 * (combined / boot_sd - 1.0)
    if wide_lo <= 0.0:
        raise ValueError("the replication-widened interval no longer excludes zero")

    digests = [(str(p), sha256_file(p)) for p in (precheck_path, analysis_path)]
    lines = [
        "% Generated by scripts/render_e70_validity_values.py; do not edit by hand.",
        f"% inputs_sha256={hashlib.sha256(''.join(d for _, d in digests).encode()).hexdigest()}",
    ]
    lines += [f"% input_sha256[{name}]={digest}" for name, digest in digests]
    lines += [
        f"% trace selected_median_future={sel_median:.17g} over n={sel_n}",
        f"% trace eligible_pool_median={pool_median:.17g} over n={pool_n}",
        f"% trace seed_level_at_mean={at_mean:.17g}",
        f"% trace replication_sd={rep_sd:.17g} flip_fraction={rep_flip:.17g}",
        f"% trace bootstrap_sd={boot_sd:.17g} eval_sd={eval_sd:.17g} combined={combined:.17g}",
        rf"\newcommand{{\ESelectedMedianFuture}}{{{sel_median:.3f}}}",
        rf"\newcommand{{\ESelectedWindows}}{{{sel_n}}}",
        rf"\newcommand{{\EEligibleMedianFuture}}{{{pool_median:.3f}}}",
        rf"\newcommand{{\EEligibleWindows}}{{{pool_n}}}",
        rf"\newcommand{{\ESeedLevelATLow}}{{{at_lo:.3f}}}",
        rf"\newcommand{{\ESeedLevelATHigh}}{{{at_hi:.3f}}}",
        rf"\newcommand{{\EReplicationSd}}{{{rep_sd:.3f}}}",
        rf"\newcommand{{\EReplicationFlipPercent}}{{{round(100 * rep_flip)}}}",
        rf"\newcommand{{\EReplicationWidenPercent}}{{{widen_pct:.1f}}}",
        rf"\newcommand{{\EReplicationATLow}}{{{wide_lo:.3f}}}",
        rf"\newcommand{{\EReplicationATHigh}}{{{wide_hi:.3f}}}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck", type=pathlib.Path, required=True)
    parser.add_argument("--analysis", type=pathlib.Path, required=True)
    parser.add_argument("--replication-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rendered = latex_macros(
        precheck_path=args.precheck,
        analysis_path=args.analysis,
        replication_root=args.replication_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.out)
    print(f"rendered E70 validity values -> {args.out}")


if __name__ == "__main__":
    main()
