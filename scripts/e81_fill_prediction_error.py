#!/usr/bin/env python
"""E81-A — how well can each outage fill predict the reference, and where does it fail?

Program context: `docs/COMMAND_INTERFACE_SYNTHESIS_2026-08-16.md` §III.2. E79 measured the
policy-level effect of four fills on the two E70 walks; this script measures the *upstream*
quantity — reference-prediction error — across many clips, so the regime boundary between
model-free and learned fills can be located before any policy is trained.

Fills compared (all strictly causal: they use only what the channel delivered before it failed):

* ``hold``   — replay the last valid sample;
* ``cv``     — dead-reckon ``q_ref + q̇_ref·Δt`` using the velocities the goal already carries;
* ``cycle``  — replay the most recent matching cycle, lag chosen by minimising the distance
  between the last ``match`` ticks and the same window ``L`` ticks earlier.

The question E81-A answers: cycle continuation was flat in horizon on cyclic walks (0.15–0.25 rad
from 0.1 s to 1.5 s). Is that a property of the method, or of walking? If it degrades on aperiodic
motion, then that regime is where a *learned* motion prior has something to contribute — and that,
not "an alternative command channel", is where the retargeting model earns a place in the deployed
stack.

Usage:
  e81_fill_prediction_error.py --motions DIR [--pattern '*.npz'] [--limit 40] [--out report.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re

import numpy as np

APERIODIC = re.compile(r"kick|punch|fight|dance|jump|throw|sit|crawl|box|martial|roll|climb", re.I)
PERIODIC = re.compile(r"walk|run|jog|treadmill|march", re.I)


def load_joint_series(path: str) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return (q, qdot, fps) for a WBT-schema or pool-schema motion NPZ."""
    try:
        d = np.load(path)
    except Exception:
        return None
    if "joint_pos" not in d or "joint_vel" not in d:
        return None
    q = np.asarray(d["joint_pos"], dtype=float)
    qd = np.asarray(d["joint_vel"], dtype=float)
    fps = int(np.asarray(d["fps"]).reshape(-1)[0])
    if q.shape[1] == 36:          # holosoma WBT schema: 7 root + 29 joints
        q, qd = q[:, 7:], qd[:, 6:]
    if q.shape[1] != qd.shape[1] or q.shape[0] < 300:
        return None
    return q, qd, fps


def fill_errors(
    q: np.ndarray, qd: np.ndarray, fps: int, horizons: tuple[int, ...],
    *, match_ticks: int = 20, min_lag: int | None = None, max_lag: int | None = None,
    samples: int = 200, seed: int = 0,
) -> dict[str, list[float]]:
    """Mean per-horizon RMSE (rad) of each causal fill against the true future reference."""
    dt = 1.0 / fps
    min_lag = min_lag or int(0.5 * fps)
    max_lag = max_lag or int(1.6 * fps)
    need = max_lag + match_ticks
    horizon_max = max(horizons)
    lo, hi = need, len(q) - horizon_max - 1
    if hi <= lo:
        return {}
    rng = np.random.default_rng(seed)
    t0s = rng.choice(np.arange(lo, hi), size=min(samples, hi - lo), replace=False)

    lags = np.arange(min_lag, max_lag + 1)
    # windows[i, l] = squared distance between the recent window at t0s[i] and the one lags[l] back
    recent = np.stack([q[t0s - k] for k in range(match_ticks)], axis=1)         # (S, M, D)
    best_lag = np.empty(len(t0s), dtype=int)
    err = np.empty((len(t0s), len(lags)))
    for j, lag in enumerate(lags):
        past = np.stack([q[t0s - k - lag] for k in range(match_ticks)], axis=1)
        err[:, j] = ((recent - past) ** 2).mean(axis=(1, 2))
    best_lag = lags[err.argmin(axis=1)]

    out: dict[str, list[float]] = {"hold": [], "cv": [], "cycle": []}
    for h in horizons:
        target = q[t0s + h]
        out["hold"].append(float(np.sqrt(((target - q[t0s]) ** 2).mean(1)).mean()))
        cv = q[t0s] + qd[t0s] * (h * dt)
        out["cv"].append(float(np.sqrt(((target - cv) ** 2).mean(1)).mean()))
        out["cycle"].append(float(np.sqrt(((target - q[t0s + h - best_lag]) ** 2).mean(1)).mean()))
    out["median_lag_s"] = [float(np.median(best_lag) / fps)]
    out["lag_iqr_s"] = [float((np.percentile(best_lag, 75) - np.percentile(best_lag, 25)) / fps)]
    return out


def classify(name: str) -> str:
    if APERIODIC.search(name):
        return "aperiodic"
    if PERIODIC.search(name):
        return "periodic"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motions", required=True, help="directory of motion NPZs (comma-separated ok)")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--limit", type=int, default=40, help="clips per class")
    ap.add_argument("--samples", type=int, default=200, help="random start frames per clip")
    ap.add_argument("--horizons-s", default="0.1,0.2,0.5,1.0,1.5")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    files: list[str] = []
    for directory in args.motions.split(","):
        files.extend(sorted(glob.glob(str(pathlib.Path(directory.strip()) / args.pattern))))
    buckets: dict[str, list[str]] = {"periodic": [], "aperiodic": [], "other": []}
    for f in files:
        buckets[classify(pathlib.Path(f).name)].append(f)

    results: dict[str, dict] = {}
    per_clip: list[dict] = []
    for label in ("periodic", "aperiodic"):
        rows: dict[str, list[list[float]]] = {"hold": [], "cv": [], "cycle": []}
        lags: list[float] = []
        used = 0
        for path in buckets[label]:
            loaded = load_joint_series(path)
            if loaded is None:
                continue
            q, qd, fps = loaded
            horizons = tuple(int(round(float(h) * fps)) for h in args.horizons_s.split(","))
            e = fill_errors(q, qd, fps, horizons, samples=args.samples)
            if not e:
                continue
            for k in rows:
                rows[k].append(e[k])
            lags.append(e["median_lag_s"][0])
            per_clip.append({"clip": pathlib.Path(path).stem, "class": label,
                             **{k: e[k] for k in ("hold", "cv", "cycle")}})
            used += 1
            if used >= args.limit:
                break
        if used:
            results[label] = {
                "clips": used,
                "horizons_s": [float(h) for h in args.horizons_s.split(",")],
                **{k: np.mean(np.asarray(v), axis=0).round(4).tolist() for k, v in rows.items()},
                "median_cycle_lag_s": float(np.median(lags)),
            }

    summary = {"motions": args.motions, "results": results, "per_clip": per_clip}
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    for label, r in results.items():
        print(f"\n{label}: {r['clips']} clips, median cycle lag {r['median_cycle_lag_s']:.2f} s")
        print(f"{'horizon':>9s}" + "".join(f"{k:>10s}" for k in ("hold", "cv", "cycle")))
        for i, h in enumerate(r["horizons_s"]):
            print(f"{h:8.2f}s" + "".join(f"{r[k][i]:10.3f}" for k in ("hold", "cv", "cycle")))


if __name__ == "__main__":
    main()
