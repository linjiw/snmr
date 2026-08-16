#!/usr/bin/env python
"""E78 frozen-sanity gate: the derived trainer must reproduce the seed-exact E70 report values.

Before any E78 cell is interpreted, the frozen E70 students are re-evaluated through the
derived trainer (``flag_dim=0``, no masking).  Their clean general-grid and ambiguity-start
completion must match the hash-bound seed-exact E70 reports — not the three-seed aggregates —
within the measured evaluation-noise tolerance, and the per-rollout start grids must be identical
(pairing invariant).

**Tolerance is completion-dependent, and the two tiers are measured, not assumed.** E76's
single-run sd 0.0083 was measured on the high-completion explicit arm; closed-loop rollouts flip
near the termination threshold, so an arm sitting at mid completion is noisier. Three fresh
replays of one frozen null arm (seed-0 time code, clean completion ≈ 0.60) spanned 0.026 on the
general grid and 0.018 on the ambiguity grid — range over n = 3 implies sd ≈ 0.016, so 3 sd ≈ 0.05.
Hence: **0.02 for arms with frozen completion ≥ 0.85, 0.05 below that** (``--tolerance`` and
``--tolerance-mid``). Recorded 2026-08-16 after two null-arm cells failed the flat 0.02 gate while
their own general grids reproduced to ≤ 0.007.

Exit 0 on pass, 3 on failure.  Writes ``<replay-dir>/SANITY.json`` either way.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def tolerance_for(frozen_completion: float, tolerance: float, tolerance_mid: float) -> float:
    """High-completion arms get the tight tolerance; mid-completion arms the measured wider one."""
    return tolerance if frozen_completion >= 0.85 else tolerance_mid


def compare(frozen: dict, replay: dict, tolerance: float, tolerance_mid: float | None = None) -> dict:
    if tolerance_mid is None:
        tolerance_mid = tolerance
    tolerance = tolerance_for(frozen["completion_rate"], tolerance, tolerance_mid)
    out = {
        "applied_tolerance": tolerance,
        "frozen_completion": frozen["completion_rate"],
        "replay_completion": replay["completion_rate"],
        "abs_diff": abs(frozen["completion_rate"] - replay["completion_rate"]),
        "starts_identical": frozen["start_steps"] == replay["start_steps"]
        and frozen["motion_ids"] == replay["motion_ids"],
        "num_rollouts": replay["num_rollouts"],
    }
    out["pass"] = out["starts_identical"] and out["abs_diff"] <= tolerance
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-dir", type=pathlib.Path, required=True)
    ap.add_argument("--replay-dir", type=pathlib.Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tolerance", type=float, default=0.02, help="arms with frozen completion >= 0.85")
    ap.add_argument("--tolerance-mid", type=float, default=0.05, help="arms below 0.85 (measured; see docstring)")
    args = ap.parse_args()

    result = {"tolerance": args.tolerance, "tolerance_mid": args.tolerance_mid, "cells": {}}
    ok = True
    for label, suffix in (("general", ""), ("ambiguity", "_ambiguity")):
        f = args.frozen_dir / f"{args.arm}_eval{suffix}.json"
        r = args.replay_dir / f"{args.arm}_eval{suffix}.json"
        if not f.exists() or not r.exists():
            result["cells"][label] = {"missing": [str(p) for p in (f, r) if not p.exists()], "pass": False}
            ok = False
            continue
        cell = compare(json.loads(f.read_text()), json.loads(r.read_text()), args.tolerance, args.tolerance_mid)
        result["cells"][label] = cell
        ok &= cell["pass"]
    result["pass"] = ok
    (args.replay_dir / "SANITY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
