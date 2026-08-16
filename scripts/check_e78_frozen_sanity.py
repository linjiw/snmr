#!/usr/bin/env python
"""E78 frozen-sanity gate: the derived trainer must reproduce the seed-exact E70 report values.

Before any E78 cell is interpreted, the frozen E70 students are re-evaluated through the
derived trainer (``flag_dim=0``, no masking).  Their clean general-grid and ambiguity-start
completion must match the hash-bound seed-exact E70 reports — not the three-seed aggregates —
within the E76 evaluation-noise tolerance (single-run sd 0.0083; default tolerance 0.02 ≈ 2.5 sd),
and the per-rollout start grids must be identical (pairing invariant).

Exit 0 on pass, 3 on failure.  Writes ``<replay-dir>/SANITY.json`` either way.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def compare(frozen: dict, replay: dict, tolerance: float) -> dict:
    out = {
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
    ap.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()

    result = {"tolerance": args.tolerance, "cells": {}}
    ok = True
    for label, suffix in (("general", ""), ("ambiguity", "_ambiguity")):
        f = args.frozen_dir / f"{args.arm}_eval{suffix}.json"
        r = args.replay_dir / f"{args.arm}_eval{suffix}.json"
        if not f.exists() or not r.exists():
            result["cells"][label] = {"missing": [str(p) for p in (f, r) if not p.exists()], "pass": False}
            ok = False
            continue
        cell = compare(json.loads(f.read_text()), json.loads(r.read_text()), args.tolerance)
        result["cells"][label] = cell
        ok &= cell["pass"]
    result["pass"] = ok
    (args.replay_dir / "SANITY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
