#!/usr/bin/env python
"""Compare two physics-verification reports under the same candidate contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snmr.provenance import ArtifactSnapshot, source_revision_manifest  # noqa: E402
from snmr.verification import VerificationReport, compare_solver_reports  # noqa: E402


def _load_report(snapshot: ArtifactSnapshot, report_key: str | None) -> VerificationReport:
    payload = json.loads(snapshot.data)
    if report_key is not None:
        payload = payload["reports"][report_key]
    return VerificationReport.from_dict(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-key", help="optional key under a top-level reports object")
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-key", help="optional key under a top-level reports object")
    parser.add_argument("--newton-root", type=Path)
    parser.add_argument("--isaac-lab-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.out}")

    left_snapshot = ArtifactSnapshot.capture(args.left)
    right_snapshot = ArtifactSnapshot.capture(args.right)
    left = _load_report(left_snapshot, args.left_key)
    right = _load_report(right_snapshot, args.right_key)
    comparison = compare_solver_reports(left, right)
    payload = {
        "schema_version": "snmr.cross-solver-comparison.v0.1",
        "command": [sys.executable, *sys.argv],
        "environment": {"python": platform.python_version()},
        "comparison": asdict(comparison),
        "input_artifacts": {
            "left": left_snapshot.manifest(),
            "right": right_snapshot.manifest(),
        },
        "left": {
            "backend": asdict(left.backend),
            "report_hash": left.report_hash,
            "verification_level": left.verification_level,
        },
        "right": {
            "backend": asdict(right.backend),
            "report_hash": right.report_hash,
            "verification_level": right.verification_level,
        },
        **source_revision_manifest(
            snmr_path=ROOT,
            newton_path=args.newton_root,
            isaac_lab_path=args.isaac_lab_root,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not comparison.same_candidate_contract:
        raise SystemExit("reports do not describe the same candidate contract")


if __name__ == "__main__":
    main()
