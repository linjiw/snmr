#!/usr/bin/env python
"""Create a hash-bound E70 visual-review record after watching the final media."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_e70_final_bundle import REQUIRED_VISUAL_CHECKS, VISUAL_REVIEW_PROTOCOL
from scripts.compose_e70_video import sha256_file


def review_payload(
    *,
    video: pathlib.Path,
    contact_sheet: pathlib.Path,
    reviewer: str,
    full_video_watched: bool,
    checks: dict[str, bool],
    reviewed_utc: str | None = None,
) -> dict[str, Any]:
    """Validate a completed review and bind it to the exact inspected files."""
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("visual review requires a reviewer identity")
    if not full_video_watched:
        raise ValueError("visual review requires watching the full final MP4")
    if set(checks) != set(REQUIRED_VISUAL_CHECKS):
        raise ValueError("visual review checklist does not match the frozen requirements")
    failed = [name for name in REQUIRED_VISUAL_CHECKS if checks[name] is not True]
    if failed:
        raise ValueError("visual review did not pass: " + ", ".join(failed))
    if not video.is_file() or not contact_sheet.is_file():
        raise FileNotFoundError("final MP4 and contact sheet must both exist")
    timestamp = reviewed_utc or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        dt.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ValueError("visual review timestamp must be UTC")
    return {
        "protocol": VISUAL_REVIEW_PROTOCOL,
        "video": str(video.resolve()),
        "video_sha256": sha256_file(video),
        "contact_sheet": str(contact_sheet.resolve()),
        "contact_sheet_sha256": sha256_file(contact_sheet),
        "reviewed_utc": timestamp,
        "reviewer": reviewer,
        "full_video_watched": True,
        "checks": {name: True for name in REQUIRED_VISUAL_CHECKS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=pathlib.Path, required=True)
    parser.add_argument("--contact-sheet", type=pathlib.Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--confirm-full-video-watched", action="store_true")
    for name in REQUIRED_VISUAL_CHECKS:
        parser.add_argument(f"--pass-{name.replace('_', '-')}", dest=name, action="store_true")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite visual review: {args.out}")
    checks = {name: bool(getattr(args, name)) for name in REQUIRED_VISUAL_CHECKS}
    payload = review_payload(
        video=args.video,
        contact_sheet=args.contact_sheet,
        reviewer=args.reviewer,
        full_video_watched=args.confirm_full_video_watched,
        checks=checks,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"recorded hash-bound E70 visual review -> {args.out}")


if __name__ == "__main__":
    main()
