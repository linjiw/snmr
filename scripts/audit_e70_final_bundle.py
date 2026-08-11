#!/usr/bin/env python
"""Certify the final E70 paper/video bundle after hash-bound visual review."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_e70_video_code_hashes import validate_hash_manifest
from scripts.compose_e70_video import result_summary, sha256_file
from scripts.render_e70_paper_values import latex_macros


PROTOCOL = "E70 final paper-video bundle v1"
VISUAL_REVIEW_PROTOCOL = "E70 full-video visual review v1"
REQUIRED_VISUAL_CHECKS = (
    "framing_and_camera_tracking",
    "labels_and_outcome_status",
    "no_reset_leakage",
    "no_clipping_or_unreadable_text",
    "no_misleading_synchronization",
)


def validate_machine_evidence(
    *,
    analysis: dict[str, Any],
    analysis_sha256: str,
    paper_values: str,
    video_validation: dict[str, Any],
    video_sha256: str,
    capture_index: dict[str, Any],
    capture_index_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Cross-check analyzer, paper macros, capture index, and composed video."""
    summary = result_summary(analysis)
    expected_macros = latex_macros(analysis, analysis_sha256=analysis_sha256)
    if paper_values != expected_macros:
        raise ValueError("paper values are not an exact rendering of the final analyzer")

    if video_validation.get("protocol") != "E70 ICRA video validation v1":
        raise ValueError("unexpected final-video validation protocol")
    if video_validation.get("passes") is not True:
        raise ValueError("final video does not pass its mechanical submission checks")
    expected_validation = {
        "analysis_sha256": analysis_sha256,
        "capture_index_sha256": capture_index_sha256,
        "manifest_sha256": manifest_sha256,
        "result_summary": summary,
    }
    for key, expected in expected_validation.items():
        if video_validation.get(key) != expected:
            raise ValueError(f"video validation {key} is not bound to the final evidence")
    if video_validation.get("observed", {}).get("sha256") != video_sha256:
        raise ValueError("video validation hash does not match the final MP4")

    if capture_index.get("protocol") != "E70 raw simulation capture index v1":
        raise ValueError("unexpected raw-capture index protocol")
    if capture_index.get("selection_uses_policy_outcomes") is not False:
        raise ValueError("raw capture selection must remain outcome-independent")
    if capture_index.get("manifest_sha256") != manifest_sha256:
        raise ValueError("raw capture index is not bound to the frozen manifest")
    captures = capture_index.get("captures")
    if not isinstance(captures, list) or len(captures) != 6:
        raise ValueError("final E70 video requires exactly six frozen raw captures")
    indexed_hashes = {item.get("name"): item.get("raw_video_sha256") for item in captures}
    validated_raw = video_validation.get("raw_captures", {})
    if set(indexed_hashes) != set(validated_raw):
        raise ValueError("video validation and capture index name different raw captures")
    for name, digest in indexed_hashes.items():
        if validated_raw[name].get("sha256") != digest:
            raise ValueError(f"composed video is not bound to indexed raw capture {name}")
    return summary


def validate_visual_review(
    review: dict[str, Any], *, video_sha256: str, contact_sheet_sha256: str
) -> None:
    """Require an explicit visual review bound to the exact final media."""
    if review.get("protocol") != VISUAL_REVIEW_PROTOCOL:
        raise ValueError("unexpected full-video visual-review protocol")
    if review.get("video_sha256") != video_sha256:
        raise ValueError("visual review is not bound to the final MP4")
    if review.get("contact_sheet_sha256") != contact_sheet_sha256:
        raise ValueError("visual review is not bound to the final contact sheet")
    if review.get("full_video_watched") is not True:
        raise ValueError("visual review does not confirm watching the full final MP4")
    timestamp = str(review.get("reviewed_utc", ""))
    try:
        dt.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ValueError("visual review lacks a UTC completion timestamp")
    if not str(review.get("reviewer", "")).strip():
        raise ValueError("visual review does not identify its reviewer")
    checks = review.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("visual review has no checklist")
    failed = [name for name in REQUIRED_VISUAL_CHECKS if checks.get(name) is not True]
    if failed:
        raise ValueError("visual review did not pass: " + ", ".join(failed))


def validate_indexed_capture_files(capture_index: dict[str, Any]) -> list[str]:
    """Replay raw-video, report, and checkpoint hashes from the final capture index."""
    captures = capture_index.get("captures")
    if not isinstance(captures, list) or len(captures) != 6:
        raise ValueError("final E70 video requires exactly six indexed captures")
    validated: list[str] = []
    for item in captures:
        name = str(item.get("name", ""))
        if not name:
            raise ValueError("indexed capture has no name")
        for path_key, hash_key in (
            ("raw_video", "raw_video_sha256"),
            ("report", "report_sha256"),
            ("checkpoint", "checkpoint_sha256"),
        ):
            path = pathlib.Path(str(item.get(path_key, "")))
            expected = item.get(hash_key)
            if not path.is_file() or not isinstance(expected, str):
                raise ValueError(f"indexed capture {name} lacks {path_key} provenance")
            if sha256_file(path) != expected:
                raise ValueError(f"indexed capture {name} {path_key} hash changed")
            validated.append(f"{name}:{path_key}")
    return validated


def validate_paper_pdf(path: pathlib.Path) -> dict[str, Any]:
    """Apply the final page, media, and font-embedding requirements."""
    info = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, check=True
    ).stdout
    fields = {
        key.strip(): value.strip()
        for line in info.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    pages = int(fields.get("Pages", "0"))
    if not 1 <= pages <= 8:
        raise ValueError(f"paper has invalid page count: {pages}")
    if fields.get("Page size") != "612 x 792 pts (letter)":
        raise ValueError("paper is not US letter")

    fonts = subprocess.run(
        ["pdffonts", str(path)], capture_output=True, text=True, check=True
    ).stdout.splitlines()[2:]
    if not fonts:
        raise ValueError("paper contains no inspectable fonts")
    if any(len(row.split()) < 7 or row.split()[-5:-3] != ["yes", "yes"] for row in fonts):
        raise ValueError("paper contains a non-embedded or non-subset font")
    return {"pages": pages, "page_size": fields["Page size"], "fonts": len(fonts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--analysis", type=pathlib.Path, required=True)
    parser.add_argument("--paper-values", type=pathlib.Path, required=True)
    parser.add_argument("--paper", type=pathlib.Path, required=True)
    parser.add_argument("--video", type=pathlib.Path, required=True)
    parser.add_argument("--video-validation", type=pathlib.Path, required=True)
    parser.add_argument("--capture-index", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--contact-sheet", type=pathlib.Path, required=True)
    parser.add_argument("--visual-review", type=pathlib.Path, required=True)
    parser.add_argument("--code-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    code_manifest = json.loads(args.code_manifest.read_text())
    frozen_files = validate_hash_manifest(code_manifest, args.repo_root)
    analysis = json.loads(args.analysis.read_text())
    capture_index = json.loads(args.capture_index.read_text())
    validation = json.loads(args.video_validation.read_text())
    video_sha256 = sha256_file(args.video)
    contact_sheet_sha256 = sha256_file(args.contact_sheet)
    summary = validate_machine_evidence(
        analysis=analysis,
        analysis_sha256=sha256_file(args.analysis),
        paper_values=args.paper_values.read_text(),
        video_validation=validation,
        video_sha256=video_sha256,
        capture_index=capture_index,
        capture_index_sha256=sha256_file(args.capture_index),
        manifest_sha256=sha256_file(args.manifest),
    )
    indexed_capture_files = validate_indexed_capture_files(capture_index)
    review = json.loads(args.visual_review.read_text())
    validate_visual_review(
        review, video_sha256=video_sha256, contact_sheet_sha256=contact_sheet_sha256
    )
    paper_checks = validate_paper_pdf(args.paper)
    if not summary["explicit_general_gate"]:
        outcome = "invalid_assay"
    elif summary["positive_content_gate"]:
        outcome = "positive_content"
    else:
        outcome = "scoped_null"

    artifacts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in (
            ("analysis", args.analysis),
            ("paper_values", args.paper_values),
            ("paper", args.paper),
            ("video", args.video),
            ("video_validation", args.video_validation),
            ("capture_index", args.capture_index),
            ("contact_sheet", args.contact_sheet),
            ("visual_review", args.visual_review),
            ("code_manifest", args.code_manifest),
        )
    }
    bundle = {
        "protocol": PROTOCOL,
        "completed_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outcome": outcome,
        "result_summary": summary,
        "paper_checks": paper_checks,
        "indexed_capture_files": indexed_capture_files,
        "frozen_code_files": frozen_files,
        "artifacts": artifacts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle, indent=2) + "\n")
    temporary.replace(args.out)
    print(f"certified final E70 bundle ({outcome}) -> {args.out}")


if __name__ == "__main__":
    main()
