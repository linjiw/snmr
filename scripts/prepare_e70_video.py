#!/usr/bin/env python
"""Freeze policy-independent E70 rollout starts for the paper video.

The exemplar is selected from the reference-only ambiguity precheck.  Student
completion, survival, and tracking errors are deliberately not inspected while
choosing it.  An already frozen ambiguity report is used only to translate the
pair/side identity into the exact concatenated-motion start step used by the
simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics


CAPTURE_SPECS = (
    ("snmr", "a_prior_snmr", False, False, "none"),
    ("time", "a_prior_snmr", True, False, "none"),
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_reference_median_window(precheck: dict) -> tuple[str, int, dict]:
    """Select the window nearest reference-only state/future medians."""
    if precheck.get("protocol") != "E70 reference-only ambiguity precheck v1":
        raise ValueError("input is not the frozen E70 reference-only precheck")
    pair_name = precheck.get("preferred_pair")
    pair = precheck.get("pairs", {}).get(pair_name)
    if not pair or pair.get("passes_floor") is not True:
        raise ValueError("preferred ambiguity pair is missing or did not pass")
    windows = pair.get("windows", [])
    if not windows:
        raise ValueError("preferred ambiguity pair has no windows")

    state_median = statistics.median(float(item["state_distance"]) for item in windows)
    future_median = statistics.median(float(item["future_distance"]) for item in windows)
    pair_id, window = min(
        enumerate(windows),
        key=lambda item: (
            abs(float(item[1]["future_distance"]) - future_median),
            abs(float(item[1]["state_distance"]) - state_median),
            int(item[1]["frame_first"]),
            int(item[1]["frame_second"]),
        ),
    )
    selection = {
        "rule": (
            "nearest to median reference future distance, then median reference "
            "state distance, then lexical frame index"
        ),
        "state_distance_median": state_median,
        "future_distance_median": future_median,
    }
    return str(pair_name), pair_id, {**window, "selection": selection}


def resolve_start_step(report: dict, *, pair_id: int, side: int) -> int:
    """Resolve one unique simulator start without reading policy outcomes."""
    required = ("start_steps", "motion_ids", "ambiguity_pair_ids", "ambiguity_sides")
    lengths = {len(report.get(key, [])) for key in required}
    if len(lengths) != 1 or lengths == {0}:
        raise ValueError("ambiguity report rollout identity arrays are missing or misaligned")
    matches = {
        int(start)
        for start, motion_id, observed_pair, observed_side in zip(
            report["start_steps"],
            report["motion_ids"],
            report["ambiguity_pair_ids"],
            report["ambiguity_sides"],
        )
        if int(observed_pair) == pair_id
        and int(observed_side) == side
        and int(motion_id) == side
    }
    if len(matches) != 1:
        raise ValueError(
            f"pair {pair_id} side {side} maps to {len(matches)} unique start steps"
        )
    return matches.pop()


def build_video_manifest(
    precheck: dict,
    ambiguity_report: dict,
    *,
    seed: int,
    student_root: pathlib.Path,
    precheck_path: pathlib.Path | None = None,
    report_path: pathlib.Path | None = None,
) -> dict:
    pair_name, pair_id, selected = select_reference_median_window(precheck)
    pair = precheck["pairs"][pair_name]
    clips = list(pair["clips"])
    if len(clips) != 2:
        raise ValueError("E70 video protocol requires exactly two clips")

    sides = []
    for side, clip in enumerate(clips):
        sides.append(
            {
                "side": side,
                "clip": clip,
                "start_step": resolve_start_step(
                    ambiguity_report, pair_id=pair_id, side=side
                ),
                "time_seconds": float(selected[f"time_seconds_{'first' if side == 0 else 'second'}"]),
            }
        )

    captures = []
    for tag, arm, phase_only, shuffle_latent, destroy_zcmd in CAPTURE_SPECS:
        checkpoint = student_root / f"seed{seed}_{tag}" / f"{arm}_student.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for side in sides:
            captures.append(
                {
                    "name": f"{tag}_{side['clip']}",
                    "tag": tag,
                    "arm": arm,
                    "phase_only": phase_only,
                    "shuffle_latent": shuffle_latent,
                    "destroy_zcmd": destroy_zcmd,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    **side,
                }
            )

    explicit_checkpoint = (
        student_root / f"seed{seed}_explicit" / "c_prior_explicit_student.pt"
    )
    if not explicit_checkpoint.is_file():
        raise FileNotFoundError(explicit_checkpoint)
    for destroy_zcmd in ("none", "marginal_random"):
        captures.append(
            {
                "name": f"explicit_{'clean' if destroy_zcmd == 'none' else 'destroy_marginal'}_{clips[0]}",
                "tag": "explicit",
                "arm": "c_prior_explicit",
                "phase_only": False,
                "shuffle_latent": False,
                "destroy_zcmd": destroy_zcmd,
                "checkpoint": str(explicit_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(explicit_checkpoint),
                **sides[0],
            }
        )

    result = {
        "protocol": "E70 paper-video rollout selection v1",
        "selection_uses_policy_outcomes": False,
        "training_seed": seed,
        "pair": pair_name,
        "pair_id": pair_id,
        "reference_window": selected,
        "sides": sides,
        "captures": captures,
    }
    if precheck_path is not None:
        result["precheck"] = {
            "path": str(precheck_path.resolve()),
            "sha256": sha256_file(precheck_path),
        }
    if report_path is not None:
        result["start_step_source"] = {
            "path": str(report_path.resolve()),
            "sha256": sha256_file(report_path),
            "fields_used": [
                "start_steps",
                "motion_ids",
                "ambiguity_pair_ids",
                "ambiguity_sides",
            ],
            "outcome_fields_used": [],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck", type=pathlib.Path, required=True)
    parser.add_argument("--ambiguity-report", type=pathlib.Path, required=True)
    parser.add_argument("--student-root", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    manifest = build_video_manifest(
        json.loads(args.precheck.read_text()),
        json.loads(args.ambiguity_report.read_text()),
        seed=args.seed,
        student_root=args.student_root,
        precheck_path=args.precheck,
        report_path=args.ambiguity_report,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.out)
    print(
        f"pair={manifest['pair']} pair_id={manifest['pair_id']} "
        f"captures={len(manifest['captures'])} out={args.out}"
    )


if __name__ == "__main__":
    main()
