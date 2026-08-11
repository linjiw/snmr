#!/usr/bin/env python
"""Convert the frozen E69 screen into an E70 loader-order ambiguity precheck.

E69 measured every window as ``anchor -> candidate``.  The motion loader sorts the
two E70 files alphabetically, which places the selected ``walk1_subject1`` before the
anchor ``walk1_subject5``.  This script swaps every side-specific field and records
the source hash so evaluation cannot silently attach starts to the wrong trajectory.
It reads reference-screen data only; no policy or student result enters the output.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.precheck_e67_ambiguity import sha256_file


SIDE_FIELDS = (
    "frame",
    "time_seconds",
    "normalized_time",
)


def build_e70_precheck(screen: dict, *, source_path: pathlib.Path | None = None) -> dict:
    if screen.get("protocol") != "E69 exhaustive reference-only pair screen v1":
        raise ValueError("input is not the frozen E69 screen")
    if screen.get("gate_passed") is not True or not screen.get("selected_clip"):
        raise ValueError("E69 did not select a passing candidate")

    anchor = str(screen["anchor"])
    selected = str(screen["selected_clip"])
    matches = [item for item in screen["candidates"] if item.get("clip") == selected]
    if len(matches) != 1:
        raise ValueError("selected E69 candidate is missing or duplicated")
    candidate = matches[0]
    if candidate.get("passes_ambiguity") is not True:
        raise ValueError("selected candidate did not pass the ambiguity floor")

    pair = copy.deepcopy(candidate["ambiguity"])
    for window in pair["windows"]:
        for field in SIDE_FIELDS:
            first = f"{field}_first"
            second = f"{field}_second"
            window[first], window[second] = window[second], window[first]

    # This is the exact order produced by sorted *.npz loading in E70.
    clips = sorted((selected, anchor))
    if clips != [selected, anchor]:
        raise ValueError(
            "selected/anchor lexical order changed; register a new explicit ordering"
        )
    pair["clips"] = clips
    pair["passes_floor"] = bool(
        pair["num_selected_windows"] >= screen["thresholds"]["min_windows"]
    )
    pair["inputs"] = [
        {
            "clip": selected,
            "path": candidate["path"],
            "sha256": candidate["sha256"],
        },
        {
            "clip": anchor,
            "path": screen["anchor_input"]["path"],
            "sha256": screen["anchor_input"]["sha256"],
        },
    ]
    preferred_pair = ",".join(clips)
    result = {
        "protocol": "E70 reference-only ambiguity precheck v1",
        "preferred_pair": preferred_pair,
        "thresholds": copy.deepcopy(screen["thresholds"]),
        "state_proxy": "previous_dof_target + root_angular_velocity + dof_pos + dof_vel",
        "future_goal": "dof_pos + dof_vel, globally standardized across the pair",
        "source_order": [anchor, selected],
        "loaded_motion_order": clips,
        "side_transform": "swap first/second for loader lexical order",
        "pairs": {preferred_pair: pair},
    }
    if source_path is not None:
        result["source_screen"] = {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    screen_path = pathlib.Path(args.screen)
    result = build_e70_precheck(
        json.loads(screen_path.read_text()), source_path=screen_path
    )
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    pair = result["pairs"][result["preferred_pair"]]
    print(
        f"preferred_pair={result['preferred_pair']}; "
        f"windows={pair['num_selected_windows']}; report={output}"
    )


if __name__ == "__main__":
    main()
