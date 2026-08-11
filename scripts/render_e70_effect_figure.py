#!/usr/bin/env python
"""Render a hash-traceable forest plot for the frozen E70 paired effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_e67_results import (
    clustered_paired_interval,
    hierarchical_paired_interval,
)


PROTOCOL = "E70 preregistered analysis v1"
REGISTERED_SEEDS = [0, 1, 2]
ARMS = {
    "snmr": "a_prior_snmr",
    "time": "a_prior_snmr",
    "shuffled": "a_prior_snmr",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_interval(value: Any, *, label: str) -> dict[str, float]:
    try:
        interval = {
            "difference": float(value["difference"]),
            "ci95_low": float(value["ci95_low"]),
            "ci95_high": float(value["ci95_high"]),
        }
        clusters = int(value["clusters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"analysis lacks a valid {label} interval") from exc
    if (
        clusters != 69
        or not all(math.isfinite(item) and -1.0 <= item <= 1.0 for item in interval.values())
        or not interval["ci95_low"] <= interval["difference"] <= interval["ci95_high"]
    ):
        raise ValueError(f"analysis has an invalid {label} interval")
    return interval


def _resolve_hashed_inputs(
    analysis: dict[str, Any], *, analysis_path: pathlib.Path
) -> dict[pathlib.Path, str]:
    records = analysis.get("inputs")
    if not isinstance(records, list) or not records:
        raise ValueError("analysis has no hash-bound inputs")
    resolved: dict[pathlib.Path, str] = {}
    for record in records:
        try:
            raw_path = pathlib.Path(record["path"])
            wanted = str(record["sha256"])
        except (KeyError, TypeError) as exc:
            raise ValueError("analysis contains a malformed input record") from exc
        path = raw_path if raw_path.is_absolute() else analysis_path.parent / raw_path
        path = path.resolve()
        if not path.is_file() or sha256_file(path) != wanted:
            raise ValueError(f"analysis input hash mismatch: {path}")
        resolved[path] = wanted
    return resolved


def _load_ambiguity_reports(
    inputs: dict[pathlib.Path, str], seeds: list[int]
) -> dict[int, dict[str, dict[str, Any]]]:
    loaded: dict[int, dict[str, dict[str, Any]]] = {}
    cross_seed_grid: tuple[Any, Any, Any] | None = None
    for seed in seeds:
        per_arm: dict[str, dict[str, Any]] = {}
        arm_grid: tuple[Any, Any, Any] | None = None
        for tag in ARMS:
            matches = [
                path
                for path in inputs
                if path.parent.name == f"seed{seed}_{tag}"
                and path.name == f"{ARMS[tag]}_eval_ambiguity.json"
            ]
            if len(matches) != 1:
                raise ValueError(f"analysis does not bind exactly one seed {seed} {tag} report")
            report = json.loads(matches[0].read_text())
            if report.get("evaluation_seed") != 404:
                raise ValueError(f"seed {seed} {tag} report has the wrong evaluation seed")
            required = ("start_steps", "ambiguity_pair_ids", "ambiguity_sides", "completed")
            if any(key not in report for key in required):
                raise ValueError(f"seed {seed} {tag} report lacks per-rollout fields")
            if len({len(report[key]) for key in required}) != 1:
                raise ValueError(f"seed {seed} {tag} report has an unaligned rollout grid")
            grid = tuple(report[key] for key in required[:3])
            if arm_grid is None:
                arm_grid = grid
            elif grid != arm_grid:
                raise ValueError(f"seed {seed} arms do not share the paired rollout grid")
            per_arm[tag] = report
        assert arm_grid is not None
        pair_ids = np.asarray(arm_grid[1], dtype=np.int64)
        sides = np.asarray(arm_grid[2], dtype=np.int64)
        if set(pair_ids.tolist()) != set(range(69)):
            raise ValueError(f"seed {seed} does not contain exactly 69 pair IDs")
        if any(set(sides[pair_ids == pair_id].tolist()) != {0, 1} for pair_id in range(69)):
            raise ValueError(f"seed {seed} does not contain both sides for every pair")
        if cross_seed_grid is None:
            cross_seed_grid = arm_grid
        elif arm_grid != cross_seed_grid:
            raise ValueError("training seeds do not share the paired rollout grid")
        loaded[seed] = per_arm
    return loaded


def _difference_vectors(
    reports: dict[str, dict[str, Any]], second: str, *, side: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    first_report = reports["snmr"]
    pair_ids = np.asarray(first_report["ambiguity_pair_ids"], dtype=np.int64)
    sides = np.asarray(first_report["ambiguity_sides"], dtype=np.int64)
    first = np.asarray(first_report["completed"], dtype=np.float64)
    control = np.asarray(reports[second]["completed"], dtype=np.float64)
    mask = np.ones_like(pair_ids, dtype=bool) if side is None else sides == side
    return (first - control)[mask], pair_ids[mask]


def _same_interval(first: dict[str, float], second: dict[str, Any]) -> bool:
    return all(
        np.isclose(first[key], float(second[key]), rtol=0.0, atol=1e-12)
        for key in ("difference", "ci95_low", "ci95_high")
    )


def _same_point(first: dict[str, float], second: dict[str, Any]) -> bool:
    return bool(
        np.isclose(first["difference"], float(second["difference"]), rtol=0.0, atol=1e-12)
    )


def effect_rows(
    analysis: dict[str, Any], *, analysis_path: pathlib.Path, preview: bool
) -> list[dict[str, Any]]:
    if analysis.get("protocol") != PROTOCOL:
        raise ValueError("effect figure requires the frozen E70 analyzer protocol")
    seeds = analysis.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(seed not in REGISTERED_SEEDS for seed in seeds):
        raise ValueError("analysis has an invalid training-seed list")
    if preview:
        if seeds == REGISTERED_SEEDS:
            raise ValueError("--preview is only for a non-final seed subset")
    elif seeds != REGISTERED_SEEDS:
        raise ValueError("non-final analysis requires --preview")

    inputs = _resolve_hashed_inputs(analysis, analysis_path=analysis_path)
    reports = _load_ambiguity_reports(inputs, seeds)
    rows: list[dict[str, Any]] = []
    contrast_specs = (
        ("A--T", "snmr_minus_time", "time"),
        ("A--S", "snmr_minus_shuffled", "shuffled"),
    )
    for contrast_label, analysis_key, second in contrast_specs:
        aggregate = _validate_interval(analysis.get(analysis_key), label=contrast_label)
        differences_by_seed: list[np.ndarray] = []
        clusters_by_seed: list[np.ndarray] = []
        for seed in seeds:
            differences, clusters = _difference_vectors(reports[seed], second)
            differences_by_seed.append(differences)
            clusters_by_seed.append(clusters)
        derived_aggregate = hierarchical_paired_interval(
            differences_by_seed, clusters_by_seed
        )
        # The archived seed-0 preview predates the final hierarchical implementation and used a
        # different deterministic bootstrap stream.  Its point must match the hash-bound reports,
        # while its displayed interval remains the authoritative archived analyzer field.  The
        # final three-seed artifact must reproduce the complete tuple exactly.
        if not _same_point(aggregate, derived_aggregate) or (
            not preview and not _same_interval(aggregate, derived_aggregate)
        ):
            raise ValueError(f"hash-bound reports disagree with analyzer {contrast_label}")
        rows.append(
            {
                "contrast": contrast_label,
                "label": "Aggregate",
                **aggregate,
                "source": f"analysis.{analysis_key}",
                "aggregate": True,
            }
        )

        per_seed_declared = analysis.get("per_seed_differences", {})
        for index, seed in enumerate(seeds):
            interval = clustered_paired_interval(
                differences_by_seed[index], clusters_by_seed[index]
            )
            if seeds == REGISTERED_SEEDS:
                try:
                    declared = float(per_seed_declared[str(seed)][analysis_key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"analysis lacks seed {seed} {contrast_label}") from exc
                if not np.isclose(declared, interval["difference"], rtol=0.0, atol=1e-12):
                    raise ValueError(f"seed {seed} {contrast_label} disagrees with analyzer")
            rows.append(
                {
                    "contrast": contrast_label,
                    "label": f"Seed~{seed}",
                    **{key: interval[key] for key in ("difference", "ci95_low", "ci95_high")},
                    "source": f"analysis.inputs seed{seed} pair-cluster bootstrap",
                    "aggregate": False,
                }
            )

        for side, side_key, clip_label in (
            (0, "first", r"walk1\_subject1"),
            (1, "second", r"walk1\_subject5"),
        ):
            side_differences, side_clusters = [], []
            for seed in seeds:
                differences, clusters = _difference_vectors(reports[seed], second, side=side)
                side_differences.append(differences)
                side_clusters.append(clusters)
            derived = hierarchical_paired_interval(side_differences, side_clusters)
            if analysis_key == "snmr_minus_time":
                try:
                    declared_value = analysis["snmr_minus_time_per_clip"][side_key]
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"analysis lacks {clip_label} A--T") from exc
                declared = _validate_interval(declared_value, label=f"{clip_label} A--T")
                if not _same_point(declared, derived) or (
                    not preview and not _same_interval(declared, derived)
                ):
                    raise ValueError(f"{clip_label} A--T disagrees with analyzer")
                interval = declared
                source = f"analysis.snmr_minus_time_per_clip.{side_key}"
            else:
                interval = {
                    key: float(derived[key])
                    for key in ("difference", "ci95_low", "ci95_high")
                }
                source = f"analysis.inputs {clip_label} hierarchical pair bootstrap"
            rows.append(
                {
                    "contrast": contrast_label,
                    "label": clip_label,
                    **interval,
                    "source": source,
                    "aggregate": False,
                }
            )
    return rows


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def render_latex(
    rows: list[dict[str, Any]], *, analysis_sha256: str, preview: bool
) -> str:
    if not rows:
        raise ValueError("effect figure has no rows")
    minimum = min(0.0, *(float(row["ci95_low"]) for row in rows))
    maximum = max(0.0, *(float(row["ci95_high"]) for row in rows))
    axis_min = math.floor((minimum - 0.02) * 10.0) / 10.0
    axis_max = math.ceil((maximum + 0.02) * 10.0) / 10.0
    if axis_max - axis_min < 0.2:
        axis_max = axis_min + 0.2
    axis_width = 5.6

    def xpos(value: float) -> float:
        return (value - axis_min) / (axis_max - axis_min) * axis_width

    lines = [
        "% Generated by scripts/render_e70_effect_figure.py; do not edit by hand.",
        f"% analysis_sha256={analysis_sha256}",
        f"% preview={'true' if preview else 'false'}",
    ]
    for row in rows:
        lines.append(
            "% trace "
            f"{row['contrast']} {row['label']}: difference={row['difference']:.17g} "
            f"ci95_low={row['ci95_low']:.17g} ci95_high={row['ci95_high']:.17g} "
            f"source={row['source']}"
        )
    lines.extend(
        [
            r"\begin{figure}[t]",
            r"\centering",
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tikzpicture}[x=1cm,y=0.43cm,font=\scriptsize]",
        ]
    )
    current_y = 0
    plotted: list[tuple[dict[str, Any], int]] = []
    for contrast in ("A--T", "A--S"):
        lines.append(
            rf"\node[anchor=west,font=\bfseries\scriptsize] at (-1.85,{current_y}) {{{contrast}}};"
        )
        current_y -= 1
        for row in (item for item in rows if item["contrast"] == contrast):
            plotted.append((row, current_y))
            current_y -= 1
        current_y -= 1
    bottom_y = current_y + 1
    zero_x = xpos(0.0)
    lines.append(
        rf"\draw[cink2!45,densely dotted,line width=0.7pt] ({zero_x:.4f},{bottom_y}) -- ({zero_x:.4f},0.4);"
    )
    for row, y in plotted:
        low_x, point_x, high_x = (
            xpos(float(row[key])) for key in ("ci95_low", "difference", "ci95_high")
        )
        color = "cblue!85!black" if row["contrast"] == "A--T" else "corange!85!black"
        dash = "" if row["contrast"] == "A--T" else ",densely dashed"
        width = "1.0pt" if row["aggregate"] else "0.65pt"
        lines.extend(
            [
                rf"\node[anchor=east] at (-0.12,{y}) {{{row['label']}}};",
                rf"\draw[{color}{dash},line width={width}] ({low_x:.4f},{y}) -- ({high_x:.4f},{y});",
                rf"\draw[{color},line width={width}] ({low_x:.4f},{y - 0.17}) -- ({low_x:.4f},{y + 0.17});",
                rf"\draw[{color},line width={width}] ({high_x:.4f},{y - 0.17}) -- ({high_x:.4f},{y + 0.17});",
            ]
        )
        if row["contrast"] == "A--T":
            radius = "2.2pt" if row["aggregate"] else "1.65pt"
            lines.append(rf"\fill[{color}] ({point_x:.4f},{y}) circle ({radius});")
        else:
            size = 0.09 if row["aggregate"] else 0.07
            lines.append(
                rf"\draw[{color},fill=white,line width={width}] "
                rf"({point_x - size:.4f},{y - size * 1.5:.4f}) rectangle "
                rf"({point_x + size:.4f},{y + size * 1.5:.4f});"
            )
        lines.append(
            rf"\node[anchor=west,font=\tiny] at ({axis_width + 0.16:.4f},{y}) "
            rf"{{{_signed(row['difference'])} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]}};"
        )

    tick = axis_min
    while tick <= axis_max + 1e-9:
        tick_x = xpos(tick)
        lines.extend(
            [
                rf"\draw[cink2!60] ({tick_x:.4f},{bottom_y - 0.12}) -- ({tick_x:.4f},{bottom_y + 0.12});",
                rf"\node[anchor=north,font=\tiny] at ({tick_x:.4f},{bottom_y - 0.18}) {{{tick:+.1f}}};",
            ]
        )
        tick += 0.1
    lines.extend(
        [
            rf"\draw[cink2!60] (0,{bottom_y}) -- ({axis_width:.4f},{bottom_y});",
            rf"\node[anchor=north,font=\scriptsize] at ({axis_width / 2:.4f},{bottom_y - 0.85}) "
            r"{paired completion effect};",
            r"\end{tikzpicture}%",
            r"}",
        ]
    )
    prefix = r"\textbf{PREVIEW---seed 0 only.} " if preview else ""
    lines.extend(
        [
            r"\caption{" + prefix
            + r"\textbf{Paired E70 effects.} Circles/solid intervals are A--T; squares/dashed intervals are A--S. "
            + r"Aggregate rows reproduce the frozen analyzer's hierarchical intervals. Seed and clip rows are generated from the analyzer's hash-bound rollout inputs with the same 69-pair clustered bootstrap; they are descriptive and do not change the registered gate.}",
            r"\label{fig:e70-effects}",
            r"\end{figure}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    rows = effect_rows(analysis, analysis_path=args.analysis, preview=args.preview)
    rendered = render_latex(
        rows, analysis_sha256=sha256_file(args.analysis), preview=args.preview
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.out)
    print(f"rendered E70 effect figure -> {args.out}")


if __name__ == "__main__":
    main()
