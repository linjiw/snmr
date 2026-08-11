from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_e67_results import (
    hierarchical_paired_interval,
)
from scripts.render_e70_effect_figure import effect_rows, render_latex


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_e70_effect_figure.py"


def _write_report(path: Path, completed: list[bool]) -> dict:
    report = {
        "evaluation_seed": 404,
        "start_steps": list(range(138)),
        "ambiguity_pair_ids": [pair for pair in range(69) for _ in (0, 1)],
        "ambiguity_sides": [side for _ in range(69) for side in (0, 1)],
        "completed": completed,
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    return report


def _analysis(tmp_path: Path, seeds: list[int]) -> tuple[Path, dict]:
    inputs = []
    loaded = {}
    for seed in seeds:
        snmr = [True] * 138
        time = [(pair + seed) % 4 != 0 for pair in range(69) for _ in (0, 1)]
        shuffled = [(pair + seed) % 3 != 0 for pair in range(69) for _ in (0, 1)]
        loaded[seed] = {}
        for tag, values in (("snmr", snmr), ("time", time), ("shuffled", shuffled)):
            path = tmp_path / f"seed{seed}_{tag}" / "a_prior_snmr_eval_ambiguity.json"
            loaded[seed][tag] = _write_report(path, values)
            inputs.append(
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )

    def vectors(seed: int, second: str, side: int | None = None):
        report = loaded[seed]["snmr"]
        pair_ids = report["ambiguity_pair_ids"]
        sides = report["ambiguity_sides"]
        differences = [
            float(first) - float(second_value)
            for first, second_value in zip(
                report["completed"], loaded[seed][second]["completed"], strict=True
            )
        ]
        selected = [
            index for index, value in enumerate(sides) if side is None or value == side
        ]
        return (
            [differences[index] for index in selected],
            [pair_ids[index] for index in selected],
        )

    analysis = {
        "protocol": "E70 preregistered analysis v1",
        "seeds": seeds,
        "inputs": inputs,
        "per_seed_differences": {},
    }
    for key, second in (("snmr_minus_time", "time"), ("snmr_minus_shuffled", "shuffled")):
        values, clusters = zip(*(vectors(seed, second) for seed in seeds), strict=True)
        analysis[key] = hierarchical_paired_interval(
            [np.asarray(item) for item in values],
            [np.asarray(item) for item in clusters],
        )
        for index, seed in enumerate(seeds):
            analysis["per_seed_differences"].setdefault(str(seed), {})[key] = analysis[key][
                "per_seed_difference"
            ][index]
    analysis["snmr_minus_time_per_clip"] = {}
    for side, name in ((0, "first"), (1, "second")):
        values, clusters = zip(*(vectors(seed, "time", side) for seed in seeds), strict=True)
        analysis["snmr_minus_time_per_clip"][name] = hierarchical_paired_interval(
            [np.asarray(item) for item in values],
            [np.asarray(item) for item in clusters],
        )
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analysis))
    return path, analysis


def test_nonfinal_analysis_requires_preview(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path, [0])
    with pytest.raises(ValueError, match="requires --preview"):
        effect_rows(analysis, analysis_path=path, preview=False)


def test_preview_fragment_is_labeled_and_traces_seed_and_clip_intervals(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path, [0])
    rows = effect_rows(analysis, analysis_path=path, preview=True)
    rendered = render_latex(rows, analysis_sha256="abc", preview=True)
    assert "PREVIEW---seed 0 only" in rendered
    assert "% analysis_sha256=abc" in rendered
    assert "% trace A--T Seed~0:" in rendered
    assert "walk1\\_subject1" in rendered
    assert "Circles/solid intervals are A--T" in rendered


def test_final_fragment_contains_three_seeds_both_clips_and_aggregate(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path, [0, 1, 2])
    rows = effect_rows(analysis, analysis_path=path, preview=False)
    rendered = render_latex(rows, analysis_sha256="def", preview=False)
    assert len(rows) == 12
    assert "Seed~2" in rendered
    assert sum(row["label"] == "Aggregate" for row in rows) == 2
    assert "PREVIEW" not in rendered


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path, [0])
    tampered = Path(analysis["inputs"][0]["path"])
    tampered.write_text(tampered.read_text() + "\n")
    with pytest.raises(ValueError, match="input hash mismatch"):
        effect_rows(analysis, analysis_path=path, preview=True)


def test_analyzer_interval_must_match_hash_bound_rollouts(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path, [0])
    changed = copy.deepcopy(analysis)
    changed["snmr_minus_time"]["difference"] += 0.01
    with pytest.raises(ValueError, match="disagree with analyzer"):
        effect_rows(changed, analysis_path=path, preview=True)


def test_direct_entrypoint_help_uses_repository_scripts_package(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--analysis" in completed.stdout
