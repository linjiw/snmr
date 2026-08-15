#!/usr/bin/env python
"""Render the frozen all-seed E70 command-destruction quantities as auditable LaTeX macros.

Sibling of scripts/render_e70_paper_values.py (frozen): that renderer binds the preregistered
three-seed *analysis* artifact, this one binds the twelve per-seed explicit-arm rollout reports
that make up the command-destruction control (three baselines + three destruction modes x three
training seeds).  Every displayed number is computed here from the hash-stamped JSON reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


PROTOCOL_ARM = "c_prior_explicit"
EVALUATION_SEED = 404
NUM_ROLLOUTS = 1024
SEEDS = (0, 1, 2)
DESTROY_MODES = ("zero", "shuffle", "marginal_random")
SEED_DIR = "seed{seed}_explicit"
BASELINE_FILE = "c_prior_explicit_eval.json"
DESTROY_FILE = "c_prior_explicit_eval_destroy_{mode}.json"
# A destroyed command must collapse survival far below the intact baseline; the guard is
# structural (a ratio against the hash-bound baselines), never a transcribed constant.
MAX_DESTROYED_SURVIVAL_FRACTION = 0.2


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completion(value: Any) -> str:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"E70 completion is outside [0, 1]: {value}")
    return f"{value:.3f}"


def _survival(value: Any) -> str:
    value = float(value)
    if not 0.0 <= value <= 10.01:
        raise ValueError(f"E70 survival is outside [0, 10.01]: {value}")
    return f"{value:.3f}"


def _report_key(seed: int, mode: str | None) -> str:
    return f"seed{seed}/{mode or 'baseline'}"


def collect_reports(
    students_root: pathlib.Path,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    """Load the twelve frozen explicit-arm reports and stamp a SHA-256 for each."""
    reports: dict[str, dict[str, Any]] = {}
    digests: list[tuple[str, str]] = []
    for seed in SEEDS:
        directory = students_root / SEED_DIR.format(seed=seed)
        wanted = [(None, BASELINE_FILE)] + [
            (mode, DESTROY_FILE.format(mode=mode)) for mode in DESTROY_MODES
        ]
        for mode, name in wanted:
            path = directory / name
            if not path.is_file():
                raise ValueError(f"destruction control is missing its input: {path}")
            try:
                report = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"destruction input is not readable JSON: {path}") from exc
            if not isinstance(report, dict):
                raise ValueError(f"destruction input is not a report object: {path}")
            reports[_report_key(seed, mode)] = report
            digests.append((f"{directory.name}/{name}", sha256_file(path)))
    return reports, digests


def _validated(reports: dict[str, dict[str, Any]], seed: int, mode: str | None) -> dict[str, Any]:
    key = _report_key(seed, mode)
    try:
        report = reports[key]
    except KeyError as exc:
        raise ValueError(f"destruction control is missing the {key} report") from exc
    try:
        arm = report["arm"]
        eval_z = report["eval_z"]
        destroy = report["destroy_zcmd"]
        evaluation_seed = int(report["evaluation_seed"])
        rollouts = int(report["num_rollouts"])
        completion = float(report["completion_rate"])
        survival = float(report["mean_survival_s"])
        noise = tuple(float(report[field]) for field in ("noise_cmd", "noise_zret", "noise_proprio"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"the {key} report lacks the frozen destruction fields") from exc
    if arm != PROTOCOL_ARM or eval_z != "prior":
        raise ValueError(f"the {key} report is not a prior-conditioned explicit-arm evaluation")
    if destroy != (mode or "none"):
        raise ValueError(f"the {key} report has the wrong destroy_zcmd: {destroy}")
    if evaluation_seed != EVALUATION_SEED or rollouts != NUM_ROLLOUTS:
        raise ValueError(f"the {key} report does not use the frozen evaluation grid")
    if any(value != 0.0 for value in noise):
        raise ValueError(f"the {key} report is not the noise-free evaluation")
    _completion(completion)
    _survival(survival)
    return {"completion": completion, "survival": survival}


def latex_macros(
    reports: dict[str, dict[str, Any]], *, input_digests: list[tuple[str, str]]
) -> str:
    if not input_digests:
        raise ValueError("destruction macros require hash-stamped inputs")

    baselines = [_validated(reports, seed, None) for seed in SEEDS]
    destroyed = {
        mode: [_validated(reports, seed, mode) for seed in SEEDS] for mode in DESTROY_MODES
    }

    if any(entry["completion"] <= 0.5 for entry in baselines):
        raise ValueError("the intact explicit baseline must complete the majority of rollouts")
    if any(entry["survival"] <= 9.0 for entry in baselines):
        raise ValueError("the intact explicit baseline must survive nearly the full episode")

    flattened = [entry for mode in DESTROY_MODES for entry in destroyed[mode]]
    if len(flattened) != len(SEEDS) * len(DESTROY_MODES):
        raise ValueError("destruction control requires every seed x mode cell")
    if any(entry["completion"] != 0.0 for entry in flattened):
        raise ValueError("command destruction must drive completion to exactly zero")

    baseline_completion = sum(entry["completion"] for entry in baselines) / len(baselines)
    destroyed_completion = sum(entry["completion"] for entry in flattened) / len(flattened)
    max_survival = max(entry["survival"] for entry in flattened)
    min_baseline_survival = min(entry["survival"] for entry in baselines)
    if max_survival >= MAX_DESTROYED_SURVIVAL_FRACTION * min_baseline_survival:
        raise ValueError("destroyed survival is not catastrophically below the intact baseline")

    lines = [
        "% Generated by scripts/render_e70_destruction_values.py; do not edit by hand.",
        f"% inputs_sha256={hashlib.sha256(''.join(digest for _, digest in input_digests).encode()).hexdigest()}",
    ]
    lines.extend(f"% input_sha256[{name}]={digest}" for name, digest in input_digests)
    for seed in SEEDS:
        for mode in (None, *DESTROY_MODES):
            entry = _validated(reports, seed, mode)
            lines.append(
                f"% trace {_report_key(seed, mode)}: "
                f"completion_rate={entry['completion']:.17g} "
                f"mean_survival_s={entry['survival']:.17g}"
            )
    lines.extend(
        [
            rf"\newcommand{{\EDestroySeeds}}{{{len(SEEDS)}}}",
            rf"\newcommand{{\EDestroyModes}}{{{len(DESTROY_MODES)}}}",
            rf"\newcommand{{\EDestroyBaselineCompletion}}{{{_completion(baseline_completion)}}}",
            rf"\newcommand{{\EDestroyCompletion}}{{{_completion(destroyed_completion)}}}",
            rf"\newcommand{{\EDestroyMaxSurvival}}{{{_survival(max_survival)}}}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    reports, digests = collect_reports(args.students_root)
    rendered = latex_macros(reports, input_digests=digests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.out)
    print(f"rendered frozen E70 destruction values -> {args.out}")


if __name__ == "__main__":
    main()
