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
import dataclasses
import json
import pathlib
from typing import Any


EVALUATION_SEED = 404
NUM_ROLLOUTS = 1024
SEEDS = (0, 1, 2)
DESTROY_MODES = ("zero", "shuffle", "marginal_random")
# A destroyed command must collapse survival far below the intact baseline; the guard is
# structural (a ratio against the hash-bound baselines), never a transcribed constant.
MAX_DESTROYED_SURVIVAL_FRACTION = 0.2


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    """One arm's destruction battery: where its reports live and how strong its baseline is.

    The baseline guards differ per arm because the arms are not equally capable -- the explicit
    controller is at teacher parity while the SNMR controller is the research interface.  Both are
    floors on the INTACT baseline, so they can only reject a degraded input; neither can make a
    destroyed cell look better than it is.
    """

    arm: str
    seed_dir: str
    baseline_file: str
    destroy_file: str
    macro_infix: str
    min_baseline_completion: float
    min_baseline_survival: float


EXPLICIT = ArmSpec(
    arm="c_prior_explicit",
    seed_dir="seed{seed}_explicit",
    baseline_file="c_prior_explicit_eval.json",
    destroy_file="c_prior_explicit_eval_destroy_{mode}.json",
    macro_infix="",
    min_baseline_completion=0.5,
    min_baseline_survival=9.0,
)

# E75: the same frozen protocol applied to the arm the paper is actually about.
# Preregistration: docs/E75_SNMR_DESTRUCTION_PREREG.md.
SNMR = ArmSpec(
    arm="a_prior_snmr",
    seed_dir="seed{seed}_snmr",
    baseline_file="a_prior_snmr_eval.json",
    destroy_file="a_prior_snmr_eval_destroy_{mode}.json",
    macro_infix="Snmr",
    min_baseline_completion=0.5,
    min_baseline_survival=8.0,
)


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
    spec: ArmSpec = EXPLICIT,
    destroy_root: pathlib.Path | None = None,
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    """Load one arm's twelve reports and stamp a SHA-256 for each.

    ``destroy_root`` defaults to ``students_root``.  The SNMR arm needs them separated: its intact
    baselines are frozen E70 artifacts while its destruction reports are E75 outputs under a new
    root, because nothing may be written under the frozen E70 tree.
    """
    reports: dict[str, dict[str, Any]] = {}
    digests: list[tuple[str, str]] = []
    destroy_root = students_root if destroy_root is None else destroy_root
    for seed in SEEDS:
        wanted = [(None, spec.baseline_file, students_root)] + [
            (mode, spec.destroy_file.format(mode=mode), destroy_root) for mode in DESTROY_MODES
        ]
        for mode, name, root in wanted:
            directory = root / spec.seed_dir.format(seed=seed)
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


def _validated(
    reports: dict[str, dict[str, Any]], seed: int, mode: str | None, spec: ArmSpec = EXPLICIT
) -> dict[str, Any]:
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
    if arm != spec.arm or eval_z != "prior":
        raise ValueError(f"the {key} report is not a prior-conditioned {spec.arm} evaluation")
    if destroy != (mode or "none"):
        raise ValueError(f"the {key} report has the wrong destroy_zcmd: {destroy}")
    if evaluation_seed != EVALUATION_SEED or rollouts != NUM_ROLLOUTS:
        raise ValueError(f"the {key} report does not use the frozen evaluation grid")
    if any(value != 0.0 for value in noise):
        raise ValueError(f"the {key} report is not the noise-free evaluation")
    _completion(completion)
    _survival(survival)
    return {"completion": completion, "survival": survival}


def _arm_block(
    reports: dict[str, dict[str, Any]],
    input_digests: list[tuple[str, str]],
    spec: ArmSpec,
) -> tuple[list[str], list[str]]:
    """Return (trace comment lines, macro lines) for one arm."""
    baselines = [_validated(reports, seed, None, spec) for seed in SEEDS]
    destroyed = {
        mode: [_validated(reports, seed, mode, spec) for seed in SEEDS] for mode in DESTROY_MODES
    }

    if any(entry["completion"] <= spec.min_baseline_completion for entry in baselines):
        raise ValueError(f"the intact {spec.arm} baseline must complete the majority of rollouts")
    if any(entry["survival"] <= spec.min_baseline_survival for entry in baselines):
        raise ValueError(f"the intact {spec.arm} baseline must survive most of the episode")

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

    traces = [
        f"% trace[{spec.arm}] {_report_key(seed, mode)}: "
        f"completion_rate={_validated(reports, seed, mode, spec)['completion']:.17g} "
        f"mean_survival_s={_validated(reports, seed, mode, spec)['survival']:.17g}"
        for seed in SEEDS
        for mode in (None, *DESTROY_MODES)
    ]
    infix = spec.macro_infix
    macros = [
        rf"\newcommand{{\EDestroy{infix}BaselineCompletion}}{{{_completion(baseline_completion)}}}",
        rf"\newcommand{{\EDestroy{infix}Completion}}{{{_completion(destroyed_completion)}}}",
        rf"\newcommand{{\EDestroy{infix}MaxSurvival}}{{{_survival(max_survival)}}}",
    ]
    return traces, macros


def latex_macros(
    reports: dict[str, dict[str, Any]],
    *,
    input_digests: list[tuple[str, str]],
    spec: ArmSpec = EXPLICIT,
    snmr_reports: dict[str, dict[str, Any]] | None = None,
    snmr_digests: list[tuple[str, str]] | None = None,
) -> str:
    if not input_digests:
        raise ValueError("destruction macros require hash-stamped inputs")

    all_digests = list(input_digests)
    if snmr_reports is not None:
        if not snmr_digests:
            raise ValueError("the SNMR destruction battery requires hash-stamped inputs")
        all_digests += list(snmr_digests)

    traces, macros = _arm_block(reports, input_digests, spec)
    arms = 1
    if snmr_reports is not None:
        snmr_traces, snmr_macros = _arm_block(snmr_reports, snmr_digests or [], SNMR)
        traces += snmr_traces
        macros += snmr_macros
        arms = 2

    lines = [
        "% Generated by scripts/render_e70_destruction_values.py; do not edit by hand.",
        f"% inputs_sha256={hashlib.sha256(''.join(d for _, d in all_digests).encode()).hexdigest()}",
    ]
    lines.extend(f"% input_sha256[{name}]={digest}" for name, digest in all_digests)
    lines.extend(traces)
    lines.extend(
        [
            rf"\newcommand{{\EDestroySeeds}}{{{len(SEEDS)}}}",
            rf"\newcommand{{\EDestroyModes}}{{{len(DESTROY_MODES)}}}",
            rf"\newcommand{{\EDestroyArms}}{{{arms}}}",
        ]
    )
    lines.extend(macros)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--snmr-destroy-root",
        type=pathlib.Path,
        default=None,
        help=(
            "E75 students root holding a_prior_snmr_eval_destroy_*.json. The SNMR intact baselines "
            "are read from --students-root, because they are frozen E70 artifacts."
        ),
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    reports, digests = collect_reports(args.students_root)
    snmr_reports = snmr_digests = None
    if args.snmr_destroy_root is not None:
        snmr_reports, snmr_digests = collect_reports(
            args.students_root, SNMR, destroy_root=args.snmr_destroy_root
        )
    rendered = latex_macros(
        reports,
        input_digests=digests,
        snmr_reports=snmr_reports,
        snmr_digests=snmr_digests,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.out)
    arms = "explicit + SNMR" if snmr_reports is not None else "explicit"
    print(f"rendered frozen destruction values ({arms}) -> {args.out}")


if __name__ == "__main__":
    main()
