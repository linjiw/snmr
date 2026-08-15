#!/usr/bin/env python
"""Render the E72 latent-substitution result as hash-stamped LaTeX macros.

E72 asks whether the SNMR command carries a static clip label or time-aligned trajectory state.
A phase shift preserves clip identity exactly -- same clip, real latent trajectory, same marginal
statistics, same internal 0.1 s window structure -- and changes only alignment to the physics.  A
one-bit clip label therefore predicts no effect under a shift.

Every displayed number is computed here from the frozen per-run reports and stamped with the
SHA-256 of each input, so no E72 value is ever hand-transcribed into the manuscript.

Protocol and registered interpretation: docs/E72_LATENT_SUBSTITUTION_PROTOCOL.md.
Noise floor and the amended control gate: docs/E76_EVALUATION_REPLICATION.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
from typing import Any


ARM = "a_prior_snmr"
REPORT = f"{ARM}_eval_ambiguity.json"
SEEDS = (0, 1, 2)
EVALUATION_SEED = 404
NUM_ROLLOUTS = 1024

CONTROL = "control"
# (directory name, macro infix, nominal offset in seconds).  Registered in the protocol; the
# realized offsets are truncated toward zero at 50 Hz, so +-0.25 s is realized as +-0.24 s.
DELTA_ARMS = (
    ("shift_m0250", "ShiftMinusQuarter", -0.24),
    ("shift_p0250", "ShiftPlusQuarter", +0.24),
    ("shift_p0500", "ShiftPlusHalf", +0.50),
)
STATIC_ARMS = (("first_frame", "FirstFrame"), ("clip_mean", "ClipMean"))

# Registered in docs/E72_LATENT_SUBSTITUTION_PROTOCOL.md, Amendment 1.  A difference from control
# smaller than this is reported as "within evaluation noise", never as an effect or a direction.
DETECTABILITY_FLOOR = 0.02
# Measured in E76 over 8 repeats per arm.
REPLICATION_SD = 0.008310


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completion(value: float) -> str:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"E72 completion is outside [0, 1]: {value}")
    return f"{float(value):.3f}"


def _signed(value: float) -> str:
    return f"{float(value):+.3f}"


def collect_arm(
    students_root: pathlib.Path, arm_dir: str
) -> tuple[list[float], list[tuple[str, str]]]:
    """Return (per-run completions across seeds and repeats, hash-stamped inputs)."""
    values: list[float] = []
    digests: list[tuple[str, str]] = []
    for seed in SEEDS:
        seed_dir = students_root / arm_dir / f"seed{seed}_snmr"
        repeats = sorted(seed_dir.glob("repeat*"))
        if not repeats:
            raise ValueError(f"E72 arm has no replicated cells: {seed_dir}")
        for repeat in repeats:
            path = repeat / REPORT
            if not path.is_file():
                raise ValueError(f"E72 arm is missing a report: {path}")
            report: dict[str, Any] = json.loads(path.read_text())
            if report.get("arm") != ARM or report.get("eval_z") != "prior":
                raise ValueError(f"{path} is not a prior-conditioned {ARM} evaluation")
            if report.get("destroy_zcmd") != "none":
                raise ValueError(f"{path} is a destruction report, not a substitution report")
            if int(report["evaluation_seed"]) != EVALUATION_SEED:
                raise ValueError(f"{path} does not use the frozen evaluation seed")
            if int(report["num_rollouts"]) != NUM_ROLLOUTS:
                raise ValueError(f"{path} does not use the frozen rollout count")
            if any(float(report[k]) != 0.0 for k in ("noise_cmd", "noise_zret", "noise_proprio")):
                raise ValueError(f"{path} is not the noise-free evaluation")
            if "ambiguity_precheck" not in report:
                raise ValueError(f"{path} is not an ambiguity-grid evaluation")
            values.append(float(report["completion_rate"]))
            digests.append((f"{arm_dir}/seed{seed}/{repeat.name}", sha256_file(path)))
    return values, digests


def latex_macros(
    students_root: pathlib.Path, *, time_completion: float
) -> str:
    control, digests = collect_arm(students_root, CONTROL)
    control_mean = statistics.fmean(control)
    advantage = control_mean - time_completion
    if advantage <= 0.0:
        raise ValueError("the control arm must exceed the clock null for the ladder to be defined")

    arms: dict[str, tuple[float, float, int]] = {
        CONTROL: (control_mean, statistics.stdev(control), len(control))
    }
    lost_fraction: dict[str, float] = {}
    for arm_dir, _infix, _seconds in DELTA_ARMS:
        values, arm_digests = collect_arm(students_root, arm_dir)
        digests += arm_digests
        mean = statistics.fmean(values)
        arms[arm_dir] = (mean, statistics.stdev(values), len(values))
        delta = control_mean - mean
        if delta <= DETECTABILITY_FLOOR:
            raise ValueError(
                f"{arm_dir} moved {delta:.4f}, at or below the registered {DETECTABILITY_FLOOR} "
                "floor; it must be reported as within evaluation noise, not rendered as an effect"
            )
        lost_fraction[arm_dir] = delta / advantage
    for arm_dir, _infix in STATIC_ARMS:
        values, arm_digests = collect_arm(students_root, arm_dir)
        digests += arm_digests
        arms[arm_dir] = (statistics.fmean(values), statistics.stdev(values), len(values))

    # The registered claim rests on monotonicity in |delta|: the half-second arm must lose more
    # than either quarter-second arm.  Assert it rather than trusting the prose.
    if not (
        lost_fraction["shift_p0500"] > lost_fraction["shift_p0250"]
        and lost_fraction["shift_p0500"] > lost_fraction["shift_m0250"]
    ):
        raise ValueError("E72 degradation is not monotone in |delta|; the registered reading fails")

    lines = [
        "% Generated by scripts/render_e72_phase_values.py; do not edit by hand.",
        f"% inputs_sha256={hashlib.sha256(''.join(d for _, d in digests).encode()).hexdigest()}",
        f"% detectability_floor={DETECTABILITY_FLOOR} replication_sd={REPLICATION_SD}",
    ]
    lines += [f"% input_sha256[{name}]={digest}" for name, digest in digests]
    for arm_dir, (mean, sd, n) in arms.items():
        lines.append(f"% trace[{arm_dir}] n={n} mean={mean:.17g} sd={sd:.17g}")

    macros = [
        rf"\newcommand{{\EPhaseSeeds}}{{{len(SEEDS)}}}",
        rf"\newcommand{{\EPhaseRepeats}}{{{arms[CONTROL][2] // len(SEEDS)}}}",
        rf"\newcommand{{\EPhaseControlCompletion}}{{{_completion(arms[CONTROL][0])}}}",
        rf"\newcommand{{\EPhaseClockAdvantage}}{{{_signed(advantage)}}}",
        rf"\newcommand{{\EPhaseFloor}}{{{DETECTABILITY_FLOOR:.2f}}}",
    ]
    for arm_dir, infix, seconds in DELTA_ARMS:
        mean = arms[arm_dir][0]
        macros += [
            rf"\newcommand{{\EPhase{infix}Seconds}}{{{seconds:+.2f}}}",
            rf"\newcommand{{\EPhase{infix}Completion}}{{{_completion(mean)}}}",
            rf"\newcommand{{\EPhase{infix}Lost}}{{{_signed(mean - arms[CONTROL][0])}}}",
            rf"\newcommand{{\EPhase{infix}LostPercent}}{{{round(100 * lost_fraction[arm_dir])}}}",
        ]
    for arm_dir, infix in STATIC_ARMS:
        macros.append(
            rf"\newcommand{{\EPhase{infix}Completion}}{{{_completion(arms[arm_dir][0])}}}"
        )
    return "\n".join(lines + macros) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--time-completion",
        type=float,
        required=True,
        help="frozen clock-null ambiguity completion, from the E70 analyzer",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rendered = latex_macros(args.students_root, time_completion=args.time_completion)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(args.out)
    print(f"rendered E72 phase-sensitivity values -> {args.out}")


if __name__ == "__main__":
    main()
