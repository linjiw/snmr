#!/usr/bin/env python3
"""Evaluate Gate 1 calibration runs against the preregistered contract."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import numpy as np


TERM_RULES = {
    "contact_bce": {
        "kind": "bce",
        "median_min": 0.03,
        "median_max": 0.5,
        "p90_max": 1.0,
        "target": 0.1,
    },
    "edge_velocity": {
        "kind": "movement",
        "median_min": 0.1,
        "median_max": 1.0,
        "p90_max": 2.0,
        "target": 0.3,
    },
    "teacher_stance_velocity": {
        "kind": "movement",
        "median_min": 0.1,
        "median_max": 1.0,
        "p90_max": 2.0,
        "target": 0.3,
    },
    "teacher_velocity": {
        "kind": "movement",
        "median_min": 0.1,
        "median_max": 1.0,
        "p90_max": 2.0,
        "target": 0.3,
    },
}
EXPECTED_ARM_TERMS = {
    "c0_seed0": set(),
    "c1_bce_seed0": {"contact_bce"},
    "c2_edge_seed0": {"contact_bce", "edge_velocity"},
    "c3_stance_seed0": {"teacher_stance_velocity"},
    "c4_teacher_velocity_seed0": {"teacher_velocity"},
}
IGNORED_RUN_DIRECTORIES = {"__pycache__"}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _term_weight(rows: list[dict[str, Any]], term: str) -> float | None:
    weights = [
        row.get("loss_terms", {}).get(term, {}).get("weight")
        for row in rows
    ]
    finite = [float(weight) for weight in weights if _finite_number(weight)]
    if len(finite) != len(rows) or not finite:
        return None
    if any(not math.isclose(weight, finite[0], rel_tol=0.0, abs_tol=1e-12) for weight in finite):
        return None
    return finite[0]


def _recalibrated_weight(weight: float, target: float, median_ratio: float) -> float | None:
    if not (math.isfinite(weight) and weight > 0 and math.isfinite(median_ratio) and median_ratio > 0):
        return None
    proposed = weight * target / median_ratio
    return min(max(proposed, weight / 4.0), weight * 4.0)


def analyze_run(
    run_dir: pathlib.Path,
    *,
    expected_events: int = 10,
    final_events: int = 5,
) -> dict[str, Any]:
    """Analyze one immutable diagnostic run."""
    errors: list[str] = []
    diagnostics_path = run_dir / "diagnostics.jsonl"
    manifest_path = run_dir / "manifest.json"
    if not diagnostics_path.is_file():
        return {"arm": run_dir.name, "passed": False, "errors": ["missing diagnostics.jsonl"]}
    if not manifest_path.is_file():
        return {"arm": run_dir.name, "passed": False, "errors": ["missing manifest.json"]}

    rows = _read_jsonl(diagnostics_path)
    manifest = _read_json(manifest_path)
    if len(rows) != expected_events:
        errors.append(f"expected {expected_events} diagnostic events, found {len(rows)}")
    if len(rows) < final_events:
        return {
            "arm": run_dir.name,
            "passed": False,
            "errors": errors + [f"need at least {final_events} diagnostic events"],
            "event_count": len(rows),
        }

    steps = [row.get("step") for row in rows]
    if any(not isinstance(step, int) for step in steps):
        errors.append("diagnostic steps must be integers")
    elif steps != sorted(steps) or len(set(steps)) != len(steps):
        errors.append("diagnostic steps must be unique and increasing")

    if manifest.get("status") != "completed":
        errors.append(f"manifest status is {manifest.get('status')!r}, expected 'completed'")
    progress = manifest.get("progress", {})
    planned_steps = manifest.get("training", {}).get("planned_optimizer_steps")
    if progress.get("completion_state") != "completed":
        errors.append("manifest progress is not completed")
    if isinstance(planned_steps, int) and progress.get("step") != planned_steps:
        errors.append("manifest did not complete its planned optimizer steps")

    git = manifest.get("git", {})
    revision = git.get("sha")
    if not revision:
        errors.append("manifest has no git revision")
    if git.get("dirty") is not False:
        errors.append("manifest source worktree was not clean")

    tail = rows[-final_events:]
    support_events = sum(
        isinstance(row.get("contact_labels", {}).get("samples"), int)
        and row["contact_labels"]["samples"] > 0
        for row in tail
    )
    if support_events < 4:
        errors.append(f"contact support is nonempty in only {support_events}/{final_events} final events")

    distill_valid = True
    for row in tail:
        distill = row.get("loss_terms", {}).get("distill", {})
        values = [
            distill.get("raw"),
            distill.get("weighted"),
            distill.get("gradient_norm", {}).get("shared_trunk"),
        ]
        if any(not _finite_number(value) or value <= 0 for value in values):
            distill_valid = False
        weighted_terms = [
            term.get("weighted")
            for term in row.get("loss_terms", {}).values()
            if isinstance(term, dict)
        ]
        if not weighted_terms or any(not _finite_number(value) for value in weighted_terms):
            errors.append(f"step {row.get('step')}: total loss has a nonfinite component")
    if not distill_valid:
        errors.append("final distillation losses and shared-trunk gradients must be finite and nonzero")

    active_terms = sorted(set(TERM_RULES) & set(rows[-1].get("loss_terms", {})))
    base_arm = run_dir.name.removesuffix("-r1")
    expected_terms = EXPECTED_ARM_TERMS.get(base_arm)
    if expected_terms is not None and set(active_terms) != expected_terms:
        errors.append(
            f"active factorized terms are {active_terms}, expected {sorted(expected_terms)}"
        )
    terms: dict[str, Any] = {}
    for term in active_terms:
        rules = TERM_RULES[term]
        ratios = []
        cosines = []
        for row in tail:
            loss_terms = row.get("loss_terms", {})
            base = loss_terms.get("distill", {}).get("gradient_norm", {}).get("shared_trunk")
            added = loss_terms.get(term, {}).get("gradient_norm", {}).get("shared_trunk")
            if not _finite_number(base) or base <= 0 or not _finite_number(added):
                continue
            ratios.append(float(added) / float(base))
            cosine = row.get("gradient_cosine", {}).get("shared_trunk", {}).get(f"distill|{term}")
            if _finite_number(cosine):
                cosines.append(float(cosine))

        term_errors = []
        if len(ratios) != final_events or any(not math.isfinite(ratio) for ratio in ratios):
            term_errors.append(f"expected {final_events} finite ratios, found {len(ratios)}")
            median_ratio = None
            p90_ratio = None
        else:
            median_ratio = float(np.median(ratios))
            p90_ratio = float(np.percentile(ratios, 90))
            if not rules["median_min"] <= median_ratio <= rules["median_max"]:
                term_errors.append(
                    f"median ratio {median_ratio:.6g} outside "
                    f"[{rules['median_min']}, {rules['median_max']}]"
                )
            if p90_ratio > rules["p90_max"]:
                term_errors.append(f"p90 ratio {p90_ratio:.6g} exceeds {rules['p90_max']}")

        weight = _term_weight(tail, term)
        if weight is None or weight <= 0:
            term_errors.append("weight is missing, nonpositive, or changes across final events")
        suggested_weight = (
            _recalibrated_weight(weight, rules["target"], median_ratio)
            if term_errors and weight is not None and median_ratio is not None
            else None
        )
        terms[term] = {
            "kind": rules["kind"],
            "weight": weight,
            "ratios": ratios,
            "median_ratio": median_ratio,
            "p90_ratio": p90_ratio,
            "distill_cosines": cosines,
            "median_distill_cosine": float(np.median(cosines)) if cosines else None,
            "passed": not term_errors,
            "errors": term_errors,
            "suggested_recalibrated_weight": suggested_weight,
        }
        errors.extend(f"{term}: {message}" for message in term_errors)

    return {
        "arm": run_dir.name,
        "passed": not errors,
        "errors": errors,
        "event_count": len(rows),
        "final_steps": [row.get("step") for row in tail],
        "contact_support_events": support_events,
        "revision": revision,
        "terms": terms,
    }


def analyze_root(
    root: pathlib.Path,
    *,
    expected_events: int = 10,
    final_events: int = 5,
) -> dict[str, Any]:
    run_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name not in IGNORED_RUN_DIRECTORIES
    )
    arms = [
        analyze_run(path, expected_events=expected_events, final_events=final_events)
        for path in run_dirs
    ]
    revisions = {arm.get("revision") for arm in arms if arm.get("revision")}
    cross_arm_errors = []
    if not arms:
        cross_arm_errors.append("no diagnostic run directories found")
    arms_by_name = {arm["arm"]: arm for arm in arms}
    expected_initial = set(EXPECTED_ARM_TERMS)
    expected_retries = {f"{name}-r1" for name in expected_initial}
    unexpected_arms = set(arms_by_name) - expected_initial - expected_retries
    if unexpected_arms:
        cross_arm_errors.append(f"unexpected arms: {sorted(unexpected_arms)}")

    present_initial_arms = set(arms_by_name) & expected_initial
    missing_initial_arms = expected_initial - present_initial_arms
    if missing_initial_arms:
        cross_arm_errors.append(f"missing initial arms: {sorted(missing_initial_arms)}")
    if len(revisions) > 1:
        cross_arm_errors.append(f"runs use multiple revisions: {sorted(revisions)}")

    effective_arms = {}
    for base_name in sorted(expected_initial & set(arms_by_name)):
        initial = arms_by_name[base_name]
        retry_name = f"{base_name}-r1"
        retry = arms_by_name.get(retry_name)
        effective_arms[base_name] = retry_name if retry is not None else base_name
        if retry is None:
            continue
        if initial["passed"]:
            cross_arm_errors.append(f"{retry_name}: retry is not allowed after a passing initial arm")
            continue

        term_prefixes = tuple(f"{term}: " for term in initial.get("terms", {}))
        non_term_errors = [
            error for error in initial["errors"]
            if not error.startswith(term_prefixes)
        ]
        if non_term_errors:
            cross_arm_errors.append(
                f"{retry_name}: retry cannot replace structural errors in {base_name}: "
                f"{non_term_errors}"
            )

        initial_terms = initial.get("terms", {})
        retry_terms = retry.get("terms", {})
        for term, initial_result in initial_terms.items():
            retry_result = retry_terms.get(term)
            if retry_result is None:
                continue
            expected_weight = initial_result["weight"]
            if not initial_result["passed"]:
                expected_weight = initial_result["suggested_recalibrated_weight"]
                if expected_weight is None:
                    cross_arm_errors.append(
                        f"{retry_name}: {term} has no valid deterministic recalibration"
                    )
                    continue
            actual_weight = retry_result["weight"]
            if (
                expected_weight is None
                or actual_weight is None
                or not math.isclose(
                    actual_weight,
                    expected_weight,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                cross_arm_errors.append(
                    f"{retry_name}: {term} weight {actual_weight!r}, "
                    f"expected {expected_weight!r}"
                )

    selected = [
        arms_by_name[name]
        for name in effective_arms.values()
    ]
    accepted_arms = [
        base_name
        for base_name, selected_name in effective_arms.items()
        if arms_by_name[selected_name]["passed"]
    ]
    dropped_arms = [
        base_name
        for base_name, selected_name in effective_arms.items()
        if selected_name.endswith("-r1") and not arms_by_name[selected_name]["passed"]
    ]
    recalibration_required_arms = [
        base_name
        for base_name, selected_name in effective_arms.items()
        if selected_name == base_name and not arms_by_name[selected_name]["passed"]
    ]
    calibration_complete = (
        len(selected) == len(expected_initial)
        and not cross_arm_errors
        and not recalibration_required_arms
    )
    if cross_arm_errors or len(selected) != len(expected_initial):
        status = "invalid"
    elif recalibration_required_arms:
        status = "recalibration_required"
    elif dropped_arms:
        status = "complete_with_dropped_arms"
    else:
        status = "pass"
    return {
        "root": str(root),
        "passed": status == "pass",
        "calibration_complete": calibration_complete,
        "status": status,
        "cross_arm_errors": cross_arm_errors,
        "revision": next(iter(revisions)) if len(revisions) == 1 else None,
        "effective_arms": effective_arms,
        "accepted_arms": accepted_arms,
        "dropped_arms": dropped_arms,
        "recalibration_required_arms": recalibration_required_arms,
        "arms": arms,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("arm\tterm\tweight\tmedian_ratio\tp90_ratio\tresult\trecalibrated_weight")
    for arm in report["arms"]:
        terms = arm.get("terms", {})
        if not terms:
            print(f"{arm['arm']}\t-\t-\t-\t-\t{'PASS' if arm['passed'] else 'FAIL'}\t-")
            continue
        for term, result in terms.items():
            values = [
                arm["arm"],
                term,
                result["weight"],
                result["median_ratio"],
                result["p90_ratio"],
                "PASS" if result["passed"] and arm["passed"] else "FAIL",
                result["suggested_recalibrated_weight"],
            ]
            print("\t".join("-" if value is None else str(value) for value in values))
    for error in report["cross_arm_errors"]:
        print(f"ERROR: {error}")
    for arm in report["arms"]:
        for error in arm["errors"]:
            print(f"ERROR: {arm['arm']}: {error}")
    print(f"overall: {report['status'].upper()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--expected-events", type=int, default=10)
    parser.add_argument("--final-events", type=int, default=5)
    args = parser.parse_args()

    report = analyze_root(
        args.root,
        expected_events=args.expected_events,
        final_events=args.final_events,
    )
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["calibration_complete"] else 1)


if __name__ == "__main__":
    main()
