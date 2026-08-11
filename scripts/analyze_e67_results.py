#!/usr/bin/env python
"""Analyze the preregistered E67 multi-trajectory result without changing its gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

ARMS = {
    "explicit": "c_prior_explicit",
    "snmr": "a_prior_snmr",
    "time": "a_prior_snmr",
    "proprio": "b_prior_proprio",
    "shuffled": "a_prior_snmr",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clustered_paired_interval(
    differences: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int = 6704,
    replicates: int = 10_000,
) -> dict[str, float]:
    """Paired cluster bootstrap, weighting each ambiguity pair equally."""
    differences = np.asarray(differences, dtype=np.float64)
    clusters = np.asarray(clusters)
    if differences.ndim != 1 or clusters.shape != differences.shape:
        raise ValueError("differences and clusters must be aligned vectors")
    unique = np.unique(clusters)
    if len(unique) < 2:
        raise ValueError("paired interval requires at least two clusters")
    means = np.asarray([differences[clusters == item].mean() for item in unique])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(means), size=(replicates, len(means)))
    boot = means[sampled].mean(axis=1)
    return {
        "difference": float(means.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "clusters": int(len(unique)),
    }


def hierarchical_paired_interval(
    differences_by_seed: list[np.ndarray],
    clusters_by_seed: list[np.ndarray],
    *,
    seed: int = 6704,
    replicates: int = 10_000,
) -> dict[str, float | int]:
    """Two-stage paired bootstrap over training seeds and ambiguity-pair clusters.

    Each training seed and each reference-only ambiguity pair receive equal weight.  With
    one training seed this reduces to the preregistered paired cluster bootstrap.
    """
    if not differences_by_seed or len(differences_by_seed) != len(clusters_by_seed):
        raise ValueError("aligned nonempty per-seed differences and clusters are required")
    per_seed_means: list[np.ndarray] = []
    reference_clusters: np.ndarray | None = None
    for differences, clusters in zip(differences_by_seed, clusters_by_seed, strict=True):
        differences = np.asarray(differences, dtype=np.float64)
        clusters = np.asarray(clusters)
        if differences.ndim != 1 or clusters.shape != differences.shape:
            raise ValueError("each seed's differences and clusters must be aligned vectors")
        unique = np.unique(clusters)
        if len(unique) < 2:
            raise ValueError("paired interval requires at least two clusters per seed")
        if reference_clusters is None:
            reference_clusters = unique
        elif not np.array_equal(unique, reference_clusters):
            raise ValueError("training seeds do not share the same ambiguity-pair clusters")
        per_seed_means.append(
            np.asarray([differences[clusters == item].mean() for item in unique])
        )

    means = np.stack(per_seed_means)
    num_seeds, num_clusters = means.shape
    if num_seeds == 1:
        interval = clustered_paired_interval(
            np.asarray(differences_by_seed[0]),
            np.asarray(clusters_by_seed[0]),
            seed=seed,
            replicates=replicates,
        )
        return {
            **interval,
            "training_seeds": 1,
            "per_seed_difference": [float(means[0].mean())],
        }
    rng = np.random.default_rng(seed)
    sampled_seeds = rng.integers(0, num_seeds, size=(replicates, num_seeds))
    sampled_clusters = rng.integers(
        0, num_clusters, size=(replicates, num_seeds, num_clusters)
    )
    selected_seed_means = means[sampled_seeds]
    boot = np.take_along_axis(selected_seed_means, sampled_clusters, axis=2).mean(
        axis=(1, 2)
    )
    return {
        "difference": float(means.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "clusters": int(num_clusters),
        "training_seeds": int(num_seeds),
        "per_seed_difference": [float(value) for value in means.mean(axis=1)],
    }


def _load_seed(root: pathlib.Path, seed: int) -> tuple[dict, dict]:
    ambiguity, general = {}, {}
    for tag, arm in ARMS.items():
        directory = root / f"seed{seed}_{tag}"
        ambiguity_path = directory / f"{arm}_eval_ambiguity.json"
        general_path = directory / f"{arm}_eval.json"
        if not ambiguity_path.is_file() or not general_path.is_file():
            raise FileNotFoundError(f"seed {seed} arm {tag} is incomplete")
        ambiguity[tag] = json.loads(ambiguity_path.read_text())
        general[tag] = json.loads(general_path.read_text())
        if (
            ambiguity[tag].get("evaluation_seed") != 404
            or general[tag].get("evaluation_seed") != 404
        ):
            raise ValueError(f"seed {seed} arm {tag} was not evaluated at seed 404")
    reference_starts = ambiguity["explicit"]["start_steps"]
    reference_pairs = ambiguity["explicit"]["ambiguity_pair_ids"]
    reference_sides = ambiguity["explicit"]["ambiguity_sides"]
    for tag, report in ambiguity.items():
        if (
            report["start_steps"] != reference_starts
            or report["ambiguity_pair_ids"] != reference_pairs
            or report["ambiguity_sides"] != reference_sides
        ):
            raise ValueError(f"seed {seed} arm {tag} does not share the paired start grid")
    return ambiguity, general


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students_root", required=True)
    parser.add_argument("--teacher_reports_root")
    parser.add_argument(
        "--teacher_reports",
        nargs="+",
        help=(
            "exact specialist reports in loaded-motion order; defaults to the two "
            "original E67 report names under --teacher_reports_root"
        ),
    )
    parser.add_argument("--protocol", default="E67 preregistered analysis v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = pathlib.Path(args.students_root)
    if args.teacher_reports:
        teacher_paths = [pathlib.Path(path) for path in args.teacher_reports]
    elif args.teacher_reports_root:
        teacher_root = pathlib.Path(args.teacher_reports_root)
        teacher_paths = [
            teacher_root / "walk1_subject5_eval404.json",
            teacher_root / "walk3_subject1_eval404.json",
        ]
    else:
        parser.error("provide --teacher_reports or --teacher_reports_root")
    if len(teacher_paths) != 2:
        parser.error("the registered analysis requires exactly two specialist reports")
    teacher_reports = [json.loads(path.read_text()) for path in teacher_paths]
    if not all(report.get("passes_gate") for report in teacher_reports):
        raise ValueError("at least one specialist teacher failed its gate")
    teacher_macro = float(
        np.mean([report["completion_rate"] for report in teacher_reports])
    )
    loaded = {seed: _load_seed(root, seed) for seed in args.seeds}
    first_seed = args.seeds[0]
    reference_report = loaded[first_seed][0]["explicit"]
    for seed in args.seeds[1:]:
        report = loaded[seed][0]["explicit"]
        if any(
            report[key] != reference_report[key]
            for key in ("start_steps", "ambiguity_pair_ids", "ambiguity_sides")
        ):
            raise ValueError(f"seed {seed} does not share the cross-seed paired start grid")
    arm_summary = {}
    for tag in ARMS:
        reports = [loaded[seed][0][tag] for seed in args.seeds]
        arm_summary[tag] = {
            "ambiguity_completion": float(
                np.mean([report["completion_rate"] for report in reports])
            ),
            "ambiguity_survival_s": float(
                np.mean([report["mean_survival_s"] for report in reports])
            ),
            "teacher_action_rmse": float(
                np.mean([report["teacher_action_rmse"] for report in reports])
            ),
            "general_completion": float(
                np.mean([loaded[seed][1][tag]["completion_rate"] for seed in args.seeds])
            ),
        }

    def comparison(
        first: str, second: str, side: int | None = None
    ) -> dict[str, float | int | list[float]]:
        values, cluster_ids = [], []
        for seed in args.seeds:
            reports = loaded[seed][0]
            a = np.asarray(reports[first]["completed"], dtype=np.float64)
            b = np.asarray(reports[second]["completed"], dtype=np.float64)
            pairs = np.asarray(reports[first]["ambiguity_pair_ids"], dtype=np.int64)
            sides = np.asarray(reports[first]["ambiguity_sides"], dtype=np.int64)
            mask = np.ones_like(pairs, dtype=bool) if side is None else sides == side
            values.append((a - b)[mask])
            cluster_ids.append(pairs[mask])
        return hierarchical_paired_interval(values, cluster_ids)

    snmr_minus_time = comparison("snmr", "time")
    snmr_minus_shuffled = comparison("snmr", "shuffled")
    per_clip = {
        "first": comparison("snmr", "time", side=0),
        "second": comparison("snmr", "time", side=1),
    }
    per_seed = {}
    for index, seed in enumerate(args.seeds):
        per_seed[str(seed)] = {
            "snmr_minus_time": snmr_minus_time["per_seed_difference"][index],
            "snmr_minus_shuffled": snmr_minus_shuffled["per_seed_difference"][index],
            "snmr_minus_time_first_clip": per_clip["first"]["per_seed_difference"][index],
            "snmr_minus_time_second_clip": per_clip["second"]["per_seed_difference"][index],
        }
    explicit_general = arm_summary["explicit"]["general_completion"]
    explicit_gate = bool(
        explicit_general >= 0.80 or explicit_general >= teacher_macro - 0.05
    )
    positive_gate = bool(
        explicit_gate
        and snmr_minus_time["difference"] >= 0.10
        and snmr_minus_time["ci95_low"] > 0.0
        and snmr_minus_shuffled["ci95_low"] > 0.0
        and all(value["difference"] > 0.0 for value in per_clip.values())
    )
    summary = {
        "protocol": args.protocol,
        "seeds": args.seeds,
        "arms": arm_summary,
        "teacher_macro_completion": teacher_macro,
        "snmr_minus_time": snmr_minus_time,
        "snmr_minus_shuffled": snmr_minus_shuffled,
        "snmr_minus_time_per_clip": per_clip,
        "per_seed_differences": per_seed,
        "explicit_general_gate": explicit_gate,
        "positive_content_gate": positive_gate,
        "interpretation": (
            "invalid student experiment: explicit control failed"
            if not explicit_gate
            else (
                "control-usable content beyond time"
                if positive_gate
                else "positive gate not met; report the scoped interface result"
            )
        ),
        "inputs": [],
    }
    for seed in args.seeds:
        for tag, arm in ARMS.items():
            for suffix in ("_eval.json", "_eval_ambiguity.json"):
                path = root / f"seed{seed}_{tag}" / f"{arm}{suffix}"
                summary["inputs"].append(
                    {"path": str(path.resolve()), "sha256": sha256_file(path)}
                )
    for path in teacher_paths:
        summary["inputs"].append(
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
        )
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
