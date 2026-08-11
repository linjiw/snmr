#!/usr/bin/env python
"""Preregistered secondary temporal-block analysis for the frozen E70 assay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from collections import defaultdict
from typing import Any

import numpy as np


PROTOCOL = "E70 secondary temporal-block analysis v1"
PRECHECK_PROTOCOL = "E70 reference-only ambiguity precheck v1"
REGISTERED_SEEDS = (0, 1, 2)
PREVIEW_SEEDS = (0, 1)
REGISTERED_PAIRS = 69
BLOCK_SECONDS = 10.0
BOOTSTRAP_SEED = 7017
BOOTSTRAP_REPLICATES = 10_000
ARM_BASENAMES = {
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


def validate_seed_request(seeds: list[int], *, preview: bool) -> None:
    if len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be unique")
    if preview:
        if not seeds or tuple(seeds) not in ((0,), PREVIEW_SEEDS):
            raise ValueError("preview accepts exactly --seeds 0 or --seeds 0 1")
    elif tuple(seeds) != REGISTERED_SEEDS:
        raise ValueError(
            "final analysis requires exactly --seeds 0 1 2; use --preview for seed 0/1"
        )


def temporal_block_partition(
    windows: list[dict[str, Any]], *, block_seconds: float = BLOCK_SECONDS
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Partition paired windows through connected per-clip start-time blocks.

    Each pair is an edge between its first-clip and second-clip atomic blocks.  Connected
    components are the resampling units, so two pairs can never be resampled independently when
    they share an atomic temporal block on either clip.
    """
    if not windows:
        raise ValueError("at least one ambiguity window is required")
    if not math.isfinite(block_seconds) or block_seconds <= 0.0:
        raise ValueError("block_seconds must be positive and finite")

    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: tuple[str, int], second: tuple[str, int]) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    pair_nodes: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for pair_id, window in enumerate(windows):
        try:
            first_time = float(window["time_seconds_first"])
            second_time = float(window["time_seconds_second"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"window {pair_id} has invalid start times") from exc
        if not all(math.isfinite(value) and value >= 0.0 for value in (first_time, second_time)):
            raise ValueError(f"window {pair_id} has invalid start times")
        first = ("first", int(math.floor(first_time / block_seconds)))
        second = ("second", int(math.floor(second_time / block_seconds)))
        union(first, second)
        pair_nodes.append((first, second))

    component_pairs: dict[tuple[str, int], list[int]] = defaultdict(list)
    component_nodes: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for pair_id, (first, second) in enumerate(pair_nodes):
        root = find(first)
        component_pairs[root].append(pair_id)
        component_nodes[root].update((first, second))

    roots = sorted(
        component_pairs,
        key=lambda root: (
            min(index for _, index in component_nodes[root]),
            min(component_pairs[root]),
        ),
    )
    block_ids = np.empty(len(windows), dtype=np.int64)
    summaries: list[dict[str, Any]] = []
    for block_id, root in enumerate(roots):
        pairs = sorted(component_pairs[root])
        for pair_id in pairs:
            block_ids[pair_id] = block_id
        nodes = sorted(component_nodes[root], key=lambda item: (item[1], item[0]))
        summaries.append(
            {
                "block_id": block_id,
                "pair_ids": pairs,
                "num_pairs": len(pairs),
                "atomic_blocks": [
                    {
                        "clip_side": side,
                        "index": index,
                        "start_s": index * block_seconds,
                        "end_s": (index + 1) * block_seconds,
                    }
                    for side, index in nodes
                ],
            }
        )
    return block_ids, summaries


def hierarchical_temporal_block_interval(
    pair_effects_by_seed: np.ndarray,
    block_ids: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Bootstrap training seeds, then temporal blocks, retaining all pairs in each block."""
    effects = np.asarray(pair_effects_by_seed, dtype=np.float64)
    blocks = np.asarray(block_ids, dtype=np.int64)
    if effects.ndim != 2 or blocks.ndim != 1 or effects.shape[1] != len(blocks):
        raise ValueError("effects must be [seed, pair] and align with block_ids")
    if effects.shape[0] < 1 or not np.isfinite(effects).all():
        raise ValueError("finite effects for at least one training seed are required")
    if not isinstance(replicates, int) or replicates < 100:
        raise ValueError("replicates must be an integer >= 100")
    unique = np.unique(blocks)
    if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("at least two contiguous temporal block IDs are required")

    block_sums = np.stack(
        [effects[:, blocks == block_id].sum(axis=1) for block_id in unique], axis=1
    )
    block_counts = np.asarray([(blocks == block_id).sum() for block_id in unique])
    num_seeds, num_blocks = effects.shape[0], len(unique)
    rng = np.random.default_rng(seed)
    sampled_seeds = rng.integers(0, num_seeds, size=(replicates, num_seeds))
    sampled_blocks = rng.integers(
        0, num_blocks, size=(replicates, num_seeds, num_blocks)
    )
    sampled_sums = block_sums[sampled_seeds[..., None], sampled_blocks].sum(axis=2)
    sampled_counts = block_counts[sampled_blocks].sum(axis=2)
    bootstrap = (sampled_sums / sampled_counts).mean(axis=1)
    return {
        "difference": float(effects.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "training_seeds": int(num_seeds),
        "temporal_blocks": int(num_blocks),
        "pairs": int(effects.shape[1]),
        "per_seed_difference": [float(value) for value in effects.mean(axis=1)],
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(replicates),
    }


def _load_precheck(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(path.read_text())
    if report.get("protocol") != PRECHECK_PROTOCOL:
        raise ValueError("unexpected ambiguity precheck protocol")
    if report.get("loaded_motion_order") != ["walk1_subject1", "walk1_subject5"]:
        raise ValueError("ambiguity precheck has the wrong loaded-motion order")
    if float(report.get("thresholds", {}).get("rollout_seconds", -1.0)) != BLOCK_SECONDS:
        raise ValueError("registered rollout duration no longer matches the block length")
    pair_key = "walk1_subject1,walk1_subject5"
    try:
        windows = report["pairs"][pair_key]["windows"]
    except (KeyError, TypeError) as exc:
        raise ValueError("ambiguity precheck lacks the registered pair windows") from exc
    if len(windows) != REGISTERED_PAIRS:
        raise ValueError(f"expected exactly {REGISTERED_PAIRS} ambiguity windows")
    return report, windows


def _ambiguity_path(root: pathlib.Path, seed: int, tag: str) -> pathlib.Path:
    return root / f"seed{seed}_{tag}" / f"{ARM_BASENAMES[tag]}_eval_ambiguity.json"


def _load_seed_reports(
    root: pathlib.Path, seed: int, *, num_pairs: int
) -> tuple[dict[str, dict[str, Any]], list[pathlib.Path]]:
    reports: dict[str, dict[str, Any]] = {}
    paths: list[pathlib.Path] = []
    reference_grid: tuple[Any, Any, Any] | None = None
    for tag in ARM_BASENAMES:
        path = _ambiguity_path(root, seed, tag)
        if not path.is_file():
            raise FileNotFoundError(f"seed {seed} arm {tag} is incomplete: {path}")
        report = json.loads(path.read_text())
        if report.get("evaluation_seed") != 404:
            raise ValueError(f"seed {seed} arm {tag} was not evaluated at seed 404")
        required = ("start_steps", "ambiguity_pair_ids", "ambiguity_sides", "completed")
        if any(key not in report for key in required):
            raise ValueError(f"seed {seed} arm {tag} lacks per-rollout fields")
        lengths = {len(report[key]) for key in required}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2 * num_pairs:
            raise ValueError(f"seed {seed} arm {tag} has an invalid rollout grid")
        grid = tuple(report[key] for key in required[:3])
        if reference_grid is None:
            reference_grid = grid
        elif grid != reference_grid:
            raise ValueError(f"seed {seed} arm {tag} does not share the paired start grid")
        reports[tag] = report
        paths.append(path)

    assert reference_grid is not None
    pair_ids = np.asarray(reference_grid[1], dtype=np.int64)
    sides = np.asarray(reference_grid[2], dtype=np.int64)
    if set(pair_ids.tolist()) != set(range(num_pairs)):
        raise ValueError(f"seed {seed} does not contain exactly pair IDs 0..{num_pairs - 1}")
    for pair_id in range(num_pairs):
        if set(sides[pair_ids == pair_id].tolist()) != {0, 1}:
            raise ValueError(f"seed {seed} pair {pair_id} does not contain both clip sides")
    return reports, paths


def _pair_effects(
    reports: dict[str, dict[str, Any]], first: str, second: str, *, num_pairs: int
) -> np.ndarray:
    pair_ids = np.asarray(reports[first]["ambiguity_pair_ids"], dtype=np.int64)
    first_values = np.asarray(reports[first]["completed"], dtype=np.float64)
    second_values = np.asarray(reports[second]["completed"], dtype=np.float64)
    differences = first_values - second_values
    return np.asarray(
        [differences[pair_ids == pair_id].mean() for pair_id in range(num_pairs)],
        dtype=np.float64,
    )


def analyze(
    students_root: pathlib.Path,
    ambiguity_precheck: pathlib.Path,
    seeds: list[int],
    *,
    preview: bool,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    validate_seed_request(seeds, preview=preview)
    _, windows = _load_precheck(ambiguity_precheck)
    block_ids, block_summaries = temporal_block_partition(windows)

    loaded: dict[int, dict[str, dict[str, Any]]] = {}
    input_paths = [ambiguity_precheck]
    reference_grid: tuple[Any, Any, Any] | None = None
    for training_seed in seeds:
        reports, paths = _load_seed_reports(
            students_root, training_seed, num_pairs=len(windows)
        )
        grid = tuple(
            reports["snmr"][key]
            for key in ("start_steps", "ambiguity_pair_ids", "ambiguity_sides")
        )
        if reference_grid is None:
            reference_grid = grid
        elif grid != reference_grid:
            raise ValueError(f"seed {training_seed} does not share the cross-seed start grid")
        loaded[training_seed] = reports
        input_paths.extend(paths)

    contrasts = {}
    for name, second in (
        ("snmr_minus_time", "time"),
        ("snmr_minus_shuffled", "shuffled"),
    ):
        effects = np.stack(
            [
                _pair_effects(loaded[training_seed], "snmr", second, num_pairs=len(windows))
                for training_seed in seeds
            ]
        )
        interval = hierarchical_temporal_block_interval(
            effects, block_ids, replicates=replicates
        )
        interval["positive_direction"] = bool(interval["difference"] > 0.0)
        interval["ci_excludes_zero_positive"] = bool(interval["ci95_low"] > 0.0)
        contrasts[name] = interval

    return {
        "protocol": PROTOCOL,
        "analysis_status": "non-final preview" if preview else "final secondary analysis",
        "preview": preview,
        "seeds": seeds,
        "primary_verdict_effect": "none; this secondary analysis cannot change the primary gate",
        "block_definition": {
            "atomic_block_seconds": BLOCK_SECONDS,
            "origin_seconds": 0.0,
            "assignment": "floor(start_seconds / 10.0) separately for each loaded clip",
            "resampling_unit": (
                "connected components of the bipartite first-clip/second-clip atomic-block graph"
            ),
            "temporal_blocks": len(block_summaries),
            "pair_counts": [item["num_pairs"] for item in block_summaries],
            "blocks": block_summaries,
        },
        "comparisons": contrasts,
        "directionally_consistent": bool(
            all(value["positive_direction"] for value in contrasts.values())
        ),
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in input_paths
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students-root", type=pathlib.Path, required=True)
    parser.add_argument("--ambiguity-precheck", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(REGISTERED_SEEDS))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    summary = analyze(
        args.students_root,
        args.ambiguity_precheck,
        args.seeds,
        preview=args.preview,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
