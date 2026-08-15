#!/usr/bin/env python
"""Analyze the frozen E71 source-valid command-swap assay.

The primary branch coordinate is the normalized difference in squared error,

``C = (Q_A - Q_B) / (Q_AB + 1e-8)``.

For a fixed physical start and frozen controller, replacing command A with command B
should increase ``C``.  Explicit-reference controllers are used only to freeze the set
of starts on which both source-valid commands choose their corresponding branches.
The SNMR interval conditions on the three frozen controllers and resamples the frozen
temporal components with one common set of component weights across controllers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import pathlib
import uuid
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL = "E71 same-state valid-command swap v1"
GATE_PROTOCOL = "E71 explicit feasibility gate v1"
PRIMARY_SCORE = "branch_coordinate_1p0_s"
REGISTERED_TRAINING_SEEDS = (0, 1, 2)
EVALUATION_SEED = 404
EXPECTED_PAIRS = 69
EXPECTED_TEMPORAL_COMPONENTS = 12
PRECHECK_PROTOCOL = "E70 reference-only ambiguity precheck v1"
EXPECTED_CLIPS = ("walk1_subject1", "walk1_subject5")
MIN_ELIGIBLE_PAIRS = 20
MIN_ELIGIBLE_TEMPORAL_COMPONENTS = 6
BLOCK_SECONDS = 10.0
MOTION_FPS = 50.0
BOOTSTRAP_SEED = 7104
BOOTSTRAP_REPLICATES = 10_000
COORDINATE_EPSILON = 1.0e-8

ARM_TO_STUDENT_ARM = {
    "explicit": "c_prior_explicit",
    "snmr": "a_prior_snmr",
}
GRID_FIELDS = (
    "pair_ids",
    "state_sides",
    "command_sides",
    "state_start_steps",
    "command_start_steps",
    "state_motion_ids",
    "command_motion_ids",
)
EXPECTED_EVALUATION_CONDITIONS: dict[str, Any] = {
    "actor_observation_noise": False,
    "startup_physics_randomization": False,
    "reset_state_randomization": False,
    "external_pushes": False,
    "action_delay": False,
    "pd_gain_randomization": False,
    "torque_rfi": False,
    "initial_pose_noise_scale": 0.0,
    "tracking_termination_suppressed_steps": 50,
    "reference_termination_after_primary_horizon": True,
    "nonfinite_state_guard": True,
    "mujoco_warp_overflow_guard": True,
    "rollout_task_update_before_termination": False,
    "passed": True,
}
COMMON_PROVENANCE_FIELDS = (
    "teacher_manifest",
    "teacher_manifest_sha256",
    "teacher_ckpts",
    "teacher_checkpoint_sha256",
    "motion_files",
    "motion_sha256",
    "ambiguity_precheck",
    "ambiguity_precheck_sha256",
    "runtime",
    "runtime_sha256",
    "reset_runtime",
    "reset_runtime_sha256",
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(report: Mapping[str, Any], key: str, dtype: Any) -> np.ndarray:
    if key not in report:
        raise ValueError(f"report is missing required field {key}")
    values = np.asarray(report[key], dtype=dtype)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{key} must be a nonempty vector")
    return values


def _integer_vector(report: Mapping[str, Any], key: str) -> np.ndarray:
    raw = report.get(key)
    if not isinstance(raw, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in raw
    ):
        raise ValueError(f"{key} must be a vector of integers")
    return _vector(report, key, np.int64)


def _finite_vector(report: Mapping[str, Any], key: str) -> np.ndarray:
    raw = report.get(key)
    if not isinstance(raw, list) or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) for value in raw
    ):
        raise ValueError(f"{key} must be a vector of numbers")
    values = _vector(report, key, np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must contain only finite values")
    return values


def _finite_matrix(
    report: Mapping[str, Any], key: str, *, rows: int, columns: int
) -> np.ndarray:
    raw = report.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be a finite {rows} x {columns} matrix")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite {rows} x {columns} matrix") from exc
    if values.shape != (rows, columns) or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must be a finite {rows} x {columns} matrix")
    return values


def _validate_difference_audit(report: Mapping[str, Any], key: str) -> None:
    audit = report.get(key)
    if not isinstance(audit, Mapping) or audit.get("passed") is not True:
        raise ValueError(f"{key} did not pass")
    try:
        maximum = float(audit["max_abs_difference"])
        tolerance = float(audit["tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} lacks a valid difference bound") from exc
    if (
        not math.isfinite(maximum)
        or not math.isfinite(tolerance)
        or maximum < 0.0
        or tolerance < 0.0
        or maximum > tolerance
    ):
        raise ValueError(f"{key} has an invalid or failed difference bound")


def _validate_full_state_audit(report: Mapping[str, Any]) -> None:
    audit = report.get("full_state_audit")
    if not isinstance(audit, Mapping) or audit.get("passed") is not True:
        raise ValueError("full_state_audit did not pass")
    for key in ("num_tensors", "num_tensor_state_comparisons"):
        value = audit.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"full_state_audit {key} must be a positive integer")
    per_tensor = audit.get("per_tensor")
    if not isinstance(per_tensor, Mapping) or not per_tensor:
        raise ValueError("full_state_audit must contain per-tensor results")
    try:
        maximum = float(audit["max_abs_difference"])
        tolerance = float(audit["tolerance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("full_state_audit lacks a valid difference bound") from exc
    if (
        not math.isfinite(maximum)
        or not math.isfinite(tolerance)
        or maximum < 0.0
        or tolerance < 0.0
        or maximum > tolerance
    ):
        raise ValueError("full_state_audit has an invalid or failed difference bound")
    for name, item in per_tensor.items():
        if not isinstance(name, str) or not name or not isinstance(item, Mapping):
            raise ValueError("full_state_audit contains an invalid per-tensor record")
        item_maximum = item.get("max_abs_difference")
        comparisons = item.get("num_state_comparisons")
        if (
            not isinstance(item_maximum, (int, float))
            or isinstance(item_maximum, bool)
            or not math.isfinite(float(item_maximum))
            or float(item_maximum) < 0.0
            or float(item_maximum) > tolerance
            or not isinstance(comparisons, int)
            or isinstance(comparisons, bool)
            or comparisons <= 0
            or item.get("passed") is not True
        ):
            raise ValueError(f"full_state_audit tensor {name!r} did not pass")


def _validate_evaluation_conditions(report: Mapping[str, Any]) -> None:
    conditions = report.get("evaluation_conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("evaluation_conditions must be a mapping")
    for key, expected in EXPECTED_EVALUATION_CONDITIONS.items():
        observed = conditions.get(key)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, int):
            matches = (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed == expected
            )
        else:
            matches = (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and float(observed) == expected
            )
        if not matches:
            raise ValueError(
                f"evaluation_conditions {key} must be exactly {expected!r}"
            )


def _validate_provenance_shape(report: Mapping[str, Any]) -> None:
    digest_fields = (
        "student_checkpoint_sha256",
        "teacher_manifest_sha256",
        "ambiguity_precheck_sha256",
        "runtime_sha256",
        "reset_runtime_sha256",
    )
    for key in digest_fields:
        value = report.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{key} must be a SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"{key} must be a SHA-256 digest") from exc
    for key in (
        "student_checkpoint",
        "teacher_manifest",
        "ambiguity_precheck",
        "runtime",
        "reset_runtime",
    ):
        if not isinstance(report.get(key), str) or not report[key]:
            raise ValueError(f"{key} must be a nonempty path")
    for path_key, digest_key in (
        ("teacher_ckpts", "teacher_checkpoint_sha256"),
        ("motion_files", "motion_sha256"),
    ):
        paths = report.get(path_key)
        digests = report.get(digest_key)
        if (
            not isinstance(paths, list)
            or not isinstance(digests, list)
            or len(paths) != 2
            or len(digests) != 2
            or any(not isinstance(path, str) or not path for path in paths)
            or any(not isinstance(value, str) or len(value) != 64 for value in digests)
        ):
            raise ValueError(f"{path_key} and {digest_key} must contain two records")
        try:
            for value in digests:
                int(value, 16)
        except ValueError as exc:
            raise ValueError(f"{digest_key} must contain SHA-256 digests") from exc


def validate_report(report: dict[str, Any], *, score_key: str = PRIMARY_SCORE) -> None:
    """Fail closed on the frozen four-cell schema, audits, and branch metric."""

    if score_key != PRIMARY_SCORE:
        raise ValueError(f"the frozen E71 primary score is {PRIMARY_SCORE}")
    if report.get("protocol") != PROTOCOL:
        raise ValueError(f"report protocol must be {PROTOCOL!r}")
    arm = report.get("arm")
    if arm not in ARM_TO_STUDENT_ARM:
        raise ValueError("report arm must be canonical explicit or snmr")
    if report.get("student_arm") != ARM_TO_STUDENT_ARM[arm]:
        raise ValueError("report student_arm does not match its canonical arm")
    seed = report.get("training_seed")
    if seed not in REGISTERED_TRAINING_SEEDS or isinstance(seed, bool):
        raise ValueError("training_seed must be one of 0, 1, or 2")
    if report.get("evaluation_seed") != EVALUATION_SEED:
        raise ValueError("evaluation_seed must be 404")

    grid = {key: _integer_vector(report, key) for key in GRID_FIELDS}
    lengths = {len(values) for values in grid.values()}
    if lengths != {EXPECTED_PAIRS * 4}:
        raise ValueError(f"grid fields must each contain {EXPECTED_PAIRS * 4} cells")
    if report.get("num_rollouts") != EXPECTED_PAIRS * 4:
        raise ValueError("num_rollouts does not match the frozen four-cell grid")
    _finite_matrix(
        report,
        "first_student_action",
        rows=EXPECTED_PAIRS * 4,
        columns=29,
    )
    _finite_matrix(
        report,
        "first_teacher_action",
        rows=EXPECTED_PAIRS * 4,
        columns=29,
    )
    pair_ids = grid["pair_ids"]
    states = grid["state_sides"]
    commands = grid["command_sides"]
    if set(np.unique(pair_ids)) != set(range(EXPECTED_PAIRS)):
        raise ValueError("pair IDs must be exactly 0 through 68")
    if set(np.unique(states)) != {0, 1} or set(np.unique(commands)) != {0, 1}:
        raise ValueError("state and command sides must each contain exactly 0 and 1")
    if not np.array_equal(grid["state_motion_ids"], states):
        raise ValueError("state motion IDs must match state sides")
    if not np.array_equal(grid["command_motion_ids"], commands):
        raise ValueError("command motion IDs must match command sides")
    if np.any(grid["state_start_steps"] < 0) or np.any(
        grid["command_start_steps"] < 0
    ):
        raise ValueError("state and command cursors must be nonnegative")

    cells = np.stack((pair_ids, states, commands), axis=1)
    unique, counts = np.unique(cells, axis=0, return_counts=True)
    if not np.all(counts == 1) or len(unique) != EXPECTED_PAIRS * 4:
        raise ValueError("each pair must contain exactly one state x command cell")
    for pair_id in range(EXPECTED_PAIRS):
        pair_mask = pair_ids == pair_id
        pair_cells = unique[unique[:, 0] == pair_id, 1:]
        if set(map(tuple, pair_cells.tolist())) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError(f"pair {pair_id} does not contain the complete four-cell grid")
        state_steps: dict[int, int] = {}
        command_steps: dict[int, int] = {}
        for side in (0, 1):
            observed_state = np.unique(
                grid["state_start_steps"][pair_mask & (states == side)]
            )
            observed_command = np.unique(
                grid["command_start_steps"][pair_mask & (commands == side)]
            )
            if len(observed_state) != 1 or len(observed_command) != 1:
                raise ValueError(f"pair {pair_id} does not keep cursors fixed by side")
            state_steps[side] = int(observed_state[0])
            command_steps[side] = int(observed_command[0])
        if state_steps != command_steps:
            raise ValueError(f"pair {pair_id} state and command branch cursors disagree")

    metric_vectors: dict[str, np.ndarray] = {}
    for horizon, samples in (("0p5_s", 25), ("1p0_s", 50)):
        q_a = _finite_vector(report, f"q_a_{horizon}")
        q_b = _finite_vector(report, f"q_b_{horizon}")
        q_ab = _finite_vector(report, f"q_ab_{horizon}")
        coordinate = _finite_vector(report, f"branch_coordinate_{horizon}")
        d_a = _finite_vector(report, f"d_a_{horizon}")
        d_b = _finite_vector(report, f"d_b_{horizon}")
        if any(
            len(values) != len(pair_ids)
            for values in (q_a, q_b, q_ab, coordinate, d_a, d_b)
        ):
            raise ValueError(f"branch metric vectors at {horizon} do not align with the grid")
        if np.any(q_a < 0.0) or np.any(q_b < 0.0) or np.any(q_ab <= 0.0):
            raise ValueError(
                f"branch squared errors at {horizon} must be nonnegative with q_ab > 0"
            )
        expected_coordinate = (q_a - q_b) / (q_ab + COORDINATE_EPSILON)
        if not np.allclose(coordinate, expected_coordinate, rtol=1.0e-7, atol=1.0e-9):
            raise ValueError(
                f"branch coordinate at {horizon} is inconsistent with (q_a-q_b)/(q_ab+1e-8)"
            )
        if not np.allclose(d_a, np.sqrt(q_a), rtol=1.0e-7, atol=1.0e-9) or not np.allclose(
            d_b, np.sqrt(q_b), rtol=1.0e-7, atol=1.0e-9
        ):
            raise ValueError(f"branch distances at {horizon} are inconsistent with sqrt(q)")
        if report.get(f"branch_samples_{horizon}") != samples:
            raise ValueError(f"branch_samples_{horizon} must be future-only {samples}")
        for pair_id in range(EXPECTED_PAIRS):
            pair_q_ab = q_ab[pair_ids == pair_id]
            if not np.allclose(pair_q_ab, pair_q_ab[0], rtol=0.0, atol=1.0e-12):
                raise ValueError(f"pair {pair_id} q_ab at {horizon} changed across cells")
        metric_vectors[horizon] = coordinate

    completed = report.get("completed")
    if (
        not isinstance(completed, list)
        or len(completed) != len(pair_ids)
        or any(
            not (
                isinstance(value, bool)
                or (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value in (0, 1)
                )
            )
            for value in completed
        )
    ):
        raise ValueError("completed must be a boolean four-cell vector")
    survival = _finite_vector(report, "survival_s")
    if len(survival) != len(pair_ids) or np.any(survival < 0.0) or np.any(survival > 10.0):
        raise ValueError("survival_s must align with the grid and lie in [0, 10]")

    _validate_difference_audit(report, "proprio_audit")
    _validate_difference_audit(report, "raw_proprio_audit")
    _validate_full_state_audit(report)
    warmup = report.get("warmup_audit")
    if not isinstance(warmup, Mapping) or warmup.get("passed") is not True:
        raise ValueError("warmup_audit did not pass")
    termination = report.get("termination_audit")
    if (
        not isinstance(termination, Mapping)
        or termination.get("passed") is not True
        or termination.get("primary_horizon_done_count") != 0
        or termination.get("suppressed_steps") != 50
    ):
        raise ValueError("termination_audit did not prove an uncensored primary horizon")
    _validate_evaluation_conditions(report)
    try:
        policy_dt = float(report["policy_dt_s"])
        motion_fps = float(report["motion_fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("policy_dt_s and motion_fps must be finite numbers") from exc
    if not math.isfinite(policy_dt) or policy_dt != 0.02:
        raise ValueError("policy_dt_s must be 0.02")
    if not math.isfinite(motion_fps) or motion_fps != MOTION_FPS:
        raise ValueError("motion_fps must be 50")
    _validate_provenance_shape(report)


def _grid_signature(report: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in report[key]) for key in GRID_FIELDS)


def _common_provenance_signature(report: Mapping[str, Any]) -> str:
    return json.dumps(
        {key: report.get(key) for key in COMMON_PROVENANCE_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_report_set(
    reports: Sequence[dict[str, Any]], *, expected_arm: str
) -> None:
    if len(reports) != len(REGISTERED_TRAINING_SEEDS):
        raise ValueError(f"{expected_arm} requires exactly three frozen reports")
    for report in reports:
        validate_report(report)
        if report.get("arm") != expected_arm:
            raise ValueError(f"expected only canonical {expected_arm} reports")
    seeds = tuple(int(report["training_seed"]) for report in reports)
    if seeds != REGISTERED_TRAINING_SEEDS:
        raise ValueError(f"{expected_arm} reports must be ordered as seeds 0, 1, and 2")
    reference_grid = _grid_signature(reports[0])
    reference_provenance = _common_provenance_signature(reports[0])
    for report in reports[1:]:
        if _grid_signature(report) != reference_grid:
            raise ValueError(f"{expected_arm} reports do not share the exact frozen grid")
        if _common_provenance_signature(report) != reference_provenance:
            raise ValueError(f"{expected_arm} reports do not share frozen provenance")


def _cell_matrix(report: dict[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted pair IDs and a ``pair x state x command`` value matrix."""

    validate_report(report)
    pairs = _integer_vector(report, "pair_ids")
    states = _integer_vector(report, "state_sides")
    commands = _integer_vector(report, "command_sides")
    values = _finite_vector(report, key)
    if len(values) != len(pairs):
        raise ValueError(f"{key} does not align with the four-cell grid")
    unique_pairs = np.unique(pairs)
    matrix = np.empty((len(unique_pairs), 2, 2), dtype=np.float64)
    for pair_index, pair_id in enumerate(unique_pairs):
        for state_side in (0, 1):
            for command_side in (0, 1):
                mask = (
                    (pairs == pair_id)
                    & (states == state_side)
                    & (commands == command_side)
                )
                indexes = np.flatnonzero(mask)
                if len(indexes) != 1:
                    raise ValueError("four-cell lookup is not unique")
                matrix[pair_index, state_side, command_side] = values[indexes[0]]
    return unique_pairs, matrix


def command_swap_effects(
    report: dict[str, Any], *, score_key: str = PRIMARY_SCORE
) -> tuple[np.ndarray, np.ndarray]:
    """Return pair IDs and ``pair x state`` command-B-minus-command-A effects."""

    if score_key != PRIMARY_SCORE:
        raise ValueError(f"the frozen E71 primary score is {PRIMARY_SCORE}")
    pairs, coordinates = _cell_matrix(report, score_key)
    return pairs, coordinates[:, :, 1] - coordinates[:, :, 0]


def first_action_teacher_alignment(
    report: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return commanded-teacher closeness margins as ``pair x state x command``.

    Positive values mean the student's first action is closer to the routed teacher
    for the supplied command than to the opposite-command teacher evaluated from the
    same physical start side.
    """

    validate_report(report)
    pairs = _integer_vector(report, "pair_ids")
    states = _integer_vector(report, "state_sides")
    commands = _integer_vector(report, "command_sides")
    student = _finite_matrix(
        report, "first_student_action", rows=EXPECTED_PAIRS * 4, columns=29
    )
    teacher = _finite_matrix(
        report, "first_teacher_action", rows=EXPECTED_PAIRS * 4, columns=29
    )
    unique_pairs = np.unique(pairs)
    margins = np.empty((len(unique_pairs), 2, 2), dtype=np.float64)
    for pair_index, pair_id in enumerate(unique_pairs):
        for state_side in (0, 1):
            indexes: dict[int, int] = {}
            for command_side in (0, 1):
                matched = np.flatnonzero(
                    (pairs == pair_id)
                    & (states == state_side)
                    & (commands == command_side)
                )
                if len(matched) != 1:
                    raise ValueError("first-action four-cell lookup is not unique")
                indexes[command_side] = int(matched[0])
            for command_side in (0, 1):
                cell = indexes[command_side]
                commanded = np.linalg.norm(student[cell] - teacher[cell])
                opposite = np.linalg.norm(
                    student[cell] - teacher[indexes[1 - command_side]]
                )
                margins[pair_index, state_side, command_side] = opposite - commanded
    return unique_pairs, margins


def _temporal_component_partition(
    report: dict[str, Any], *, block_seconds: float = BLOCK_SECONDS
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Reproduce E70's connected two-clip temporal-block partition exactly."""

    validate_report(report)
    if not math.isfinite(block_seconds) or block_seconds <= 0.0:
        raise ValueError("block_seconds must be positive and finite")
    precheck_path = pathlib.Path(report["ambiguity_precheck"]).resolve()
    if not precheck_path.is_file():
        raise ValueError(f"frozen ambiguity precheck is missing: {precheck_path}")
    if sha256_file(precheck_path) != report["ambiguity_precheck_sha256"]:
        raise ValueError("frozen ambiguity precheck hash does not match the report")
    try:
        precheck = json.loads(precheck_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen ambiguity precheck is not valid JSON") from exc
    pair_key = ",".join(EXPECTED_CLIPS)
    thresholds = precheck.get("thresholds")
    try:
        rollout_seconds = float(
            thresholds["rollout_seconds"] if isinstance(thresholds, Mapping) else math.nan
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen ambiguity precheck has an invalid rollout duration") from exc
    if (
        precheck.get("protocol") != PRECHECK_PROTOCOL
        or precheck.get("preferred_pair") != pair_key
        or precheck.get("loaded_motion_order") != list(EXPECTED_CLIPS)
        or not math.isfinite(rollout_seconds)
        or rollout_seconds != BLOCK_SECONDS
    ):
        raise ValueError("frozen ambiguity precheck changed its registered contract")
    pair_report = precheck.get("pairs", {}).get(pair_key)
    windows = pair_report.get("windows") if isinstance(pair_report, Mapping) else None
    if not isinstance(windows, list) or len(windows) != EXPECTED_PAIRS:
        raise ValueError("frozen ambiguity precheck must contain exactly 69 windows")
    pair_ids = _integer_vector(report, "pair_ids")
    states = _integer_vector(report, "state_sides")
    starts = _integer_vector(report, "state_start_steps")

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
    inferred_motion_offsets: list[set[int]] = [set(), set()]
    for pair_id, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise ValueError(f"ambiguity window {pair_id} must be a mapping")
        try:
            first_time = float(window["time_seconds_first"])
            second_time = float(window["time_seconds_second"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"ambiguity window {pair_id} has invalid start times") from exc
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (first_time, second_time)
        ):
            raise ValueError(f"ambiguity window {pair_id} has invalid start times")
        for side, local_time in enumerate((first_time, second_time)):
            observed = np.unique(starts[(pair_ids == pair_id) & (states == side)])
            if len(observed) != 1:
                raise ValueError(f"pair {pair_id} does not have one start per state side")
            inferred_motion_offsets[side].add(
                int(observed[0]) - int(round(local_time * MOTION_FPS))
            )
        first = (
            "first",
            int(math.floor(first_time / block_seconds)),
        )
        second = (
            "second",
            int(math.floor(second_time / block_seconds)),
        )
        union(first, second)
        pair_nodes.append((first, second))
    if inferred_motion_offsets[0] != {0} or len(inferred_motion_offsets[1]) != 1:
        raise ValueError("report cursors do not match the frozen local-time ambiguity windows")

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
    component_ids = np.empty(EXPECTED_PAIRS, dtype=np.int64)
    summaries: list[dict[str, Any]] = []
    for component_id, root in enumerate(roots):
        pairs = sorted(component_pairs[root])
        component_ids[pairs] = component_id
        nodes = sorted(component_nodes[root], key=lambda item: (item[1], item[0]))
        summaries.append(
            {
                "component_id": component_id,
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
    return component_ids, summaries


def eligible_pairs(
    explicit_reports: Sequence[dict[str, Any]],
    *,
    score_key: str = PRIMARY_SCORE,
) -> dict[str, Any]:
    """Freeze starts where explicit commands choose both branches in >=2/3 seeds."""

    if score_key != PRIMARY_SCORE:
        raise ValueError(f"the frozen E71 primary score is {PRIMARY_SCORE}")
    _validate_report_set(explicit_reports, expected_arm="explicit")
    pairs_by_seed: list[np.ndarray] = []
    coordinates_by_seed: list[np.ndarray] = []
    for report in explicit_reports:
        pairs, coordinates = _cell_matrix(report, score_key)
        pairs_by_seed.append(pairs)
        coordinates_by_seed.append(coordinates)
    reference = pairs_by_seed[0]
    if any(not np.array_equal(pairs, reference) for pairs in pairs_by_seed[1:]):
        raise ValueError("explicit seeds do not share the same pair grid")

    coordinates = np.stack(coordinates_by_seed)  # seed x pair x state x command
    required_seeds = int(math.ceil(2 * len(explicit_reports) / 3))
    source_valid_selection = (coordinates[:, :, :, 0] < 0.0) & (
        coordinates[:, :, :, 1] > 0.0
    )
    selection_seed_count_by_state = source_valid_selection.sum(axis=0)
    selection_pass_by_state = selection_seed_count_by_state >= required_seeds
    passed = selection_pass_by_state.all(axis=1)

    component_ids, component_summaries = _temporal_component_partition(explicit_reports[0])
    if len(component_summaries) != EXPECTED_TEMPORAL_COMPONENTS:
        raise ValueError(
            "the frozen E71 grid must reproduce exactly 12 E70 temporal components"
        )
    eligible_component_ids = np.unique(component_ids[passed])
    return {
        "pair_ids": reference.tolist(),
        "eligible_pair_ids": reference[passed].tolist(),
        "num_pairs": int(len(reference)),
        "num_eligible_pairs": int(passed.sum()),
        "required_explicit_seeds": required_seeds,
        "selection_rule": (
            "within each physical state side, C(command A)<0 and C(command B)>0 "
            "jointly in at least two of three explicit seeds"
        ),
        "source_valid_selection_seed_count_by_state": selection_seed_count_by_state.tolist(),
        "source_valid_selection_pass_by_state": selection_pass_by_state.tolist(),
        "explicit_mean_coordinate_by_state_and_command": coordinates.mean(axis=0).tolist(),
        "temporal_component_ids": component_ids.tolist(),
        "temporal_components": component_summaries,
        "num_temporal_components": int(len(component_summaries)),
        "eligible_temporal_component_ids": eligible_component_ids.tolist(),
        "num_eligible_temporal_components": int(len(eligible_component_ids)),
    }


def _validate_interval_inputs(
    per_seed_pair_effects: np.ndarray, component_ids: np.ndarray, replicates: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    effects = np.asarray(per_seed_pair_effects, dtype=np.float64)
    components = np.asarray(component_ids, dtype=np.int64)
    if effects.ndim != 2 or effects.shape[0] != 3 or effects.shape[1] < 2:
        raise ValueError("interval requires a 3 x pair matrix with at least two pairs")
    if components.ndim != 1 or len(components) != effects.shape[1]:
        raise ValueError("component_ids must align with the pair dimension")
    if not np.all(np.isfinite(effects)):
        raise ValueError("interval effects must be finite")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 100:
        raise ValueError("replicates must be an integer >= 100")
    unique = np.unique(components)
    if len(unique) < 2:
        raise ValueError("at least two eligible temporal components are required")
    return effects, components, unique


def _component_sums(
    effects: np.ndarray, components: np.ndarray, unique: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.stack(
        [effects[:, components == component].sum(axis=1) for component in unique],
        axis=1,
    )
    counts = np.asarray(
        [(components == component).sum() for component in unique], dtype=np.int64
    )
    return sums, counts


def fixed_controller_temporal_component_interval(
    per_seed_pair_effects: np.ndarray,
    component_ids: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Condition on three controllers; resample common temporal components."""

    effects, components, unique = _validate_interval_inputs(
        per_seed_pair_effects, component_ids, replicates
    )
    block_sums, block_counts = _component_sums(effects, components, unique)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(replicates, len(unique)))
    sampled_sums = np.take(block_sums, sampled, axis=1).sum(axis=2)
    sampled_counts = block_counts[sampled].sum(axis=1)
    per_seed_bootstrap = sampled_sums / sampled_counts[None, :]
    bootstrap = per_seed_bootstrap.mean(axis=0)
    return {
        "difference": float(effects.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "training_seeds": int(effects.shape[0]),
        "controllers_conditioned_fixed": True,
        "pairs": int(effects.shape[1]),
        "temporal_components": int(len(unique)),
        "per_seed_difference": effects.mean(axis=1).tolist(),
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(replicates),
        "resampling": "common temporal-component weights across all fixed controllers",
    }


def crossed_seed_temporal_component_interval(
    per_seed_pair_effects: np.ndarray,
    component_ids: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Sensitivity only: resample seeds and common temporal components as crossed factors."""

    effects, components, unique = _validate_interval_inputs(
        per_seed_pair_effects, component_ids, replicates
    )
    block_sums, block_counts = _component_sums(effects, components, unique)
    rng = np.random.default_rng(seed)
    sampled_components = rng.integers(
        0, len(unique), size=(replicates, len(unique))
    )
    sampled_sums = np.take(block_sums, sampled_components, axis=1).sum(axis=2)
    sampled_counts = block_counts[sampled_components].sum(axis=1)
    per_seed_bootstrap = (sampled_sums / sampled_counts[None, :]).T
    sampled_seeds = rng.integers(
        0, effects.shape[0], size=(replicates, effects.shape[0])
    )
    bootstrap = np.take_along_axis(
        per_seed_bootstrap, sampled_seeds, axis=1
    ).mean(axis=1)
    return {
        "difference": float(effects.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "training_seeds": int(effects.shape[0]),
        "pairs": int(effects.shape[1]),
        "temporal_components": int(len(unique)),
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(replicates),
        "resampling": "crossed seeds and common temporal-component weights",
        "confirmatory": False,
    }


def hierarchical_interval(
    per_seed_pair_effects: np.ndarray,
    component_ids: np.ndarray | None = None,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Compatibility alias for the fixed-controller component interval.

    Pair-wise nested resampling is intentionally no longer supported.  Callers must
    provide the frozen temporal-component ID for every pair.
    """

    if component_ids is None:
        raise ValueError("component_ids are required; pair-wise hierarchical bootstrap was retired")
    return fixed_controller_temporal_component_interval(
        per_seed_pair_effects,
        component_ids,
        seed=seed,
        replicates=replicates,
    )


def analyze(
    explicit_reports: Sequence[dict[str, Any]],
    snmr_reports: Sequence[dict[str, Any]],
    *,
    score_key: str = PRIMARY_SCORE,
    min_eligible_pairs: int = MIN_ELIGIBLE_PAIRS,
    min_eligible_temporal_components: int = MIN_ELIGIBLE_TEMPORAL_COMPONENTS,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Run the frozen explicit screen and conditional SNMR confirmatory analysis."""

    if score_key != PRIMARY_SCORE:
        raise ValueError(f"the frozen E71 primary score is {PRIMARY_SCORE}")
    if min_eligible_pairs != MIN_ELIGIBLE_PAIRS:
        raise ValueError("the minimum eligible-pair threshold is frozen at 20")
    if min_eligible_temporal_components != MIN_ELIGIBLE_TEMPORAL_COMPONENTS:
        raise ValueError("the minimum eligible-component threshold is frozen at 6")
    _validate_report_set(explicit_reports, expected_arm="explicit")
    _validate_report_set(snmr_reports, expected_arm="snmr")
    if _grid_signature(explicit_reports[0]) != _grid_signature(snmr_reports[0]):
        raise ValueError("explicit and SNMR reports do not share the exact frozen grid")
    if _common_provenance_signature(explicit_reports[0]) != _common_provenance_signature(
        snmr_reports[0]
    ):
        raise ValueError("explicit and SNMR reports do not share frozen provenance")

    gate = eligible_pairs(explicit_reports, score_key=score_key)
    eligible = np.asarray(gate["eligible_pair_ids"], dtype=np.int64)
    valid_gate = (
        len(eligible) >= MIN_ELIGIBLE_PAIRS
        and gate["num_eligible_temporal_components"]
        >= MIN_ELIGIBLE_TEMPORAL_COMPONENTS
    )
    summary: dict[str, Any] = {
        "protocol": PROTOCOL,
        "primary_score": score_key,
        "primary_estimand": (
            "mean within-controller C(command B)-C(command A), averaged equally "
            "over the two physical start sides, three frozen controllers, and "
            "explicit-feasible pairs"
        ),
        "explicit_feasibility_gate": gate,
        "minimum_eligible_pairs": MIN_ELIGIBLE_PAIRS,
        "minimum_eligible_temporal_components": MIN_ELIGIBLE_TEMPORAL_COMPONENTS,
        "valid_explicit_gate": bool(valid_gate),
        "snmr_command_swap": None,
        "secondary_first_action_teacher_alignment": None,
        "positive_shift_gate": False,
        "positive_selection_gate": False,
        "positive_target_specific_gate": False,
    }
    if not valid_gate:
        summary["interpretation"] = (
            "invalid command-swap assay: explicit feasibility pair/component gate failed"
        )
        return summary

    coordinates_by_seed: list[np.ndarray] = []
    action_margins_by_seed: list[np.ndarray] = []
    reference_pairs: np.ndarray | None = None
    for report in snmr_reports:
        pairs, coordinates = _cell_matrix(report, score_key)
        action_pairs, action_margins = first_action_teacher_alignment(report)
        if not np.array_equal(pairs, action_pairs):
            raise ValueError("SNMR branch and first-action pair grids differ")
        if reference_pairs is None:
            reference_pairs = pairs
        elif not np.array_equal(pairs, reference_pairs):
            raise ValueError("SNMR seeds do not share the same pair grid")
        coordinates_by_seed.append(coordinates)
        action_margins_by_seed.append(action_margins)
    assert reference_pairs is not None
    pair_index = {int(pair): offset for offset, pair in enumerate(reference_pairs)}
    if any(int(pair) not in pair_index for pair in eligible):
        raise ValueError("SNMR report is missing an explicit-eligible pair")
    selected_indexes = [pair_index[int(pair)] for pair in eligible]
    coordinates = np.stack(coordinates_by_seed)[:, selected_indexes]
    state_effects = coordinates[:, :, :, 1] - coordinates[:, :, :, 0]
    pair_effects = state_effects.mean(axis=2)

    all_component_ids = np.asarray(gate["temporal_component_ids"], dtype=np.int64)
    selected_component_ids = all_component_ids[eligible]
    primary_interval = fixed_controller_temporal_component_interval(
        pair_effects, selected_component_ids, replicates=replicates
    )
    sensitivity_interval = crossed_seed_temporal_component_interval(
        pair_effects, selected_component_ids, replicates=replicates
    )
    action_margins = np.stack(action_margins_by_seed)[:, selected_indexes]
    action_pair_margins = action_margins.mean(axis=(2, 3))
    action_primary_interval = fixed_controller_temporal_component_interval(
        action_pair_margins, selected_component_ids, replicates=replicates
    )
    action_sensitivity_interval = crossed_seed_temporal_component_interval(
        action_pair_margins, selected_component_ids, replicates=replicates
    )
    action_margin_by_seed_and_state = action_margins.mean(axis=(1, 3))
    positive_action_alignment = bool(
        action_primary_interval["ci95_low"] > 0.0
        and np.all(action_margin_by_seed_and_state > 0.0)
    )

    per_seed_effect_by_state = state_effects.mean(axis=1)
    mean_effect_by_state = state_effects.mean(axis=(0, 1))
    positive_by_seed_and_state = per_seed_effect_by_state > 0.0
    shift = bool(
        primary_interval["ci95_low"] > 0.0 and np.all(positive_by_seed_and_state)
    )

    mean_coordinate_by_state_and_command = coordinates.mean(axis=(0, 1))
    mean_coordinate_by_command = coordinates.mean(axis=(0, 1, 2))
    selection_sign_by_state = (
        (mean_coordinate_by_state_and_command[:, 0] < 0.0)
        & (mean_coordinate_by_state_and_command[:, 1] > 0.0)
    )
    selection = bool(shift and np.all(selection_sign_by_state))
    expected_branch_choice = np.stack(
        (coordinates[:, :, :, 0] < 0.0, coordinates[:, :, :, 1] > 0.0),
        axis=-1,
    )

    summary.update(
        {
            "snmr_command_swap": {
                **primary_interval,
                "fixed_controller_temporal_component_interval": primary_interval,
                "crossed_seed_temporal_component_sensitivity": sensitivity_interval,
                "mean_effect_by_state_side": mean_effect_by_state.tolist(),
                "per_seed_effect_by_state_side": per_seed_effect_by_state.tolist(),
                "positive_direction_by_seed_and_state_side": (
                    positive_by_seed_and_state.tolist()
                ),
                "mean_coordinate_by_state_and_command": (
                    mean_coordinate_by_state_and_command.tolist()
                ),
                "mean_coordinate_by_command": mean_coordinate_by_command.tolist(),
                "selection_sign_pass_by_state_side": selection_sign_by_state.tolist(),
                "branch_choice_rate_by_command": expected_branch_choice.mean(
                    axis=(0, 1, 2)
                ).tolist(),
                "branch_choice_rate_by_state_and_command": expected_branch_choice.mean(
                    axis=(0, 1)
                ).tolist(),
                "branch_choice_rate_by_seed_state_and_command": expected_branch_choice.mean(
                    axis=1
                ).tolist(),
                "eligible_pair_ids": eligible.tolist(),
                "eligible_temporal_component_ids": np.unique(
                    selected_component_ids
                ).tolist(),
            },
            "secondary_first_action_teacher_alignment": {
                **action_primary_interval,
                "fixed_controller_temporal_component_interval": action_primary_interval,
                "crossed_seed_temporal_component_sensitivity": (
                    action_sensitivity_interval
                ),
                "margin_definition": (
                    "||a_student-a_teacher(opposite command)||_2 - "
                    "||a_student-a_teacher(supplied command)||_2"
                ),
                "mean_margin_by_seed_and_state_side": (
                    action_margin_by_seed_and_state.tolist()
                ),
                "positive_alignment_criterion": positive_action_alignment,
                "status": "registered secondary; does not affect the primary decision",
                "scope_caveat": (
                    "specialist-teacher target alignment; does not separate teacher or "
                    "clip identity from finer trajectory content"
                ),
            },
            "positive_shift_gate": shift,
            "positive_selection_gate": selection,
            # Backward-compatible name now means the stronger selection criterion.
            "positive_target_specific_gate": selection,
            "interpretation": (
                (
                    "Within explicit-feasible starts, source-valid SNMR command replacement "
                    "shifted rollouts toward the replacement branch and met the aggregate "
                    "branch-selection sign criterion."
                )
                if selection
                else (
                    "The SNMR replacement met the directional shift criterion, but not the "
                    "stronger aggregate branch-selection sign criterion."
                    if shift
                    else (
                        "The explicit feasibility gate passed, but the preregistered SNMR "
                        "directional shift criterion was not met."
                    )
                )
            ),
        }
    )
    return summary


def _load_reports(paths: Sequence[pathlib.Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in paths]


def _input_records(paths: Sequence[pathlib.Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
    ]


def write_json_once(output: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON atomically without replacing an existing artifact."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite analysis artifact {output}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _gate_artifact(
    explicit_paths: Sequence[pathlib.Path], explicit_reports: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    gate = eligible_pairs(explicit_reports)
    valid = (
        gate["num_eligible_pairs"] >= MIN_ELIGIBLE_PAIRS
        and gate["num_eligible_temporal_components"]
        >= MIN_ELIGIBLE_TEMPORAL_COMPONENTS
    )
    return {
        "protocol": GATE_PROTOCOL,
        "analysis_protocol": PROTOCOL,
        "minimum_eligible_pairs": MIN_ELIGIBLE_PAIRS,
        "minimum_eligible_temporal_components": MIN_ELIGIBLE_TEMPORAL_COMPONENTS,
        "valid_explicit_gate": bool(valid),
        "explicit_feasibility_gate": gate,
        "inputs": _input_records(explicit_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit-reports", nargs=3, required=True, type=pathlib.Path)
    parser.add_argument("--snmr-reports", nargs=3, type=pathlib.Path)
    destinations = parser.add_mutually_exclusive_group(required=True)
    destinations.add_argument("--gate-out", type=pathlib.Path)
    destinations.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    explicit_reports = _load_reports(args.explicit_reports)
    if args.gate_out is not None:
        payload = _gate_artifact(args.explicit_reports, explicit_reports)
        write_json_once(args.gate_out, payload)
        print(json.dumps(payload, indent=2))
        return
    if args.snmr_reports is None:
        parser.error("--out requires exactly three --snmr-reports")
    summary = analyze(explicit_reports, _load_reports(args.snmr_reports))
    summary["inputs"] = _input_records([*args.explicit_reports, *args.snmr_reports])
    assert args.out is not None
    write_json_once(args.out, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
