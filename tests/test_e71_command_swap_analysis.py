from __future__ import annotations

import json
import pathlib
from typing import Callable

import numpy as np
import pytest

from scripts.analyze_e71_command_swap import (
    BOOTSTRAP_SEED,
    EXPECTED_EVALUATION_CONDITIONS,
    GATE_PROTOCOL,
    MIN_ELIGIBLE_PAIRS,
    MIN_ELIGIBLE_TEMPORAL_COMPONENTS,
    PRIMARY_SCORE,
    PROTOCOL,
    _gate_artifact,
    analyze,
    command_swap_effects,
    crossed_seed_temporal_component_interval,
    eligible_pairs,
    fixed_controller_temporal_component_interval,
    first_action_teacher_alignment,
    hierarchical_interval,
    sha256_file,
    validate_report,
    write_json_once,
)
from scripts.analyze_e70_temporal_blocks import temporal_block_partition


CoordinateFn = Callable[[int, int, int, int], float]
_PRECHECK_PATH: pathlib.Path | None = None


@pytest.fixture(autouse=True)
def _frozen_precheck(tmp_path: pathlib.Path):
    global _PRECHECK_PATH
    windows = []
    for pair_id in range(69):
        component = pair_id % 12
        within_block_offset = 2 * (pair_id // 12)
        windows.append(
            {
                "time_seconds_first": (
                    component * 500 + within_block_offset
                )
                / 50.0,
                "time_seconds_second": (
                    component * 500 + 100 + within_block_offset
                )
                / 50.0,
            }
        )
    pair_key = "walk1_subject1,walk1_subject5"
    path = tmp_path / "ambiguity_precheck.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "E70 reference-only ambiguity precheck v1",
                "preferred_pair": pair_key,
                "loaded_motion_order": ["walk1_subject1", "walk1_subject5"],
                "thresholds": {"rollout_seconds": 10.0},
                "pairs": {pair_key: {"windows": windows}},
            }
        )
    )
    _PRECHECK_PATH = path
    try:
        yield
    finally:
        _PRECHECK_PATH = None


def _default_coordinate(seed: int, pair: int, state: int, command: int) -> float:
    del seed, pair, state
    return -0.6 if command == 0 else 0.6


def _report(
    arm: str,
    seed: int,
    *,
    coordinate_fn: CoordinateFn = _default_coordinate,
) -> dict:
    assert _PRECHECK_PATH is not None
    pair_ids: list[int] = []
    states: list[int] = []
    commands: list[int] = []
    state_steps: list[int] = []
    command_steps: list[int] = []
    q_a: list[float] = []
    q_b: list[float] = []
    q_ab: list[float] = []
    coordinates: list[float] = []
    d_a: list[float] = []
    d_b: list[float] = []
    first_student_action: list[list[float]] = []
    first_teacher_action: list[list[float]] = []
    for pair_id in range(69):
        component = pair_id % 12
        within_block_offset = 2 * (pair_id // 12)
        starts = (
            component * 500 + within_block_offset,
            10_000 + component * 500 + 100 + within_block_offset,
        )
        for state in (0, 1):
            for command in (0, 1):
                numerator = float(coordinate_fn(seed, pair_id, state, command))
                qa = 1.0 + numerator / 2.0
                qb = 1.0 - numerator / 2.0
                pair_ids.append(pair_id)
                states.append(state)
                commands.append(command)
                state_steps.append(starts[state])
                command_steps.append(starts[command])
                q_a.append(qa)
                q_b.append(qb)
                q_ab.append(1.0)
                coordinates.append((qa - qb) / (1.0 + 1.0e-8))
                d_a.append(qa**0.5)
                d_b.append(qb**0.5)
                action_value = -1.0 if command == 0 else 1.0
                first_student_action.append([action_value] * 29)
                first_teacher_action.append([action_value] * 29)

    digest = "a" * 64
    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "student_arm": {
            "explicit": "c_prior_explicit",
            "snmr": "a_prior_snmr",
        }[arm],
        "training_seed": seed,
        "evaluation_seed": 404,
        "num_rollouts": 276,
        "pair_ids": pair_ids,
        "state_sides": states,
        "command_sides": commands,
        "state_start_steps": state_steps,
        "command_start_steps": command_steps,
        "state_motion_ids": states.copy(),
        "command_motion_ids": commands.copy(),
        "q_a_0p5_s": q_a,
        "q_b_0p5_s": q_b,
        "q_ab_0p5_s": q_ab,
        "branch_coordinate_0p5_s": coordinates,
        "d_a_0p5_s": d_a,
        "d_b_0p5_s": d_b,
        "branch_samples_0p5_s": 25,
        "q_a_1p0_s": q_a,
        "q_b_1p0_s": q_b,
        "q_ab_1p0_s": q_ab,
        "branch_coordinate_1p0_s": coordinates,
        "d_a_1p0_s": d_a,
        "d_b_1p0_s": d_b,
        "branch_samples_1p0_s": 50,
        "completed": [True] * 276,
        "survival_s": [10.0] * 276,
        "first_student_action": first_student_action,
        "first_teacher_action": first_teacher_action,
        "proprio_audit": {
            "passed": True,
            "max_abs_difference": 0.0,
            "tolerance": 1.0e-6,
        },
        "raw_proprio_audit": {
            "passed": True,
            "max_abs_difference": 0.0,
            "tolerance": 1.0e-6,
        },
        "full_state_audit": {
            "num_tensors": 1,
            "num_tensor_state_comparisons": 138,
            "max_abs_difference": 0.0,
            "tolerance": 1.0e-6,
            "per_tensor": {
                "qpos": {
                    "num_state_comparisons": 138,
                    "max_abs_difference": 0.0,
                    "passed": True,
                }
            },
            "passed": True,
        },
        "warmup_audit": {"passed": True},
        "termination_audit": {
            "primary_horizon_done_count": 0,
            "suppressed_steps": 50,
            "passed": True,
        },
        "evaluation_conditions": dict(EXPECTED_EVALUATION_CONDITIONS),
        "student_checkpoint": f"/frozen/{arm}-seed{seed}.pt",
        "student_checkpoint_sha256": digest,
        "teacher_manifest": "/frozen/teachers.json",
        "teacher_manifest_sha256": digest,
        "teacher_ckpts": ["/frozen/a.pt", "/frozen/b.pt"],
        "teacher_checkpoint_sha256": [digest, digest],
        "motion_files": ["/frozen/a.npz", "/frozen/b.npz"],
        "motion_sha256": [digest, digest],
        "ambiguity_precheck": str(_PRECHECK_PATH),
        "ambiguity_precheck_sha256": sha256_file(_PRECHECK_PATH),
        "runtime": "/frozen/eval.py",
        "runtime_sha256": digest,
        "reset_runtime": "/frozen/reset.py",
        "reset_runtime_sha256": digest,
        "policy_dt_s": 0.02,
        "motion_fps": 50.0,
    }


def _reports(arm: str, *, coordinate_fn: CoordinateFn = _default_coordinate) -> list[dict]:
    return [_report(arm, seed, coordinate_fn=coordinate_fn) for seed in range(3)]


def test_command_swap_effects_preserve_pair_and_state_directions() -> None:
    def coordinates(seed: int, pair: int, state: int, command: int) -> float:
        del seed, pair
        base = (-0.4, -0.2)[state]
        return base if command == 0 else base + (0.3, 0.5)[state]

    pairs, effects = command_swap_effects(_report("snmr", 0, coordinate_fn=coordinates))
    assert pairs.tolist() == list(range(69))
    np.testing.assert_allclose(effects, np.tile([0.3, 0.5], (69, 1)))


def test_first_action_alignment_prefers_the_supplied_command_teacher() -> None:
    pairs, margins = first_action_teacher_alignment(_report("snmr", 0))
    assert pairs.tolist() == list(range(69))
    assert margins.shape == (69, 2, 2)
    np.testing.assert_allclose(margins, 2.0 * np.sqrt(29.0))


def test_validate_report_recomputes_the_branch_coordinate() -> None:
    report = _report("explicit", 0)
    report["branch_coordinate_1p0_s"][0] = 9.0
    with pytest.raises(ValueError, match=r"inconsistent with \(q_a-q_b\)"):
        validate_report(report)


def test_validate_report_fails_closed_on_grid_audits_conditions_and_samples() -> None:
    report = _report("explicit", 0)
    report["command_sides"][1] = 0
    report["command_motion_ids"][1] = 0
    with pytest.raises(ValueError, match="exactly one"):
        validate_report(report)

    report = _report("explicit", 0)
    report["full_state_audit"]["passed"] = False
    with pytest.raises(ValueError, match="full_state_audit did not pass"):
        validate_report(report)

    report = _report("explicit", 0)
    report["evaluation_conditions"]["external_pushes"] = True
    with pytest.raises(ValueError, match="external_pushes"):
        validate_report(report)

    report = _report("explicit", 0)
    report["branch_samples_1p0_s"] = 51
    with pytest.raises(ValueError, match="future-only 50"):
        validate_report(report)


def test_explicit_gate_requires_joint_source_valid_selection_for_both_states() -> None:
    def coordinates(seed: int, pair: int, state: int, command: int) -> float:
        default = -0.6 if command == 0 else 0.6
        if pair == 1 and state == 0 and seed in (0, 1) and command == 1:
            return -0.1
        if pair == 2 and state == 1 and seed in (1, 2) and command == 0:
            return 0.1
        return default

    gate = eligible_pairs(_reports("explicit", coordinate_fn=coordinates))
    assert gate["required_explicit_seeds"] == 2
    assert gate["num_temporal_components"] == 12
    assert gate["num_eligible_temporal_components"] == 12
    assert 0 in gate["eligible_pair_ids"]
    assert 1 not in gate["eligible_pair_ids"]
    assert 2 not in gate["eligible_pair_ids"]
    assert gate["source_valid_selection_seed_count_by_state"][1][0] == 1


def test_temporal_components_exactly_reproduce_the_e70_partition() -> None:
    assert _PRECHECK_PATH is not None
    precheck = json.loads(_PRECHECK_PATH.read_text())
    windows = precheck["pairs"][precheck["preferred_pair"]]["windows"]
    expected_ids, expected_summaries = temporal_block_partition(windows)
    gate = eligible_pairs(_reports("explicit"))
    assert gate["temporal_component_ids"] == expected_ids.tolist()
    assert [item["num_pairs"] for item in gate["temporal_components"]] == [
        item["num_pairs"] for item in expected_summaries
    ]


def test_component_intervals_are_deterministic_and_share_component_weights() -> None:
    effects = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ]
    )
    components = np.asarray([0, 0, 1, 1, 2, 2])
    first = fixed_controller_temporal_component_interval(
        effects, components, replicates=2_000
    )
    second = fixed_controller_temporal_component_interval(
        effects, components, replicates=2_000
    )
    sensitivity = crossed_seed_temporal_component_interval(
        effects, components, replicates=2_000
    )
    assert first == second
    assert first["difference"] == pytest.approx(effects.mean())
    assert first["ci95_low"] > 0.0
    assert first["controllers_conditioned_fixed"] is True
    assert first["bootstrap_seed"] == BOOTSTRAP_SEED
    assert sensitivity["confirmatory"] is False
    with pytest.raises(ValueError, match="component_ids are required"):
        hierarchical_interval(effects, replicates=2_000)


def test_analyze_distinguishes_directional_shift_from_branch_selection() -> None:
    explicit = _reports("explicit")

    def selecting(seed: int, pair: int, state: int, command: int) -> float:
        del seed, pair, state
        return -0.3 if command == 0 else 0.4

    selected = analyze(explicit, _reports("snmr", coordinate_fn=selecting), replicates=500)
    assert selected["valid_explicit_gate"] is True
    assert selected["positive_shift_gate"] is True
    assert selected["positive_selection_gate"] is True
    assert selected["positive_target_specific_gate"] is True
    assert selected["snmr_command_swap"]["ci95_low"] > 0.0
    assert selected["snmr_command_swap"]["positive_direction_by_seed_and_state_side"] == [
        [True, True],
        [True, True],
        [True, True],
    ]

    def shifting_only(seed: int, pair: int, state: int, command: int) -> float:
        del seed, pair, state
        return 0.1 if command == 0 else 0.4

    shifted = analyze(explicit, _reports("snmr", coordinate_fn=shifting_only), replicates=500)
    assert shifted["positive_shift_gate"] is True
    assert shifted["positive_selection_gate"] is False
    assert shifted["positive_target_specific_gate"] is False
    assert shifted["snmr_command_swap"]["mean_coordinate_by_command"][0] > 0.0


def test_shift_gate_requires_every_seed_and_physical_state_direction() -> None:
    def coordinates(seed: int, pair: int, state: int, command: int) -> float:
        if seed == 2 and state == 1:
            return 0.4 if command == 0 else 0.2
        return -0.3 if command == 0 else 0.4

    result = analyze(
        _reports("explicit"),
        _reports("snmr", coordinate_fn=coordinates),
        replicates=500,
    )
    assert result["snmr_command_swap"]["difference"] > 0.0
    assert result["snmr_command_swap"]["positive_direction_by_seed_and_state_side"][2][1] is False
    assert result["positive_shift_gate"] is False


def test_gate_requires_six_temporal_components_even_with_twenty_pairs() -> None:
    def coordinates(seed: int, pair: int, state: int, command: int) -> float:
        component = pair % 12
        if component >= 5 and seed in (0, 1) and command == 1:
            return -0.1
        return -0.6 if command == 0 else 0.6

    explicit = _reports("explicit", coordinate_fn=coordinates)
    gate = eligible_pairs(explicit)
    assert gate["num_eligible_pairs"] >= MIN_ELIGIBLE_PAIRS
    assert gate["num_eligible_temporal_components"] == 5
    result = analyze(explicit, _reports("snmr"), replicates=500)
    assert result["minimum_eligible_temporal_components"] == MIN_ELIGIBLE_TEMPORAL_COMPONENTS
    assert result["valid_explicit_gate"] is False
    assert result["snmr_command_swap"] is None


def test_report_sets_fail_closed_on_grid_and_common_provenance() -> None:
    explicit = _reports("explicit")
    snmr = _reports("snmr")
    for index in (0, 1):
        snmr[2]["state_start_steps"][index] += 1
    for index in (0, 2):
        snmr[2]["command_start_steps"][index] += 1
    with pytest.raises(ValueError, match="exact frozen grid"):
        analyze(explicit, snmr, replicates=500)

    snmr = _reports("snmr")
    snmr[1]["motion_sha256"][0] = "b" * 64
    with pytest.raises(ValueError, match="frozen provenance"):
        analyze(explicit, snmr, replicates=500)


def test_gate_artifact_and_json_writer_are_bound_and_write_once(tmp_path: pathlib.Path) -> None:
    explicit = _reports("explicit")
    paths = []
    for seed in range(3):
        path = tmp_path / f"explicit-{seed}.json"
        path.write_text(f"seed={seed}\n")
        paths.append(path)
    artifact = _gate_artifact(paths, explicit)
    assert artifact["protocol"] == GATE_PROTOCOL
    assert artifact["minimum_eligible_pairs"] == MIN_ELIGIBLE_PAIRS
    assert artifact["minimum_eligible_temporal_components"] == MIN_ELIGIBLE_TEMPORAL_COMPONENTS
    assert artifact["valid_explicit_gate"] is True

    output = tmp_path / "gate.json"
    write_json_once(output, artifact)
    frozen = output.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_json_once(output, {"changed": True})
    assert output.read_bytes() == frozen


def test_primary_score_is_frozen() -> None:
    report = _report("snmr", 0)
    with pytest.raises(ValueError, match=PRIMARY_SCORE):
        validate_report(report, score_key="branch_score_1p0_s")
    with pytest.raises(ValueError, match="threshold is frozen"):
        analyze(
            _reports("explicit"),
            _reports("snmr"),
            min_eligible_pairs=19,
            replicates=500,
        )
