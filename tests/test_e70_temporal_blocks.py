import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_e70_temporal_blocks import (
    PRECHECK_PROTOCOL,
    analyze,
    hierarchical_temporal_block_interval,
    temporal_block_partition,
    validate_seed_request,
)


def test_temporal_block_partition_connects_pairs_sharing_either_clip_block():
    windows = [
        {"time_seconds_first": 1.0, "time_seconds_second": 2.0},
        {"time_seconds_first": 3.0, "time_seconds_second": 12.0},
        {"time_seconds_first": 15.0, "time_seconds_second": 14.0},
        {"time_seconds_first": 31.0, "time_seconds_second": 32.0},
    ]
    block_ids, blocks = temporal_block_partition(windows)
    assert block_ids.tolist() == [0, 0, 0, 1]
    assert [item["pair_ids"] for item in blocks] == [[0, 1, 2], [3]]


def test_temporal_block_interval_retains_equal_pair_point_estimate_and_is_reproducible():
    effects = np.asarray([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
    blocks = np.asarray([0, 0, 1, 1])
    first = hierarchical_temporal_block_interval(effects, blocks, seed=9, replicates=500)
    second = hierarchical_temporal_block_interval(effects, blocks, seed=9, replicates=500)
    assert first == second
    assert first["difference"] == pytest.approx(2.0)
    assert first["per_seed_difference"] == pytest.approx([1.5, 2.5])
    assert first["temporal_blocks"] == 2


def test_partial_seed_set_requires_explicit_preview():
    with pytest.raises(ValueError, match="final analysis requires exactly"):
        validate_seed_request([0, 1], preview=False)
    validate_seed_request([0, 1], preview=True)
    with pytest.raises(ValueError, match="preview accepts exactly"):
        validate_seed_request([0, 1, 2], preview=True)


def _write_report(path: Path, completed: list[bool]) -> None:
    pair_ids = [pair_id for pair_id in range(69) for _ in range(2)]
    sides = [side for _ in range(69) for side in (0, 1)]
    payload = {
        "evaluation_seed": 404,
        "start_steps": list(range(138)),
        "ambiguity_pair_ids": pair_ids,
        "ambiguity_sides": sides,
        "completed": completed,
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))


def test_preview_analysis_hashes_every_input_and_labels_output_nonfinal(tmp_path: Path):
    windows = [
        {
            "time_seconds_first": pair_id * 11.0,
            "time_seconds_second": pair_id * 11.0 + 0.5,
        }
        for pair_id in range(69)
    ]
    precheck = tmp_path / "precheck.json"
    precheck.write_text(
        json.dumps(
            {
                "protocol": PRECHECK_PROTOCOL,
                "loaded_motion_order": ["walk1_subject1", "walk1_subject5"],
                "thresholds": {"rollout_seconds": 10.0},
                "pairs": {
                    "walk1_subject1,walk1_subject5": {"windows": windows}
                },
            }
        )
    )
    students = tmp_path / "students"
    for seed in (0, 1):
        snmr = [True] * 138
        time = [(pair_id % 3) != 0 for pair_id in range(69) for _ in range(2)]
        shuffled = [(pair_id % 2) != 0 for pair_id in range(69) for _ in range(2)]
        for tag, basename, completed in (
            ("snmr", "a_prior_snmr", snmr),
            ("time", "a_prior_snmr", time),
            ("shuffled", "a_prior_snmr", shuffled),
        ):
            _write_report(
                students / f"seed{seed}_{tag}" / f"{basename}_eval_ambiguity.json",
                completed,
            )

    result = analyze(students, precheck, [0, 1], preview=True, replicates=200)
    assert result["analysis_status"] == "non-final preview"
    assert result["directionally_consistent"] is True
    assert len(result["inputs"]) == 7
    assert all(len(item["sha256"]) == 64 for item in result["inputs"])
    assert result["comparisons"]["snmr_minus_time"]["pairs"] == 69


def test_preview_rejects_cross_arm_grid_mismatch(tmp_path: Path):
    windows = [
        {"time_seconds_first": i * 11.0, "time_seconds_second": i * 11.0}
        for i in range(69)
    ]
    precheck = tmp_path / "precheck.json"
    precheck.write_text(
        json.dumps(
            {
                "protocol": PRECHECK_PROTOCOL,
                "loaded_motion_order": ["walk1_subject1", "walk1_subject5"],
                "thresholds": {"rollout_seconds": 10.0},
                "pairs": {"walk1_subject1,walk1_subject5": {"windows": windows}},
            }
        )
    )
    students = tmp_path / "students"
    for tag in ("snmr", "time", "shuffled"):
        _write_report(
            students / f"seed0_{tag}" / "a_prior_snmr_eval_ambiguity.json",
            [True] * 138,
        )
    time_path = students / "seed0_time" / "a_prior_snmr_eval_ambiguity.json"
    time_report = json.loads(time_path.read_text())
    time_report["start_steps"][0] = 999
    time_path.write_text(json.dumps(time_report))
    with pytest.raises(ValueError, match="does not share the paired start grid"):
        analyze(students, precheck, [0], preview=True, replicates=100)
