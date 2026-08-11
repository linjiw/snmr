import pathlib

import pytest
import torch

from scripts.prepare_e70_video import (
    build_video_manifest,
    resolve_start_step,
    select_reference_median_window,
)


def _precheck():
    windows = [
        {
            "frame_first": 10,
            "frame_second": 20,
            "time_seconds_first": 1.0,
            "time_seconds_second": 2.0,
            "state_distance": 0.2,
            "future_distance": 1.0,
        },
        {
            "frame_first": 30,
            "frame_second": 40,
            "time_seconds_first": 3.0,
            "time_seconds_second": 4.0,
            "state_distance": 0.4,
            "future_distance": 1.2,
        },
        {
            "frame_first": 50,
            "frame_second": 60,
            "time_seconds_first": 5.0,
            "time_seconds_second": 6.0,
            "state_distance": 0.6,
            "future_distance": 1.4,
        },
    ]
    return {
        "protocol": "E70 reference-only ambiguity precheck v1",
        "preferred_pair": "clip_a,clip_b",
        "pairs": {
            "clip_a,clip_b": {
                "passes_floor": True,
                "clips": ["clip_a", "clip_b"],
                "windows": windows,
            }
        },
    }


def _report():
    return {
        "start_steps": [100, 200, 110, 210, 120, 220, 110, 210],
        "motion_ids": [0, 1, 0, 1, 0, 1, 0, 1],
        "ambiguity_pair_ids": [0, 0, 1, 1, 2, 2, 1, 1],
        "ambiguity_sides": [0, 1, 0, 1, 0, 1, 0, 1],
        # These outcome fields must not affect selection or start resolution.
        "completed": [False] * 8,
        "survival_s": [0.1] * 8,
    }


def _checkpoints(root: pathlib.Path):
    for tag, arm in (
        ("snmr", "a_prior_snmr"),
        ("time", "a_prior_snmr"),
        ("explicit", "c_prior_explicit"),
    ):
        path = root / f"seed0_{tag}" / f"{arm}_student.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tag": tag}, path)


def test_video_window_selection_is_reference_median_only():
    pair, pair_id, selected = select_reference_median_window(_precheck())
    assert pair == "clip_a,clip_b"
    assert pair_id == 1
    assert selected["frame_first"] == 30
    assert "selection" in selected


def test_start_resolution_ignores_outcomes_and_requires_unique_start():
    report = _report()
    assert resolve_start_step(report, pair_id=1, side=0) == 110
    report["start_steps"][-2] = 111
    with pytest.raises(ValueError, match="2 unique"):
        resolve_start_step(report, pair_id=1, side=0)


def test_video_manifest_freezes_two_clip_comparison_and_intervention(tmp_path):
    _checkpoints(tmp_path)
    manifest = build_video_manifest(
        _precheck(), _report(), seed=0, student_root=tmp_path
    )
    assert manifest["selection_uses_policy_outcomes"] is False
    assert manifest["pair_id"] == 1
    assert [side["start_step"] for side in manifest["sides"]] == [110, 210]
    assert len(manifest["captures"]) == 6
    assert {capture["destroy_zcmd"] for capture in manifest["captures"]} == {
        "none",
        "marginal_random",
    }
    assert all(len(capture["checkpoint_sha256"]) == 64 for capture in manifest["captures"])
