import pathlib

import pytest

from scripts.prepare_e70_protocol import build_e70_precheck


def _screen():
    return {
        "protocol": "E69 exhaustive reference-only pair screen v1",
        "anchor": "walk1_subject5",
        "selected_clip": "walk1_subject1",
        "gate_passed": True,
        "thresholds": {"min_windows": 1},
        "anchor_input": {"path": "/anchor.npz", "sha256": "anchor"},
        "candidates": [
            {
                "clip": "walk1_subject1",
                "path": "/selected.npz",
                "sha256": "selected",
                "passes_ambiguity": True,
                "ambiguity": {
                    "num_selected_windows": 1,
                    "windows": [
                        {
                            "frame_first": 10,
                            "frame_second": 20,
                            "time_seconds_first": 1.0,
                            "time_seconds_second": 2.0,
                            "normalized_time_first": 0.1,
                            "normalized_time_second": 0.2,
                            "state_distance": 0.3,
                            "future_distance": 1.2,
                        }
                    ],
                },
            }
        ],
    }


def test_e70_precheck_swaps_all_side_fields_into_loader_order():
    result = build_e70_precheck(_screen())
    assert result["preferred_pair"] == "walk1_subject1,walk1_subject5"
    pair = result["pairs"][result["preferred_pair"]]
    assert pair["clips"] == ["walk1_subject1", "walk1_subject5"]
    window = pair["windows"][0]
    assert window["frame_first"] == 20
    assert window["frame_second"] == 10
    assert window["time_seconds_first"] == 2.0
    assert window["normalized_time_second"] == 0.1
    assert window["state_distance"] == 0.3
    assert pair["passes_floor"] is True


def test_e70_precheck_rejects_failed_or_wrong_screen():
    screen = _screen()
    screen["gate_passed"] = False
    with pytest.raises(ValueError, match="did not select"):
        build_e70_precheck(screen)

    screen = _screen()
    screen["protocol"] = "unregistered"
    with pytest.raises(ValueError, match="not the frozen"):
        build_e70_precheck(screen, source_path=pathlib.Path("unused"))
