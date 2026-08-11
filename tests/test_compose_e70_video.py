from __future__ import annotations

import pytest

from scripts.compose_e70_video import (
    capture_status,
    outcome_gate_label,
    result_summary,
    sha256_file,
    validated_capture_index,
)


def _analysis() -> dict:
    return {
        "protocol": "E70 preregistered analysis v1",
        "seeds": [0, 1, 2],
        "arms": {
            "explicit": {"ambiguity_completion": 0.97},
            "snmr": {"ambiguity_completion": 0.75},
            "time": {"ambiguity_completion": 0.56},
            "proprio": {"ambiguity_completion": 0.50},
            "shuffled": {"ambiguity_completion": 0.59},
        },
        "snmr_minus_time": {
            "difference": 0.19,
            "ci95_low": 0.12,
            "ci95_high": 0.25,
            "clusters": 69,
        },
        "snmr_minus_shuffled": {
            "difference": 0.16,
            "ci95_low": 0.09,
            "ci95_high": 0.22,
            "clusters": 69,
        },
        "explicit_general_gate": True,
        "positive_content_gate": True,
        "interpretation": "control-usable content beyond time",
    }


def test_result_summary_uses_frozen_fields() -> None:
    summary = result_summary(_analysis())
    assert summary["seeds"] == [0, 1, 2]
    assert summary["completions"]["snmr"] == pytest.approx(0.75)
    assert summary["contrasts"]["A-T"]["ci95_low"] == pytest.approx(0.12)
    assert summary["explicit_general_gate"] is True
    assert summary["positive_content_gate"] is True


def test_result_summary_rejects_wrong_protocol() -> None:
    analysis = _analysis()
    analysis["protocol"] = "exploratory"
    with pytest.raises(ValueError, match="frozen E70"):
        result_summary(analysis)


def test_result_summary_rejects_partial_seed_analysis() -> None:
    analysis = _analysis()
    analysis["seeds"] = [0]
    with pytest.raises(ValueError, match="completed training seeds"):
        result_summary(analysis)


def test_result_summary_rejects_invalid_completion() -> None:
    analysis = _analysis()
    analysis["arms"]["snmr"]["ambiguity_completion"] = 1.2
    with pytest.raises(ValueError, match="invalid snmr"):
        result_summary(analysis)


def test_result_summary_requires_explicit_capability_gate() -> None:
    analysis = _analysis()
    analysis.pop("explicit_general_gate")
    with pytest.raises(ValueError, match="explicit capability gate"):
        result_summary(analysis)


def test_result_summary_rejects_positive_result_with_failed_capability() -> None:
    analysis = _analysis()
    analysis["explicit_general_gate"] = False
    with pytest.raises(ValueError, match="cannot pass"):
        result_summary(analysis)


@pytest.mark.parametrize(
    "explicit_gate,positive_gate,expected",
    [
        (True, True, "Frozen positive-content gate: PASS"),
        (True, False, "Frozen positive-content gate: DOES NOT PASS"),
        (False, False, "Frozen assay validity: INVALID (explicit control failed)"),
    ],
)
def test_outcome_gate_label_covers_all_registered_outcomes(
    explicit_gate: bool, positive_gate: bool, expected: str
) -> None:
    label, _ = outcome_gate_label(
        {
            "explicit_general_gate": explicit_gate,
            "positive_content_gate": positive_gate,
        }
    )
    assert label == expected


def test_capture_status_distinguishes_completion_and_termination() -> None:
    assert capture_status({"completed": True, "survival_s": 10.0})[0] == "completed · 10.00 s"
    assert capture_status({"completed": False, "survival_s": 2.4})[0] == "terminated · 2.40 s"


def test_capture_index_binds_manifest_names_and_raw_hashes(tmp_path) -> None:
    raw = tmp_path / "example.mp4"
    raw.write_bytes(b"frozen raw capture")
    digest = sha256_file(raw)
    index = {
        "protocol": "E70 raw simulation capture index v1",
        "manifest_sha256": "manifest-hash",
        "selection_uses_policy_outcomes": False,
        "captures": [
            {
                "name": "example",
                "raw_video_sha256": digest,
                "completed": False,
                "survival_s": 2.4,
            }
        ],
    }
    records = validated_capture_index(
        index, manifest_sha256="manifest-hash", raw_paths={"example": raw}
    )
    assert records["example"]["raw_video_sha256"] == digest

    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validated_capture_index(
            index, manifest_sha256="manifest-hash", raw_paths={"example": raw}
        )
