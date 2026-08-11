from __future__ import annotations

import copy
import pathlib

import pytest

from scripts.audit_e70_final_bundle import (
    REQUIRED_VISUAL_CHECKS,
    validate_indexed_capture_files,
    validate_machine_evidence,
    validate_visual_review,
)
from scripts.compose_e70_video import result_summary, sha256_file
from scripts.render_e70_paper_values import latex_macros


def _analysis() -> dict:
    arms = {
        name: {
            "general_completion": 0.8,
            "ambiguity_completion": 0.7,
            "ambiguity_survival_s": 8.0,
        }
        for name in ("explicit", "snmr", "time", "proprio", "shuffled")
    }
    contrast = {"difference": 0.2, "ci95_low": 0.1, "ci95_high": 0.3, "clusters": 69}
    return {
        "protocol": "E70 preregistered analysis v1",
        "seeds": [0, 1, 2],
        "arms": arms,
        "snmr_minus_time": contrast,
        "snmr_minus_shuffled": contrast,
        "per_seed_differences": {
            str(seed): {"snmr_minus_time": 0.2, "snmr_minus_shuffled": 0.2}
            for seed in range(3)
        },
        "snmr_minus_time_per_clip": {"first": contrast, "second": contrast},
        "explicit_general_gate": True,
        "positive_content_gate": True,
        "interpretation": "control-usable content beyond time",
    }


def _machine_inputs() -> dict:
    analysis = _analysis()
    analysis_sha = "analysis-hash"
    capture_index = {
        "protocol": "E70 raw simulation capture index v1",
        "selection_uses_policy_outcomes": False,
        "manifest_sha256": "manifest-hash",
        "captures": [
            {"name": f"capture-{index}", "raw_video_sha256": f"raw-{index}"}
            for index in range(6)
        ],
    }
    validation = {
        "protocol": "E70 ICRA video validation v1",
        "passes": True,
        "observed": {"sha256": "video-hash"},
        "analysis_sha256": analysis_sha,
        "capture_index_sha256": "capture-index-hash",
        "manifest_sha256": "manifest-hash",
        "result_summary": result_summary(analysis),
        "raw_captures": {
            f"capture-{index}": {"sha256": f"raw-{index}"} for index in range(6)
        },
    }
    return {
        "analysis": analysis,
        "analysis_sha256": analysis_sha,
        "paper_values": latex_macros(analysis, analysis_sha256=analysis_sha),
        "video_validation": validation,
        "video_sha256": "video-hash",
        "capture_index": capture_index,
        "capture_index_sha256": "capture-index-hash",
        "manifest_sha256": "manifest-hash",
    }


def test_machine_evidence_accepts_one_fully_bound_bundle() -> None:
    summary = validate_machine_evidence(**_machine_inputs())
    assert summary["positive_content_gate"] is True


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda values: values.update(paper_values="stale"), "paper values"),
        (
            lambda values: values["video_validation"].update(analysis_sha256="stale"),
            "analysis_sha256",
        ),
        (
            lambda values: values["capture_index"].update(selection_uses_policy_outcomes=True),
            "outcome-independent",
        ),
        (lambda values: values["capture_index"]["captures"].pop(), "exactly six"),
    ],
)
def test_machine_evidence_fails_closed(mutation, match: str) -> None:
    values = copy.deepcopy(_machine_inputs())
    mutation(values)
    with pytest.raises(ValueError, match=match):
        validate_machine_evidence(**values)


def _review() -> dict:
    return {
        "protocol": "E70 full-video visual review v1",
        "video_sha256": "video-hash",
        "contact_sheet_sha256": "sheet-hash",
        "reviewed_utc": "2026-08-11T00:00:00Z",
        "reviewer": "artifact reviewer",
        "full_video_watched": True,
        "checks": {name: True for name in REQUIRED_VISUAL_CHECKS},
    }


def test_visual_review_binds_exact_media_and_all_checks() -> None:
    validate_visual_review(
        _review(), video_sha256="video-hash", contact_sheet_sha256="sheet-hash"
    )


def test_visual_review_rejects_failed_or_missing_check() -> None:
    review = _review()
    review["checks"][REQUIRED_VISUAL_CHECKS[0]] = False
    with pytest.raises(ValueError, match="did not pass"):
        validate_visual_review(
            review, video_sha256="video-hash", contact_sheet_sha256="sheet-hash"
        )


def test_visual_review_rejects_stale_video() -> None:
    with pytest.raises(ValueError, match="final MP4"):
        validate_visual_review(
            _review(), video_sha256="new-video", contact_sheet_sha256="sheet-hash"
        )


def test_visual_review_requires_full_video_confirmation() -> None:
    review = _review()
    review["full_video_watched"] = False
    with pytest.raises(ValueError, match="full final MP4"):
        validate_visual_review(
            review, video_sha256="video-hash", contact_sheet_sha256="sheet-hash"
        )


def test_visual_review_rejects_malformed_timestamp() -> None:
    review = _review()
    review["reviewed_utc"] = "eventuallyZ"
    with pytest.raises(ValueError, match="UTC completion timestamp"):
        validate_visual_review(
            review, video_sha256="video-hash", contact_sheet_sha256="sheet-hash"
        )


def _capture_index_with_files(tmp_path) -> dict:
    captures = []
    for index in range(6):
        item = {"name": f"capture-{index}"}
        for path_key, hash_key in (
            ("raw_video", "raw_video_sha256"),
            ("report", "report_sha256"),
            ("checkpoint", "checkpoint_sha256"),
        ):
            path = tmp_path / f"{index}-{path_key}"
            path.write_bytes(f"{index}-{path_key}".encode())
            item[path_key] = str(path)
            item[hash_key] = sha256_file(path)
        captures.append(item)
    return {"captures": captures}


def test_indexed_capture_files_replay_all_provenance(tmp_path) -> None:
    validated = validate_indexed_capture_files(_capture_index_with_files(tmp_path))
    assert len(validated) == 18


def test_indexed_capture_files_reject_tampered_raw_video(tmp_path) -> None:
    index = _capture_index_with_files(tmp_path)
    pathlib.Path(index["captures"][0]["raw_video"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="raw_video hash changed"):
        validate_indexed_capture_files(index)
