from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from scripts.audit_e70_final_bundle import REQUIRED_VISUAL_CHECKS
from scripts.compose_e70_video import sha256_file
from scripts.record_e70_visual_review import review_payload


def _media(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    video = tmp_path / "final.mp4"
    contact_sheet = tmp_path / "contact.png"
    video.write_bytes(b"final video")
    contact_sheet.write_bytes(b"final contact sheet")
    return video, contact_sheet


def _checks() -> dict[str, bool]:
    return {name: True for name in REQUIRED_VISUAL_CHECKS}


def test_review_payload_binds_exact_inspected_media(tmp_path: pathlib.Path) -> None:
    video, contact_sheet = _media(tmp_path)
    payload = review_payload(
        video=video,
        contact_sheet=contact_sheet,
        reviewer="reviewer",
        full_video_watched=True,
        checks=_checks(),
        reviewed_utc="2026-08-11T20:00:00Z",
    )
    assert payload["video_sha256"] == sha256_file(video)
    assert payload["contact_sheet_sha256"] == sha256_file(contact_sheet)
    assert payload["full_video_watched"] is True


def test_review_payload_requires_full_video_watch(tmp_path: pathlib.Path) -> None:
    video, contact_sheet = _media(tmp_path)
    with pytest.raises(ValueError, match="full final MP4"):
        review_payload(
            video=video,
            contact_sheet=contact_sheet,
            reviewer="reviewer",
            full_video_watched=False,
            checks=_checks(),
        )


def test_review_payload_rejects_failed_check(tmp_path: pathlib.Path) -> None:
    video, contact_sheet = _media(tmp_path)
    checks = _checks()
    checks[REQUIRED_VISUAL_CHECKS[0]] = False
    with pytest.raises(ValueError, match="did not pass"):
        review_payload(
            video=video,
            contact_sheet=contact_sheet,
            reviewer="reviewer",
            full_video_watched=True,
            checks=checks,
        )


def test_review_payload_rejects_incomplete_checklist(tmp_path: pathlib.Path) -> None:
    video, contact_sheet = _media(tmp_path)
    checks = _checks()
    checks.pop(REQUIRED_VISUAL_CHECKS[0])
    with pytest.raises(ValueError, match="frozen requirements"):
        review_payload(
            video=video,
            contact_sheet=contact_sheet,
            reviewer="reviewer",
            full_video_watched=True,
            checks=checks,
        )


def test_review_payload_requires_both_media_files(tmp_path: pathlib.Path) -> None:
    video, contact_sheet = _media(tmp_path)
    contact_sheet.unlink()
    with pytest.raises(FileNotFoundError, match="must both exist"):
        review_payload(
            video=video,
            contact_sheet=contact_sheet,
            reviewer="reviewer",
            full_video_watched=True,
            checks=_checks(),
        )


def test_review_payload_rejects_malformed_utc_timestamp(tmp_path: pathlib.Path) -> None:
    video, contact_sheet = _media(tmp_path)
    with pytest.raises(ValueError, match="timestamp must be UTC"):
        review_payload(
            video=video,
            contact_sheet=contact_sheet,
            reviewer="reviewer",
            full_video_watched=True,
            checks=_checks(),
            reviewed_utc="eventuallyZ",
        )


@pytest.mark.parametrize(
    "entrypoint", ["record_e70_visual_review.py", "audit_e70_final_bundle.py"]
)
def test_finalization_entrypoint_help_avoids_installed_scripts_collision(
    entrypoint: str, tmp_path: pathlib.Path
) -> None:
    repository = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repository / "scripts" / entrypoint), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
