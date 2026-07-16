from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_wbt_latent_phase3 import ManifestRow, analyze  # noqa: E402
from test_wbt_latent_phase2_analysis import _report  # noqa: E402


def _rows(tmp_path, spec):
    """spec: {arm: {clip: (split, completion, rmse)}}"""
    ckpt = tmp_path / "model_15999.pt"
    ckpt.write_bytes(b"x")
    rows = []
    for arm, clips in spec.items():
        for clip, (split, completion, rmse) in clips.items():
            name = f"{arm}_{clip}"
            motion = tmp_path / f"{clip}.npz"
            motion.write_bytes(b"")
            report_path = tmp_path / f"{name}.json"
            report_path.write_text(
                json.dumps(_report(name, str(motion), 404, completion, rmse))
            )
            rows.append(ManifestRow(arm, clip, split, 404, name, ckpt, report_path))
    return rows


def test_phase3_promotes_on_heldout_mean(tmp_path):
    spec = {
        "b_multi": {
            "walk1_subject5": ("heldout", 0.80, 0.26),
            "run2_subject1": ("heldout", 0.60, 0.30),
            "walk1_subject2": ("trained", 0.90, 0.24),
        },
        "s3_multi": {
            "walk1_subject5": ("heldout", 0.84, 0.244),
            "run2_subject1": ("heldout", 0.62, 0.282),
            "walk1_subject2": ("trained", 0.91, 0.235),
        },
        "l1_multi": {
            "walk1_subject5": ("heldout", 0.55, 0.29),
            "run2_subject1": ("heldout", 0.35, 0.33),
            "walk1_subject2": ("trained", 0.70, 0.28),
        },
    }
    motion_files = {
        clip: tmp_path / f"{clip}.npz"
        for clip in ("walk1_subject5", "run2_subject1", "walk1_subject2")
    }
    result = analyze(
        _rows(tmp_path, spec),
        {"b_multi": "baseline", "s3_multi": "phase0_heldout", "l1_multi": "descriptive"},
        motion_files,
    )
    assert result["passed"], result["protocol_errors"]
    # s3 held-out mean rmse_rel = mean(0.244/0.26-1, 0.282/0.30-1) = mean(-6.2%, -6%) -> promotes
    assert result["arms"]["s3_multi"]["decision"]["promoted"] is True
    l1 = result["arms"]["l1_multi"]["decision"]
    # l1 held-out completion mean 0.45 -> not collapsed (>= 0.40), gap reported
    assert l1["collapsed"] is False
    assert abs(l1["heldout_gap_pp"] - (-25.0)) < 1e-9
    assert result["verdict"] == "promoted:s3_multi"


def test_phase3_flags_l1_collapse(tmp_path):
    spec = {
        "b_multi": {
            "walk1_subject5": ("heldout", 0.80, 0.26),
            "run2_subject1": ("heldout", 0.70, 0.28),
        },
        "l1_multi": {
            "walk1_subject5": ("heldout", 0.30, 0.35),
            "run2_subject1": ("heldout", 0.35, 0.36),
        },
    }
    motion_files = {
        clip: tmp_path / f"{clip}.npz"
        for clip in ("walk1_subject5", "run2_subject1")
    }
    result = analyze(
        _rows(tmp_path, spec),
        {"b_multi": "baseline", "l1_multi": "descriptive"},
        motion_files,
    )
    assert result["passed"], result["protocol_errors"]
    assert result["arms"]["l1_multi"]["decision"]["collapsed"] is True
    assert result["verdict"] == "no_arm_promotes"


def test_phase3_rejects_incomplete_cartesian_matrix(tmp_path):
    spec = {
        "b_multi": {
            "walk1_subject5": ("heldout", 0.80, 0.26),
            "run2_subject1": ("heldout", 0.70, 0.28),
        },
        "s3_multi": {
            "walk1_subject5": ("heldout", 0.82, 0.25),
        },
    }
    motion_files = {
        clip: tmp_path / f"{clip}.npz"
        for clip in ("walk1_subject5", "run2_subject1")
    }

    result = analyze(
        _rows(tmp_path, spec),
        {
            "b_multi": "baseline",
            "s3_multi": "phase0_heldout",
        },
        motion_files,
    )

    assert not result["passed"]
    assert any(
        "missing manifest cells" in error
        for error in result["protocol_errors"]
    )
