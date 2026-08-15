from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.render_e70_temporal_block_values import PROTOCOL, latex_macros, sha256_file


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_e70_temporal_block_values.py"


def _analysis(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "rollouts" / "a_prior_snmr_eval_ambiguity.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"completed": [True, False]}))
    pair_counts = [9, 7, 14, 3, 4, 2, 13, 4, 1, 2, 7, 3]
    analysis = {
        "protocol": PROTOCOL,
        "analysis_status": "final secondary analysis",
        "preview": False,
        "seeds": [0, 1, 2],
        "block_definition": {
            "atomic_block_seconds": 10.0,
            "temporal_blocks": 12,
            "pair_counts": pair_counts,
            "blocks": [
                {"block_id": index, "num_pairs": count}
                for index, count in enumerate(pair_counts)
            ],
        },
        "comparisons": {
            "snmr_minus_time": {
                "difference": 0.19077812284334025,
                "ci95_low": 0.10583720221149008,
                "ci95_high": 0.2802602923093665,
                "training_seeds": 3,
                "temporal_blocks": 12,
                "pairs": 69,
                "positive_direction": True,
                "ci_excludes_zero_positive": True,
            },
            "snmr_minus_shuffled": {
                "difference": 0.19940476190476186,
                "ci95_low": 0.11648667640943908,
                "ci95_high": 0.281780924779332,
                "training_seeds": 3,
                "temporal_blocks": 12,
                "pairs": 69,
                "positive_direction": True,
                "ci_excludes_zero_positive": True,
            },
        },
        "directionally_consistent": True,
        "inputs": [
            {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    path = tmp_path / "secondary_temporal_block_final.json"
    path.write_text(json.dumps(analysis))
    return path, analysis


def test_macros_bind_twelve_blocks_both_contrasts_and_hash(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path)
    rendered = latex_macros(analysis, analysis_path=path, analysis_sha256=sha256_file(path))
    assert f"% analysis_sha256={sha256_file(path)}" in rendered
    assert r"\newcommand{\ETemporalBlocks}{12}" in rendered
    assert r"\newcommand{\ETemporalATDifference}{+0.191}" in rendered
    assert r"\newcommand{\ETemporalATCILow}{0.106}" in rendered
    assert r"\newcommand{\ETemporalATCIHigh}{0.280}" in rendered
    assert r"\newcommand{\ETemporalASDifference}{+0.199}" in rendered
    assert r"\newcommand{\ETemporalASCILow}{0.116}" in rendered
    assert r"\newcommand{\ETemporalASCIHigh}{0.282}" in rendered


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("protocol", "E70 preregistered analysis v1", "frozen E70 secondary protocol"),
        ("preview", True, "non-preview"),
        ("seeds", [0, 1], "completed training seeds"),
        ("directionally_consistent", False, "directionally consistent"),
    ],
)
def test_guards_reject_nonfinal_analysis(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path, analysis = _analysis(tmp_path)
    analysis[field] = value
    with pytest.raises(ValueError, match=message):
        latex_macros(analysis, analysis_path=path, analysis_sha256="abc123")


def test_mutated_hash_bound_input_fails_closed(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path)
    tampered = Path(analysis["inputs"][0]["path"])
    tampered.write_text(tampered.read_text() + "\n")
    with pytest.raises(ValueError, match="input hash mismatch"):
        latex_macros(analysis, analysis_path=path, analysis_sha256="abc123")


def test_missing_hash_bound_input_fails_closed(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path)
    Path(analysis["inputs"][0]["path"]).unlink()
    with pytest.raises(ValueError, match="input hash mismatch"):
        latex_macros(analysis, analysis_path=path, analysis_sha256="abc123")


def test_partial_block_grid_fails_closed(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path)
    analysis["block_definition"]["blocks"] = analysis["block_definition"]["blocks"][:8]
    with pytest.raises(ValueError, match="completed 12 blocks"):
        latex_macros(analysis, analysis_path=path, analysis_sha256="abc123")


def test_blocks_must_partition_the_sixty_nine_pairs(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path)
    analysis["block_definition"]["pair_counts"][0] += 1
    with pytest.raises(ValueError, match="partition the 69"):
        latex_macros(analysis, analysis_path=path, analysis_sha256="abc123")


def test_interval_must_bracket_its_point_estimate(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path)
    analysis["comparisons"]["snmr_minus_time"]["ci95_low"] = 0.25
    with pytest.raises(ValueError, match="invalid AT temporal contrast"):
        latex_macros(analysis, analysis_path=path, analysis_sha256="abc123")


def test_undirected_contrast_fails_closed(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path)
    analysis["comparisons"]["snmr_minus_shuffled"]["ci_excludes_zero_positive"] = False
    with pytest.raises(ValueError, match="not directionally resolved"):
        latex_macros(analysis, analysis_path=path, analysis_sha256="abc123")


def test_missing_contrast_fails_closed(tmp_path: Path) -> None:
    path, analysis = _analysis(tmp_path)
    analysis["comparisons"].pop("snmr_minus_shuffled")
    with pytest.raises(ValueError, match="missing the AS temporal contrast"):
        latex_macros(analysis, analysis_path=path, analysis_sha256="abc123")


def test_direct_entrypoint_help_uses_repository_scripts_package(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--analysis" in completed.stdout
