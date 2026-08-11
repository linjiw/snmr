from __future__ import annotations

import pytest

from scripts.render_e70_paper_values import latex_macros


def _analysis() -> dict:
    arms = {
        arm: {
            "general_completion": 0.5,
            "ambiguity_completion": 0.6,
            "ambiguity_survival_s": 7.5,
        }
        for arm in ("explicit", "snmr", "time", "proprio", "shuffled")
    }
    return {
        "protocol": "E70 preregistered analysis v1",
        "seeds": [0, 1, 2],
        "arms": arms,
        "snmr_minus_time": {
            "difference": 0.2,
            "ci95_low": 0.1,
            "ci95_high": 0.3,
            "clusters": 69,
        },
        "snmr_minus_shuffled": {
            "difference": 0.15,
            "ci95_low": 0.05,
            "ci95_high": 0.25,
            "clusters": 69,
        },
        "per_seed_differences": {
            str(seed): {"snmr_minus_time": 0.1 + seed / 10, "snmr_minus_shuffled": 0.15}
            for seed in range(3)
        },
        "snmr_minus_time_per_clip": {
            "first": {
                "difference": 0.18,
                "ci95_low": 0.08,
                "ci95_high": 0.28,
                "clusters": 69,
            },
            "second": {
                "difference": 0.22,
                "ci95_low": 0.12,
                "ci95_high": 0.32,
                "clusters": 69,
            },
        },
        "explicit_general_gate": True,
        "positive_content_gate": True,
    }


def test_latex_macros_bind_three_seed_endpoints_and_hash() -> None:
    rendered = latex_macros(_analysis(), analysis_sha256="abc123")
    assert "% analysis_sha256=abc123" in rendered
    assert r"\newcommand{\ESnmrAmb}{0.600}" in rendered
    assert r"\newcommand{\EATDifference}{+0.200}" in rendered
    assert r"\newcommand{\EATSeedTwo}{+0.300}" in rendered
    assert r"\newcommand{\EExplicitGate}{PASS}" in rendered
    assert r"\newcommand{\EPositiveGate}{PASS}" in rendered
    assert r"\eexplicitvalidtrue" in rendered
    assert r"\epositivecontenttrue" in rendered


@pytest.mark.parametrize("field,value", [("protocol", "exploratory"), ("seeds", [0, 1])])
def test_latex_macros_reject_nonfinal_analysis(field: str, value: object) -> None:
    analysis = _analysis()
    analysis[field] = value
    with pytest.raises(ValueError, match="frozen E70|completed training seeds"):
        latex_macros(analysis, analysis_sha256="abc123")


def test_latex_macros_rejects_nonphysical_completion() -> None:
    analysis = _analysis()
    analysis["arms"]["snmr"]["ambiguity_completion"] = 1.1
    with pytest.raises(ValueError, match="completion"):
        latex_macros(analysis, analysis_sha256="abc123")


def test_latex_macros_rejects_wrong_cluster_count() -> None:
    analysis = _analysis()
    analysis["snmr_minus_time"]["clusters"] = 68
    with pytest.raises(ValueError, match="invalid AT contrast"):
        latex_macros(analysis, analysis_sha256="abc123")


def test_latex_macros_requires_boolean_gate() -> None:
    analysis = _analysis()
    analysis.pop("positive_content_gate")
    with pytest.raises(ValueError, match="positive-content gate"):
        latex_macros(analysis, analysis_sha256="abc123")


def test_latex_macros_requires_boolean_explicit_gate() -> None:
    analysis = _analysis()
    analysis.pop("explicit_general_gate")
    with pytest.raises(ValueError, match="explicit capability gate"):
        latex_macros(analysis, analysis_sha256="abc123")


def test_latex_macros_emits_scoped_null_booleans() -> None:
    analysis = _analysis()
    analysis["positive_content_gate"] = False
    rendered = latex_macros(analysis, analysis_sha256="abc123")
    assert r"\eexplicitvalidtrue" in rendered
    assert r"\epositivecontentfalse" in rendered


def test_latex_macros_emits_invalid_assay_booleans() -> None:
    analysis = _analysis()
    analysis["explicit_general_gate"] = False
    analysis["positive_content_gate"] = False
    rendered = latex_macros(analysis, analysis_sha256="abc123")
    assert r"\eexplicitvalidfalse" in rendered
    assert r"\epositivecontentfalse" in rendered


def test_latex_macros_rejects_positive_result_with_failed_capability() -> None:
    analysis = _analysis()
    analysis["explicit_general_gate"] = False
    with pytest.raises(ValueError, match="cannot pass"):
        latex_macros(analysis, analysis_sha256="abc123")
