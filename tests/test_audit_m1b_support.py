import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_m1b_support.py"
SPEC = importlib.util.spec_from_file_location("audit_m1b_support", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_m1b_support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_m1b_support)


def test_mask_agreement_reports_exact_confusion_and_micro_metrics():
    candidate = torch.tensor([[True, True], [False, False]])
    oracle = torch.tensor([[True, False], [True, False]])

    result = audit_m1b_support.mask_agreement(candidate, oracle)

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert result["candidate_prevalence"] == pytest.approx(0.5)


def test_mask_agreement_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="candidate shape"):
        audit_m1b_support.mask_agreement(
            torch.zeros(2, 2, dtype=torch.bool),
            torch.zeros(2, 1, dtype=torch.bool),
        )


def test_select_smoke_window_uses_first_support_bearing_row():
    rows = [
        {"clip": "a", "start": 0, "candidate_samples": 0},
        {"clip": "a", "start": 10, "candidate_samples": 3},
        {"clip": "b", "start": 0, "candidate_samples": 5},
    ]

    selected = audit_m1b_support.select_smoke_window(rows)

    assert selected == rows[1]
