import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_footlock.py"
SPEC = importlib.util.spec_from_file_location("eval_footlock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
eval_footlock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_footlock)


def test_predicted_contact_mask_selects_feet_and_thresholds_probabilities():
    logits = torch.tensor([
        [-2.0, 0.0, 2.0, 1.0],
        [2.0, -1.0, 0.0, -2.0],
    ])

    mask = eval_footlock.predicted_contact_mask(
        logits,
        [1, 3],
        probability_threshold=0.5,
    )

    assert mask.equal(torch.tensor([[True, True], [False, False]]))


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_predicted_contact_mask_rejects_degenerate_threshold(threshold):
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\)"):
        eval_footlock.predicted_contact_mask(
            torch.zeros(2, 3),
            [1],
            probability_threshold=threshold,
        )
