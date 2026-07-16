import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_wbt_reference_confirmatory.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_wbt_reference_confirmatory", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
confirmatory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(confirmatory)


def _arrays(gmr_completion, snmr_completion, gmr_joint, snmr_joint):
    shape = (
        2,
        len(confirmatory.TRAINING_SEEDS),
        len(confirmatory.EVALUATION_SEEDS),
        confirmatory.ROLLOUTS,
    )
    completion = np.empty(shape)
    joint = np.empty(shape)
    completion[0] = gmr_completion
    completion[1] = snmr_completion
    joint[0] = gmr_joint
    joint[1] = snmr_joint
    return completion, joint


def test_hierarchical_effects_pass_clear_noninferiority():
    completion, joint = _arrays(0.8, 0.8, 0.2, 0.2)

    result = confirmatory.hierarchical_effects(
        completion, joint, replicates=100, seed=7
    )

    assert result["noninferior"]
    assert all(result["checks"].values())


def test_hierarchical_effects_fail_completion_margin():
    completion, joint = _arrays(0.8, 0.6, 0.2, 0.2)

    result = confirmatory.hierarchical_effects(
        completion, joint, replicates=100, seed=7
    )

    assert not result["noninferior"]
    assert not result["checks"]["completion_noninferiority"]


def test_hierarchical_effects_fail_joint_margin():
    completion, joint = _arrays(0.8, 0.8, 0.2, 0.25)

    result = confirmatory.hierarchical_effects(
        completion, joint, replicates=100, seed=7
    )

    assert not result["noninferior"]
    assert not result["checks"]["joint_rmse_noninferiority"]
