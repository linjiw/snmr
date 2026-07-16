import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_sharing_cost_screen.py"
SPEC = importlib.util.spec_from_file_location("analyze_sharing_cost_screen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sharing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sharing)


def _parameters(total, robot_specific=0):
    return {
        "total": total,
        "robot_specific": robot_specific,
        "robot_specific_fraction": robot_specific / total,
    }


def test_sharing_decision_promotes_arm_that_closes_half_of_both_gaps():
    specialist = {robot: 0.03 for robot in sharing.ROBOTS}
    base = {robot: 0.05 for robot in sharing.ROBOTS}
    wide = {robot: 0.039 for robot in sharing.ROBOTS}
    adapter = {robot: 0.041 for robot in sharing.ROBOTS}

    result = sharing.sharing_decision(
        specialist,
        {
            "shared_base_seed0": base,
            "shared_wide_seed0": wide,
            "shared_adapter_seed0": adapter,
        },
        {
            "shared_base_seed0": _parameters(sharing.BASE_PARAMETERS),
            "shared_wide_seed0": _parameters(sharing.WIDE_PARAMETERS),
            "shared_adapter_seed0": _parameters(
                sharing.ADAPTER_PARAMETERS,
                sharing.ROBOT_SPECIFIC_PARAMETERS,
            ),
        },
    )

    assert result["assay_passed"]
    assert result["promoted_arms"] == ["shared_wide_seed0"]
    assert result["winner"] == "shared_wide_seed0"
    assert result["replicate_winner_seeds"] == [1, 2]


def test_sharing_decision_requires_worst_robot_gap_closure():
    specialist = {robot: 0.03 for robot in sharing.ROBOTS}
    base = {robot: 0.05 for robot in sharing.ROBOTS}
    candidate = {robot: 0.035 for robot in sharing.ROBOTS}
    candidate[sharing.ROBOTS[-1]] = 0.045

    result = sharing.sharing_decision(
        specialist,
        {
            "shared_base_seed0": base,
            "shared_wide_seed0": candidate,
            "shared_adapter_seed0": candidate,
        },
        {
            "shared_base_seed0": _parameters(sharing.BASE_PARAMETERS),
            "shared_wide_seed0": _parameters(sharing.WIDE_PARAMETERS),
            "shared_adapter_seed0": _parameters(
                sharing.ADAPTER_PARAMETERS,
                sharing.ROBOT_SPECIFIC_PARAMETERS,
            ),
        },
    )

    assert not result["candidates"]["shared_wide_seed0"]["passed"]
    assert not result["candidates"]["shared_wide_seed0"]["checks"][
        "worst_gap_closure_ge_0.5"
    ]
    assert result["winner"] is None
