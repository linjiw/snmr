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


def _dataset(*records):
    files = [
        {"path": path, "size_bytes": size, "sha256": character * 64}
        for path, size, character in records
    ]
    return {
        "sha256": "f" * 64,
        "splits": {
            "train": {
                "sha256": "e" * 64,
                "file_count": len(files),
                "files": files,
            }
        },
    }


def test_dataset_contract_requires_specialists_to_partition_shared_data():
    robot_records = {
        robot: (f"{robot}/clip.npz", index + 1, str(index))
        for index, robot in enumerate(sharing.ROBOTS)
    }
    shared = _dataset(*robot_records.values())
    datasets = {
        f"specialist_{robot}_seed0": _dataset(robot_records[robot])
        for robot in sharing.ROBOTS
    }
    datasets.update({arm: shared for arm in sharing.SHARED_ARMS})

    assert sharing._validate_dataset_partition(datasets) == []

    bad = dict(datasets)
    bad["specialist_unitree_g1_seed0"] = _dataset(
        ("unitree_g1/changed.npz", 1, "0")
    )
    assert sharing._validate_dataset_partition(bad) == [
        "specialist datasets do not partition shared split 'train'"
    ]


def test_resume_contract_rejects_dataset_drift():
    manifest = {
        "invocations": [
            {"resume": False},
            {"resume": True},
        ],
        "resume_checks": [
            {
                "config_matches_original_except_resume": True,
                "config_differences": {},
                "dataset_matches_original": False,
                "dataset_hash": "a" * 64,
            }
        ]
    }

    assert sharing._validate_resume_checks(manifest) == [
        "resume check 0 changed the original dataset"
    ]
