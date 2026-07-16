import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_wbt_reference_dev.py"
SPEC = importlib.util.spec_from_file_location("analyze_wbt_reference_dev", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reference_dev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference_dev)


def test_development_decision_promotes_at_registered_floor():
    result = reference_dev.development_decision(0.70)

    assert result["completion_floor_passed"]
    assert result["promote_confirmatory_matrix"]


def test_development_decision_stops_below_registered_floor():
    result = reference_dev.development_decision(0.69)

    assert not result["completion_floor_passed"]
    assert not result["promote_confirmatory_matrix"]
