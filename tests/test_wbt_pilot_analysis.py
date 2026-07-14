import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_wbt_pilot.py"
SPEC = importlib.util.spec_from_file_location("analyze_wbt_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
wbt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wbt)


def test_wbt_series_summary_uses_fixed_final_window_and_full_curve():
    series = [(step, float(step)) for step in range(10)]

    summary, errors = wbt._summarize_series(
        series,
        expected_events=10,
        final_window=2,
    )

    assert not errors
    assert summary["initial_window_mean"] == 0.5
    assert summary["final_window_mean"] == 8.5
    assert summary["normalized_auc"] == 4.5


def test_wbt_series_summary_rejects_missing_and_nonfinite_events():
    summary, errors = wbt._summarize_series(
        [(0, 1.0), (2, float("nan"))],
        expected_events=3,
        final_window=1,
    )

    assert summary["event_count"] == 2
    assert len(errors) == 3
