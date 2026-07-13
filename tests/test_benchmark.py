import torch

from scripts.benchmark import _aggregate_rows, _time_inference


def test_benchmark_aggregation_preserves_window_mean_and_summarizes_clips():
    rows = [
        ("clip_a", {"metric": 1.0}),
        ("clip_a", {"metric": 3.0}),
        ("clip_b", {"metric": 6.0}),
    ]
    aggregate, distribution, per_clip = _aggregate_rows(
        rows, bootstrap_samples=100, seed=4
    )
    assert abs(aggregate["metric"] - 10.0 / 3.0) < 1e-12
    assert per_clip == {"clip_a": {"metric": 2.0}, "clip_b": {"metric": 6.0}}
    assert distribution["metric"]["clip_mean"] == 4.0
    assert distribution["metric"]["clip_median"] == 4.0
    assert distribution["metric"]["num_clips"] == 2


def test_timing_reports_repeat_distribution():
    value = torch.ones(4)
    output, timing = _time_inference(
        lambda: (value + 1,),
        frames=16,
        device="cpu",
        warmup=1,
        repeats=3,
    )
    assert torch.equal(output[0], torch.full((4,), 2.0))
    assert timing["timed_repeats"] == 3
    assert timing["fps_p10"] <= timing["fps_median"] <= timing["fps_p90"]
    assert timing["fps_median"] > 0
