import numpy as np

from scripts.precheck_e67_ambiguity import reference_features
from scripts.precheck_e69_pairs import (
    difficulty_ratio,
    reference_difficulty,
    select_candidate,
)


def test_reference_difficulty_is_finite_and_self_ratio_is_one():
    frames = 700
    fps = 50.0
    time = np.arange(frames) / fps
    qpos = np.zeros((frames, 10), dtype=np.float64)
    qpos[:, 3] = 1.0
    qpos[:, 7] = np.sin(time)
    qpos[:, 8] = np.cos(0.5 * time)
    qpos[:, 9] = 0.2 * np.sin(2.0 * time)
    difficulty = reference_difficulty(reference_features(qpos, fps))
    assert all(np.isfinite(list(difficulty.values())))
    assert difficulty_ratio(difficulty, difficulty) == 1.0


def test_select_candidate_uses_frozen_lexicographic_order():
    def record(clip, ratio, windows, future, passes=True):
        return {
            "clip": clip,
            "difficulty_ratio": ratio,
            "passes_ambiguity": passes,
            "ambiguity": {
                "num_selected_windows": windows,
                "eligible_future_distance": {"median": future},
            },
        }

    records = [
        record("hard", 1.3, 100, 2.0),
        record("few", 1.0, 20, 2.0),
        record("many", 1.0, 30, 1.0),
        record("failed", 0.5, 200, 3.0, passes=False),
    ]
    assert select_candidate(records, 1.25)["clip"] == "many"

