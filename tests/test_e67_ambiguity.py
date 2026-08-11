import numpy as np

from scripts.precheck_e67_ambiguity import (
    ReferenceFeatures,
    ambiguity_windows,
    reference_features,
)


def test_reference_features_match_actor_proprio_width_for_g1():
    frames = 80
    qpos = np.zeros((frames, 36), dtype=np.float64)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = np.linspace(0, 1, frames)[:, None]
    features = reference_features(qpos, 30.0)
    assert features.state.shape == (frames, 90)
    assert features.goal.shape == (frames, 58)
    assert np.isfinite(features.state).all()


def test_ambiguity_windows_require_similar_present_and_divergent_future():
    frames, dims = 240, 4
    time = np.arange(frames)[:, None]
    shared_state = np.concatenate(
        [np.sin(time / 10 + phase) for phase in np.arange(dims)[None, :]], axis=1
    )
    first_goal = np.concatenate((shared_state, np.zeros_like(shared_state)), axis=1)
    second_goal = np.concatenate((shared_state, np.zeros_like(shared_state)), axis=1)
    # After each present state, the two desired trajectories separate persistently.
    second_goal[:, dims:] = 3.0 * np.sin(time / 17)
    first = ReferenceFeatures(shared_state, first_goal, 30.0)
    second = ReferenceFeatures(shared_state.copy(), second_goal, 30.0)
    result = ambiguity_windows(
        first,
        second,
        time_bins=20,
        future_seconds=0.5,
        rollout_seconds=0.5,
        future_samples=6,
        max_state_distance=0.1,
        min_future_distance=0.3,
        min_spacing_seconds=0.2,
    )
    assert result["num_selected_windows"] >= 20
    assert all(window["state_distance"] <= 0.1 for window in result["windows"])
    assert all(window["future_distance"] >= 0.3 for window in result["windows"])
    assert all(
        window["frame_first"] + result["rollout_frames"] <= frames
        and window["frame_second"] + result["rollout_frames"] <= frames
        for window in result["windows"]
    )
