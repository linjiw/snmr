"""CPU tests for scripts/e1_retarget_difficulty.py label construction and CV plumbing."""

import importlib.util
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e1", ROOT / "scripts" / "e1_retarget_difficulty.py")
e1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e1)


def test_bin_labels_hazard_and_exposure_follow_survival():
    # one clip of 300 frames, bins of 50 -> 6 bins; fps 50
    reports = [{
        "start_steps": [0, 100],        # global == local (single clip, offset 0)
        "motion_ids": [0, 0],
        "completed": [False, True],
        "survival_s": [1.5, 10.0],       # first fails at frame 75 -> bin 1; second survives to end
    }]
    lab = e1.bin_labels(reports, [0], [300], bin_len=50, fps=50)
    exposure = lab["exposure"][0]
    # rollout 1 exposes frames 0..75: 50 in bin 0, 25 in bin 1; rollout 2 exposes 100..300 fully.
    assert exposure.tolist() == [50, 25, 50, 50, 50, 50]
    hazard = lab["hazard"][0]
    assert hazard[1] == 1 / 25 and hazard[0] == 0 and hazard[2:].sum() == 0
    assert lab["start_fail"][0][0] == 1.0 and lab["start_fail"][0][2] == 0.0
    assert np.isnan(lab["start_fail"][0][1])


def test_ridge_and_cv_recover_a_linear_signal():
    rng = np.random.default_rng(0)
    n = 400
    kin = rng.normal(size=(n, 3))
    ret = rng.normal(size=(n, 2))
    z_raw = rng.normal(size=(n, 32))
    y = kin[:, 0] + 2.0 * ret[:, 1] + 0.1 * rng.normal(size=n)
    feats = {"kin": kin, "ret": ret, "z_raw": z_raw, "z_scalar": rng.normal(size=(n, 2))}
    folds = [np.arange(i, n, 5) for i in range(5)]
    res = e1.cross_validate(feats, y, folds, alpha=1.0)
    assert res["kin+ret"]["r2"] > 0.95
    assert res["kin+ret"]["incremental_r2_over_kin"] > 0.5
    assert abs(res["kin+z"]["incremental_r2_over_kin"]) < 0.1   # noise features add nothing
