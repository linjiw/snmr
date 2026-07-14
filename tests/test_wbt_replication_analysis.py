import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_wbt_replication.py"
SPEC = importlib.util.spec_from_file_location("analyze_wbt_replication", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
wbt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wbt)


def _config(source, clip, seed):
    return {
        "training": {
            "name": wbt._run_name(source, clip, seed),
            "num_envs": 1024,
            "seed": seed,
        },
        "algo": {"config": {"num_learning_iterations": 1000}},
        "command": {
            "setup_terms": {
                "motion_command": {
                    "params": {
                        "motion_config": {
                            "motion_file": (
                                f"/data/{source}/{clip}_subject_mj.npz"
                            )
                        }
                    }
                }
            }
        },
    }


def test_wbt_replication_protocol_accepts_only_name_motion_and_seed_changes():
    runs = {}
    for source in wbt.SOURCES:
        for clip in wbt.CLIPS:
            for seed in wbt.SEEDS:
                name = wbt._run_name(source, clip, seed)
                runs[name] = {
                    "config": _config(source, clip, seed),
                    "checkpoint": f"/run/{name}/model_00999.pt",
                }

    errors = wbt._validate_protocol(
        runs,
        expected_events=1000,
        expected_envs=1024,
    )

    assert not errors


def test_wbt_replication_aggregates_all_clip_seed_pairs():
    paired = {}
    for clip_index, clip in enumerate(wbt.CLIPS):
        paired[clip] = {}
        for seed in wbt.SEEDS:
            paired[clip][str(seed)] = {
                "Train/mean_reward": {
                    "final_window_mean": {
                        "gmr": 1.0,
                        "snmr": 1.0 + clip_index + seed,
                        "snmr_minus_gmr": float(clip_index + seed),
                        "relative_delta": float(clip_index + seed),
                        "favorable_effect": float(clip_index + seed),
                    },
                    "normalized_auc": {
                        "gmr": 1.0,
                        "snmr": 2.0,
                        "snmr_minus_gmr": 1.0,
                        "relative_delta": 1.0,
                        "favorable_effect": 1.0,
                    },
                }
            }

    aggregate = wbt._aggregate_effects(paired)
    final = aggregate["Train/mean_reward"]["final_window_mean"]

    assert final["pair_count"] == 9
    assert final["favorable_pair_count"] == 8
    assert final["mean_favorable_effect"] == 2.0
