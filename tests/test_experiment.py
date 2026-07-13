import json
import random

import numpy as np
import torch

from snmr.experiment import (
    RunManifest,
    capture_rng_state,
    dataset_fingerprint,
    restore_rng_state,
    sha256_file,
)


def test_dataset_fingerprint_is_deterministic_and_content_sensitive(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    paths = {"validation": [second], "train": [first]}

    before = dataset_fingerprint(paths, root=tmp_path)
    reordered = dataset_fingerprint({"train": [first], "validation": [second]}, root=tmp_path)
    assert before["sha256"] == reordered["sha256"]
    assert before["splits"]["train"]["files"][0]["path"] == "a.bin"
    assert before["splits"]["train"]["files"][0]["sha256"] == sha256_file(first)

    first.write_bytes(b"changed")
    after = dataset_fingerprint(paths, root=tmp_path)
    assert before["sha256"] != after["sha256"]


def test_run_manifest_tracks_invocations_checkpoint_and_completion(tmp_path):
    dataset = dataset_fingerprint({"train": []}, root=tmp_path)
    path = tmp_path / "run" / "manifest.json"
    manifest = RunManifest.start(
        path,
        trainer="test",
        repo_root=tmp_path,
        argv=["train.py", "--steps", "2"],
        config={"steps": 2, "device": "cpu"},
        dataset=dataset,
        training={"seed": 3},
        objectives={"distill": {"weight": 1.0}},
        resume=False,
    )
    checkpoint = tmp_path / "run" / "ckpt.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest.update_progress(
        step=2,
        robot_exposures={"g1": 2},
        checkpoint_path=checkpoint,
        complete=True,
    )

    written = json.loads(path.read_text())
    assert written["status"] == "completed"
    assert written["git"]["available"] is False
    assert written["source"]["splits"]["source"]["file_count"] == 0
    assert written["progress"]["robot_exposures"] == {"g1": 2}
    assert written["checkpoints"][0]["sha256"] == sha256_file(checkpoint)
    checkpoint.write_bytes(b"new checkpoint")
    manifest.update_progress(
        step=3,
        robot_exposures={"g1": 3},
        checkpoint_path=checkpoint,
    )
    assert len(manifest.data["checkpoints"]) == 1
    assert manifest.data["checkpoints"][0]["step"] == 3

    resumed = RunManifest.start(
        path,
        trainer="test",
        repo_root=tmp_path,
        argv=["train.py", "--resume"],
        config={"steps": 2, "device": "cpu"},
        dataset=dataset,
        training={"seed": 3},
        objectives={},
        resume=True,
    )
    assert resumed.data["status"] == "running"
    assert len(resumed.data["invocations"]) == 2
    assert resumed.data["resume_checks"][0]["config_matches_original_except_resume"]
    assert resumed.data["resume_checks"][0]["config_differences"] == {}


def test_rng_state_roundtrip_restores_all_cpu_generators():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))

    random.random()
    np.random.rand()
    torch.rand(())
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected
