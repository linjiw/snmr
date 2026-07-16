import importlib.util
from pathlib import Path
import subprocess
import sys

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_gate1b_m3.py"
SPEC = importlib.util.spec_from_file_location("analyze_gate1b_m3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
m3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m3)


def test_compare_checkpoint_backbone_accepts_only_new_contact_head():
    source = {"encoder.weight": torch.tensor([1.0])}
    trained = {
        **source,
        "decoder.contact_head.0.weight": torch.tensor([2.0]),
        "decoder.contact_head.0.bias": torch.tensor([3.0]),
        "decoder.contact_head.2.weight": torch.tensor([4.0]),
        "decoder.contact_head.2.bias": torch.tensor([5.0]),
    }

    result = m3.compare_checkpoint_backbone(source, trained)

    assert result["passed"]
    assert not result["changed_inherited_keys"]
    assert result["new_keys"] == sorted(m3.EXPECTED_TRAINABLE)


def test_compare_checkpoint_backbone_rejects_changed_inherited_tensor():
    source = {"encoder.weight": torch.tensor([1.0])}
    trained = {
        "encoder.weight": torch.tensor([2.0]),
        **{
            name: torch.tensor([3.0])
            for name in m3.EXPECTED_TRAINABLE
        },
    }

    result = m3.compare_checkpoint_backbone(source, trained)

    assert not result["passed"]
    assert result["changed_inherited_keys"] == ["encoder.weight"]


def test_validate_artifact_revision_accepts_clean_launch_revision(monkeypatch):
    class Result:
        returncode = 0

    monkeypatch.setattr(m3.subprocess, "run", lambda *args, **kwargs: Result())

    errors = m3.validate_artifact_revision(
        "launch",
        {"sha": "artifact", "dirty": False},
    )

    assert errors == []


def test_validate_artifact_revision_rejects_dirty_artifact(monkeypatch):
    class Result:
        returncode = 0

    monkeypatch.setattr(m3.subprocess, "run", lambda *args, **kwargs: Result())

    errors = m3.validate_artifact_revision(
        "launch",
        {"sha": "artifact", "dirty": True},
    )

    assert errors == ["artifact revision is dirty"]


def test_analyzer_copy_can_import_package_outside_repository(tmp_path):
    copied = tmp_path / SCRIPT.name
    copied.write_bytes(SCRIPT.read_bytes())

    result = subprocess.run(
        [sys.executable, str(copied), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
