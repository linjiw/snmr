import importlib.util
from pathlib import Path

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
