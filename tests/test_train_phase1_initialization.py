import importlib.util
from pathlib import Path

import torch

from snmr.model import SNMR, SNMRConfig


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_phase1.py"
SPEC = importlib.util.spec_from_file_location("train_phase1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
train_phase1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_phase1)


def _model(*, predict_contact: bool) -> SNMR:
    return SNMR(
        SNMRConfig(
            latent_dim=8,
            enc_hidden=16,
            dec_hidden=16,
            predict_contact=predict_contact,
        )
    )


def test_contact_head_initialization_preserves_backbone_and_freezes_it(tmp_path):
    source = _model(predict_contact=False)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "step": 50,
            "xy_scale": 0.875,
            "config": {"predict_contact": False},
        },
        checkpoint,
    )
    target = _model(predict_contact=True)

    state, provenance = train_phase1.load_initial_checkpoint(
        target, checkpoint, "cpu"
    )
    parameters, names = train_phase1.configure_trainable_parameters(
        target, contact_head_only=True
    )

    assert state["step"] == 50
    assert provenance["source_step"] == 50
    assert provenance["new_parameter_keys"]
    assert all(name.startswith("decoder.contact_head.") for name in names)
    assert len(parameters) == len(names)
    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)
    for name, parameter in target.named_parameters():
        assert parameter.requires_grad == name.startswith("decoder.contact_head.")


def test_contact_head_only_mode_keeps_frozen_backbone_in_eval_mode():
    model = _model(predict_contact=True)

    train_phase1.set_model_training_mode(model, contact_head_only=True)

    assert not model.training
    assert model.decoder.contact_head is not None
    assert model.decoder.contact_head.training
