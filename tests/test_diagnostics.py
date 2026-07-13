import torch

from snmr.diagnostics import binary_contact_metrics, loss_gradient_diagnostics


def test_loss_gradient_diagnostics_reports_norms_cosines_without_mutating_grad():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    terms = {
        "aligned": parameter.sum(),
        "opposed": -parameter.sum(),
        "orthogonal": parameter[0] - parameter[1],
    }
    result = loss_gradient_diagnostics(
        terms,
        {"aligned": 2.0, "opposed": 1.0, "orthogonal": 1.0},
        {"shared_trunk": [parameter], "output_heads": []},
    )
    assert parameter.grad is None
    assert result["loss_terms"]["aligned"]["weighted"] == -2.0
    assert result["loss_terms"]["aligned"]["gradient_norm"]["shared_trunk"] > 0
    cosine = result["gradient_cosine"]["shared_trunk"]
    assert abs(cosine["aligned|opposed"] + 1.0) < 1e-7
    assert abs(cosine["aligned|orthogonal"]) < 1e-7

    sum(terms.values()).backward()
    assert parameter.grad is not None


def test_binary_contact_metrics_known_confusion_matrix():
    logits = torch.tensor([10.0, 10.0, -10.0, -10.0])
    target = torch.tensor([1, 0, 1, 0])
    metrics = binary_contact_metrics(logits, target)
    assert metrics["target_prevalence"] == 0.5
    assert metrics["predicted_prevalence"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["support"] == 2
