"""Loss-scale, gradient-conflict, and binary-contact diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def model_parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """Return disjoint shared-trunk and decoder-output parameter groups."""
    output_prefixes = (
        "decoder.angle_head.",
        "decoder.root_head.",
        "decoder.contact_head.",
    )
    groups: dict[str, list[torch.nn.Parameter]] = {
        "shared_trunk": [],
        "output_heads": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = "output_heads" if name.startswith(output_prefixes) else "shared_trunk"
        groups[group].append(parameter)
    return groups


def loss_gradient_diagnostics(
    terms: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    parameter_groups: Mapping[str, Sequence[torch.nn.Parameter]],
) -> dict:
    """Measure each weighted term's gradient without changing ``parameter.grad``.

    Cosines are reported for every active term pair in each parameter group. A cosine is ``None``
    when either term has zero gradient in that group.
    """
    ordered_parameters: list[torch.nn.Parameter] = []
    parameter_index: dict[int, int] = {}
    group_indices: dict[str, list[int]] = {}
    for group_name, parameters in parameter_groups.items():
        indices = []
        for parameter in parameters:
            key = id(parameter)
            if key not in parameter_index:
                parameter_index[key] = len(ordered_parameters)
                ordered_parameters.append(parameter)
            indices.append(parameter_index[key])
        group_indices[group_name] = indices

    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    term_records = {}
    for name, term in terms.items():
        weight = float(weights.get(name, 1.0))
        weighted = term * weight
        if ordered_parameters and weighted.requires_grad:
            grads = torch.autograd.grad(
                weighted,
                ordered_parameters,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            grads = tuple(None for _ in ordered_parameters)
        gradients[name] = grads
        term_records[name] = {
            "raw": float(term.detach()),
            "weight": weight,
            "weighted": float(weighted.detach()),
            "gradient_norm": {
                group_name: _gradient_norm(grads, indices)
                for group_name, indices in group_indices.items()
            },
        }

    cosines: dict[str, dict[str, float | None]] = {}
    names = list(terms)
    for group_name, indices in group_indices.items():
        group_cosines = {}
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                group_cosines[f"{left}|{right}"] = _gradient_cosine(
                    gradients[left], gradients[right], indices
                )
        cosines[group_name] = group_cosines
    return {"loss_terms": term_records, "gradient_cosine": cosines}


def _gradient_norm(
    gradients: Sequence[torch.Tensor | None],
    indices: Sequence[int],
) -> float:
    squared_norm = 0.0
    for index in indices:
        gradient = gradients[index]
        if gradient is not None:
            squared_norm += float(gradient.detach().double().square().sum())
    return math.sqrt(squared_norm)


def _gradient_cosine(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    indices: Sequence[int],
) -> float | None:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for index in indices:
        grad_left = left[index]
        grad_right = right[index]
        if grad_left is not None:
            left_sq += float(grad_left.detach().double().square().sum())
        if grad_right is not None:
            right_sq += float(grad_right.detach().double().square().sum())
        if grad_left is not None and grad_right is not None:
            dot += float((grad_left.detach().double() * grad_right.detach().double()).sum())
    if left_sq == 0.0 or right_sq == 0.0:
        return None
    return dot / math.sqrt(left_sq * right_sq)


def binary_contact_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    from_logits: bool = True,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Contact prevalence, precision, recall, and F1 for equally shaped tensors."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    probability = torch.sigmoid(prediction) if from_logits else prediction
    predicted = probability >= threshold
    actual = target.to(dtype=torch.bool, device=predicted.device)
    true_positive = int((predicted & actual).sum())
    false_positive = int((predicted & ~actual).sum())
    false_negative = int((~predicted & actual).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "target_prevalence": float(actual.float().mean()),
        "predicted_prevalence": float(predicted.float().mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(actual.sum()),
    }
