"""Dependency-gated advanced analyses and targeted attribution contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gradient_x_input(
    model, inputs: dict, target_positions: list[int], target_labels: list[int]
) -> pd.DataFrame:
    """Targeted gradient×input attribution; never invoked corpus-wide by the pipeline."""
    import torch
    embedding_layer = model.get_input_embeddings()
    embedded = embedding_layer(inputs["input_ids"]).detach().requires_grad_(True)
    forwarded = {key: value for key, value in inputs.items() if key != "input_ids"}
    output = model(inputs_embeds=embedded, **forwarded)
    objective = sum(output.logits[0, position, label] for position, label in zip(target_positions, target_labels))
    objective.backward()
    scores = (embedded.grad * embedded).sum(dim=-1).detach().cpu().numpy()[0]
    return pd.DataFrame({"token_position": np.arange(len(scores)), "attribution": scores,
                         "method": "gradient_x_input"})


def integrated_gradients(
    score_from_embeddings, embeddings, baseline=None, steps: int = 32,
) -> np.ndarray:
    """Integrated Gradients for a caller-supplied scalar target-score function."""
    import torch
    if steps < 2:
        raise ValueError("Integrated Gradients requires at least two integration steps.")
    baseline = torch.zeros_like(embeddings) if baseline is None else baseline
    total_gradient = torch.zeros_like(embeddings)
    for alpha in torch.linspace(0.0, 1.0, steps, device=embeddings.device):
        interpolated = (baseline + alpha * (embeddings - baseline)).detach().requires_grad_(True)
        score = score_from_embeddings(interpolated)
        gradient = torch.autograd.grad(score, interpolated)[0]
        total_gradient += gradient
    attribution = (embeddings - baseline) * total_gradient / steps
    return attribution.detach().cpu().numpy()


def token_occlusion(
    token_ids: np.ndarray, score_function, replacement_id: int,
) -> pd.DataFrame:
    """Targeted token-occlusion score drops for a caller-selected token/span score."""
    token_ids = np.asarray(token_ids, dtype=int)
    original = float(score_function(token_ids))
    rows = []
    for position in range(len(token_ids)):
        occluded = token_ids.copy()
        occluded[position] = replacement_id
        rows.append({"token_position": position,
                     "attribution": original - float(score_function(occluded)),
                     "method": "token_occlusion"})
    return pd.DataFrame(rows)


def attribution_faithfulness(
    original_score: float, comprehensiveness_score: float, sufficiency_score: float
) -> dict[str, float]:
    return {
        "comprehensiveness": float(original_score - comprehensiveness_score),
        "sufficiency": float(original_score - sufficiency_score),
    }


def require_optional(module: str, analysis: str):
    try:
        return __import__(module)
    except ImportError as error:
        raise ImportError(f"{analysis} requires the optional dependency {module!r}.") from error


def targeted_influence(
    checkpoints: list[dict[str, np.ndarray]], validation_gradient: np.ndarray,
    training_gradients: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    """TracIn-style checkpoint gradient-dot-product scores for preselected cases."""
    del checkpoints
    rows = []
    for example_id, gradients in sorted(training_gradients.items()):
        score = sum(float(np.dot(validation_gradient.ravel(), gradient.ravel())) for gradient in gradients)
        rows.append({"training_example_id": example_id, "influence_score": score,
                     "method": "tracin_gradient_dot_product"})
    return pd.DataFrame(rows).sort_values("influence_score", ascending=False, kind="stable")
