from typing import Optional

import torch


CLASS_NAMES = ["Low", "Medium", "High"]


def get_demo_output(
    logits: torch.Tensor,
    model_validated: bool = False,
) -> dict:
    """
    Format model output for a demonstration.

    This does not establish clinical cardiovascular risk.
    Set model_validated=True only after appropriate training
    and independent validation have been completed.
    """
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(
            "Expected logits with shape (batch_size, 3)."
        )

    if not torch.isfinite(logits).all():
        raise ValueError("Model output contains invalid values.")

    probabilities = torch.softmax(logits, dim=1)
    predicted_indices = probabilities.argmax(dim=1)

    results = []

    for i, index in enumerate(predicted_indices.tolist()):
        results.append({
            "label": CLASS_NAMES[index],
            "probabilities": {
                CLASS_NAMES[j]: round(
                    probabilities[i, j].item(), 4
                )
                for j in range(3)
            },
            "status": (
                "Model validated"
                if model_validated
                else "DEMO ONLY — not a clinical risk prediction"
            ),
        })

    return {
        "results": results,
        "is_clinically_validated": model_validated,
    }