
"""
MediTwin-AI: Multimodal Embedding Pipeline V2

Connects embeddings produced by the existing encoders
to MediTwinFusionV2.

Expected embedding dimensions:
Image: 64
ECG: 128
Clinical text: 64
Tabular: 32
"""

from pathlib import Path
from typing import Optional

import torch

from src.models.fusion_v2 import (
    MediTwinFusionV2,
    load_fusion_v2,
)


CLASS_NAMES = [
    "Low Risk",
    "Medium Risk",
    "High Risk",
]

MODALITY_DIMENSIONS = {
    "image": 64,
    "ecg": 128,
    "text": 64,
    "tabular": 32,
}


def get_device():
    """Select GPU when available, otherwise CPU."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def prepare_embedding(
    embedding,
    modality_name: str,
    device=None,
):
    """
    Convert an existing encoder output into a [1, features] tensor.

    Accepts:
        - PyTorch tensors
        - Lists
        - NumPy arrays

    Does not generate or modify clinical features.
    """

    if embedding is None:
        return None

    expected_dim = MODALITY_DIMENSIONS[modality_name]

    if device is None:
        device = get_device()

    tensor = torch.as_tensor(
        embedding,
        dtype=torch.float32,
        device=device,
    )

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError(
            f"{modality_name} embedding must have shape "
            f"[{expected_dim}] or [1, {expected_dim}]. "
            f"Received {tuple(tensor.shape)}."
        )

    if tensor.shape[1] != expected_dim:
        raise ValueError(
            f"{modality_name} expected {expected_dim} dimensions, "
            f"but received {tensor.shape[1]}."
        )

    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{modality_name} embedding contains NaN or infinity."
        )

    return tensor


def prepare_patient_embeddings(
    image_embedding=None,
    ecg_embedding=None,
    text_embedding=None,
    tabular_embedding=None,
    device=None,
):
    """
    Prepare all available embeddings for one patient.

    Missing modalities remain None and are handled
    by the fusion model's modality mask.
    """

    if device is None:
        device = get_device()

    embeddings = {
        "image": prepare_embedding(
            image_embedding, "image", device
        ),
        "ecg": prepare_embedding(
            ecg_embedding, "ecg", device
        ),
        "text": prepare_embedding(
            text_embedding, "text", device
        ),
        "tabular": prepare_embedding(
            tabular_embedding, "tabular", device
        ),
    }

    if all(value is None for value in embeddings.values()):
        raise ValueError(
            "No patient embeddings were provided."
        )

    modality_mask = torch.tensor(
        [[embeddings[name] is not None for name in (
            "image", "ecg", "text", "tabular"
        )]],
        dtype=torch.bool,
        device=device,
    )

    return embeddings, modality_mask


def predict_with_fusion_v2(
    model: MediTwinFusionV2,
    image_embedding=None,
    ecg_embedding=None,
    text_embedding=None,
    tabular_embedding=None,
):
    """
    Run multimodal fusion for one patient's existing embeddings.

    Returns research-model logits, softmax scores,
    modality weights and the fused representation.

    Scores are not clinically calibrated probabilities.
    """

    device = next(model.parameters()).device

    embeddings, modality_mask = prepare_patient_embeddings(
        image_embedding=image_embedding,
        ecg_embedding=ecg_embedding,
        text_embedding=text_embedding,
        tabular_embedding=tabular_embedding,
        device=device,
    )

    model.eval()

    with torch.inference_mode():
        output = model(
            image=embeddings["image"],
            ecg=embeddings["ecg"],
            text=embeddings["text"],
            tabular=embeddings["tabular"],
            modality_mask=modality_mask,
            return_details=True,
        )

        logits = output["logits"]
        scores = torch.softmax(logits, dim=1)

    predicted_class = int(scores.argmax(dim=1).item())

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "scores": scores.squeeze(0).cpu().tolist(),
        "logits": logits.squeeze(0).cpu().tolist(),
        "fused_embedding": output[
            "embedding"
        ].squeeze(0).cpu(),
        "modality_weights": {
            name: float(output["modality_weights"][0, index].item())
            for index, name in enumerate(
                ("image", "ecg", "text", "tabular")
            )
            if modality_mask[0, index].item()
        },
        "modality_mask": modality_mask.squeeze(0).cpu().tolist(),
    }


def load_patient_fusion_model(
    checkpoint_path="models/checkpoints/fusion_v2.pth",
):
    """
    Load the separately trained Fusion V2 model.

    This does not replace the existing fusion_model.pth.
    """

    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fusion V2 checkpoint not found: {path}. "
            "Train Fusion V2 before using this prediction function."
        )

    device = get_device()

    model = load_fusion_v2(
        str(path),
        device=str(device),
    )

    return model
