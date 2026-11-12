
"""
MediTwin-AI: Multimodal Fusion V2
Combines image, ECG, clinical text and tabular embeddings.

Expected dimensions:
    Image:   64
    ECG:    128
    Text:    64
    Tabular: 32

Output:
    Shared patient embedding: 128
    Classification logits: 3
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn


class MediTwinFusionV2(nn.Module):
    def __init__(
        self,
        image_dim: int = 64,
        ecg_dim: int = 128,
        text_dim: int = 64,
        tabular_dim: int = 32,
        shared_dim: int = 128,
        num_classes: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if shared_dim % num_heads != 0:
            raise ValueError(
                "shared_dim must be divisible by num_heads."
            )

        self.expected_dims = {
            "image": image_dim,
            "ecg": ecg_dim,
            "text": text_dim,
            "tabular": tabular_dim,
        }

        self.shared_dim = shared_dim

        # Project all modalities into a common embedding space.
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
        )

        self.ecg_projection = nn.Sequential(
            nn.Linear(ecg_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
        )

        self.tabular_projection = nn.Sequential(
            nn.Linear(tabular_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
        )

        # Learnable identity embeddings for each modality.
        self.modality_embeddings = nn.Parameter(
            torch.randn(1, 4, shared_dim) * 0.02
        )

        # Cross-modal attention.
        self.attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(shared_dim)

        # Feed-forward refinement.
        self.feed_forward = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Dropout(dropout),
        )

        self.feed_forward_norm = nn.LayerNorm(shared_dim)

        # Learn which modalities contribute most.
        self.gate = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.GELU(),
            nn.Linear(shared_dim // 2, 1),
        )

        # Final patient-level representation.
        self.embedding_norm = nn.LayerNorm(shared_dim)

        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def _validate_input(
        self,
        name: str,
        value: Optional[torch.Tensor],
        expected_dim: int,
    ):
        if value is None:
            return None

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None.")

        if value.ndim != 2:
            raise ValueError(
                f"{name} must have shape [batch, {expected_dim}]. "
                f"Received {tuple(value.shape)}."
            )

        if value.shape[1] != expected_dim:
            raise ValueError(
                f"{name} expected {expected_dim} features, "
                f"received {value.shape[1]}."
            )

        if not torch.is_floating_point(value):
            value = value.float()

        return value

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        ecg: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        tabular: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        inputs = {
            "image": self._validate_input(
                "image", image, self.expected_dims["image"]
            ),
            "ecg": self._validate_input(
                "ecg", ecg, self.expected_dims["ecg"]
            ),
            "text": self._validate_input(
                "text", text, self.expected_dims["text"]
            ),
            "tabular": self._validate_input(
                "tabular", tabular, self.expected_dims["tabular"]
            ),
        }

        available = [
            value for value in inputs.values()
            if value is not None
        ]

        if not available:
            raise ValueError("At least one modality must be provided.")

        batch_size = available[0].shape[0]
        device = available[0].device

        for name, value in inputs.items():
            if value is not None:
                if value.shape[0] != batch_size:
                    raise ValueError(
                        f"{name} has a different batch size."
                    )
                if value.device != device:
                    raise ValueError(
                        f"{name} is on a different device."
                    )

        # Build embeddings. Missing modalities use zero placeholders.
        projections = []

        projection_layers = {
            "image": self.image_projection,
            "ecg": self.ecg_projection,
            "text": self.text_projection,
            "tabular": self.tabular_projection,
        }

        for name in ("image", "ecg", "text", "tabular"):
            value = inputs[name]

            if value is None:
                projected = torch.zeros(
                    batch_size,
                    self.shared_dim,
                    device=device,
                    dtype=self.modality_embeddings.dtype,
                )
            else:
                projected = projection_layers[name](value)

            projections.append(projected)

        tokens = torch.stack(projections, dim=1)

        # Mask order: image, ECG, text, tabular.
        if modality_mask is None:
            mask = torch.tensor(
                [
                    inputs[name] is not None
                    for name in ("image", "ecg", "text", "tabular")
                ],
                device=device,
                dtype=torch.bool,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            mask = torch.as_tensor(
                modality_mask, device=device, dtype=torch.bool
            )

            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(batch_size, -1)

            if tuple(mask.shape) != (batch_size, 4):
                raise ValueError(
                    "modality_mask must have shape [batch, 4]."
                )

            # A mask cannot mark an unprovided tensor as available.
            provided = torch.tensor(
                [
                    inputs[name] is not None
                    for name in ("image", "ecg", "text", "tabular")
                ],
                device=device,
                dtype=torch.bool,
            ).unsqueeze(0)

            if torch.any(mask & ~provided):
                raise ValueError(
                    "A modality mask marks a missing input as available."
                )

        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError(
                "Every patient must have at least one available modality."
            )

        tokens = tokens + self.modality_embeddings

        # Prevent missing modalities from participating in attention.
        attended, attention_weights = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~mask,
            need_weights=True,
            average_attn_weights=False,
        )

        tokens = self.attention_norm(tokens + attended)

        refined = self.feed_forward(tokens)
        tokens = self.feed_forward_norm(tokens + refined)

        # Gated modality weighting.
        gate_scores = self.gate(tokens).squeeze(-1)
        gate_scores = gate_scores.masked_fill(~mask, -1e9)

        modality_weights = torch.softmax(gate_scores, dim=1)

        fused_embedding = torch.sum(
            tokens * modality_weights.unsqueeze(-1),
            dim=1,
        )

        fused_embedding = self.embedding_norm(fused_embedding)

        logits = self.classifier(fused_embedding)

        if not return_details:
            return logits

        return {
            "logits": logits,
            "embedding": fused_embedding,
            "modality_weights": modality_weights,
            "attention_weights": attention_weights,
            "modality_mask": mask,
        }


def load_fusion_v2(
    checkpoint_path: str,
    device: Optional[str] = None,
):
    """Load a trained MediTwinFusionV2 checkpoint."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MediTwinFusionV2()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        architecture = checkpoint.get("architecture")

        if architecture != "MediTwinFusionV2":
            raise ValueError(
                "Checkpoint is not a MediTwinFusionV2 checkpoint."
            )

        state_dict = checkpoint.get("model_state_dict")

        if state_dict is None:
            raise ValueError(
                "Checkpoint does not contain model_state_dict."
            )
    else:
        raise ValueError("Invalid Fusion V2 checkpoint format.")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
