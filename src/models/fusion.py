
import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """
    MediTwin-AI multimodal fusion model.

    Inputs:
        image:   (batch, 64)
        ecg:     (batch, 128)
        text:    (batch, 64)
        tabular: (batch, 32)

    Output:
        logits:  (batch, 3)
    """

    def __init__(self, num_classes=3):
        super().__init__()

        # Convert every modality to a common 64-dimensional space.
        self.image_projection = nn.Linear(64, 64)
        self.ecg_projection = nn.Linear(128, 64)
        self.text_projection = nn.Linear(64, 64)
        self.tabular_projection = nn.Linear(32, 64)

        # 4 modalities × 64 features = 256 features.
        self.fusion = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, ecg, text, tabular):

        inputs = {
            "image": (image, 64),
            "ecg": (ecg, 128),
            "text": (text, 64),
            "tabular": (tabular, 32),
        }

        batch_size = image.shape[0]

        for name, (tensor, expected_dim) in inputs.items():
            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} must have shape (batch, features)"
                )

            if tensor.shape[1] != expected_dim:
                raise ValueError(
                    f"{name} must have {expected_dim} features; "
                    f"received {tensor.shape[1]}"
                )

            if tensor.shape[0] != batch_size:
                raise ValueError(
                    "All modalities must have the same batch size"
                )

            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains invalid values")

        # Project all modalities to 64 dimensions.
        image = self.image_projection(image.float())
        ecg = self.ecg_projection(ecg.float())
        text = self.text_projection(text.float())
        tabular = self.tabular_projection(tabular.float())

        combined = torch.cat(
            [image, ecg, text, tabular],
            dim=1
        )

        fused_features = self.fusion(combined)
        logits = self.classifier(fused_features)

        return logits
