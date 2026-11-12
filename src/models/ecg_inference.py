from pathlib import Path

import torch
import torch.nn as nn

from src.models.ecg_encoder import ECGEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints" / "ecg_encoder.pth"
)


class ECGClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = ECGEncoder(output_dim=128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, ecg):
        features = self.encoder(ecg)
        return self.classifier(features)


def load_ecg_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"ECG checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]
    model = ECGClassifier(num_classes=len(classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes


def extract_ecg_features(ecg_tensor):
    """
    Extract features from a preprocessed ECG tensor.
    Expected shape: (batch, 12, time).
    """
    model, classes = load_ecg_model()

    if ecg_tensor.ndim != 3 or ecg_tensor.shape[1] != 12:
        raise ValueError(
            "Expected ECG tensor shape (batch, 12, time)."
        )

    with torch.no_grad():
        features = model.encoder(ecg_tensor.float())
        logits = model.classifier(features)

    return {
        "features": features,
        "logits": logits,
        "classes": classes,
    }