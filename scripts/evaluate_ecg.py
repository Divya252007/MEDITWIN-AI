import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
from src.training.dataset import PTBXLDataset
from src.models.ecg_encoder import ECGEncoder

DATA_PATH = (
    "data/raw/ptbxl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
CHECKPOINT = "models/checkpoints/ecg_encoder.pth"
BATCH_SIZE = 16

class ECGClassifier(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.encoder = ECGEncoder(output_dim=128)
        self.classifier = nn.Linear(128, class_count)

    def forward(self, ecg):
        return self.classifier(self.encoder(ecg))

def main():
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PTBXLDataset(DATA_PATH, sampling_rate="lr")

    # Use official PTB-XL fold 10 as a provisional evaluation set.
    # This checkpoint was trained on a random subset, so fold 10
    # may overlap with training data. Results are NOT independent.
    metadata = dataset.df
    test_indices = metadata.index[metadata["strat_fold"] == 10].tolist()

    if not test_indices:
        raise ValueError("No fold-10 records found.")

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    classes = checkpoint.get("classes", PTBXLDataset.CLASSES)

    if list(classes) != list(PTBXLDataset.CLASSES):
        raise ValueError(f"Class mismatch: {classes}")

    model = ECGClassifier(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_targets, all_predictions = [], []

    with torch.no_grad():
        for batch in test_loader:
            ecg = batch["ecg"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(ecg)
            predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy()

            all_targets.append(labels)
            all_predictions.append(predictions)

    y_true = np.vstack(all_targets).astype(int)
    y_pred = np.vstack(all_predictions).astype(int)

    print(f"\nDevice: {device}")
    print(f"Evaluated records: {len(test_indices)}")
    print("\nPer-class multilabel report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=classes,
        zero_division=0,
    ))
    print(
        "\nWARNING: This is a provisional diagnostic evaluation. "
        "The existing checkpoint used random splitting, so fold 10 "
        "may overlap with training data. Do not report these results "
        "as independent test performance."
    )

if __name__ == "__main__":
    main()