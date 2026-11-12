import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, f1_score

from src.training.dataset import PTBXLDataset
from src.models.ecg_encoder import ECGEncoder

DATA_PATH = (
    "data/raw/ptbxl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
CHECKPOINT_PATH = "models/checkpoints/ecg_encoder_folded.pth"
BATCH_SIZE = 16


class ECGClassifier(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.encoder = ECGEncoder(output_dim=128)
        self.classifier = nn.Linear(128, class_count)

    def forward(self, ecg):
        return self.classifier(self.encoder(ecg))


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    dataset = PTBXLDataset(DATA_PATH, sampling_rate="lr")
    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=device, weights_only=True
    )

    metadata_path = f"{DATA_PATH}/ptbxl_database.csv"
    import pandas as pd
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")
    folds = dataset.df["ecg_id"].map(metadata["strat_fold"])

    test_indices = np.flatnonzero((folds == 10).to_numpy())
    if len(test_indices) == 0:
        raise ValueError("No records found for test fold 10.")

    model = ECGClassifier(len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            ecg = batch["ecg"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(ecg)
            predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy()

            all_targets.append(labels)
            all_predictions.append(predictions)

    y_true = np.concatenate(all_targets).astype(int)
    y_pred = np.concatenate(all_predictions).astype(int)

    print("\nTest fold: 10")
    print("Test records evaluated:", len(y_true))
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=checkpoint["classes"],
        zero_division=0,
    ))

    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_f1 = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    print(f"Macro F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(f"Micro F1: {micro_f1:.4f} ({micro_f1 * 100:.2f}%)")
    print(
        "\nResearch-only evaluation. This model was trained on a "
        "small subset and is not validated for clinical use."
    )


if __name__ == "__main__":
    main()