import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.training.dataset import PTBXLDataset
from src.models.ecg_encoder import ECGEncoder

DATA_PATH = (
    "data/raw/ptbxl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
OUTPUT_PATH = "models/checkpoints/ecg_encoder_hyp_experiment.pth"

BATCH_SIZE = 16
MAX_TRAIN_RECORDS = 2000
MAX_VAL_RECORDS = 400
EPOCHS = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Device:", device)


class ECGClassifier(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.encoder = ECGEncoder(output_dim=128)
        self.classifier = nn.Linear(128, class_count)

    def forward(self, ecg):
        return self.classifier(self.encoder(ecg))


def save_checkpoint(state):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    temp_path = OUTPUT_PATH + ".tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, OUTPUT_PATH)


def main():
    dataset = PTBXLDataset(DATA_PATH, sampling_rate="lr")

    metadata_path = os.path.join(DATA_PATH, "ptbxl_database.csv")
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")

    folds = dataset.df["ecg_id"].map(metadata["strat_fold"])

    train_pool = np.flatnonzero(folds.isin(range(1, 9)).to_numpy())
    val_pool = np.flatnonzero((folds == 9).to_numpy())
    test_count = int((folds == 10).sum())

    if len(train_pool) == 0 or len(val_pool) == 0 or test_count == 0:
        raise ValueError("An official PTB-XL fold is empty.")

    rng = np.random.default_rng(SEED)
    train_indices = np.sort(
        rng.choice(
            train_pool,
            size=min(MAX_TRAIN_RECORDS, len(train_pool)),
            replace=False,
        )
    )
    val_indices = np.sort(
        rng.choice(
            val_pool,
            size=min(MAX_VAL_RECORDS, len(val_pool)),
            replace=False,
        )
    )

    print("Training subset:", len(train_indices))
    print("Validation subset:", len(val_indices))
    print("Fold 10 reserved records:", test_count)
    print("Starting small CPU training...")

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ECGClassifier(len(dataset.CLASSES)).to(device)
    class_counts = dataset.labels[train_indices].sum(axis=0)
    class_weights = torch.tensor(
        class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1)),
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
    pos_weight=class_weights
)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_total = 0.0

        for batch in train_loader:
            ecg = batch["ecg"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(ecg), labels)
            loss.backward()
            optimizer.step()
            train_total += loss.item()

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                ecg = batch["ecg"].to(device)
                labels = batch["labels"].to(device)
                val_total += criterion(model(ecg), labels).item()

        train_loss = train_total / max(len(train_loader), 1)
        val_loss = val_total / max(len(val_loader), 1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "model_state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "classes": dataset.CLASSES,
                "input_leads": 12,
                "embedding_dim": 128,
                "sampling_rate": "lr",
                "train_folds": list(range(1, 9)),
                "validation_fold": 9,
                "test_fold": 10,
                "training_subset_size": len(train_indices),
                "validation_subset_size": len(val_indices),
                "best_validation_loss": best_val_loss,
            })
            print("Checkpoint saved:", OUTPUT_PATH)

    print("Training finished.")
    print("Fold 10 was not used for training or validation.")
    print("This small run checks the pipeline; it does not establish clinical accuracy.")


if __name__ == "__main__":
    main()