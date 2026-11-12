import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.training.dataset import PTBXLDataset
from src.models.ecg_encoder import ECGEncoder


# Update this path to match your actual downloaded folder.
DATA_PATH = (
    "data/raw/ptbxl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

BATCH_SIZE = 16
EPOCHS = 5
MAX_RECORDS = 2000  # Initial pipeline test, not final evaluation
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)

dataset = PTBXLDataset(DATA_PATH, sampling_rate="lr")

# Use a small subset for the first run to test the pipeline.
if MAX_RECORDS and len(dataset) > MAX_RECORDS:
    dataset, _ = random_split(
        dataset,
        [MAX_RECORDS, len(dataset) - MAX_RECORDS],
        generator=torch.Generator().manual_seed(SEED)
    )

if len(dataset) < 2:
    raise ValueError("Not enough labeled ECG records to train.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

class_count = len(PTBXLDataset.CLASSES)

class ECGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ECGEncoder(output_dim=128)
        self.classifier = nn.Linear(128, class_count)

    def forward(self, ecg):
        features = self.encoder(ecg)
        return self.classifier(features)


model = ECGClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("ECG diagnostic training started...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        ecg = batch["ecg"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(ecg)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            ecg = batch["ecg"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ecg)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            predictions = torch.sigmoid(logits) >= 0.5
            correct += (predictions == labels.bool()).sum().item()
            total += labels.numel()

    train_loss = total_loss / max(len(train_loader), 1)
    val_loss = val_loss / max(len(val_loader), 1)
    element_accuracy = 100.0 * correct / max(total, 1)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Validation loss: {val_loss:.4f} | "
        f"Label-wise accuracy: {element_accuracy:.2f}%"
    )

os.makedirs("models/checkpoints", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": PTBXLDataset.CLASSES,
        "input_leads": 12,
        "embedding_dim": 128,
        "sampling_rate": "lr",
    },
    "models/checkpoints/ecg_encoder.pth"
)

print("Training finished.")
print("Checkpoint saved: models/checkpoints/ecg_encoder.pth")
print("Note: this is an initial diagnostic-classification experiment.")