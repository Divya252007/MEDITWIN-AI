import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

from src.models.image_encoder import ImageEncoder


# -----------------------------
# Paths and settings
# -----------------------------
CSV_FILE = "data/processed/image/cxr8_available.csv"
IMAGE_DIR = "data/raw/cxr8/images"
CHECKPOINT = "models/checkpoints/image_encoder.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 2
MAX_IMAGES = 128  # Small pipeline test first
LEARNING_RATE = 0.001
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", DEVICE)


# -----------------------------
# Load metadata
# -----------------------------
df = pd.read_csv(CSV_FILE)

required_columns = {"Image Index", "Finding Labels"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain these columns: {required_columns}"
    )

df = df.drop_duplicates(subset=["Image Index"]).copy()

# Discover labels from the available metadata.
classes = sorted({
    label.strip()
    for value in df["Finding Labels"].fillna("No Finding")
    for label in str(value).split("|")
    if label.strip()
})

if not classes:
    raise ValueError("No labels found in the dataset CSV.")

class_to_index = {name: i for i, name in enumerate(classes)}

# Verify actual image paths.
df["image_path"] = df["Image Index"].apply(
    lambda name: os.path.join(IMAGE_DIR, str(name))
)
df = df[df["image_path"].map(os.path.isfile)].reset_index(drop=True)

if len(df) < 2:
    raise ValueError(
        "Fewer than two images were found. Check IMAGE_DIR."
    )

print("Available images:", len(df))
print("Finding classes:", classes)


# -----------------------------
# Dataset
# -----------------------------
class CXRDataset(Dataset):
    def __init__(self, dataframe, transform, classes):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.classes = classes
        self.class_to_index = {
            name: i for i, name in enumerate(classes)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")

        image = self.transform(image)

        target = torch.zeros(len(self.classes), dtype=torch.float32)

        for label in str(row["Finding Labels"]).split("|"):
            label = label.strip()
            if label in self.class_to_index:
                target[self.class_to_index[label]] = 1.0

        return image, target


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Limit the initial run to verify the complete pipeline.
if MAX_IMAGES and len(df) > MAX_IMAGES:
    df = df.sample(
        n=MAX_IMAGES, random_state=SEED
    ).reset_index(drop=True)

dataset = CXRDataset(df, transform, classes)

train_size = max(1, int(0.8 * len(dataset)))
val_size = len(dataset) - train_size

if val_size == 0:
    raise ValueError("Need at least two images for train/validation.")

train_set, val_set = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)


# -----------------------------
# Model
# -----------------------------
print("Loading Vision Transformer...")
print("Pretrained weights may need to download on first run.")

class CXRClassifier(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()

        self.encoder = ImageEncoder(
            embedding_dim=64, pretrained=True
        )

        # Freeze the pretrained ViT backbone for the first test.
        for parameter in self.encoder.encoder.parameters():
            parameter.requires_grad = False

        self.classifier = nn.Linear(64, number_of_classes)

    def forward(self, images):
        embeddings = self.encoder(images)
        return self.classifier(embeddings)


model = CXRClassifier(len(classes)).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)


# -----------------------------
# Training and validation
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, targets)
            val_loss += loss.item()

            predictions = torch.sigmoid(logits) >= 0.5
            correct += (
                predictions == targets.bool()
            ).sum().item()
            total += targets.numel()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss / len(train_loader):.4f} | "
        f"Validation loss: {val_loss / len(val_loader):.4f} | "
        f"Label-wise accuracy: {100 * correct / max(total, 1):.2f}%"
    )


# -----------------------------
# Save checkpoint
# -----------------------------
os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "embedding_dim": 64,
        "image_size": IMAGE_SIZE,
        "model_name": "ViT_B_16",
    },
    CHECKPOINT
)

print("Image training pipeline completed.")
print("Checkpoint saved to:", CHECKPOINT)
print("Initial subset run only; not final model evaluation.")