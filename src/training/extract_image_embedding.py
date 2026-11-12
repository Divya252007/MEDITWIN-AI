
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pathlib import Path
import torch
from src.models.image_encoder import ImageEncoder

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = ROOT / "models" / "checkpoints" / "image_encoder.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_encoder():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = ImageEncoder(
        embedding_dim=checkpoint.get("embedding_dim", 64),
        pretrained=False
    )

    saved_state = checkpoint["model_state_dict"]

    encoder_state = {
        key[len("encoder."):]: value
        for key, value in saved_state.items()
        if key.startswith("encoder.")
    }

    model.load_state_dict(encoder_state, strict=True)

    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "image"
    / "cxr8_processed.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "image_encoder.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multimodal"
)

OUTPUT_PATH = OUTPUT_DIR / "image_embeddings.pt"

BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Same 15 classes used by your image checkpoint.
DEFAULT_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


# =========================================================
# IMAGE TRANSFORMATION
# Matches demo_encoders.py exactly.
# =========================================================

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# IMAGE DATASET
# =========================================================

class CXRImageDataset(Dataset):

    def __init__(self, dataframe, classes):

        self.dataframe = dataframe.reset_index(drop=True)
        self.classes = classes

        self.class_to_index = {
            name: index
            for index, name in enumerate(classes)
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = Path(str(row["Processed Image"]))

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        # Load and preprocess image.
        try:
            with Image.open(image_path) as image:
                image_tensor = IMAGE_TRANSFORM(
                    image.convert("RGB")
                )

        except Exception as error:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from error

        # Convert original disease labels to multi-hot vector.
        label_vector = np.zeros(
            len(self.classes),
            dtype=np.float32
        )

        finding_labels = str(row["Finding Labels"])

        for label in finding_labels.split("|"):

            label = label.strip()

            if label in self.class_to_index:

                label_index = self.class_to_index[label]

                label_vector[label_index] = 1.0

        # Preserve original patient ID.
        patient_id = str(int(row["Patient ID"]))

        return {
            "image": image_tensor,
            "labels": torch.tensor(
                label_vector,
                dtype=torch.float32
            ),
            "patient_id": patient_id,
            "image_name": str(row["Image Index"]),
            "image_path": str(image_path)
        }


# =========================================================
# MAIN EXTRACTION
# =========================================================

def main():

    print("=" * 60)
    print("MEDTWIN-AI: CHEST X-RAY EMBEDDING EXTRACTION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # 1. Validate files
    # -----------------------------------------------------

    print("\n[1/5] Checking dataset and checkpoint...")

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Image CSV not found: {CSV_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Image checkpoint not found: {CHECKPOINT_PATH}"
        )

    dataframe = pd.read_csv(CSV_PATH)

    required_columns = [
        "Image Index",
        "Finding Labels",
        "Patient ID",
        "Processed Image"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    dataframe = dataframe.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # 2. Read checkpoint metadata
    # -----------------------------------------------------

    print("\n[2/5] Reading image checkpoint metadata...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    classes = checkpoint.get(
        "classes",
        DEFAULT_CLASSES
    )

    embedding_dim = checkpoint.get(
        "embedding_dim",
        64
    )

    image_size = checkpoint.get(
        "image_size",
        224
    )

    if classes != DEFAULT_CLASSES:
        raise ValueError(
            "Checkpoint disease classes do not match "
            "the expected ChestX-ray8 class order."
        )

    if embedding_dim != 64:
        raise ValueError(
            f"Expected 64-dimensional embeddings, "
            f"got {embedding_dim}."
        )

    if image_size != 224:
        raise ValueError(
            f"Expected image size 224, got {image_size}."
        )

    print("Model:", checkpoint.get("model_name"))
    print("Image size:", image_size)
    print("Embedding dimension:", embedding_dim)
    print("Disease classes:", len(classes))

    del checkpoint

    # -----------------------------------------------------
    # 3. Load image encoder
    # -----------------------------------------------------

    print("\n[3/5] Loading trained ViT-B/16 encoder...")

    model = load_image_encoder()

    model.eval()

    model_device = next(model.parameters()).device

    print("Model device:", model_device)

    # -----------------------------------------------------
    # 4. Prepare image loader
    # -----------------------------------------------------

    print("\n[4/5] Preparing image dataset...")

    dataset = CXRImageDataset(
        dataframe=dataframe,
        classes=classes
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("Total images:", len(dataset))
    print(
        "Unique patients:",
        dataframe["Patient ID"].nunique()
    )

    # -----------------------------------------------------
    # 5. Extract image embeddings
    # -----------------------------------------------------

    print("\nStarting feature extraction...")

    all_embeddings = []
    all_labels = []

    patient_ids = []
    image_names = []
    image_paths = []

    with torch.inference_mode():

        for batch_index, batch in enumerate(loader):

            images = batch["image"].to(
                model_device,
                dtype=torch.float32
            )

            embeddings = model(images)

            all_embeddings.append(
                embeddings.cpu()
            )

            all_labels.append(
                batch["labels"].cpu()
            )

            patient_ids.extend(
                batch["patient_id"]
            )

            image_names.extend(
                batch["image_name"]
            )

            image_paths.extend(
                batch["image_path"]
            )

            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(dataset)
            )

            if (
                (batch_index + 1) % 10 == 0
                or processed == len(dataset)
            ):

                print(
                    f"Processed {processed}/{len(dataset)} images"
                )

    # -----------------------------------------------------
    # Combine extracted features
    # -----------------------------------------------------

    embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    # -----------------------------------------------------
    # Validate output
    # -----------------------------------------------------

    print("\nValidating extracted embeddings...")

    if embeddings.shape != (len(dataset), 64):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if labels.shape != (len(dataset), len(classes)):
        raise ValueError(
            f"Unexpected label shape: {labels.shape}"
        )

    if not torch.isfinite(embeddings).all():
        raise ValueError(
            "Image embeddings contain NaN or infinite values."
        )

    if len(patient_ids) != len(embeddings):
        raise ValueError(
            "Patient ID count does not match embedding count."
        )

    # -----------------------------------------------------
    # Save image embedding bundle
    # -----------------------------------------------------

    bundle = {

        "patient_ids": patient_ids,

        "image_names": image_names,

        "image_paths": image_paths,

        "image_embeddings": embeddings,

        "labels": labels,

        "class_names": classes,

        "embedding_dim": 64,

        "image_size": 224,

        "model_name": "ViT_B_16",

        "modality": "Chest X-ray",

        "dataset": "ChestX-ray8",

        "num_images": len(dataset),

        "num_unique_patients": len(set(patient_ids)),

    }

    print("\nSaving image embedding bundle...")

    torch.save(
        bundle,
        OUTPUT_PATH
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("CHEST X-RAY EXTRACTION COMPLETED!")
    print("=" * 60)

    print("Total images:", len(embeddings))
    print("Unique patients:", len(set(patient_ids)))
    print("Embedding shape:", embeddings.shape)
    print("Label shape:", labels.shape)
    print("Embedding dimension:", embedding_dim)
    print("Disease classes:", len(classes))

    print("\nSaved file:")
    print(OUTPUT_PATH)

    print("\nIMAGE EMBEDDING EXTRACTION SUCCESSFUL!")


if __name__ == "__main__":
    main()
