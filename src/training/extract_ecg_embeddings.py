
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.dataset import PTBXLDataset
from src.models.ecg_inference import load_ecg_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ptbxl"
PROCESSED_ECG_DIR = PROJECT_ROOT / "data" / "processed" / "ecg"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "multimodal"
OUTPUT_PATH = OUTPUT_DIR / "ecg_embeddings.pt"

BATCH_SIZE = 32


def main():

    print("=" * 55)
    print("MEDTWIN-AI: ECG EMBEDDING EXTRACTION")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset using processed ECG files.
    print("\n[1/4] Loading PTB-XL dataset...")

    dataset = PTBXLDataset(
        data_path=str(DATA_PATH),
        sampling_rate="lr",
        processed_ecg_dir=str(PROCESSED_ECG_DIR)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Total records:", len(dataset))

    # 2. Load trained ECG model.
    print("\n[2/4] Loading ECG encoder...")

    model, classes = load_ecg_model()
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    print("Device:", device)
    print("Classes:", classes)

    # 3. Extract embeddings.
    print("\n[3/4] Extracting ECG embeddings...")

    all_embeddings = []
    all_labels = []

    patient_ids = []
    ecg_ids = []
    record_names = []

    with torch.inference_mode():

        for batch_index, batch in enumerate(loader):

            ecg = batch["ecg"].to(
                device,
                dtype=torch.float32
            )

            # Extract 128-dimensional features.
            features = model.encoder(ecg)

            all_embeddings.append(features.cpu())
            all_labels.append(batch["labels"].cpu())

            patient_ids.extend(
                int(value) for value in batch["patient_id"]
            )

            ecg_ids.extend(
                int(value) for value in batch["ecg_id"]
            )

            record_names.extend(batch["record_name"])

            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(dataset)
            )

            if (
                (batch_index + 1) % 25 == 0
                or processed == len(dataset)
            ):
                print(
                    f"Processed {processed}/{len(dataset)} records"
                )

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # 4. Validate and save.
    print("\n[4/4] Validating and saving...")

    if embeddings.shape != (len(dataset), 128):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if labels.shape != (len(dataset), len(classes)):
        raise ValueError(
            f"Unexpected label shape: {labels.shape}"
        )

    if not torch.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    bundle = {
        "patient_ids": patient_ids,
        "ecg_ids": ecg_ids,
        "record_names": record_names,
        "ecg_embeddings": embeddings,
        "labels": labels,
        "class_names": classes,
        "embedding_dim": 128,
        "modality": "ECG",
        "dataset": "PTB-XL",
        "sampling_rate": "lr",
        "signal_source": "processed_HR_resampled"
    }

    torch.save(bundle, OUTPUT_PATH)

    print("\n" + "=" * 55)
    print("ECG EMBEDDING EXTRACTION COMPLETED!")
    print("=" * 55)

    print("Records:", len(patient_ids))
    print("Unique patients:", len(set(patient_ids)))
    print("Embedding shape:", embeddings.shape)
    print("Labels shape:", labels.shape)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
