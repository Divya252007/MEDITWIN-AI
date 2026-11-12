
"""
MediTwin-AI: Fusion V2 Training

Trains the multimodal fusion model using patient-linked embeddings.

Input:
data/processed/multimodal/embeddings.pt

Expected dictionary:
{
    "patient_ids": [...],
    "labels": [...],
    "image": Tensor[N, 64] or None,
    "ecg": Tensor[N, 128] or None,
    "text": Tensor[N, 64] or None,
    "tabular": Tensor[N, 32] or None,
}

Labels:
0 = Low Risk
1 = Medium Risk
2 = High Risk

IMPORTANT:
Rows must represent the same patient across modalities.
Never match separate datasets by row number.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from src.models.fusion_v2 import MediTwinFusionV2


SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

CLASS_NAMES = ["Low Risk", "Medium Risk", "High Risk"]

DIMENSIONS = {
    "image": 64,
    "ecg": 128,
    "text": 64,
    "tabular": 32,
}

MODALITIES = ["image", "ecg", "text", "tabular"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bundle(path):
    """
    Load patient-linked embeddings.

    Do not use randomly generated embeddings or labels.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding bundle not found: {path}"
        )

    bundle = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required = ["patient_ids", "labels"]

    for key in required:
        if key not in bundle:
            raise ValueError(
                f"Missing required field: {key}"
            )

    patient_ids = np.asarray(bundle["patient_ids"])
    labels = torch.as_tensor(
        bundle["labels"], dtype=torch.long
    ).reshape(-1)

    if len(patient_ids) != len(labels):
        raise ValueError(
            "Patient ID count and label count do not match."
        )

    if len(patient_ids) < 20:
        raise ValueError(
            "At least 20 patient records are required "
            "for this training workflow."
        )

    if len(np.unique(patient_ids)) < 3:
        raise ValueError(
            "At least 3 unique patients are required."
        )

    if not torch.all((labels >= 0) & (labels <= 2)):
        raise ValueError(
            "Labels must be 0, 1 or 2."
        )

    embeddings = {}

    for name in MODALITIES:
        value = bundle.get(name)

        if value is None:
            embeddings[name] = None
            continue

        tensor = torch.as_tensor(
            value, dtype=torch.float32
        )

        expected = DIMENSIONS[name]

        if tensor.shape != (len(labels), expected):
            raise ValueError(
                f"{name} must have shape "
                f"({len(labels)}, {expected}); "
                f"received {tuple(tensor.shape)}."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or infinity."
            )

        embeddings[name] = tensor

    if all(embeddings[name] is None for name in MODALITIES):
        raise ValueError("No modality embeddings found.")

    # Build mask for missing modalities.
    modality_mask = torch.stack(
        [
            torch.full(
                (len(labels),),
                embeddings[name] is not None,
                dtype=torch.bool,
            )
            for name in MODALITIES
        ],
        dim=1,
    )

    if "modality_mask" in bundle:
        supplied_mask = torch.as_tensor(
            bundle["modality_mask"], dtype=torch.bool
        )

        if supplied_mask.shape != modality_mask.shape:
            raise ValueError(
                "modality_mask must have shape [N, 4]."
            )

        if torch.any(supplied_mask & ~modality_mask):
            raise ValueError(
                "Mask marks a missing modality as available."
            )

        modality_mask = supplied_mask

    if torch.any(modality_mask.sum(dim=1) == 0):
        raise ValueError(
            "Every patient must have at least one modality."
        )

    # Check each patient has consistent labels.
    patient_label_map = {}

    for patient, label in zip(patient_ids, labels.tolist()):
        patient = str(patient)

        if patient in patient_label_map:
            if patient_label_map[patient] != label:
                raise ValueError(
                    f"Inconsistent labels for patient {patient}."
                )
        else:
            patient_label_map[patient] = label

    return patient_ids, labels, embeddings, modality_mask


class PatientEmbeddingDataset(Dataset):
    def __init__(
        self,
        indices,
        labels,
        embeddings,
        modality_mask,
    ):
        self.indices = torch.as_tensor(
            indices, dtype=torch.long
        )
        self.labels = labels
        self.embeddings = embeddings
        self.modality_mask = modality_mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])

        sample = {}

        for name in MODALITIES:
            value = self.embeddings[name]

            sample[name] = (
                torch.zeros(DIMENSIONS[name])
                if value is None
                else value[row]
            )

        sample["mask"] = self.modality_mask[row]
        sample["label"] = self.labels[row]

        return sample


def split_patients(patient_ids):
    """
    70% train, 15% validation, 15% test.
    All records belonging to one patient stay in one split.
    """

    indices = np.arange(len(patient_ids))

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=SEED,
    )

    train_idx, temporary_idx = next(
        first_split.split(
            indices,
            groups=patient_ids,
        )
    )

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED,
    )

    val_relative, test_relative = next(
        second_split.split(
            temporary_idx,
            groups=patient_ids[temporary_idx],
        )
    )

    val_idx = temporary_idx[val_relative]
    test_idx = temporary_idx[test_relative]

    train_patients = set(patient_ids[train_idx])
    val_patients = set(patient_ids[val_idx])
    test_patients = set(patient_ids[test_idx])

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)

    return train_idx, val_idx, test_idx


def run_epoch(model, loader, optimizer, criterion, device, training):
    model.train(training)

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch in loader:
        image = (
            batch["image"].to(device)
            if batch["image"].shape[1] == 64
            and "image" in batch
            else None
        )

        # The dataset uses zero placeholders for missing
        # global modalities; the mask prevents them being used.
        # Inputs are passed consistently to the fusion model.
        image = batch["image"].to(device)
        ecg = batch["ecg"].to(device)
        text = batch["text"].to(device)
        tabular = batch["tabular"].to(device)

        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        # If an entire modality is absent from the bundle,
        # its all-false mask tells the model to ignore it.
        # The model still needs a tensor or None for each input.
        logits = model(
            image=image,
            ecg=ecg,
            text=text,
            tabular=tabular,
            modality_mask=mask,
        )

        loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/processed/multimodal/embeddings.pt",
    )

    parser.add_argument(
        "--output",
        default="models/checkpoints/fusion_v2.pth",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    args = parser.parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")

    patient_ids, labels, embeddings, mask = load_bundle(
        Path(args.data)
    )

    train_idx, val_idx, test_idx = split_patients(
        patient_ids
    )

    print(
        f"Patients: {len(np.unique(patient_ids))} | "
        f"Train records: {len(train_idx)} | "
        f"Validation records: {len(val_idx)} | "
        f"Test records: {len(test_idx)}"
    )

    train_dataset = PatientEmbeddingDataset(
        train_idx, labels, embeddings, mask
    )

    val_dataset = PatientEmbeddingDataset(
        val_idx, labels, embeddings, mask
    )

    test_dataset = PatientEmbeddingDataset(
        test_idx, labels, embeddings, mask
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MediTwinFusionV2().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            training=True,
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model.")

    model.load_state_dict(best_state)

    test_loss, test_acc, test_f1 = run_epoch(
        model,
        test_loader,
        optimizer,
        criterion,
        device,
        training=False,
    )

    print("\nFinal held-out test results")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro F1: {test_f1:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "architecture": "MediTwinFusionV2",
            "model_state_dict": model.state_dict(),
            "expected_dims": DIMENSIONS,
            "shared_dim": 128,
            "num_classes": 3,
            "class_names": CLASS_NAMES,
            "best_validation_macro_f1": best_val_f1,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "seed": SEED,
        },
        output_path,
    )

    print(f"\nSaved Fusion V2 checkpoint: {output_path}")


if __name__ == "__main__":
    main()
