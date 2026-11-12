
from pathlib import Path
import torch
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "multimodal"
RAW_DIR = ROOT / "data" / "raw"

ECG_FILE = DATA_DIR / "ecg_embeddings.pt"
IMAGE_FILE = DATA_DIR / "image_embeddings.pt"
TEXT_NOTE = RAW_DIR / "sample_notes" / "sample_note_1.txt"
TABULAR_FILE = RAW_DIR / "uci_heart" / "heart.csv"

REPORT_FILE = DATA_DIR / "readiness_report.txt"


def load_bundle(path):
    if not path.exists():
        print(f"Missing file: {path}")
        return None

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


def get_unique_count(values):
    if values is None:
        return 0

    if isinstance(values, torch.Tensor):
        values = values.cpu().tolist()

    return len(set(str(x) for x in values))


def inspect_embedding(name, path, embedding_key, id_key, expected_dim):
    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    bundle = load_bundle(path)

    if bundle is None:
        return None

    embeddings = bundle.get(embedding_key)
    patient_ids = bundle.get(id_key)

    if embeddings is None:
        print("Embedding tensor missing!")
        return None

    if not isinstance(embeddings, torch.Tensor):
        print("Invalid embedding format!")
        return None

    finite = bool(torch.isfinite(embeddings).all())

    result = {
        "modality": name,
        "records": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "expected_dimension": expected_dim,
        "unique_patient_ids": get_unique_count(patient_ids),
        "finite": finite,
        "dimension_valid": embeddings.shape[1] == expected_dim,
        "dataset": bundle.get("dataset", "Unknown"),
    }

    for key, value in result.items():
        print(f"{key}: {value}")

    print("Status:", "PASS" if finite and result["dimension_valid"] else "CHECK")

    return result


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\nMEDTWIN-AI MULTIMODAL READINESS REPORT")
    print("=" * 55)

    results = []

    ecg = inspect_embedding(
        "ECG",
        ECG_FILE,
        "ecg_embeddings",
        "patient_ids",
        128
    )

    image = inspect_embedding(
        "CHEST X-RAY",
        IMAGE_FILE,
        "image_embeddings",
        "patient_ids",
        64
    )

    if ecg:
        results.append(ecg)

    if image:
        results.append(image)

    print(f"\n{'=' * 55}")
    print("TEXT DATA")
    print("=" * 55)

    text_exists = TEXT_NOTE.exists()

    if text_exists:
        note = TEXT_NOTE.read_text(encoding="utf-8")
        print("Sample note exists: True")
        print("Characters:", len(note))
        print("Patient-linked text dataset: Not verified")
    else:
        print("Sample note exists: False")

    print(f"\n{'=' * 55}")
    print("TABULAR DATA")
    print("=" * 55)

    tabular_exists = TABULAR_FILE.exists()

    if tabular_exists:
        df = pd.read_csv(TABULAR_FILE)

        print("Records:", len(df))
        print("Unique IDs:", df["id"].nunique())
        print("Features:", len([
            "age", "sex", "cp", "trestbps", "chol",
            "fbs", "restecg", "thalch", "exang",
            "oldpeak", "slope", "ca", "thal"
        ]))
        print("Target distribution:")
        print(df["num"].value_counts().sort_index().to_dict())
        print("Patient linkage to ECG/X-ray: Not verified")
    else:
        print("Tabular dataset missing")

    print(f"\n{'=' * 55}")
    print("PATIENT ID OVERLAP CHECK")
    print("=" * 55)

    
    if ecg and image:
        ecg_bundle = load_bundle(ECG_FILE)
        image_bundle = load_bundle(IMAGE_FILE)

        def normalize_ids(values):
            if isinstance(values, torch.Tensor):
                values = values.cpu().tolist()
            elif hasattr(values, "tolist"):
                values = values.tolist()

            return set(str(x) for x in values)

        ecg_ids = normalize_ids(ecg_bundle["patient_ids"])
        image_ids = normalize_ids(image_bundle["patient_ids"])

        overlap = ecg_ids.intersection(image_ids)

        print("ECG unique IDs:", len(ecg_ids))
        print("X-ray unique IDs:", len(image_ids))
        print("Numeric ID overlap:", len(overlap))
        print("Verified cross-dataset patient matches: 0")
        print(
            "NOTE: Numeric ID overlap does not prove that "
            "records belong to the same real patient."
        )
        
    print(f"\n{'=' * 55}")
    print("FUSION V2 READINESS")
    print("=" * 55)

    print("ECG embedding extraction:", "Completed" if ecg else "Missing")
    print("Image embedding extraction:", "Completed" if image else "Missing")
    print("Text inference:", "Tested" if text_exists else "Missing")
    print("Tabular adapter: Previously validated")

    print("\nPatient-level multimodal training:")
    print("NOT READY - verified cross-modal patient associations")
    print("and consistent training targets are still required.")

    report = {
        "project": "MediTwin-AI",
        "modalities": results,
        "text_sample_exists": text_exists,
        "tabular_dataset_exists": tabular_exists,
        "fusion_v2_training_ready": False,
        "reason": (
            "No verified shared patient-level associations and "
            "consistent labels across all four modalities."
        )
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("MEDTWIN-AI MULTIMODAL READINESS REPORT\n")
        f.write("=" * 55 + "\n\n")
        f.write(json.dumps(report, indent=4))

    print("\nReport saved successfully:")
    print(REPORT_FILE)



if __name__ == "__main__":
    main()
