
from pathlib import Path
import torch
from transformers import AutoTokenizer
from src.models.text_encoder import TextEncoder

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
NOTES_DIR = Path("data/processed/text")
CHECKPOINT_PATH = Path("models/checkpoints/text_encoder.pth")


def main():
    files = list(NOTES_DIR.glob("*.txt"))

    if not files:
        print(f"No cleaned text files found in {NOTES_DIR}")
        return

    note = files[0].read_text(encoding="utf-8").strip()
    if not note:
        print("The sample note is empty.")
        return

    print("Loading BioBERT tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TextEncoder(MODEL_NAME, embedding_dim=64)

    inputs = tokenizer(
        note,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    model.eval()
    with torch.no_grad():
        embedding = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "embedding_dim": 64,
        },
        CHECKPOINT_PATH,
    )

    print("Text encoder test completed.")
    print("Embedding shape:", tuple(embedding.shape))
    print("Checkpoint saved:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()