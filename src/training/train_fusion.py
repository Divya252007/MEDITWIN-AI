from pathlib import Path

CHECKPOINT_DIR = Path("models/checkpoints")

def main():
    print("Fusion training has been paused.")
    print()
    print("The previous fusion model was trained on synthetic")
    print("embeddings and random labels. It is not a valid")
    print("cardiovascular risk predictor.")
    print()
    print("No checkpoint was changed.")
    print()
    print("Next requirements for real fusion training:")
    print("1. Compatible, trained modality feature extractors.")
    print("2. Patient-linked multimodal records.")
    print("3. One consistent, clinically justified target.")
    print("4. Patient-level train/validation/test splits.")
    print("5. Evaluation on held-out patient data.")

    checkpoint = CHECKPOINT_DIR / "fusion_model.pth"
    print()
    print("Existing checkpoint exists:", checkpoint.exists())
    print("Existing checkpoint was not loaded or modified.")

if __name__ == "__main__":
    main()