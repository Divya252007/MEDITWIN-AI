import os
import re
from pathlib import Path

# Input and output folders
INPUT_DIR = Path("data/raw/sample_notes")
OUTPUT_DIR = Path("data/processed/text")

def clean_text(text):
    """Clean a clinical text note while preserving useful medical information."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def preprocess_text_files():
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(INPUT_DIR.glob("*.txt"))

    if not files:
        print(f"No .txt files found in {INPUT_DIR}")
        print("Add at least one sample clinical note before running.")
        return

    processed_count = 0

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            cleaned_text = clean_text(text)

            output_path = OUTPUT_DIR / file_path.name
            output_path.write_text(cleaned_text, encoding="utf-8")

            processed_count += 1
            print(f"Processed: {file_path.name}")

        except OSError as error:
            print(f"Could not process {file_path.name}: {error}")

    print(f"\nPreprocessing complete. Files processed: {processed_count}")
    print(f"Output folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_text_files()