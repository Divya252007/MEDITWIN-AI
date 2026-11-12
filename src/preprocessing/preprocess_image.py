import os
import pandas as pd
from PIL import Image

# Paths
BASE_DIR = "data/raw/cxr8"
IMAGE_DIR = os.path.join(BASE_DIR, "images")

INPUT_CSV = "data/processed/image/cxr8_available.csv"

OUTPUT_DIR = "data/processed/image/images"
OUTPUT_CSV = "data/processed/image/cxr8_processed.csv"

# Image size for the deep learning model
IMAGE_SIZE = (224, 224)

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load available-image CSV
df = pd.read_csv(INPUT_CSV)

print("CXR8 dataset loaded!")
print("Images to process:", len(df))

processed_rows = []
failed = []

# Process images
for count, (_, row) in enumerate(df.iterrows(), start=1):

    image_name = str(row["Image Index"])

    input_path = os.path.join(IMAGE_DIR, image_name)

    output_name = os.path.splitext(image_name)[0] + ".jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    try:
        # Open image
        image = Image.open(input_path)

        # Convert to RGB
        image = image.convert("RGB")

        # Resize to 224 x 224
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        # Save processed image
        image.save(output_path, "JPEG", quality=90)

        # Store processed path
        new_row = row.copy()
        new_row["Processed Image"] = output_path.replace("\\", "/")

        processed_rows.append(new_row)

    except Exception as e:
        failed.append((image_name, str(e)))

    # Progress message
    if count % 500 == 0:
        print(f"Processed {count}/{len(df)} images")

# Create processed dataframe
processed_df = pd.DataFrame(processed_rows)

# Save processed CSV
processed_df.to_csv(OUTPUT_CSV, index=False)

print("\nImage preprocessing completed!")
print("Successfully processed:", len(processed_df))
print("Failed:", len(failed))
print("Saved CSV:", OUTPUT_CSV)
print("Saved images:", OUTPUT_DIR)