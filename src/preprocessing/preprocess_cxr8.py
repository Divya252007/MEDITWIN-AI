import os
import pandas as pd

# Dataset paths
BASE_DIR = "data/raw/cxr8"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CSV_FILE = os.path.join(BASE_DIR, "Data_Entry_2017_v2020.csv")

# Read CSV
df = pd.read_csv(CSV_FILE)

print("Dataset loaded successfully!")
print("Total records:", len(df))
print(df.head())

# Check if images exist
missing = []

for img in df["Image Index"]:
    path = os.path.join(IMAGE_DIR, img)
    if not os.path.exists(path):
        missing.append(img)

print("Missing images:", len(missing))
print("Available images:", len(df) - len(missing))
# Keep only images that are actually available
available_df = df[~df["Image Index"].isin(missing)].copy()

# Create processed image folder
OUTPUT_DIR = "data/processed/image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save filtered dataset information
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "cxr8_available.csv")
available_df.to_csv(OUTPUT_CSV, index=False)

print("Available dataset CSV saved!")
print("Records saved:", len(available_df))
print("Saved to:", OUTPUT_CSV)