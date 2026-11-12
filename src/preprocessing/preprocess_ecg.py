
import os
import numpy as np
import wfdb
from scipy.signal import resample


INPUT_DIR = "data/raw/ecg/WFDB"
OUTPUT_DIR = "data/processed/ecg"

TARGET_LENGTH = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_ecg(record_path):

    # Read ECG record
    record = wfdb.rdrecord(record_path)

    signal = record.p_signal

    # Replace missing values
    signal = np.nan_to_num(signal)

    # Normalize ECG
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)

    std[std == 0] = 1

    signal = (signal - mean) / std

    # Convert to fixed length
    signal = resample(signal, TARGET_LENGTH, axis=0)

    return signal


def main():

    print("Starting ECG preprocessing...")

    hea_files = [
        file for file in os.listdir(INPUT_DIR)
        if file.endswith(".hea")
    ]

    print("ECG records found:", len(hea_files))

    for file in hea_files:

        record_name = os.path.splitext(file)[0]

        record_path = os.path.join(
            INPUT_DIR,
            record_name
        )

        try:

            print("Processing:", record_name)

            signal = preprocess_ecg(record_path)

            output_path = os.path.join(
                OUTPUT_DIR,
                record_name + ".npy"
            )

            np.save(output_path, signal)

            print("Saved:", output_path)

        except Exception as e:

            print("Error:", record_name, e)

    print("\nECG preprocessing completed!")


if __name__ == "__main__":
    main()
