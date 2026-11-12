
import os
import ast
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
import wfdb


class PTBXLDataset(Dataset):
    """
    PTB-XL dataset loader.

    Loads ECG signals either:
      1. From original WFDB records, or
      2. From processed .npy files when processed_ecg_dir is provided.

    Each sample returns:
      - ecg:        Tensor [12, 1000]
      - labels:     Multi-label diagnostic tensor [5]
      - patient_id: Original PTB-XL patient ID
      - ecg_id:     Original PTB-XL ECG record ID
      - record_name: Original PTB-XL record path
    """

    CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

    def __init__(
        self,
        data_path,
        sampling_rate="lr",
        processed_ecg_dir=None
    ):
        self.data_path = data_path
        self.processed_ecg_dir = processed_ecg_dir

        if sampling_rate not in ("lr", "hr"):
            raise ValueError(
                "sampling_rate must be either 'lr' or 'hr'."
            )

        self.sampling_rate = sampling_rate
        self.filename_column = (
            "filename_lr" if sampling_rate == "lr"
            else "filename_hr"
        )

        database_path = os.path.join(
            data_path, "ptbxl_database.csv"
        )
        statements_path = os.path.join(
            data_path, "scp_statements.csv"
        )

        if not os.path.exists(database_path):
            raise FileNotFoundError(
                f"PTB-XL metadata not found: {database_path}"
            )

        if not os.path.exists(statements_path):
            raise FileNotFoundError(
                f"SCP statements not found: {statements_path}"
            )

        self.metadata = pd.read_csv(database_path)
        self.scp_statements = pd.read_csv(
            statements_path,
            index_col=0
        )

        if "diagnostic" not in self.scp_statements.columns:
            raise ValueError(
                "The SCP statements file has no 'diagnostic' column."
            )

        if "diagnostic_class" not in self.scp_statements.columns:
            raise ValueError(
                "The SCP statements file has no 'diagnostic_class' column."
            )

        # Keep only diagnostic SCP codes.
        self.diagnostic_statements = self.scp_statements[
            self.scp_statements["diagnostic"] == 1
        ]

        required_columns = [
            "ecg_id",
            "patient_id",
            "scp_codes",
            self.filename_column
        ]

        missing_columns = [
            column for column in required_columns
            if column not in self.metadata.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing PTB-XL metadata columns: {missing_columns}"
            )

        # Generate diagnostic labels and retain usable records.
        self.samples = []

        for _, row in self.metadata.iterrows():
            try:
                scp_codes = row["scp_codes"]

                if isinstance(scp_codes, str):
                    scp_codes = ast.literal_eval(scp_codes)

                if not isinstance(scp_codes, dict):
                    continue

                label_vector = np.zeros(
                    len(self.CLASSES),
                    dtype=np.float32
                )

                for scp_code in scp_codes.keys():
                    if scp_code not in self.diagnostic_statements.index:
                        continue

                    diagnostic_class = self.diagnostic_statements.loc[
                        scp_code, "diagnostic_class"
                    ]

                    # Handle duplicate SCP statement rows safely.
                    if isinstance(diagnostic_class, pd.Series):
                        diagnostic_class = diagnostic_class.iloc[0]

                    if pd.isna(diagnostic_class):
                        continue

                    if diagnostic_class in self.CLASSES:
                        class_index = self.CLASSES.index(
                            diagnostic_class
                        )
                        label_vector[class_index] = 1.0

                # Ignore records without a recognized diagnostic label.
                if label_vector.sum() == 0:
                    continue

                if pd.isna(row["patient_id"]):
                    continue

                if pd.isna(row["ecg_id"]):
                    continue

                record_name = row[self.filename_column]

                if pd.isna(record_name):
                    continue

                self.samples.append({
                    "ecg_id": int(row["ecg_id"]),
                    "patient_id": int(row["patient_id"]),
                    "record_name": str(record_name),
                    "labels": label_vector
                })

            except (ValueError, SyntaxError, TypeError, KeyError):
                continue

        if not self.samples:
            raise ValueError(
                "No usable PTB-XL records were found. "
                "Check the metadata and diagnostic labels."
            )

        print(f"Usable ECG records: {len(self.samples)}")
        print(f"Diagnostic classes: {self.CLASSES}")

        if self.processed_ecg_dir is not None:
            if not os.path.isdir(self.processed_ecg_dir):
                raise FileNotFoundError(
                    "Processed ECG directory not found: "
                    f"{self.processed_ecg_dir}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]

        ecg_id = row["ecg_id"]
        patient_id = row["patient_id"]
        record_name = row["record_name"]

        # -------------------------------------------------
        # Load from processed .npy files when configured.
        # -------------------------------------------------
        if self.processed_ecg_dir is not None:
            processed_path = os.path.join(
                self.processed_ecg_dir,
                f"HR{ecg_id:05d}.npy"
            )

            if not os.path.exists(processed_path):
                raise FileNotFoundError(
                    f"Processed ECG not found: {processed_path}"
                )

            signal_array = np.load(processed_path)

            if signal_array.shape != (1000, 12):
                raise ValueError(
                    f"Unexpected ECG shape for ECG ID {ecg_id}: "
                    f"{signal_array.shape}. Expected (1000, 12)."
                )

            # Convert [1000, 12] into [12, 1000].
            signal = np.asarray(
                signal_array,
                dtype=np.float32
            ).T.copy()

        # -------------------------------------------------
        # Otherwise load the original WFDB record.
        # -------------------------------------------------
        else:
            record_path = os.path.join(
                self.data_path,
                record_name
            )

            record = wfdb.rdrecord(record_path)

            signal = np.asarray(
                record.p_signal,
                dtype=np.float32
            ).T.copy()

        # Expected shape: [12 leads, 1000 samples].
        if signal.shape != (12, 1000):
            raise ValueError(
                f"Unexpected ECG shape for ECG ID {ecg_id}: "
                f"{signal.shape}. Expected (12, 1000)."
            )

        # Replace invalid values.
        signal = np.nan_to_num(
            signal,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Per-lead normalization.
        mean = np.mean(signal, axis=1, keepdims=True)
        std = np.std(signal, axis=1, keepdims=True)

        std[std < 1e-8] = 1.0

        signal = (signal - mean) / std

        signal = np.nan_to_num(
            signal,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return {
            "ecg": torch.tensor(
                signal,
                dtype=torch.float32
            ),
            "labels": torch.tensor(
                row["labels"],
                dtype=torch.float32
            ),
            "patient_id": patient_id,
            "ecg_id": ecg_id,
            "record_name": record_name
        }
