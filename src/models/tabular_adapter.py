import torch
import torch.nn as nn


class TabularAdapter(nn.Module):
    """Extract 32-dimensional features from the trained tabular model."""

    def __init__(self, checkpoint_path):
        super().__init__()

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        self.preprocessor = checkpoint["preprocessor"]
        self.feature_columns = checkpoint["feature_columns"]
        input_dim = checkpoint["input_dim"]

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.network.load_state_dict({
            key.replace("network.", "", 1): value
            for key, value in checkpoint["model_state_dict"].items()
        })

        # Use the trained hidden layers as the feature extractor.
        self.feature_extractor = nn.Sequential(
            self.network[0],
            self.network[1],
            self.network[2],
            self.network[3],
            self.network[4],
        )

        self.eval()

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, patient_data):
        if not hasattr(patient_data, "columns"):
            raise ValueError(
                "Expected patient data as a pandas DataFrame."
            )

        if patient_data.empty:
            raise ValueError("Patient data is empty.")

        missing = [
            column for column in self.feature_columns
            if column not in patient_data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required patient features: {missing}"
            )

        patient_data = patient_data.reindex(
            columns=self.feature_columns
        )

        processed = self.preprocessor.transform(patient_data)

        x = torch.as_tensor(
            processed,
            dtype=torch.float32,
            device=next(self.feature_extractor.parameters()).device,
        )

        return self.feature_extractor(x)