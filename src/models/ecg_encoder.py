import torch
import torch.nn as nn


class ECGEncoder(nn.Module):
    """Extract features from a 12-lead ECG signal."""

    def __init__(self, output_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Expected shape: (batch, 12 leads, time samples)
        if x.ndim != 3 or x.shape[1] != 12:
            raise ValueError(
                f"Expected ECG shape (batch, 12, time), got {tuple(x.shape)}"
            )

        return self.features(x)