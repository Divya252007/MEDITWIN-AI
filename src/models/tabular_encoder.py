
import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                "Expected tabular input shape (batch_size, features)"
            )

        return self.encoder(x.float())


if __name__ == "__main__":
    model = TabularEncoder(input_dim=10)
    sample = torch.randn(4, 10)
    print("Output shape:", model(sample).shape)