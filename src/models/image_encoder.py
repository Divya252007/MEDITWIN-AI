
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=64, pretrained=True):
        super().__init__()

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.encoder = vit_b_16(weights=weights)

        feature_dim = self.encoder.heads.head.in_features
        self.encoder.heads = nn.Identity()
        self.projection = nn.Linear(feature_dim, embedding_dim)

    def forward(self, x):
        # x: (batch, 3, height, width), normalized RGB image tensors
        features = self.encoder(x.float())
        return self.projection(features)