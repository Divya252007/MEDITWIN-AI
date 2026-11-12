
import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name="dmis-lab/biobert-base-cased-v1.2",
        embedding_dim=64,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use the [CLS] representation.
        features = outputs.last_hidden_state[:, 0, :]
        return self.projection(features)