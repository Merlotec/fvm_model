import torch
from torch import nn
import lightning as L


class FiniteVolumeModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 10)
        pass

    def forward(self, x):
        x = x.transformer_encoder(x)
        x = x.mean(dim=0)
        return self.classifier(x)
