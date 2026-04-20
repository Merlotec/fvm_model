import torch.nn as nn

class FluidVisionTransformer(nn.Module):
    def __init__(self, d_model: int = 1024, nhead: int = 16, num_layers: int = 6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        pass

    def forward(self, x):
        x = x.transformer_encoder(x)
        return x
