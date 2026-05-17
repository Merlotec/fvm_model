import math
import torch
import torch.nn as nn

from stacking_patch import StackingPatchEmbedding
from axial_transformer import FluidAxialTransformer
from cnn_decoder import CNNDecoder


class FluidVisionModel(nn.Module):
    def __init__(self, num_obs: int, num_patches: int, patch_size: int, emb_dim: int,
                 num_channels: int = 3, num_layers: int = 12):
        super().__init__()
        grid_size = int(math.isqrt(num_patches))
        assert grid_size * grid_size == num_patches, "num_patches must be a perfect square"

        self.patch_embed = StackingPatchEmbedding(num_obs, num_channels, patch_size, emb_dim)
        self.transformer = FluidAxialTransformer(emb_dim, grid_size=grid_size, num_layers=num_layers)
        self.decoder     = CNNDecoder(emb_dim, num_channels, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)             # (B, T*C, H, W) — stack timesteps as channels
        x = self.patch_embed(x)                    # (B, num_patches, emb_dim)
        x = self.transformer(x)                    # (B, num_patches, emb_dim)
        return self.decoder(x)                     # (B, num_channels, H, W)
