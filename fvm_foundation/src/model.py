import math
import torch
import torch.nn as nn

from temporal_patch import TemporalPatchEmbedding
from axial_transformer import FluidAxialTransformer
from linear_decoder import LinearDecoder


class FluidVisionModel(nn.Module):
    def __init__(self, num_obs: int, num_patches: int, patch_size: int, emb_dim: int,
                 num_channels: int = 3, num_layers: int = 12):
        super().__init__()
        grid_size = int(math.isqrt(num_patches))
        assert grid_size * grid_size == num_patches, "num_patches must be a perfect square"

        self.num_patches = num_patches
        self.patch_embed = TemporalPatchEmbedding(num_channels, num_obs, patch_size, emb_dim)
        self.transformer = FluidAxialTransformer(emb_dim, grid_size=grid_size,
                                                 num_obs=num_obs, num_layers=num_layers)
        self.decoder     = LinearDecoder(emb_dim, num_channels, patch_size, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)                                    # (B, T*num_patches, emb_dim)
        x = self.transformer(x)                                    # (B, T*num_patches, emb_dim)
        x = x[:, -self.num_patches:, :]                            # last frame only: (B, num_patches, emb_dim)
        return self.decoder(x)                                     # (B, num_channels, H, W)
