import math
import torch.nn as nn

from temporal_patch import TemporalPatchEmbedding
from transformer import FluidVisionTransformer
from linear_overflow_decoder import LinearOverflowDecoder


class FluidVisionModel(nn.Module):
    def __init__(self, num_obs: int, num_patches: int, patch_size: int, emb_dim: int, num_channels: int = 3):
        super().__init__()
        grid_size = int(math.isqrt(num_patches))
        assert grid_size * grid_size == num_patches, "num_patches must be a perfect square"

        self.num_patches        = num_patches   # per frame, needed to slice decoder input
        self.patch_embed        = TemporalPatchEmbedding(num_obs, num_channels, patch_size, emb_dim)
        self.vision_transformer = FluidVisionTransformer(emb_dim, grid_size=grid_size)
        self.decoder            = LinearOverflowDecoder(emb_dim, num_channels, patch_size, grid_size)

    def forward(self, x):
        x = self.patch_embed(x)          # (B, num_obs * num_patches, emb_dim)
        x = self.vision_transformer(x)   # (B, num_obs * num_patches, emb_dim)
        x = x[:, -self.num_patches:, :]  # (B, num_patches, emb_dim) — last frame's tokens
        x = self.decoder(x)              # (B, num_channels, H, W)
        return x
