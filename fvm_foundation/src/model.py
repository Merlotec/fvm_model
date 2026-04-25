import math
import torch.nn as nn

from patch import StackingPatchEmbedding
from transformer import FluidVisionTransformer
from decoder import FluidDecoder


class FluidVisionModel(nn.Module):
    def __init__(self, num_obs: int, num_patches: int, patch_size: int, emb_dim: int, num_channels: int = 3):
        super().__init__()
        grid_size = int(math.isqrt(num_patches))
        assert grid_size * grid_size == num_patches, "num_patches must be a perfect square"

        self.patch_embed       = StackingPatchEmbedding(num_obs, num_channels, patch_size, emb_dim)
        self.vision_transformer = FluidVisionTransformer(emb_dim, grid_size=grid_size)
        self.decoder           = FluidDecoder(emb_dim, num_channels)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.vision_transformer(x)
        x = self.decoder(x)
        return x
