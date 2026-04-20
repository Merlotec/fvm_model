import torch
import torch.nn as nn

from patch import StackingPatchEmbedding
from transformer import FluidVisionTransformer
from decoder import FluidDecoder


class FluidVisionModel(nn.Module):
    def __init__(self, num_obs: int, num_patches: int, patch_size: int, emb_dim: int, num_channels: int = 3):
        super().__init__()

        self.num_patches = num_patches
        self.emb_dim = emb_dim
        self.num_obs = num_obs
        
        self.patch_embed = StackingPatchEmbedding(num_obs, num_channels, patch_size, emb_dim)
        self.vision_transformer = FluidVisionTransformer(emb_dim)
        self.decoder = FluidDecoder(emb_dim, num_channels)

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))
        pass

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.vision_transformer(x)
        x = self.decoder(x)
        return x
