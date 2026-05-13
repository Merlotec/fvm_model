import torch
import torch.nn as nn


class TemporalPatchEmbedding(nn.Module):
    """
    Projects each frame's patches independently using a shared Conv2d,
    then adds a learned temporal position embedding per timestep.

    Input:  (B, T, C, H, W)
    Output: (B, T * num_patches, emb_dim)
    """

    def __init__(self, in_channels: int, num_obs: int, patch_size: int = 16, emb_dim: int = 768):
        super().__init__()
        self.projection     = nn.Conv2d(in_channels, emb_dim,
                                        kernel_size=patch_size, stride=patch_size)
        self.temporal_embed = nn.Embedding(num_obs, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.projection(x)                   # (B*T, emb_dim, G, G)
        x = x.flatten(2).transpose(1, 2)         # (B*T, num_patches, emb_dim)
        num_patches = x.shape[1]
        x = x.reshape(B, T, num_patches, x.shape[2])

        t_idx = torch.arange(T, device=x.device)
        t_emb = self.temporal_embed(t_idx)       # (T, emb_dim)
        x = x + t_emb.unsqueeze(0).unsqueeze(2)  # (1, T, 1, emb_dim) broadcast

        return x.reshape(B, T * num_patches, x.shape[3])
