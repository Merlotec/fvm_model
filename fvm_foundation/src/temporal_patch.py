import torch
import torch.nn as nn


class TemporalPatchEmbedding(nn.Module):
    """
    Projects each frame's patches independently using a shared Conv2d.

    Temporal position is encoded by RoPE inside AxisAttention (temporal axis),
    so no learned positional embedding is added here.

    Input:  (B, T, C, H, W)
    Output: (B, T * num_patches, emb_dim)
    """

    def __init__(self, in_channels: int, patch_size: int = 16, emb_dim: int = 768):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, emb_dim,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.projection(x)                  # (B*T, emb_dim, G, G)
        x = x.flatten(2).transpose(1, 2)        # (B*T, num_patches, emb_dim)
        return x.reshape(B, T * x.shape[1], x.shape[2])
