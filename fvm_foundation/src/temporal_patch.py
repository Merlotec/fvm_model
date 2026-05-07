import torch
import torch.nn as nn


class TemporalPatchEmbedding(nn.Module):
    """
    Late-fusion patch embedding: each timestep's frame is projected independently
    with a shared Conv2d, then a learned temporal positional encoding is added so
    the transformer can distinguish which timestep each token belongs to.

    Output sequence length: num_obs * num_patches_per_frame
    (vs. StackingPatchEmbedding which produces num_patches_per_frame tokens).
    """

    def __init__(self, num_obs: int, in_channels: int, patch_size: int = 16, emb_dim: int = 768):
        super().__init__()
        self.num_obs     = num_obs
        self.in_channels = in_channels

        # Shared spatial projection — same weights applied to every timestep
        self.projection  = nn.Conv2d(in_channels, emb_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # One learned vector per timestep position
        self.temporal_pos = nn.Embedding(num_obs, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_obs * in_channels, H, W)
        B, _, H, W = x.shape

        # Split stacked channels back into (B*num_obs, in_channels, H, W)
        x = x.reshape(B * self.num_obs, self.in_channels, H, W)

        # Shared patch projection → (B*num_obs, emb_dim, G, G)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)          # (B*num_obs, num_patches, emb_dim)

        num_patches = x.shape[1]

        # Reshape to (B, num_obs, num_patches, emb_dim)
        x = x.reshape(B, self.num_obs, num_patches, x.shape[2])

        # Add temporal positional encoding: (num_obs, emb_dim) → (1, num_obs, 1, emb_dim)
        t_idx = torch.arange(self.num_obs, device=x.device)
        t_emb = self.temporal_pos(t_idx).unsqueeze(0).unsqueeze(2)
        x = x + t_emb

        # Flatten to (B, num_obs * num_patches, emb_dim)
        return x.reshape(B, self.num_obs * num_patches, x.shape[3])
