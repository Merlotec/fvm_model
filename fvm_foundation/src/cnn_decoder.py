import math
import torch.nn as nn
import torch.nn.functional as F


class CNNDecoder(nn.Module):
    """
    Upsample patch tokens back to pixel resolution.

    The number of stride-2 ConvTranspose stages is derived from patch_size:
        n_ups = log2(patch_size)
    so the grid goes G×G → (G*patch_size)×(G*patch_size) = H×W exactly.

    Channel schedule per stage: 256 → 128 → 64 → 32 → 16 → 8 (truncated to n_ups stages).
    """

    _CHANNEL_SCHEDULE = [256, 128, 64, 32, 16, 8]

    def __init__(self, emb_dim: int = 768, out_channels: int = 3, patch_size: int = 16):
        super().__init__()
        assert patch_size >= 2 and (patch_size & (patch_size - 1)) == 0, \
            "patch_size must be a power of 2"

        n_ups = int(math.log2(patch_size))
        assert n_ups <= len(self._CHANNEL_SCHEDULE), \
            f"patch_size={patch_size} requires {n_ups} upsample stages but only " \
            f"{len(self._CHANNEL_SCHEDULE)} are defined in _CHANNEL_SCHEDULE"

        channel_sizes = self._CHANNEL_SCHEDULE[:n_ups]
        ups = []
        in_ch = emb_dim
        for out_ch in channel_sizes:
            ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            in_ch = out_ch

        self.ups        = nn.ModuleList(ups)
        self.final_proj = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, num_patches, emb_dim)
        x    = x.transpose(1, 2)                 # (B, emb_dim, num_patches)
        grid = int(x.shape[2] ** 0.5)
        x    = x.unflatten(2, (grid, grid))       # (B, emb_dim, G, G)
        for up in self.ups:
            x = F.elu(up(x))
        return self.final_proj(x)                # (B, out_channels, H, W)
