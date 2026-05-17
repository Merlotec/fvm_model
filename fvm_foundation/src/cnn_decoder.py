import math
import torch.nn as nn


class CNNDecoder(nn.Module):
    """
    Upsample patch tokens back to pixel resolution.

    Uses bilinear upsample + Conv2d(3×3) at each doubling stage instead of
    ConvTranspose2d, which eliminates patch-boundary checkerboard artifacts.

    Channel schedule: emb_dim → emb_dim//2 → ... → out_channels*2 → out_channels
    (halving at each stage, floored at out_channels*2 until the final stage).
    """

    def __init__(self, emb_dim: int = 768, out_channels: int = 3, patch_size: int = 16):
        super().__init__()
        assert patch_size >= 2 and (patch_size & (patch_size - 1)) == 0, \
            "patch_size must be a power of 2"

        n_ups = int(math.log2(patch_size))
        layers: list[nn.Module] = []
        c_in = emb_dim
        for i in range(n_ups):
            is_last = (i == n_ups - 1)
            c_out   = out_channels if is_last else max(emb_dim >> (i + 1), out_channels * 2)
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, padding_mode='replicate'))
            if not is_last:
                layers.append(nn.GELU())
            c_in = c_out

        self.upsampler = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, num_patches, emb_dim)
        x    = x.transpose(1, 2)                  # (B, emb_dim, num_patches)
        grid = int(x.shape[2] ** 0.5)
        x    = x.unflatten(2, (grid, grid))        # (B, emb_dim, G, G)
        return self.upsampler(x)                   # (B, out_channels, H, W)
