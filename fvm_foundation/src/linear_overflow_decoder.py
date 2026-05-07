import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearOverflowDecoder(nn.Module):
    """
    Each patch token projects to an extended (3P/2 × 3P/2) pixel region —
    P/4 pixels of overlap into each neighbouring patch on every side.

    Overlapping contributions are blended via a linear distance weight
    (bilinear tent centred on the native patch region) and then normalised
    by the accumulated weight at each pixel so the result is a proper
    weighted average.
    """

    def __init__(self, emb_dim: int = 768, out_channels: int = 3,
                 patch_size: int = 16, grid_size: int = 14):
        super().__init__()
        assert patch_size % 4 == 0, "patch_size must be divisible by 4"

        self.patch_size   = patch_size
        self.out_channels = out_channels
        self.grid_size    = grid_size

        ext            = patch_size // 4        # P/4 buffer on each side
        ext_patch      = patch_size + 2 * ext   # 3P/2  (e.g. 24 for P=16)
        self.ext       = ext
        self.ext_patch = ext_patch

        self.proj = nn.Linear(emb_dim, ext_patch * ext_patch * out_channels)

        # 2-D bilinear tent weight: 1 at native-patch centre, 0 at extended edges.
        # Separable: w2d(y, x) = w1d(y) * w1d(x)
        center = (ext_patch - 1) / 2.0
        p      = torch.arange(ext_patch, dtype=torch.float32)
        w1d    = (1.0 - (p - center).abs() / center).clamp(min=0)
        w2d    = w1d.unsqueeze(1) * w1d.unsqueeze(0)   # (ext_patch, ext_patch)
        self.register_buffer('weight_map', w2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, G*G, emb_dim)
        B       = x.shape[0]
        G, P, C = self.grid_size, self.patch_size, self.out_channels
        E, ext  = self.ext_patch, self.ext
        H = W   = G * P

        # Project each token to its extended pixel block
        x = self.proj(x)                                        # (B, G*G, E*E*C)
        x = x.reshape(B, G * G, C, E, E)

        # Weight each extended block by the distance map
        x = x * self.weight_map                                 # broadcasts (E, E)

        # Fold weighted patches into the output image
        # fold expects (B, C*E*E, L) where L = G*G
        x = x.reshape(B, G * G, C * E * E).permute(0, 2, 1)    # (B, C*E*E, G*G)
        out = F.fold(x, output_size=(H, W),
                     kernel_size=E, stride=P, padding=ext)      # (B, C, H, W)

        # Fold the weight map itself to get per-pixel normalisation
        w    = self.weight_map.reshape(E * E)
        w    = w.unsqueeze(0).unsqueeze(-1).expand(B, E * E, G * G)
        norm = F.fold(w, output_size=(H, W),
                      kernel_size=E, stride=P, padding=ext)     # (B, 1, H, W)

        return out / norm.clamp(min=1e-6)
