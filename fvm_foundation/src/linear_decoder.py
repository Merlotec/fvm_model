import torch.nn as nn


class LinearDecoder(nn.Module):
    def __init__(self, emb_dim=768, out_channels=3, patch_size=16, grid_size=14):
        super().__init__()
        self.proj         = nn.Linear(emb_dim, patch_size * patch_size * out_channels)
        self.patch_size   = patch_size
        self.out_channels = out_channels
        self.grid_size    = grid_size

    def forward(self, x):
        # x: (B, num_patches, emb_dim)
        B       = x.shape[0]
        G, P, C = self.grid_size, self.patch_size, self.out_channels
        x = self.proj(x)                      # (B, G*G, P*P*C)
        x = x.reshape(B, G, G, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5)      # (B, C, G, P, G, P)
        x = x.reshape(B, C, G * P, G * P)     # (B, C, H, W)
        return x
