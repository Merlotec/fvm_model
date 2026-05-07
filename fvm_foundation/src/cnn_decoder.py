import torch.nn as nn
import torch.nn.functional as F


class CNNDecoder(nn.Module):
    def __init__(self, emb_dim=768, out_channels=3):
        super().__init__()
        self.up1        = nn.ConvTranspose2d(emb_dim, 256, kernel_size=2, stride=2)
        self.up2        = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3        = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2)
        self.up4        = nn.ConvTranspose2d( 64,  32, kernel_size=2, stride=2)
        self.final_proj = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, num_patches, emb_dim)
        x    = x.transpose(1, 2)                    # (B, emb_dim, num_patches)
        grid = int(x.shape[2] ** 0.5)
        x    = x.unflatten(2, (grid, grid))          # (B, emb_dim, G, G)
        x    = F.elu(self.up1(x))
        x    = F.elu(self.up2(x))
        x    = F.elu(self.up3(x))
        x    = F.elu(self.up4(x))
        return self.final_proj(x)                    # (B, out_channels, H, W)
