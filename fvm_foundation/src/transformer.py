import torch
import torch.nn as nn
import torch.nn.functional as F

from position import RotaryEmbedding2D


class RoPEAttention(nn.Module):
    def __init__(self, emb_dim: int, nhead: int, grid_size: int, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % nhead == 0
        self.nhead    = nhead
        self.head_dim = emb_dim // nhead
        self.scale    = self.head_dim ** -0.5

        self.qkv     = nn.Linear(emb_dim, 3 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout  = nn.Dropout(dropout)
        self.rope     = RotaryEmbedding2D(self.head_dim, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each (B, nhead, N, head_dim)

        q, k = self.rope(q, k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(x)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        emb_dim:         int,
        nhead:           int,
        grid_size:       int,
        dim_feedforward: int   = 3072,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.attn  = RoPEAttention(emb_dim, nhead, grid_size, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, emb_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class FluidVisionTransformer(nn.Module):
    def __init__(
        self,
        emb_dim:    int = 768,
        nhead:      int = 16,
        num_layers: int = 12,
        grid_size:  int = 14,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(emb_dim, nhead, grid_size)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
