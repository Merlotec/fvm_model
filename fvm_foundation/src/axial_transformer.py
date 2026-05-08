"""
Axial + temporal attention transformer for spatiotemporal patch sequences.

Token layout: (B, T*G*G, D), ordered as [t0_patches..., t1_patches..., ..., tT_patches...]
where patches within each frame are row-major (row 0 col 0, row 0 col 1, ...).

Each layer applies three focused attention operations then a FFN:
  - Row attention:      each token attends over the G tokens in its row (same frame, same row)
  - Column attention:   each token attends over the G tokens in its column (same frame, same col)
  - Temporal attention: each token attends over the T tokens at its spatial position across time

1D RoPE is applied to row and column attention (relative spatial position).
No positional encoding for temporal attention — the learned temporal embedding
added by TemporalPatchEmbedding already differentiates timesteps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_rope_buffers(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) each of shape (seq_len, head_dim)."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos   = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)          # (seq_len, head_dim//2)
    emb   = torch.cat([freqs, freqs], dim=-1)   # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(q, k, cos, sin):
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class AxisAttention(nn.Module):
    """
    Single-axis multi-head attention.

    axis='row'      — attend over the G column positions in the same row and frame
    axis='col'      — attend over the G row positions in the same column and frame
    axis='temporal' — attend over the T timesteps at the same spatial (row, col)
    """

    cos_cache: torch.Tensor
    sin_cache: torch.Tensor

    def __init__(self, emb_dim: int, nhead: int, axis: str,
                 grid_size: int, num_obs: int, dropout: float = 0.1):
        super().__init__()
        assert axis in ('row', 'col', 'temporal')
        assert emb_dim % nhead == 0

        self.axis      = axis
        self.nhead     = nhead
        self.head_dim  = emb_dim // nhead
        self.scale     = self.head_dim ** -0.5
        self.grid_size = grid_size
        self.num_obs   = num_obs

        self.qkv      = nn.Linear(emb_dim, 3 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout  = nn.Dropout(dropout)

        if axis != 'temporal':
            cos, sin = _build_rope_buffers(grid_size, self.head_dim)
            self.register_buffer('cos_cache', cos)  # (G, head_dim)
            self.register_buffer('sin_cache', sin)

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        """Standard MHA. x: (groups, seq, D)."""
        BG, L, D = x.shape
        qkv = self.qkv(x).reshape(BG, L, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # (BG, nhead, L, head_dim)

        if self.axis != 'temporal':
            cos = self.cos_cache[:L].unsqueeze(0).unsqueeze(0)   # (1, 1, L, head_dim)
            sin = self.sin_cache[:L].unsqueeze(0).unsqueeze(0)
            q, k = _apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(BG, L, D)
        return self.out_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T*G*G, D)
        B, _, D = x.shape
        T, G    = self.num_obs, self.grid_size

        if self.axis == 'row':
            # Group: (B*T*G_rows, G_cols, D)
            x = x.reshape(B, T, G, G, D)
            x = x.reshape(B * T * G, G, D)
            x = self._attend(x)
            x = x.reshape(B, T, G, G, D).reshape(B, T * G * G, D)

        elif self.axis == 'col':
            # Transpose rows↔cols so columns become the sequence axis
            x = x.reshape(B, T, G, G, D)
            x = x.permute(0, 1, 3, 2, 4).contiguous()  # (B, T, G_cols, G_rows, D)
            x = x.reshape(B * T * G, G, D)
            x = self._attend(x)
            x = x.reshape(B, T, G, G, D)
            x = x.permute(0, 1, 3, 2, 4).contiguous()  # restore row-major order
            x = x.reshape(B, T * G * G, D)

        else:  # temporal
            # Group: (B*G*G, T, D)
            x = x.reshape(B, T, G * G, D)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, G*G, T, D)
            x = x.reshape(B * G * G, T, D)
            x = self._attend(x)
            x = x.reshape(B, G * G, T, D)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, G*G, D)
            x = x.reshape(B, T * G * G, D)

        return x


class AxialTransformerLayer(nn.Module):
    """
    One axial transformer layer: row attn → col attn → temporal attn → FFN.
    Each sub-operation has its own pre-norm and residual connection.
    """

    def __init__(self, emb_dim: int, nhead: int, grid_size: int, num_obs: int,
                 dim_feedforward: int = 3072, dropout: float = 0.1, num_layers: int = 12):
        super().__init__()

        mk = lambda axis: AxisAttention(emb_dim, nhead, axis, grid_size, num_obs, dropout)
        self.row_attn  = mk('row')
        self.col_attn  = mk('col')
        self.temp_attn = mk('temporal')

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, emb_dim),
            nn.Dropout(dropout),
        )

        self.norm_row  = nn.LayerNorm(emb_dim)
        self.norm_col  = nn.LayerNorm(emb_dim)
        self.norm_temp = nn.LayerNorm(emb_dim)
        self.norm_ff   = nn.LayerNorm(emb_dim)

        # 4 residual branches per layer (row, col, temp, ff) — scale init accordingly
        scale = (2 * num_layers) ** -0.5
        s_attn = scale * (emb_dim ** -0.5)
        nn.init.normal_(self.row_attn.out_proj.weight,  std=s_attn)
        nn.init.normal_(self.col_attn.out_proj.weight,  std=s_attn)
        nn.init.normal_(self.temp_attn.out_proj.weight, std=s_attn)
        nn.init.normal_(self.ff[3].weight, std=scale * (dim_feedforward ** -0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.row_attn(self.norm_row(x))
        x = x + self.col_attn(self.norm_col(x))
        x = x + self.temp_attn(self.norm_temp(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class FluidAxialTransformer(nn.Module):
    def __init__(self, emb_dim: int = 768, nhead: int = 16, num_layers: int = 12,
                 grid_size: int = 14, num_obs: int = 5):
        super().__init__()
        self.layers = nn.ModuleList([
            AxialTransformerLayer(emb_dim, nhead, grid_size, num_obs, num_layers=num_layers)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
