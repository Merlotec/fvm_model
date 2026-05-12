"""
Axial + temporal attention transformer for spatiotemporal patch sequences.

Token layout: (B, T*G*G, D), ordered as [t0_patches..., t1_patches..., ..., tT_patches...]
where patches within each frame are row-major (row 0 col 0, row 0 col 1, ...).

Each layer applies three focused attention operations then a FFN:
  - Row attention:      each token attends over the G tokens in its row (same frame, same row)
  - Column attention:   each token attends over the G tokens in its column (same frame, same col)
  - Temporal attention: each token attends causally over a window of num_obs past timesteps at
                        its spatial position. T is derived from the actual input length so the
                        transformer generalises to any sequence length at train and inference time.
                        Uses flex_attention on CUDA; falls back to SDPA + manual mask elsewhere.

1D RoPE is applied to all three axes (row, col, temporal) so position is always
encoded as a relative offset in the attention dot product rather than an absolute index.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention
    _FLEX_AVAILABLE = True
except ImportError:
    _FLEX_AVAILABLE = False

_TEMPORAL_ROPE_MAX = 4096   # maximum supported sequence length for temporal RoPE


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


def _make_causal_window_score_mod(window: int):
    """
    Returns a score_mod for flex_attention: causal + sliding window of `window` steps.
    Position k is attended by q only when  k <= q  and  q - k < window.
    """
    def score_mod(score, b, h, q_idx, k_idx):
        causal = k_idx <= q_idx
        within = (q_idx - k_idx) < window
        return torch.where(causal & within, score, score.new_full((), float('-inf')))
    return score_mod


def _causal_window_attn_mask(T: int, window: int, device, dtype) -> torch.Tensor:
    """(T, T) additive mask for SDPA fallback on non-CUDA devices."""
    q_idx   = torch.arange(T, device=device).unsqueeze(1)
    k_idx   = torch.arange(T, device=device).unsqueeze(0)
    allowed = (k_idx <= q_idx) & ((q_idx - k_idx) < window)
    return torch.zeros(T, T, device=device, dtype=dtype).masked_fill(~allowed, float('-inf'))


class AxisAttention(nn.Module):
    """
    Single-axis multi-head attention with RoPE on all three axes.

    axis='row'      — attend over the G column positions in the same row and frame
    axis='col'      — attend over the G row positions in the same column and frame
    axis='temporal' — causal window attention over T timesteps at each spatial position
                      (T is inferred from the input at runtime, not fixed at init time)
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
        self.grid_size = grid_size
        self.num_obs   = num_obs
        self.dropout_p = dropout

        self.qkv      = nn.Linear(emb_dim, 3 * emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        if axis in ('row', 'col'):
            cos, sin = _build_rope_buffers(grid_size, self.head_dim)
        else:
            cos, sin = _build_rope_buffers(_TEMPORAL_ROPE_MAX, self.head_dim)
            self._score_mod = _make_causal_window_score_mod(num_obs)

        self.register_buffer('cos_cache', cos)
        self.register_buffer('sin_cache', sin)

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        """x: (groups, seq, D) → (groups, seq, D)."""
        BG, L, D = x.shape
        qkv = self.qkv(x).reshape(BG, L, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # (BG, nhead, L, head_dim)

        # RoPE applied on all axes
        cos = self.cos_cache[:L].unsqueeze(0).unsqueeze(0)   # (1, 1, L, head_dim)
        sin = self.sin_cache[:L].unsqueeze(0).unsqueeze(0)
        q, k = _apply_rope(q, k, cos, sin)

        if self.axis != 'temporal':
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0
            )
        elif _FLEX_AVAILABLE and q.device.type == 'cuda':
            out = flex_attention(q, k, v, score_mod=self._score_mod)
        else:
            mask = _causal_window_attn_mask(L, self.num_obs, q.device, q.dtype)
            out  = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        return self.out_proj(out.transpose(1, 2).reshape(BG, L, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T*G*G, D) — T is inferred from the input, not from self.num_obs
        B, S, D = x.shape
        G = self.grid_size
        T = S // (G * G)

        if self.axis == 'row':
            x = x.reshape(B, T, G, G, D).reshape(B * T * G, G, D)
            x = self._attend(x)
            x = x.reshape(B, T, G, G, D).reshape(B, T * G * G, D)

        elif self.axis == 'col':
            x = x.reshape(B, T, G, G, D)
            x = x.permute(0, 1, 3, 2, 4).contiguous()  # (B, T, G_cols, G_rows, D)
            x = x.reshape(B * T * G, G, D)
            x = self._attend(x)
            x = x.reshape(B, T, G, G, D)
            x = x.permute(0, 1, 3, 2, 4).contiguous()  # restore row-major
            x = x.reshape(B, T * G * G, D)

        else:  # temporal
            x = x.reshape(B, T, G * G, D).permute(0, 2, 1, 3).contiguous()  # (B, G*G, T, D)
            x = x.reshape(B * G * G, T, D)
            x = self._attend(x)
            x = x.reshape(B, G * G, T, D).permute(0, 2, 1, 3).contiguous()  # (B, T, G*G, D)
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

        # 4 residual branches per layer — scale init accordingly
        scale  = (2 * num_layers) ** -0.5
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
