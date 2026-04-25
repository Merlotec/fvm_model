"""
2D Rotary Position Embeddings (RoPE) for grid-arranged vision patches.

The head dimension is split in half: the first half encodes the row axis
and the second half encodes the column axis.  Standard 1D RoPE is applied
independently to each half, so the full rotation is:

    q' = [RoPE_row(q[:D/2]), RoPE_col(q[D/2:])]

This keeps the implementation simple while giving the model distinct
positional signals for both spatial axes.

Requirements: head_dim (emb_dim // nhead) must be divisible by 4.
"""

import torch
import torch.nn as nn


class RotaryEmbedding2D(nn.Module):
    """
    Precomputes and applies 2D RoPE to query/key tensors.

    Args:
        head_dim:  Dimension per attention head.  Must be divisible by 4.
        grid_size: Patches per spatial axis (e.g. 14 for a 224/16 grid).
    """

    inv_freq:  torch.Tensor
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor

    def __init__(self, head_dim: int, grid_size: int):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        half = head_dim // 2

        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))
        self.register_buffer('inv_freq', inv_freq)   # (half/2,)

        self.grid_size = grid_size
        self._build_cache(grid_size)

    def _build_cache(self, grid_size: int) -> None:
        pos   = torch.arange(grid_size, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq)         # (grid_size, half/2)
        emb   = torch.cat([freqs, freqs], dim=-1)       # (grid_size, half)
        self.register_buffer('cos_cache', emb.cos())    # (grid_size, half)
        self.register_buffer('sin_cache', emb.sin())    # (grid_size, half)

    @staticmethod
    def _rotate_half_2d(x: torch.Tensor) -> torch.Tensor:
        """rotate_half applied independently to each D//2 chunk."""
        D = x.shape[-1]
        h = D // 2
        q = h // 2
        return torch.cat([
            -x[..., q:h], x[..., :q],      # rotated row half
            -x[..., h + q:], x[..., h:h + q],  # rotated col half
        ], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D RoPE in-place to queries and keys.

        Args:
            q, k: (batch, nhead, seq_len, head_dim)
                  where seq_len == grid_size ** 2.

        Returns:
            Rotated (q, k) with the same shape.
        """
        g = self.grid_size
        N = q.shape[2]
        assert N == g * g, f"seq_len {N} != grid_size^2 {g*g}"

        rows = torch.arange(g, device=q.device).repeat_interleave(g)  # (N,)
        cols = torch.arange(g, device=q.device).repeat(g)             # (N,)

        # (N, head_dim): row RoPE for first half, col RoPE for second half
        cos = torch.cat([self.cos_cache[rows], self.cos_cache[cols]], dim=-1)
        sin = torch.cat([self.sin_cache[rows], self.sin_cache[cols]], dim=-1)

        # Broadcast over batch and head dims: (1, 1, N, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = q * cos + self._rotate_half_2d(q) * sin
        k = k * cos + self._rotate_half_2d(k) * sin
        return q, k
