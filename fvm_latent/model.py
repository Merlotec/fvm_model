"""
Multi-Level Same-Resolution Fluid Prediction Model.

Architecture
------------
K levels, each with P = (img_size/patch_size)^2 patch tokens covering the
full image at the same spatial resolution.

  1. PatchEmbed (shared)  — (B,T*C,H,W) → (B,P,d) base spatial features.
  2. Level embedding      — each level gets base features + a different learned
                            offset (n_levels, d), so the transformer can tell
                            which level each token belongs to.
  3. HierarchicalTransformer — (B,K*P,d) with axial + cross-level mask:
                            Within a level: axial attention (row OR column only).
                            Across levels: token (k,p) attends to (j,p) only —
                            same spatial position, any lower level j<k.
                            2D RoPE encodes (row, col) into q and k so the
                            transformer knows spatial layout without learned pos bias.
  4. PerLevelDecoder      — each level k projects d → d features (head_k).
                            Gate-weighted features are summed into a spatial map
                            (B, d, n, n), then a CNN upsampler (bilinear+conv ×log₂p)
                            decodes to full resolution (B, C, H, W).  Cross-patch conv
                            operations allow spatially coherent reconstructions that a
                            per-patch linear head cannot produce.

Training
--------
  Phase 1 (curriculum): n_active_levels steps from 1 → K, adding one level
  every N optimiser steps.  With 1 level the model is just a standard ViT-like
  patch predictor; each new level is a learned refinement layer.

  Loss = reconstruction (L1+MSE).  All levels contribute with uniform weight 1.

Inference
---------
  All K*P tokens present, all levels contributing equally.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask
    _FLEX_AVAILABLE = True
except ImportError:
    _flex_attention = None        # type: ignore[assignment]
    _create_block_mask = None     # type: ignore[assignment]
    _FLEX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared building blocks (unchanged from previous version)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Conv2d patch projection + learned spatial positional embedding."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0
        n = img_size // patch_size
        self.n_patches = n * n
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos  = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos


class FFN(nn.Module):
    def __init__(self, d: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * expansion, d), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding (RoPE) for spatial patch tokens.

    head_dim is split into quarters: the first half encodes row position,
    the second half encodes column position.  The same embedding is tiled
    across levels so that tokens at the same spatial position but different
    levels carry identical positional signals — level identity comes from
    level_embed, not from RoPE.

    Requires head_dim % 4 == 0.
    """

    def __init__(self, head_dim: int, n_rows: int, n_cols: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        quarter = head_dim // 4
        inv_freq = 1.0 / (base ** (torch.arange(quarter).float() / quarter))

        rows = torch.arange(n_rows).float()
        cols = torch.arange(n_cols).float()

        freqs_row = torch.outer(rows, inv_freq)                             # (n_rows, quarter)
        freqs_col = torch.outer(cols, inv_freq)                             # (n_cols, quarter)

        P       = n_rows * n_cols
        pos_row = freqs_row.unsqueeze(1).expand(-1, n_cols, -1).reshape(P, quarter)
        pos_col = freqs_col.unsqueeze(0).expand(n_rows, -1, -1).reshape(P, quarter)

        freqs = torch.cat([pos_row, pos_col], dim=-1)   # (P, head_dim//2)
        emb   = torch.cat([freqs, freqs],     dim=-1)   # (P, head_dim) — duplicated for rotate_half

        self.register_buffer('cos_emb', emb.cos())      # (P, head_dim)
        self.register_buffer('sin_emb', emb.sin())      # (P, head_dim)
        self.n_patches = P

    cos_emb: torch.Tensor
    sin_emb: torch.Tensor

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, n_levels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, n_heads, n_levels*P, head_dim)
        cos = self.cos_emb.repeat(n_levels, 1).unsqueeze(0).unsqueeze(0)  # (1,1,n_levels*P,hd)
        sin = self.sin_emb.repeat(n_levels, 1).unsqueeze(0).unsqueeze(0)
        return (
            q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin,
        )


class RoPESelfAttnLayer(nn.Module):
    """Pre-norm self-attention with 2D RoPE applied to q and k."""

    def __init__(self, d: int, n_heads: int, rope: RotaryEmbedding2D, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = d // n_heads
        self.rope      = rope
        self.dropout_p = dropout

        self.qkv      = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d)
        self.ffn      = FFN(d, dropout=dropout)
        self.norm1    = nn.LayerNorm(d)
        self.norm2    = nn.LayerNorm(d)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], n_levels: int,
    ) -> torch.Tensor:
        B, N, d = x.shape
        nh, hd  = self.n_heads, self.head_dim

        res  = x
        qkv  = self.qkv(self.norm1(x)).reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                       # (B, nh, N, hd)

        q, k = self.rope(q, k, n_levels)

        out: torch.Tensor
        if _FLEX_AVAILABLE and not isinstance(mask, torch.Tensor):
            out = _flex_attention(q, k, v, block_mask=mask, return_lse=False)  # type: ignore[assignment]
        elif q.is_cuda:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            # CPU / MPS: SDPA backward with bool masks is broken on these backends;
            # use explicit matmul attention instead.
            scores = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5)
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            if self.dropout_p > 0.0 and self.training:
                attn = F.dropout(attn, p=self.dropout_p)
            out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, d)
        x   = res + self.out_proj(out)
        return x + self.ffn(self.norm2(x))


class HierarchicalTransformer(nn.Module):
    def __init__(self, d: int, n_heads: int, n_layers: int, n_per_side: int, dropout: float = 0.0):
        super().__init__()
        self.rope   = RotaryEmbedding2D(d // n_heads, n_per_side, n_per_side)
        self.layers = nn.ModuleList([
            RoPESelfAttnLayer(d, n_heads, self.rope, dropout) for _ in range(n_layers)
        ])

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, n_levels: int = 1,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask, n_levels)
        return x


class SkipEncoder(nn.Module):
    """
    Lightweight multi-scale feature extractor for U-Net-style skip connections.

    Produces one feature map per decoder upsampling stage, starting at full
    input resolution and halving spatial size with each stage.  The decoder
    injects feats[-1] at its coarsest stage and feats[0] at its finest,
    giving the CNN upsampler direct access to high-frequency pixel content
    it cannot recover from patch tokens alone.
    """

    def __init__(self, in_channels: int, skip_ch: int, n_doublings: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, skip_ch, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
        )
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(skip_ch, skip_ch, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            for _ in range(n_doublings - 1)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Returns [full_res, half_res, ..., coarsest_res] feature maps."""
        feats: list[torch.Tensor] = []
        f = self.stem(x)
        feats.append(f)
        for down in self.downs:
            f = down(f)
            feats.append(f)
        return feats


class PerLevelDecoder(nn.Module):
    """
    Gate-weighted feature aggregation + CNN upsampler with skip connections.

    Each level k projects its tokens to a d-dim feature map via a small head
    (LayerNorm → Linear).  The gate-weighted sum across levels produces a spatial
    feature map at the coarse token resolution (n×n).  A CNN upsampler then
    decodes this to full image resolution using bilinear upsampling + 3×3 conv
    repeated log₂(patch_size) times, halving channels at each stage.

    At each upsampling stage, skip features from the SkipEncoder are concatenated
    before the conv, giving the decoder direct access to sub-patch spatial detail.
    """

    def __init__(
        self, d: int, out_channels: int, img_size: int, patch_size: int, n_levels: int,
        skip_ch: int = 0,
    ):
        super().__init__()
        assert math.log2(patch_size) == int(math.log2(patch_size)), \
            "patch_size must be a power of 2"
        n = img_size // patch_size
        self.n_per_side  = n
        self.n_patches   = n * n
        self.d           = d
        n_doublings      = int(math.log2(patch_size))
        self.n_doublings = n_doublings

        # Per-level feature projections: d → d, one per level
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
            for _ in range(n_levels)
        ])

        # Upsampler stages as explicit lists so skip features can be injected
        # between the bilinear upsample and the conv at each stage.
        self.up_convs: nn.ModuleList = nn.ModuleList()
        self.up_acts:  nn.ModuleList = nn.ModuleList()
        c_in = d
        for i in range(n_doublings):
            is_last = (i == n_doublings - 1)
            c_out   = out_channels if is_last else max(d >> (i + 1), out_channels * 2)
            self.up_convs.append(
                nn.Conv2d(c_in + skip_ch, c_out, kernel_size=3, padding=1, padding_mode='replicate')
            )
            self.up_acts.append(nn.GELU() if not is_last else nn.Identity())
            c_in = c_out

    def forward(
        self,
        transformer_out: torch.Tensor,
        gate_weights: torch.Tensor,
        skip_feats: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # transformer_out: (B, n_levels*P, d)
        # gate_weights:    (B, n_levels*P)
        # skip_feats:      [full_res, ..., coarsest_res] from SkipEncoder
        B        = transformer_out.size(0)
        P        = self.n_patches
        n        = self.n_per_side
        d        = self.d
        n_levels = transformer_out.size(1) // P

        feat = transformer_out.new_zeros(B, d, n, n)
        for k in range(n_levels):
            tok = transformer_out[:, k * P : (k + 1) * P]   # (B, P, d)
            w   = gate_weights[:, k * P : (k + 1) * P]      # (B, P)
            f_k = self.heads[k](tok)                         # (B, P, d)
            feat = feat + (w.unsqueeze(-1) * f_k).reshape(B, n, n, d).permute(0, 3, 1, 2)
        feat = feat / n_levels

        for i, (conv, act) in enumerate(zip(self.up_convs, self.up_acts)):
            feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
            if skip_feats is not None:
                feat = torch.cat([feat, skip_feats[self.n_doublings - 1 - i]], dim=1)
            feat = act(conv(feat))

        return feat  # (B, out_channels, H, W)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_axial_cross_level_mask(
    n_levels: int, n_patches: int, n_per_side: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Additive attention mask: 0 = may attend, -inf = blocked.

    Token (k, p) may attend to token (j, q) iff:
      Within-level  (j == k): row(p) == row(q)  OR  col(p) == col(q)   [axial]
      Cross-level   (j <  k): p == q                                     [same position only]

    Axial within-level attention keeps lateral spatial context cheap.
    Cross-level attention is strictly vertical so each position refines
    itself using its own lower-level representation only.
    """
    N   = n_levels * n_patches
    n   = n_per_side
    idx = torch.arange(N)

    lv  = idx // n_patches          # level index
    pos = idx % n_patches           # spatial index within level
    row = pos // n
    col = pos % n

    li, lj = lv.unsqueeze(1),  lv.unsqueeze(0)
    ri, rj = row.unsqueeze(1), row.unsqueeze(0)
    ci, cj = col.unsqueeze(1), col.unsqueeze(0)
    pi, pj = pos.unsqueeze(1), pos.unsqueeze(0)

    axial = (li == lj) & ((ri == rj) | (ci == cj))
    cross = (li >  lj) & (pi == pj)

    mask = axial | cross  # True = may attend
    return mask if device is None else mask.to(device)


def _make_mask_mod(n_per_side: int, n_patches: int):
    """flex_attention mask_mod for the axial + cross-level pattern."""
    def mask_mod(b, h, q_idx, kv_idx):
        q_level = q_idx // n_patches
        k_level = kv_idx // n_patches
        q_pos   = q_idx % n_patches
        k_pos   = kv_idx % n_patches
        q_row   = q_pos // n_per_side
        q_col   = q_pos % n_per_side
        k_row   = k_pos // n_per_side
        k_col   = k_pos % n_per_side
        within  = (q_level == k_level) & ((q_row == k_row) | (q_col == k_col))
        cross   = (q_level > k_level)  & (q_pos == k_pos)
        return within | cross
    return mask_mod


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MultiLevelFluidModel(nn.Module):
    """
    Multi-Level Same-Resolution Fluid Prediction Model.

    Args:
        img_size:             Spatial resolution (square).
        patch_size:           Patch size (must divide img_size).
        in_channels:          Per-frame channels (e.g. 4 for Vx,Vy,rho,T).
        window_size:          Temporal frames stacked as input (T).
        n_levels:             Number of refinement levels (K).
        d_model:              Token embedding dimension.
        n_heads:              Attention heads.
        n_transformer_layers: Transformer depth.
        dropout:              Dropout probability.
    """

    def __init__(
        self,
        img_size: int             = 512,
        patch_size: int           = 32,
        in_channels: int          = 4,
        window_size: int          = 5,
        n_levels: int             = 5,
        d_model: int              = 512,
        n_heads: int              = 8,
        n_transformer_layers: int = 6,
        dropout: float            = 0.0,
        skip_ch: int              = 32,
    ):
        super().__init__()
        n           = img_size // patch_size
        P           = n * n
        self.n_patches      = P
        self.n_per_side     = n
        self.n_levels       = n_levels
        self.d_model        = d_model

        n_doublings = int(math.log2(patch_size))

        # Shared patch embedding — extracts local patch features once
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels * window_size, d_model)

        # Per-level linear projections — each level gets a distinct learned readout
        self.level_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_levels)
        ])

        # Transformer with 2D RoPE and axial+cross-level mask
        self.transformer = HierarchicalTransformer(d_model, n_heads, n_transformer_layers, n, dropout)

        # Skip encoder: multi-scale features from raw input for the decoder
        self.skip_encoder = SkipEncoder(in_channels * window_size, skip_ch, n_doublings)

        # Decoder — one linear head per level, equally weighted sum + skip connections
        self.decoder = PerLevelDecoder(d_model, in_channels, img_size, patch_size, n_levels, skip_ch=skip_ch)

        # Mask caches keyed by n_levels (built lazily, reused across steps)
        # _mask_cache: float additive mask for CPU / SDPA fallback
        # _block_mask_cache: BlockMask for flex_attention on CUDA
        self._mask_cache: dict[int, torch.Tensor] = {}
        self._block_mask_cache: dict[int, object] = {}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the last conv so predictions start at zero delta.
        last_conv = self.decoder.up_convs[-1]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, n_levels: Optional[int] = None) -> dict:
        """
        x:        (B,C,H,W) or (B,T,C,H,W) normalised input
        n_levels: active levels for this step (curriculum); defaults to self.n_levels

        Returns:
            pred: (B, C, H, W) predicted normalised frame delta
        """
        if n_levels is None:
            n_levels = self.n_levels

        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B, T * C, H, W)

        B = x.size(0)
        P = self.n_patches

        # Multi-scale skip features from raw input (before patch projection)
        skip_feats = self.skip_encoder(x)

        # Shared patch features, projected differently per level
        base   = self.patch_embed(x)                                      # (B, P, d)
        tokens = torch.cat(
            [self.level_projs[k](base) for k in range(n_levels)],
            dim=1,
        )                                                                  # (B, K*P, d)

        # Axial + cross-level mask (cached per n_levels)
        if _FLEX_AVAILABLE and x.is_cuda:
            if n_levels not in self._block_mask_cache:
                self._block_mask_cache[n_levels] = _create_block_mask(
                    _make_mask_mod(self.n_per_side, P),
                    B=None, H=None, Q_LEN=n_levels * P, KV_LEN=n_levels * P,
                    device=str(x.device),
                )
            mask = self._block_mask_cache[n_levels]
        else:
            if n_levels not in self._mask_cache:
                self._mask_cache[n_levels] = build_axial_cross_level_mask(
                    n_levels, P, self.n_per_side,
                )
            mask = self._mask_cache[n_levels].to(x.device)
        out = self.transformer(tokens, mask, n_levels)   # (B, K*P, d)

        weights = torch.ones(B, n_levels * P, device=x.device)
        pred    = self.decoder(out, weights, skip_feats) # (B, C, H, W)

        return {'pred': pred}

# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

def _recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    l1  = F.l1_loss(pred, target, reduction='none')
    mse = F.mse_loss(pred, target, reduction='none')
    if pixel_mask is not None:
        # Avoid boolean indexing — its backward is broken on CPU/MPS backends.
        m = pixel_mask.float().expand_as(pred)
        n = m.sum().clamp(min=1)
        return (l1 * m).sum() / n + (mse * m).sum() / n
    return l1.mean() + mse.mean()
