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
  4. GateNetwork          — after each level, a lightweight MLP produces a
                            per-position importance score in (0,1).
                            Effective weight at level k = product of gates 0..k-1
                            (cascade: dropping a position at level j also drops it
                            at all higher levels).
  5. PerLevelDecoder      — each level k has its own linear head d → C·p².
                            Final patch = Σ_k gate_weight[k,p] · head_k(out[k,p]).
                            Level 0 always contributes (weight=1); higher levels
                            contribute proportionally to their cascade gate score.
                            At hard-gate inference, zero-gated levels are free to skip.

Training
--------
  Phase 1 (curriculum): n_active_levels steps from 1 → K, adding one level
  every N optimiser steps.  With 1 level the model is just a standard ViT-like
  patch predictor; each new level is a learned refinement layer.

  Loss = reconstruction (MSE+L1) + sparsity_weight * (mean_active - gate_budget)²
  The sparsity term targets a chosen fraction of level-1+ tokens being active,
  rather than driving all gates to zero.

Inference
---------
  Soft (training-compatible): all K*P tokens present, gate scores applied as
  value weights in the decoder.

  Sparse (efficient): run levels sequentially, caching each level's KV.  After
  each level apply hard gate threshold; only surviving positions advance.  The
  level-causal mask ensures this is identical to the full-K forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
        self.scale     = self.head_dim ** -0.5
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

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale      # (B, nh, N, N)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        if self.dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, d)
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


class PerLevelDecoder(nn.Module):
    """
    Per-level linear decode + gated residual sum + convolutional boundary smoother.

    Each level k projects its token to a patch: LayerNorm → Linear (d → C·p²).
    The gated sum is unfolded back to image space, then a small conv refinement
    (two 3×3 layers, same padding) blends across patch boundaries so the 32×32
    grid artefacts are not baked into the output.

        patches[p] = Σ_k  gate_weight[k,p] · head_k(transformer_out[k,p])
        output     = refine(unfold(patches))
    """

    def __init__(
        self, d: int, out_channels: int, img_size: int, patch_size: int, n_levels: int,
    ):
        super().__init__()
        n = img_size // patch_size
        self.n_per_side   = n
        self.n_patches    = n * n
        self.patch_size   = patch_size
        self.out_channels = out_channels

        hidden = d * 4
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_channels * patch_size * patch_size),
            )
            for _ in range(n_levels)
        ])

        # Two conv layers with kernel=3 span patch boundaries and learn to blend them.
        # GroupNorm over channels keeps this stable without batch statistics.
        C = out_channels
        self.refine = nn.Sequential(
            nn.Conv2d(C, C * 4, kernel_size=3, padding=1),
            nn.GroupNorm(C, C * 4),
            nn.GELU(),
            nn.Conv2d(C * 4, C, kernel_size=3, padding=1),
        )

    def forward(self, transformer_out: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        # transformer_out: (B, n_levels*P, d)
        # gate_weights:    (B, n_levels*P)  — level 0 slice is all 1s
        B        = transformer_out.size(0)
        P        = self.n_patches
        C, p, n  = self.out_channels, self.patch_size, self.n_per_side
        n_levels = transformer_out.size(1) // P

        patches = transformer_out.new_zeros(B, P, C * p * p)
        for k in range(n_levels):
            tok = transformer_out[:, k * P : (k + 1) * P]            # (B, P, d)
            w   = gate_weights[:, k * P : (k + 1) * P].unsqueeze(-1) # (B, P, 1)
            patches = patches + w * self.heads[k](tok)

        x = patches.view(B, n, n, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, n * p, n * p)
        return x + self.refine(x)


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

    mask = torch.full((N, N), float('-inf'))
    mask[axial | cross] = 0.0
    return mask if device is None else mask.to(device)


# ---------------------------------------------------------------------------
# Gate network
# ---------------------------------------------------------------------------

class GateNetwork(nn.Module):
    """
    Per-position importance gate.

    Takes level-k transformer output (B, P, d) → scalar score per position (B, P)
    in (0, 1).  A score near 0 means "level k+1 is not needed at this position";
    near 1 means "keep refining here."

    The gate reads the FULL transformer output at level k, which includes
    attended context from all levels 0..k.  This gives the gate the richest
    possible signal for deciding whether further refinement is worthwhile.
    """

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B, P)


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
        gate_threshold:       Hard gate cutoff at inference (default 0.5).
        gate_budget:          Target fraction of level-1+ tokens to keep during
                              training (default 0.4).  The sparsity loss drives
                              mean gate weight toward this target rather than 0.
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
        gate_threshold: float     = 0.5,
        gate_budget: float        = 0.4,
        dropout: float            = 0.0,
    ):
        super().__init__()
        n           = img_size // patch_size
        P           = n * n
        self.n_patches      = P
        self.n_per_side     = n
        self.n_levels       = n_levels
        self.gate_threshold = gate_threshold
        self.gate_budget    = gate_budget
        self.d_model        = d_model

        # Shared patch embedding — sees T*C channels
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels * window_size, d_model)

        # Level embeddings: distinguish which level each token belongs to
        self.level_embed = nn.Parameter(torch.randn(n_levels, d_model) * 0.02)

        # Transformer with 2D RoPE and axial+cross-level mask
        self.transformer = HierarchicalTransformer(d_model, n_heads, n_transformer_layers, n, dropout)

        # Gate[k]: reads level-k output → importance of level-k+1 at each position
        self.gates = nn.ModuleList([GateNetwork(d_model) for _ in range(n_levels - 1)])

        # Decoder — one linear head per level, gated residual sum
        self.decoder = PerLevelDecoder(d_model, in_channels, img_size, patch_size, n_levels)

        # Mask cache keyed by n_levels (built lazily, reused across steps)
        self._mask_cache: dict[int, torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the last conv in refine so it starts as identity (output = 0).
        # Without this, refine(x) ≠ 0 at init and corrupts early reconstruction.
        last_conv = self.decoder.refine[-1]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    # ------------------------------------------------------------------
    # Gate weight computation
    # ------------------------------------------------------------------

    def effective_gate_weights(
        self, transformer_out: torch.Tensor, n_levels: int,
    ) -> torch.Tensor:
        """
        Per-token effective weights, (B, n_levels*P).

          Level 0 : weight = 1.0  (always contributes, no gate)
          Level k : weight = gates[k-1](transformer_out[k-1])  at each position

        Each level's weight is independent — gate[k-1] reads level-(k-1) output
        and decides how much level-k's head should contribute to the output.

        Independent gates avoid the cascade gradient problem where level-k heads
        would otherwise receive gradient ≈ 0.5^k of level-0.  Each head now
        sees a roughly uniform gradient scale regardless of depth.
        """
        P   = self.n_patches
        B   = transformer_out.size(0)
        dev = transformer_out.device
        weights = torch.ones(B, n_levels * P, device=dev)

        for k in range(n_levels - 1):
            gate_k = self.gates[k](transformer_out[:, k * P : (k + 1) * P])  # (B, P)
            weights[:, (k + 1) * P : (k + 2) * P] = gate_k

        return weights   # (B, n_levels*P)

    def active_token_fraction(
        self, gate_weights: torch.Tensor, n_levels: int,
    ) -> torch.Tensor:
        """Mean fraction of level-1+ tokens above the hard gate threshold."""
        if n_levels == 1:
            return gate_weights.new_zeros(1).squeeze()
        P = self.n_patches
        return (gate_weights[:, P:] > self.gate_threshold).float().mean()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, n_levels: Optional[int] = None) -> dict:
        """
        x:        (B,C,H,W) or (B,T,C,H,W) normalised input
        n_levels: active levels for this step (curriculum); defaults to self.n_levels

        Returns:
            pred:         (B, C, H, W) predicted normalised frame delta
            gate_weights: (B, n_levels*P) effective per-token weights
            sparsity:     scalar — mean weight of level-1+ tokens (target: gate_budget)
            active_frac:  scalar — hard-threshold fraction active at inference
        """
        if n_levels is None:
            n_levels = self.n_levels

        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B, T * C, H, W)

        B = x.size(0)
        P = self.n_patches

        # Base patch features, shared across all levels
        base = self.patch_embed(x)    # (B, P, d)

        # Stack K copies with different level embeddings
        tokens = torch.cat(
            [base + self.level_embed[k] for k in range(n_levels)],
            dim=1,
        )                              # (B, K*P, d)

        # Axial + cross-level mask (cached per n_levels)
        if n_levels not in self._mask_cache:
            self._mask_cache[n_levels] = build_axial_cross_level_mask(
                n_levels, P, self.n_per_side,
            )
        mask = self._mask_cache[n_levels].to(x.device)
        out = self.transformer(tokens, mask, n_levels)   # (B, K*P, d)

        # Gate weights (soft, differentiable)
        weights = self.effective_gate_weights(out, n_levels)   # (B, K*P)

        # Decode: each level's head weighted by its gate score, summed per position
        pred = self.decoder(out, weights)                      # (B, C, H, W)

        # Sparsity metrics
        sparsity    = weights[:, P:].mean() if n_levels > 1 else weights.new_zeros(1).squeeze()
        active_frac = self.active_token_fraction(weights, n_levels)

        return {
            'pred':         pred,
            'gate_weights': weights,
            'sparsity':     sparsity,
            'active_frac':  active_frac,
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict_with_budget(
        self, x: torch.Tensor, max_active_fraction: float = 0.3,
    ) -> dict:
        """
        Run full forward pass, then report which positions would be dropped
        at each level if using max_active_fraction as a budget threshold.
        Useful for profiling without implementing the sparse kernel.
        """
        out = self.forward(x)
        P   = self.n_patches
        report = []
        for k in range(self.n_levels):
            w      = out['gate_weights'][:, k * P : (k + 1) * P]
            active = (w > max_active_fraction).float().mean().item()
            report.append({'level': k, 'mean_weight': w.mean().item(), 'active_frac': active})
        out['level_report'] = report
        return out


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

def _recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pixel_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pixel_mask is not None:
        m      = pixel_mask.expand_as(pred)
        pred   = pred[m]
        target = target[m]
    return F.mse_loss(pred, target) + 0.1 * F.l1_loss(pred, target)
