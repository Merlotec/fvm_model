"""
Hierarchical Latent Fluid Simulation Model.

Architecture:
  1. PatchEmbed        — (B,C,H,W) → (B,P,d) image patch tokens
  2. CrossAttnEncoder  — N learned query tokens cross-attend to patch tokens
                         → (B,N,d) latent tokens  [tokens are independent: no latent self-attn]
  3. HierarchicalTransformer — self-attention over latent with block-causal mask
                         → (B,N,d) predicted next-state latent
  4. LatentDecoder     — predicted latent cross-attended to by patch queries → (B,C,H,W)

Token hierarchy:
  Group 0  : tokens [0, n_base)           — always active, encode most critical info
  Group k≥1: tokens [n_base+(k-1)*s, n_base+k*s) — progressively finer detail

Causal mask: group k can attend to groups 0..k; earlier groups cannot attend to later ones.
This enables KV-caching: group-0 KV is computed once and reused for all subsequent groups.

Training phases:
  Phase 1 — Curriculum: start with n_base tokens, expand by step_size every N steps.
  Phase 2 — Value learning: freeze backbone, train value_head to predict per-group
             marginal loss decrease (L_{k-1} - L_k). Each latent token in group k is
             supervised to produce its group's marginal improvement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Conv2d patch projection + learned 1-D positional embedding."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0
        n = img_size // patch_size
        self.n_patches = n * n
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos  = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, P, d)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos


class FFN(nn.Module):
    def __init__(self, d: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * expansion, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttnLayer(nn.Module):
    """
    One cross-attention layer: latent queries independently attend to context.

    Intentionally has NO self-attention between latent tokens, so that
    latent[i] depends only on query_tokens[i] and the image context.
    This independence lets Phase-2 evaluation slice the latent safely.
    """

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn     = FFN(d, dropout=dropout)
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.norm_ff = nn.LayerNorm(d)

    def forward(self, latent: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        nkv = self.norm_kv(ctx)
        x   = latent + self.attn(self.norm_q(latent), nkv, nkv)[0]
        return x + self.ffn(self.norm_ff(x))


class CrossAttnEncoder(nn.Module):
    """Stack of CrossAttnLayers: (B,N,d) queries + (B,P,d) image → (B,N,d) latent."""

    def __init__(self, d: int, n_heads: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttnLayer(d, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, latent: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            latent = layer(latent, ctx)
        return latent


class SelfAttnLayer(nn.Module):
    """Pre-norm transformer block with optional additive attention mask."""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn   = FFN(d, dropout=dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, attn_mask=mask)[0]
        return x + self.ffn(self.norm2(x))


class HierarchicalTransformer(nn.Module):
    """
    Standard transformer whose attention is shaped by a block-causal mask.

    The mask enforces: group k sees groups 0..k but NOT k+1, k+2, ...
    This is equivalent to causal attention where the "time axis" is the
    token-group hierarchy instead of sequence position.
    """

    def __init__(self, d: int, n_heads: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([SelfAttnLayer(d, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class LatentDecoder(nn.Module):
    """
    Decode N latent tokens → (B, C, H, W) via one cross-attention layer.

    Learned patch queries attend to all N latent tokens, then each patch
    token is projected to its C*p*p pixel block and the blocks are tiled.
    Handles any N naturally (cross-attention is flexible in key/value length).
    """

    def __init__(
        self, d: int, out_channels: int, img_size: int, patch_size: int,
        n_heads: int, dropout: float = 0.0,
    ):
        super().__init__()
        n  = img_size // patch_size
        self.n_per_side   = n
        self.patch_size   = patch_size
        self.out_channels = out_channels

        self.patch_q = nn.Parameter(torch.randn(1, n * n, d) * 0.02)
        self.attn    = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn     = FFN(d, dropout=dropout)
        self.norm_q  = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.norm_ff = nn.LayerNorm(d)
        self.proj    = nn.Linear(d, out_channels * patch_size * patch_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B   = latent.size(0)
        q   = self.patch_q.expand(B, -1, -1)
        nkv = self.norm_kv(latent)
        x   = q + self.attn(self.norm_q(q), nkv, nkv)[0]
        x   = x + self.ffn(self.norm_ff(x))                      # (B, P, d)

        # Fold patch pixels into image layout
        patches = self.proj(x)                                    # (B, P, C*p*p)
        p, C, n = self.patch_size, self.out_channels, self.n_per_side
        img = patches.view(B, n, n, C, p, p)
        img = img.permute(0, 3, 1, 4, 2, 5).reshape(B, C, n * p, n * p)
        return img


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_hierarchical_mask(
    n_tokens: int,
    n_base: int,
    step_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build an additive attention mask enforcing the group-causal hierarchy.

    Group assignment:
      Group 0  : tokens [0, n_base)
      Group k≥1: tokens [n_base + (k-1)*step_size, n_base + k*step_size)

    Entry (i, j):  0.0   → token i may attend to token j
                   -inf  → token i may NOT attend to token j

    Invariant: group(i) >= group(j)  ⟺  i may attend to j.
    Within the same group: full bidirectional attention.
    Across groups: only downstream-to-upstream (later → earlier) attention.
    """
    groups = torch.zeros(n_tokens, dtype=torch.long)
    if n_tokens > n_base and step_size > 0:
        extra = torch.arange(n_tokens - n_base)
        groups[n_base:] = extra // step_size + 1

    can_attend = groups.unsqueeze(1) >= groups.unsqueeze(0)       # (N, N) bool
    mask = torch.full((n_tokens, n_tokens), float('-inf'))
    mask[can_attend] = 0.0

    return mask if device is None else mask.to(device)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HierarchicalLatentModel(nn.Module):
    """
    Hierarchical Latent Fluid Simulation Model.

    Args:
        img_size:            Spatial resolution (assumed square).
        patch_size:          Patch size (must divide img_size).
        in_channels:         Per-frame channels (e.g. 4 for Vx,Vy,rho,T).
        window_size:         Number of temporal frames concatenated as input (T).
                             The patch embedder sees in_channels*window_size channels.
                             Output is always in_channels (single-frame delta).
        n_base:              Base token count — always active (default 100).
        step_size:           Tokens added per curriculum stage (s).
        n_steps:             Number of additional stages (total = n_base + n_steps*step_size).
        d_model:             Embedding dimension.
        n_heads:             Attention heads.
        n_encoder_layers:    Cross-attention encoder depth.
        n_transformer_layers: Hierarchical transformer depth.
        dropout:             Dropout probability.
    """

    def __init__(
        self,
        img_size: int             = 512,
        patch_size: int           = 32,
        in_channels: int          = 4,
        window_size: int          = 1,
        n_base: int               = 100,
        step_size: int            = 10,
        n_steps: int              = 10,
        d_model: int              = 512,
        n_heads: int              = 8,
        n_encoder_layers: int     = 3,
        n_transformer_layers: int = 6,
        dropout: float            = 0.0,
    ):
        super().__init__()
        self.n_base        = n_base
        self.step_size     = step_size
        self.n_steps       = n_steps
        self.total_tokens  = n_base + n_steps * step_size
        self.d_model       = d_model
        self.window_size   = window_size
        self.in_channels   = in_channels

        # PatchEmbed sees T*C channels; decoder always outputs C (single-frame delta)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels * window_size, d_model)

        # Learned query tokens — positional index encodes hierarchy
        self.query_tokens = nn.Parameter(torch.randn(self.total_tokens, d_model) * 0.02)

        self.encoder     = CrossAttnEncoder(d_model, n_heads, n_encoder_layers, dropout)
        self.transformer = HierarchicalTransformer(d_model, n_heads, n_transformer_layers, dropout)
        self.decoder     = LatentDecoder(d_model, in_channels, img_size, patch_size, n_heads, dropout)

        # Phase-2 value head: predicts per-token marginal loss improvement.
        # Gradient flows through this head back into the latent representation,
        # so the tokens learn to encode their own value in their embedding.
        self.value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Grouped token utilities
    # ------------------------------------------------------------------

    def group_boundaries(self, n_tokens: int) -> list[int]:
        """
        Return cumulative token counts at each group boundary.

        Example (n_base=100, step_size=10, n_tokens=130):
          → [100, 110, 120, 130]
        """
        bounds = [min(self.n_base, n_tokens)]
        n = self.n_base
        while n < n_tokens:
            n = min(n + self.step_size, n_tokens)
            if n != bounds[-1]:
                bounds.append(n)
        return bounds

    # ------------------------------------------------------------------
    # Forward components
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """
        (B,C,H,W) or (B,T,C,H,W) → (B,n_tokens,d) latent tokens.

        Temporal windows are flattened along the channel axis before patch
        embedding, giving the model access to velocity history without
        architectural changes to the encoder.

        Each latent token is independent (cross-attn only, no latent self-attn),
        so latent[:, :k] is identical whether n_tokens=k or n_tokens=N.
        This property makes Phase-2 group-boundary evaluation cheap.
        """
        if x.dim() == 5:                                                     # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.reshape(B, T * C, H, W)
        B         = x.size(0)
        patch_tok = self.patch_embed(x)                                      # (B, P, d)
        q         = self.query_tokens[:n_tokens].unsqueeze(0).expand(B, -1, -1)
        return self.encoder(q, patch_tok)                                    # (B, N, d)

    def predict(self, latent: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B,N,d) latent → (B,N,d) predicted next-state latent."""
        n = latent.size(1)
        if mask is None:
            mask = build_hierarchical_mask(n, self.n_base, self.step_size, latent.device)
        return self.transformer(latent, mask)

    def decode(self, pred_latent: torch.Tensor) -> torch.Tensor:
        """(B,N,d) predicted latent → (B,C,H,W) predicted frame delta."""
        return self.decoder(pred_latent)

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, n_tokens: Optional[int] = None) -> dict:
        """
        x:        (B, C, H, W) single frame  OR  (B, T, C, H, W) temporal window
        n_tokens: how many latent tokens to use (curriculum control; defaults to total)

        Returns:
            pred:        (B, C, H, W) predicted normalised frame delta
            latent:      (B, N, d)   encoded current-state latent
            pred_latent: (B, N, d)   predicted next-state latent
        """
        if n_tokens is None:
            n_tokens = self.total_tokens

        latent      = self.encode(x, n_tokens)
        device      = x.device if x.dim() == 4 else x.device
        mask        = build_hierarchical_mask(n_tokens, self.n_base, self.step_size, device)
        pred_latent = self.predict(latent, mask)
        pred        = self.decode(pred_latent)

        return {'pred': pred, 'latent': latent, 'pred_latent': pred_latent}

    # ------------------------------------------------------------------
    # Phase-2 helpers
    # ------------------------------------------------------------------

    def value_predictions(self, latent: torch.Tensor) -> torch.Tensor:
        """(B,N,d) → (B,N) per-token value scores."""
        return self.value_head(latent).squeeze(-1)

    def compute_group_losses(
        self,
        latent: torch.Tensor,
        target_norm: torch.Tensor,
        pixel_mask: Optional[torch.Tensor],
    ) -> list[float]:
        """
        For each group boundary, run transformer + decoder and compute loss.

        Because latent tokens are independent (no encoder self-attn), slicing
        latent[:, :n_g] is exact — no re-encoding needed.

        Returns list of scalar losses, one per group boundary.
        """
        losses = []
        for n_g in self.group_boundaries(latent.size(1)):
            mask        = build_hierarchical_mask(n_g, self.n_base, self.step_size, latent.device)
            pred_latent = self.predict(latent[:, :n_g], mask)
            pred        = self.decode(pred_latent)
            losses.append(_recon_loss(pred, target_norm, pixel_mask).item())
        return losses


# ---------------------------------------------------------------------------
# Loss helper (shared between model and lightning wrapper)
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
