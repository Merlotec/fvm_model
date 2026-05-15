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
  3. HierarchicalTransformer — (B,K*P,d) with level-causal mask:
                            level k can attend to levels 0..k only.
                            This enforces that level 0 must predict well alone,
                            level 1 can refine using level 0's full-image context,
                            and so on.
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


class SelfAttnLayer(nn.Module):
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
    def __init__(self, d: int, n_heads: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([SelfAttnLayer(d, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PerLevelDecoder(nn.Module):
    """
    Per-level linear decode + gated residual sum.

    Each level k has its own projection head (LayerNorm → Linear, d → C·p²).
    The final patch pixel values are:

        output[p] = Σ_k  gate_weight[k, p] · head_k(transformer_out[k, p])

    Level 0 always has gate_weight=1.0, so it provides the base prediction.
    Each subsequent level adds a gated residual correction at each position.
    Positions with near-zero cumulative gate weight contribute nothing, making
    it trivial to skip those levels at sparse-inference time.
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

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, out_channels * patch_size * patch_size),
            )
            for _ in range(n_levels)
        ])

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

        return patches.view(B, n, n, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, n * p, n * p)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_hierarchical_mask(
    n_tokens: int, n_base: int, step_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Additive attention mask: 0 = may attend, -inf = blocked.

    Group 0  : tokens [0, n_base)
    Group k≥1: tokens [n_base + (k-1)*step_size, n_base + k*step_size)

    Token i may attend to token j iff group(i) >= group(j).
    """
    groups = torch.zeros(n_tokens, dtype=torch.long)
    if n_tokens > n_base and step_size > 0:
        groups[n_base:] = torch.arange(n_tokens - n_base) // step_size + 1

    can_attend = groups.unsqueeze(1) >= groups.unsqueeze(0)
    mask = torch.full((n_tokens, n_tokens), float('-inf'))
    mask[can_attend] = 0.0
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
        P = (img_size // patch_size) ** 2
        self.n_patches     = P
        self.n_levels      = n_levels
        self.gate_threshold = gate_threshold
        self.gate_budget   = gate_budget
        self.d_model       = d_model

        # Shared patch embedding — sees T*C channels
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels * window_size, d_model)

        # Level embeddings: distinguish which level each token belongs to
        self.level_embed = nn.Parameter(torch.randn(n_levels, d_model) * 0.02)

        # Transformer — the level-causal mask is built per forward pass
        self.transformer = HierarchicalTransformer(d_model, n_heads, n_transformer_layers, dropout)

        # Gate[k]: reads level-k output → importance of level-k+1 at each position
        self.gates = nn.ModuleList([GateNetwork(d_model) for _ in range(n_levels - 1)])

        # Decoder — one linear head per level, gated residual sum
        self.decoder = PerLevelDecoder(d_model, in_channels, img_size, patch_size, n_levels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Gate weight computation
    # ------------------------------------------------------------------

    def effective_gate_weights(
        self, transformer_out: torch.Tensor, n_levels: int,
    ) -> torch.Tensor:
        """
        Per-token effective weights, (B, n_levels*P).

          Level 0 : weight = 1.0  (always active, no gate)
          Level k : weight = gate[0] * gate[1] * ... * gate[k-1]  at each position

        Cascade property: if gate[j][p] ≈ 0 for any j < k, the effective weight
        at level k and position p collapses to ≈ 0, so position p contributes
        nothing to the decoder from level k onward.

        Gradients flow correctly through the cumulative product — each gate[j]
        receives gradient proportional to the product of all other gates times
        the downstream decoder gradient.
        """
        P  = self.n_patches
        B  = transformer_out.size(0)
        dev = transformer_out.device
        weights = torch.ones(B, n_levels * P, device=dev)

        cumulative = torch.ones(B, P, device=dev)
        for k in range(n_levels - 1):
            gate_k     = self.gates[k](transformer_out[:, k * P : (k + 1) * P])  # (B, P)
            cumulative = cumulative * gate_k
            weights[:, (k + 1) * P : (k + 2) * P] = cumulative

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

        # Level-causal transformer
        mask = build_hierarchical_mask(
            n_levels * P, n_base=P, step_size=P, device=x.device,
        )
        out = self.transformer(tokens, mask)   # (B, K*P, d)

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
