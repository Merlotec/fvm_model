"""
Denoising diffusion decoder conditioned on patch tokens from the transformer backbone.

Interface mirrors CNNDecoder / LinearOverflowDecoder:
  - forward(z, n_steps=50) → (B, C, H, W)   — DDIM reverse diffusion (inference)
  - compute_loss(z, target) → scalar          — DDPM epsilon-prediction loss (training)

where z is (B, G*G, emb_dim) from FluidAxialTransformer.

Architecture: a U-Net with log2(patch_size) down/up stages.
The bottleneck sits at G×G (same spatial resolution as the patch grid), so
cross-attention between bottleneck features and patch tokens is spatially aligned.
A configurable number of up-blocks also receive cross-attention.

Training:
  Recommended — two-stage:
    Stage 1: train backbone (TemporalPatchEmbed + Transformer) with any decoder
             using MSE loss until convergence. (Already done if you have checkpoints.)
    Stage 2: freeze backbone, train only DiffusionDecoder with compute_loss.
             optimizer = Adam(diffusion_decoder.parameters(), lr=1e-4)
  Alternative — one-stage (train everything jointly from scratch):
    Replace decoder MSE with compute_loss in the lightning model, train end-to-end.
    Feasible but slower — the diffusion loss is noisier, the backbone needs ~2–3×
    more steps to settle. Lower the backbone LR by 10× relative to the decoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Noise schedule ─────────────────────────────────────────────────────────────

def _cosine_schedule(T: int):
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).
    Returns (alphas_bar, betas) each of length T.
    """
    s = 0.008
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]                                           # length T+1, ab[0]=1.0
    betas = (1 - ab[1:] / ab[:-1]).clamp(0, 0.999)         # length T
    return ab[1:], betas                                    # both length T


# ── Sinusoidal timestep embedding ──────────────────────────────────────────────

def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    x = t.float().unsqueeze(1) * freqs.unsqueeze(0)        # (B, half)
    return torch.cat([x.cos(), x.sin()], dim=-1)            # (B, dim)


# ── Building blocks ────────────────────────────────────────────────────────────

def _gn(ch: int) -> nn.GroupNorm:
    groups = 1
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            groups = g
            break
    return nn.GroupNorm(groups, ch)


class _ResBlock(nn.Module):
    """Conv ResBlock with AdaGN timestep conditioning (scale + shift)."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1  = _gn(in_ch)
        self.conv1  = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2  = _gn(out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch * 2)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        ts = self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)    # (B, 2C, 1, 1)
        scale, shift = ts.chunk(2, dim=1)
        h = F.silu(self.norm2(h) * (1 + scale) + shift)
        return self.conv2(h) + self.skip(x)


class _CrossAttn(nn.Module):
    """Spatial features (Q) cross-attend to conditioning tokens (K, V)."""

    def __init__(self, spatial_dim: int, cond_dim: int, n_heads: int):
        super().__init__()
        assert spatial_dim % n_heads == 0, \
            f"spatial_dim {spatial_dim} not divisible by n_heads {n_heads}"
        self.norm     = _gn(spatial_dim)
        self.q        = nn.Linear(spatial_dim, spatial_dim, bias=False)
        self.kv       = nn.Linear(cond_dim,    spatial_dim * 2, bias=False)
        self.proj     = nn.Linear(spatial_dim, spatial_dim)
        self.n_heads  = n_heads
        self.head_dim = spatial_dim // n_heads

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)     # (B, L, C)
        q = self.q(h)
        k, v = self.kv(cond).chunk(2, dim=-1)                       # (B, N, C)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, t.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        out = out.transpose(1, 2).reshape(B, H * W, C)
        return x + self.proj(out).permute(0, 2, 1).reshape(B, C, H, W)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.res  = _ResBlock(in_ch, out_ch, t_dim)
        self.down = nn.AvgPool2d(2)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        return self.down(x), x          # (downsampled, pre-pool skip)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 t_dim: int, cond_dim: int, use_attn: bool, n_heads: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.res  = _ResBlock(in_ch + skip_ch, out_ch, t_dim)
        self.attn = _CrossAttn(out_ch, cond_dim, n_heads) if use_attn else None

    def forward(self, x, skip, t_emb, cond):
        x = self.up(x)
        x = self.res(torch.cat([x, skip], dim=1), t_emb)
        if self.attn is not None:
            x = self.attn(x, cond)
        return x


class _MidBlock(nn.Module):
    def __init__(self, ch: int, t_dim: int, cond_dim: int, n_heads: int):
        super().__init__()
        self.res1 = _ResBlock(ch, ch, t_dim)
        self.attn = _CrossAttn(ch, cond_dim, n_heads)
        self.res2 = _ResBlock(ch, ch, t_dim)

    def forward(self, x, t_emb, cond):
        return self.res2(self.attn(self.res1(x, t_emb), cond), t_emb)


# ── Main module ────────────────────────────────────────────────────────────────

class DiffusionDecoder(nn.Module):
    """
    Args:
        emb_dim:        transformer output dimension (conditioning)
        out_channels:   output image channels (4 for this project)
        patch_size:     must match backbone patch size (power of 2)
        T:              total DDPM timesteps for training
        base_ch:        U-Net base channel count (doubles per stage, capped at max_ch)
        max_ch:         maximum channel count in the U-Net
        n_heads:        attention heads (spatial_dim and cond_dim must be divisible by this)
        attn_up_levels: how many up-blocks (counted from the bottleneck outward) receive
                        cross-attention. Default 2 → 32×32 and 64×64 levels for patch_size=32.
                        Set higher to improve quality at the cost of memory.
    """

    def __init__(
        self,
        emb_dim:        int = 512,
        out_channels:   int = 4,
        patch_size:     int = 32,
        T:              int = 1000,
        base_ch:        int = 64,
        max_ch:         int = 512,
        n_heads:        int = 8,
        attn_up_levels: int = 2,
    ):
        super().__init__()
        assert patch_size >= 2 and (patch_size & (patch_size - 1)) == 0
        n_stages = int(math.log2(patch_size))

        self.T            = T
        self.out_channels = out_channels
        self.n_stages     = n_stages

        # ── Noise schedule buffers ────────────────────────────────────────────
        alphas_bar, betas = _cosine_schedule(T)
        alphas = 1.0 - betas
        self.register_buffer('betas',             betas)
        self.register_buffer('alphas',            alphas)
        self.register_buffer('alphas_bar',        alphas_bar)
        self.register_buffer('sqrt_ab',           alphas_bar.sqrt())
        self.register_buffer('sqrt_one_minus_ab', (1 - alphas_bar).sqrt())

        # ── Timestep embedding ────────────────────────────────────────────────
        t_dim = base_ch * 4
        self.t_sin_dim = base_ch
        self.t_emb = nn.Sequential(
            nn.Linear(base_ch, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # ── Channel schedule: base_ch, 2×, 4×, … capped at max_ch ───────────
        chs = [min(base_ch * (2 ** i), max_ch) for i in range(n_stages + 1)]

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Conv2d(out_channels, chs[0], kernel_size=3, padding=1)

        # ── Down path (no cross-attention — conditioning enters at bottleneck) ─
        self.downs = nn.ModuleList(
            _DownBlock(chs[i], chs[i + 1], t_dim) for i in range(n_stages)
        )

        # ── Bottleneck (spatially aligned with patch grid G×G) ────────────────
        self.mid = _MidBlock(chs[-1], t_dim, emb_dim, n_heads)

        # ── Up path ───────────────────────────────────────────────────────────
        self.ups = nn.ModuleList()
        for j, i in enumerate(range(n_stages - 1, -1, -1)):
            use_attn = (j < attn_up_levels)
            self.ups.append(_UpBlock(
                in_ch    = chs[i + 1],
                skip_ch  = chs[i + 1],
                out_ch   = chs[i],
                t_dim    = t_dim,
                cond_dim = emb_dim,
                use_attn = use_attn,
                n_heads  = n_heads,
            ))

        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            _gn(chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], out_channels, kernel_size=3, padding=1),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _t_embed(self, t: torch.Tensor) -> torch.Tensor:
        return self.t_emb(_sinusoidal(t, self.t_sin_dim))

    def _predict_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                       z: torch.Tensor) -> torch.Tensor:
        """U-Net forward: predict noise given noisy image, timestep, and conditioning."""
        t_emb = self._t_embed(t)
        h     = self.stem(x_t)
        skips = []
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)
        h = self.mid(h, t_emb, z)
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, t_emb, z)
        return self.head(h)

    # ── Training ──────────────────────────────────────────────────────────────

    def compute_loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        DDPM epsilon-prediction loss.
          z:      (B, G*G, emb_dim)  — patch token conditioning (from frozen backbone)
          target: (B, C, H, W)       — normalised clean target
        Returns scalar MSE loss on predicted vs. actual noise.
        """
        B     = target.shape[0]
        t     = torch.randint(0, self.T, (B,), device=target.device)
        eps   = torch.randn_like(target)
        ab    = self.sqrt_ab[t][:, None, None, None]
        s1    = self.sqrt_one_minus_ab[t][:, None, None, None]
        x_t   = ab * target + s1 * eps
        return F.mse_loss(self._predict_noise(x_t, t, z), eps)

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, z: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        DDIM deterministic reverse diffusion (eta=0).
          z:       (B, G*G, emb_dim)
          n_steps: denoising steps — 50 is a good quality/speed trade-off
        Returns:   (B, C, H, W) predicted clean output
        """
        B          = z.shape[0]
        G          = int(z.shape[1] ** 0.5)
        H = W      = G * (2 ** self.n_stages)
        x          = torch.randn(B, self.out_channels, H, W, device=z.device, dtype=z.dtype)

        # evenly spaced DDIM timestep indices: T-1 → 0
        indices = torch.linspace(self.T - 1, 0, n_steps).round().long().tolist()

        for step_i, t_idx in enumerate(indices):
            t_batch  = torch.full((B,), t_idx, device=z.device, dtype=torch.long)
            eps_pred = self._predict_noise(x, t_batch, z)

            ab_t     = self.alphas_bar[t_idx]
            ab_prev  = (self.alphas_bar[indices[step_i + 1]]
                        if step_i + 1 < n_steps
                        else torch.ones(1, device=z.device, dtype=z.dtype))

            # DDIM update: x_{t-1} = sqrt(ab_prev) * x0_pred + sqrt(1-ab_prev) * eps_pred
            x0_pred = ((x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()).clamp(-4, 4)
            x       = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_pred

        return x
