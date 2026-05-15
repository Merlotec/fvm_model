"""
Lightning wrappers for the HierarchicalLatentModel.

Two classes:
  Phase1LightningModel — curriculum training (expand token count over time).
  Phase2LightningModel — value learning (freeze backbone, train value_head).

A CurriculumCallback manages stepping up the token count in Phase 1.
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import Callback
from cprint import c_print

# Make sure model.py (in the same directory) is importable regardless of cwd
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from model import HierarchicalLatentModel, _recon_loss  # type: ignore[import]  noqa: E402

# Reuse normalisation artefacts from the sibling fvm_foundation package
_FOUND           = Path(__file__).resolve().parents[1] / 'fvm_foundation'
DELTA_STATS_PATH = _FOUND / 'delta_stats.json'
INPUT_STATS_PATH = _FOUND / 'input_stats.json'
PIXEL_MASK_PATH  = _FOUND / 'pixel_mask.pt'


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(Callback):
    """
    Steps up the active token count every `steps_per_stage` optimiser steps.
    Calls pl_module.step_curriculum(), defined on Phase1LightningModel.
    """

    def __init__(self, steps_per_stage: int = 1000):
        self.steps_per_stage = steps_per_stage

    def on_train_batch_end(  # type: ignore[override]
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx,
    ) -> None:
        if (trainer.global_step + 1) % self.steps_per_stage == 0:
            pl_module.step_curriculum()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Phase 1 — curriculum training
# ---------------------------------------------------------------------------

class Phase1LightningModel(L.LightningModule):
    """
    Curriculum training for HierarchicalLatentModel.

    Starts with n_base tokens.  CurriculumCallback calls step_curriculum()
    every N steps to expand by step_size until all tokens are active.

    The block-causal mask means base tokens must always do well alone —
    they can never look at later tokens — so the curriculum naturally forces
    the most important information into the base group first.
    """

    def __init__(
        self,
        model: HierarchicalLatentModel,
        lr: float           = 1e-4,
        weight_decay: float = 1e-5,
        noise_std: float    = 0.02,
        img_size: int       = 512,
        window_size: int    = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model     = model
        self._n_tokens = model.n_base

        N = 4
        self.register_buffer('input_mean', torch.zeros(1, N, 1, 1))
        self.register_buffer('input_std',  torch.ones( 1, N, 1, 1))
        self.register_buffer('delta_mean', torch.zeros(1, N, 1, 1))
        self.register_buffer('delta_std',  torch.ones( 1, N, 1, 1))
        # Full-resolution mask so copy_() works without shape mismatch
        self.register_buffer('pixel_mask', torch.ones(1, 1, img_size, img_size, dtype=torch.bool))

    # -- type declarations so arithmetic ops are unambiguous to linters --
    input_mean: torch.Tensor
    input_std:  torch.Tensor
    delta_mean: torch.Tensor
    delta_std:  torch.Tensor
    pixel_mask: torch.Tensor

    # ------------------------------------------------------------------
    # Curriculum state
    # ------------------------------------------------------------------

    @property
    def current_n_tokens(self) -> int:
        return self._n_tokens

    def step_curriculum(self) -> None:
        new_n = min(self._n_tokens + self.model.step_size, self.model.total_tokens)
        if new_n != self._n_tokens:
            self._n_tokens = new_n
            c_print(f'Curriculum step → {new_n} tokens', color='cyan')
        self.log('curriculum/n_tokens', float(self._n_tokens), prog_bar=True)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _norm_window(self, x: torch.Tensor) -> torch.Tensor:
        # Works for both (B,C,H,W) and (B,T,C,H,W) by broadcasting over the T dim
        m = self.input_mean.unsqueeze(1) if x.dim() == 5 else self.input_mean
        s = self.input_std.unsqueeze(1)  if x.dim() == 5 else self.input_std
        return ((x - m) / s).nan_to_num(0.0)

    def _norm_delta(self, d: torch.Tensor) -> torch.Tensor:
        return (d - self.delta_mean) / self.delta_std

    def _denorm_delta(self, d: torch.Tensor) -> torch.Tensor:
        return d * self.delta_std + self.delta_mean

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        def _load(path: Path, mbuf: torch.Tensor, sbuf: torch.Tensor, label: str) -> None:
            if path.exists():
                with open(path) as f:
                    s = json.load(f)
                NC = mbuf.size(1)
                mbuf.copy_(torch.tensor(s['mean'], device=self.device).view(1, NC, 1, 1))
                sbuf.copy_( torch.tensor(s['std'],  device=self.device).view(1, NC, 1, 1))
                c_print(f'Loaded {label}', color='green')
            else:
                c_print(f'Warning: {path.name} not found — identity normalisation', color='yellow')

        _load(DELTA_STATS_PATH, self.delta_mean, self.delta_std, 'delta stats')
        _load(INPUT_STATS_PATH, self.input_mean, self.input_std, 'input stats')

        if PIXEL_MASK_PATH.exists():
            self.pixel_mask.copy_(torch.load(PIXEL_MASK_PATH, map_location=self.device))
            c_print('Loaded pixel mask', color='green')
        else:
            c_print('Warning: pixel_mask.pt not found — all pixels treated as fluid', color='yellow')

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        window, target = batch                             # (B,T,C,H,W), (B,C,H,W)
        window_n = self._norm_window(window)
        noise_std: float = self.hparams['noise_std']      # type: ignore[index]
        if noise_std > 0:
            window_n = window_n + torch.randn_like(window_n) * noise_std
        target_n = self._norm_delta(target)

        out  = self.model(window_n, n_tokens=self._n_tokens)
        loss = _recon_loss(out['pred'], target_n, self.pixel_mask)

        pred_phys = self._denorm_delta(out['pred'])
        valid     = self.pixel_mask.expand_as(target)
        rel_err   = ((target - pred_phys).abs()[valid] /
                     target.abs()[valid].clamp(min=1e-6)).mean()

        self.log('train/loss',     loss,    prog_bar=True,  on_step=True,  on_epoch=True, sync_dist=True)
        self.log('train/rel_err',  rel_err, prog_bar=True,  on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/n_tokens', float(self._n_tokens))
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        window, target = batch
        window_n = self._norm_window(window)
        target_n = self._norm_delta(target)

        # Log loss at every group boundary so we can track hierarchy compression
        for n_tok in self.model.group_boundaries(self.model.total_tokens):
            out  = self.model(window_n, n_tokens=n_tok)
            loss = _recon_loss(out['pred'], target_n, self.pixel_mask)
            self.log(f'val/loss_{n_tok}tok', loss, sync_dist=True)

        out      = self.model(window_n)
        val_loss = _recon_loss(out['pred'], target_n, self.pixel_mask)
        self.log('val/loss', val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):  # type: ignore[override]
        lr: float = self.hparams['lr']           # type: ignore[index]
        wd: float = self.hparams['weight_decay'] # type: ignore[index]
        opt   = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50_000, eta_min=1e-6)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}


# ---------------------------------------------------------------------------
# Phase 2 — value learning
# ---------------------------------------------------------------------------

class Phase2LightningModel(L.LightningModule):
    """
    Value learning: train value_head to predict per-group marginal loss decrease.

    Trainable parameters: value_head + query_tokens (everything else frozen).

    Per batch there are two encoder forward passes:
      Pass 1 (no_grad): encode window → full_latent; run transformer+decoder at
        each group boundary → group_losses; compute marginal targets V_k.
      Pass 2 (with_grad): re-encode window → full_latent; run value_head on each
        group's tokens without detaching, so gradients flow through query_tokens.

    Why query_tokens must be trainable: value_head takes the latent tokens as
    input, and latent[i] = CrossAttn(query_tokens[i], image_patches).  If we
    detach the latent before value_head (as originally written), gradients never
    reach query_tokens and the tokens cannot restructure to encode their value.
    Allowing gradients into query_tokens (but not the cross-attention weights)
    lets each token's "what to ask for" adapt so the latent naturally carries
    its group's importance signal.

    At inference: value_head(encode(window)) gives per-token value scores.
    Stop adding groups when the mean score falls below a chosen threshold.
    """

    def __init__(
        self,
        model: HierarchicalLatentModel,
        lr: float        = 3e-4,
        noise_std: float = 0.02,
        img_size: int    = 512,
        window_size: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model

        for name, param in self.model.named_parameters():
            # value_head: predicts the marginal value.
            # query_tokens: the learned "what to ask for" — must be trainable so
            #   gradients from value_head can reshape what each token encodes.
            param.requires_grad = (
                name.startswith('value_head') or name == 'query_tokens'
            )

        N = 4
        self.register_buffer('input_mean', torch.zeros(1, N, 1, 1))
        self.register_buffer('input_std',  torch.ones( 1, N, 1, 1))
        self.register_buffer('delta_mean', torch.zeros(1, N, 1, 1))
        self.register_buffer('delta_std',  torch.ones( 1, N, 1, 1))
        self.register_buffer('pixel_mask', torch.ones(1, 1, img_size, img_size, dtype=torch.bool))

    input_mean: torch.Tensor
    input_std:  torch.Tensor
    delta_mean: torch.Tensor
    delta_std:  torch.Tensor
    pixel_mask: torch.Tensor

    def _norm_window(self, x: torch.Tensor) -> torch.Tensor:
        m = self.input_mean.unsqueeze(1) if x.dim() == 5 else self.input_mean
        s = self.input_std.unsqueeze(1)  if x.dim() == 5 else self.input_std
        return ((x - m) / s).nan_to_num(0.0)

    def _norm_delta(self, d: torch.Tensor) -> torch.Tensor:
        return (d - self.delta_mean) / self.delta_std

    def on_fit_start(self) -> None:
        def _load(path: Path, mbuf: torch.Tensor, sbuf: torch.Tensor, label: str) -> None:
            if path.exists():
                with open(path) as f:
                    s = json.load(f)
                NC = mbuf.size(1)
                mbuf.copy_(torch.tensor(s['mean'], device=self.device).view(1, NC, 1, 1))
                sbuf.copy_( torch.tensor(s['std'],  device=self.device).view(1, NC, 1, 1))
                c_print(f'Loaded {label}', color='green')
            else:
                c_print(f'Warning: {path.name} not found', color='yellow')

        _load(DELTA_STATS_PATH, self.delta_mean, self.delta_std, 'delta stats')
        _load(INPUT_STATS_PATH, self.input_mean, self.input_std, 'input stats')
        if PIXEL_MASK_PATH.exists():
            self.pixel_mask.copy_(torch.load(PIXEL_MASK_PATH, map_location=self.device))

    def _compute_group_marginals(
        self, full_latent: torch.Tensor, target_n: torch.Tensor,
    ) -> tuple[list[float], list[float], list[int]]:
        """Returns (group_losses, marginals, boundaries)."""
        boundaries   = self.model.group_boundaries(full_latent.size(1))
        group_losses = self.model.compute_group_losses(full_latent, target_n, self.pixel_mask)
        marginals    = [group_losses[0]]
        for k in range(1, len(group_losses)):
            marginals.append(group_losses[k - 1] - group_losses[k])
        return group_losses, marginals, boundaries

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        window, target = batch
        window_n = self._norm_window(window)
        target_n = self._norm_delta(target)

        # Pass 1 (no_grad): compute group losses to derive marginal value targets.
        # Kept separate so the expensive transformer+decoder runs are not tracked.
        with torch.no_grad():
            full_latent_ng = self.model.encode(window_n, self.model.total_tokens)
            group_losses, marginals, boundaries = self._compute_group_marginals(
                full_latent_ng, target_n,
            )

        # Pass 2 (with_grad): re-encode so gradients flow query_tokens → value_head.
        # The cross-attention weights are frozen; only query_tokens are updated.
        full_latent = self.model.encode(window_n, self.model.total_tokens)

        value_loss = torch.tensor(0.0, device=self.device)
        prev_n = 0
        for n_g, mv in zip(boundaries, marginals):
            value_pred = self.model.value_head(full_latent[:, prev_n:n_g]).squeeze(-1)
            target_v   = torch.full_like(value_pred, mv)
            value_loss = value_loss + F.mse_loss(value_pred, target_v)
            prev_n     = n_g

        value_loss = value_loss / len(boundaries)

        self.log('train/value_loss', value_loss, prog_bar=True, sync_dist=True)
        for k, (n_g, gl) in enumerate(zip(boundaries, group_losses)):
            self.log(f'train/group_loss_g{k}_{n_g}tok', gl, sync_dist=True)
        return value_loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        window, target = batch
        window_n = self._norm_window(window)
        target_n = self._norm_delta(target)

        with torch.no_grad():
            full_latent = self.model.encode(window_n, self.model.total_tokens)
            group_losses, marginals, boundaries = self._compute_group_marginals(
                full_latent, target_n,
            )

        value_loss = torch.tensor(0.0, device=self.device)
        prev_n = 0
        for n_g, mv in zip(boundaries, marginals):
            vp         = self.model.value_head(full_latent[:, prev_n:n_g]).squeeze(-1)
            value_loss = value_loss + F.mse_loss(vp, torch.full_like(vp, mv))
            prev_n     = n_g

        self.log('val/value_loss', value_loss / len(boundaries), prog_bar=True, sync_dist=True)
        self.log('val/loss_base',  group_losses[0],              sync_dist=True)
        self.log('val/loss_full',  group_losses[-1],             sync_dist=True)

    def configure_optimizers(self):  # type: ignore[override]
        lr: float = self.hparams['lr']  # type: ignore[index]
        params    = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr)
