"""
Lightning wrapper for the MultiLevelFluidModel.

Single-phase training: curriculum expands active levels 1 → K.
All levels contribute with uniform weight — no gating.

  Loss = recon(pred, target)

CurriculumCallback steps n_active_levels every `steps_per_stage` optimiser steps.
"""

import json
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from cprint import c_print
from tqdm import tqdm

from .model import MultiLevelFluidModel, _recon_loss  # noqa: E402

_FOUND           = Path(__file__).resolve().parents[1] / 'fvm_foundation'
DELTA_STATS_PATH = _FOUND / 'delta_stats.json'
INPUT_STATS_PATH = _FOUND / 'input_stats.json'
PIXEL_MASK_PATH  = _FOUND / 'pixel_mask.pt'


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------

class CurriculumCallback(Callback):
    """Steps up the active level count every `steps_per_stage` optimiser steps."""

    def __init__(self, steps_per_stage: int = 1000):
        self.steps_per_stage = steps_per_stage

    def on_train_batch_end(  # type: ignore[override]
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx,
    ) -> None:
        if (trainer.global_step + 1) % self.steps_per_stage == 0:
            pl_module.step_curriculum()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Lightning model
# ---------------------------------------------------------------------------

class Phase1LightningModel(L.LightningModule):
    """
    Curriculum training for MultiLevelFluidModel.

    Starts with 1 active level. CurriculumCallback calls step_curriculum()
    every N steps to add one level until all K levels are active.

    Loss = reconstruction + sparsity_weight * (mean_active - gate_budget)²

    The reconstruction term teaches each level to predict the residual.
    The sparsity term targets gate_budget fraction of level-1+ tokens active,
    preventing both gate collapse and unbounded token use.
    """

    def __init__(
        self,
        model: MultiLevelFluidModel,
        lr: float           = 1e-4,
        weight_decay: float = 1e-5,
        noise_std: float    = 0.02,
        aux_weight: float   = 0.3,
        img_size: int       = 512,
        window_size: int    = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model            = model
        self._n_active_levels = 1

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

    # ------------------------------------------------------------------
    # Curriculum state
    # ------------------------------------------------------------------

    @property
    def current_n_levels(self) -> int:
        return self._n_active_levels

    def step_curriculum(self) -> None:
        new_n = min(self._n_active_levels + 1, self.model.n_levels)
        if new_n != self._n_active_levels:
            self._n_active_levels = new_n
            tqdm.write(f'Curriculum step → {new_n} active levels')
        self.log('curriculum/n_levels', float(self._n_active_levels), prog_bar=True)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _norm_window(self, x: torch.Tensor) -> torch.Tensor:
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
        window, target = batch                              # (B,T,C,H,W), (B,C,H,W)
        window_n = self._norm_window(window)
        noise_std: float = self.hparams['noise_std']       # type: ignore[index]
        if noise_std > 0:
            window_n = window_n + torch.randn_like(window_n) * noise_std
        target_n = self._norm_delta(target)

        out = self.model(window_n, n_levels=self._n_active_levels)

        recon_loss = _recon_loss(out['pred'], target_n, self.pixel_mask)
        loss       = recon_loss

        # Auxiliary lower-level pass: randomly sample one level below the current
        # active count and include its reconstruction loss. This keeps intermediate
        # level prefixes good standalone predictors throughout training.
        aux_weight: float = self.hparams['aux_weight']  # type: ignore[index]
        if aux_weight > 0.0 and self._n_active_levels > 1:
            k_aux     = int(torch.randint(1, self._n_active_levels, (1,)).item())
            out_aux   = self.model(window_n, n_levels=k_aux)
            aux_loss  = _recon_loss(out_aux['pred'], target_n, self.pixel_mask)
            loss      = loss + aux_weight * aux_loss
            self.log('train/aux_loss',  aux_loss,      on_step=True, on_epoch=False, sync_dist=True)
            self.log('train/aux_level', float(k_aux),  on_step=True, on_epoch=False)

        pred_phys = self._denorm_delta(out['pred'])
        valid     = self.pixel_mask.expand_as(target)
        rel_err   = ((target - pred_phys).abs()[valid] /
                     target.abs()[valid].clamp(min=1e-6)).mean()

        self.log('train/loss',       loss,                    prog_bar=True,  on_step=True,  on_epoch=True, sync_dist=True)
        self.log('train/recon_loss', recon_loss,              prog_bar=False, on_step=True,  on_epoch=True, sync_dist=True)
        self.log('train/rel_err',    rel_err,                 prog_bar=True,  on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/n_levels',   float(self._n_active_levels))
        return loss


    def configure_optimizers(self):  # type: ignore[override]
        lr: float = self.hparams['lr']           # type: ignore[index]
        wd: float = self.hparams['weight_decay'] # type: ignore[index]
        opt   = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50_000, eta_min=1e-6)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}
