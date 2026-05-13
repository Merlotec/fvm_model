import json
import sys
from pathlib import Path

import torch
import lightning as L
from cprint import c_print

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from model import FluidVisionModel

from helper import (
    print_histogram,
    RESOLUTION, PATCH_SIZE, EMB_DIM, N_CHANNELS, WINDOW_SIZE, NUM_LAYERS,
    DELTA_STATS_PATH, INPUT_STATS_PATH, PIXEL_MASK_PATH,
)


class FVMLightningModel(L.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        H, W        = RESOLUTION
        num_patches = (H // PATCH_SIZE) * (W // PATCH_SIZE)
        self.model = FluidVisionModel(
            num_obs      = WINDOW_SIZE,
            num_patches  = num_patches,
            patch_size   = PATCH_SIZE,
            emb_dim      = EMB_DIM,
            num_channels = N_CHANNELS,
            num_layers   = NUM_LAYERS,
        )
        self.register_buffer('delta_mean', torch.zeros(N_CHANNELS, 1, 1))
        self.register_buffer('delta_std',  torch.ones( N_CHANNELS, 1, 1))
        self.register_buffer('input_mean', torch.zeros(N_CHANNELS, 1, 1))
        self.register_buffer('input_std',  torch.ones( N_CHANNELS, 1, 1))
        self.register_buffer('pixel_mask', torch.ones(1, 1, H, W, dtype=torch.bool))

    def _normalise_window(self, window: torch.Tensor) -> torch.Tensor:
        # window: (B, T, C, H, W) — broadcast mean/std over B and T
        nm = self.input_mean.unsqueeze(0).unsqueeze(0)  # (1, 1, C, 1, 1)
        ns = self.input_std.unsqueeze(0).unsqueeze(0)
        return ((window - nm) / ns).nan_to_num(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalised delta prediction, zeroed at non-fluid pixels."""
        pred = self.model(self._normalise_window(x))
        return pred * self.pixel_mask

    def denormalise(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalised delta output back to physical space, re-zeroing non-fluid pixels."""
        delta = pred_norm * self.delta_std + self.delta_mean
        return delta * self.pixel_mask.squeeze(0)  # (C, H, W) or (B, C, H, W)

    def on_fit_start(self) -> None:
        def _load(path, mean_buf, std_buf, label):
            if path.exists():
                with open(path) as f:
                    s = json.load(f)
                mean_buf.copy_(torch.tensor(s['mean'], device=self.device).view(N_CHANNELS, 1, 1))
                std_buf.copy_( torch.tensor(s['std'],  device=self.device).view(N_CHANNELS, 1, 1))
                c_print(f'Loaded {label}', color='green')
            else:
                c_print(f'Warning: {path.name} not found', color='yellow')

        _load(DELTA_STATS_PATH, self.delta_mean, self.delta_std, 'delta normalisation stats')
        _load(INPUT_STATS_PATH, self.input_mean, self.input_std, 'input normalisation stats')

        if PIXEL_MASK_PATH.exists():
            self.pixel_mask.copy_(torch.load(PIXEL_MASK_PATH, map_location=self.device))
            c_print('Loaded pixel mask', color='green')
        else:
            c_print('Warning: pixel_mask.pt not found — all pixels treated as fluid', color='yellow')

    def on_train_epoch_start(self) -> None:
        self._epoch_errs: list[torch.Tensor] = []

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        window, target = batch                                     # (B, T, C, H, W), (B, C, H, W)
        pred           = self(window)                              # (B, C, H, W) norm delta
        target_norm    = (target - self.delta_mean) / self.delta_std

        # pixel_mask is (1,1,H,W) — broadcasts over (B,C,H,W)
        valid = self.pixel_mask.expand_as(target_norm)
        err   = (pred - target_norm)[valid]
        loss  = err.pow(2).mean() + err.abs().mean()

        pred_denorm = self.denormalise(pred)
        rel_err = ((target - pred_denorm).abs()[valid] /
                   target.abs()[valid].clamp(min=1e-6)).mean()

        self._epoch_errs.append(err.detach().cpu())

        self.log('train_loss', loss,    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('rel_err',    rel_err, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.global_rank == 0 and self._epoch_errs:
            all_errs = torch.cat(self._epoch_errs)
            print_histogram(all_errs, title=f'Epoch {self.current_epoch} error distribution')
        self._epoch_errs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
