"""
Stage-2 training: freeze the backbone, train only the DiffusionDecoder.

Usage:
    python train_diffusion.py --backbone-ckpt checkpoints/last.ckpt

The backbone (patch_embed + transformer) is loaded from an existing Lightning
checkpoint and kept frozen throughout. Only DiffusionDecoder parameters are updated.

Resume a partially-trained diffusion run:
    python train_diffusion.py --backbone-ckpt checkpoints/last.ckpt \
                              --resume checkpoints_diffusion/last.ckpt
"""

import json
import sys
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from cprint import c_print

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from model import FluidVisionModel
from diffusion_decoder import DiffusionDecoder
from data import FVMDataModule
from helper import (
    _HP, DATASET_DIR,
    RESOLUTION, PATCH_SIZE, EMB_DIM, N_CHANNELS, WINDOW_SIZE, NUM_LAYERS,
    DELTA_STATS_PATH, INPUT_STATS_PATH,
)


class DiffusionStageLightning(L.LightningModule):
    def __init__(
        self,
        backbone_ckpt: str,
        lr:            float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        H, W        = RESOLUTION
        num_patches = (H // PATCH_SIZE) * (W // PATCH_SIZE)

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = FluidVisionModel(
            num_obs     = WINDOW_SIZE,
            num_patches = num_patches,
            patch_size  = PATCH_SIZE,
            emb_dim     = EMB_DIM,
            num_channels= N_CHANNELS,
            num_layers  = NUM_LAYERS,
        )
        self._load_backbone(Path(backbone_ckpt))
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # ── Diffusion decoder ────────────────────────────────────────────────
        self.decoder = DiffusionDecoder(
            emb_dim      = EMB_DIM,
            out_channels = N_CHANNELS,
            patch_size   = PATCH_SIZE,
        )

        # ── Normalisation stats (loaded in on_fit_start) ──────────────────────
        self.register_buffer('delta_mean', torch.zeros(N_CHANNELS, 1, 1))
        self.register_buffer('delta_std',  torch.ones( N_CHANNELS, 1, 1))
        self.register_buffer('input_mean', torch.zeros(N_CHANNELS, 1, 1))
        self.register_buffer('input_std',  torch.ones( N_CHANNELS, 1, 1))

    # ── Backbone loading ──────────────────────────────────────────────────────

    def _load_backbone(self, ckpt_path: Path) -> None:
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = {}
        for k, v in raw['state_dict'].items():
            k = k.removeprefix('model.')          # strip FVMLightningModel wrapper
            k = k.replace('vision_transformer.', 'transformer.')  # old checkpoint name
            state[k] = v

        # Filter out shape-mismatched tensors so strict=False can handle the rest
        current = dict(self.backbone.named_parameters())
        current.update(dict(self.backbone.named_buffers()))
        skipped = []
        compatible = {}
        for k, v in state.items():
            if k in current and current[k].shape != v.shape:
                skipped.append(f'{k}: ckpt {tuple(v.shape)} vs model {tuple(current[k].shape)}')
            else:
                compatible[k] = v

        if skipped:
            c_print(f'Skipped {len(skipped)} shape-mismatched tensors:', color='yellow')
            for s in skipped[:5]:
                c_print(f'  {s}', color='yellow')
            if len(skipped) > 5:
                c_print(f'  ... and {len(skipped) - 5} more', color='yellow')

        missing, _ = self.backbone.load_state_dict(compatible, strict=False)
        backbone_missing = [k for k in missing if not k.startswith('decoder.')]
        loaded = len(compatible) - len(skipped)
        if backbone_missing and not skipped:
            c_print(f'WARNING: backbone keys missing: {backbone_missing}', color='yellow')
        elif skipped:
            c_print(
                f'Checkpoint architecture differs — {len(skipped)} tensors skipped '
                f'(emb_dim or patch_size changed). Run backbone training first.',
                color='yellow'
            )
        else:
            c_print(f'Backbone loaded from {ckpt_path.name}', color='green')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalise_input(self, window: torch.Tensor) -> torch.Tensor:
        nm = self.input_mean.unsqueeze(0).unsqueeze(0)
        ns = self.input_std.unsqueeze(0).unsqueeze(0)
        return ((window - nm) / ns).nan_to_num(0.0)

    @torch.no_grad()
    def _patch_tokens(self, window_norm: torch.Tensor) -> torch.Tensor:
        """Extract last-frame patch tokens from the frozen backbone."""
        x = self.backbone.patch_embed(window_norm)          # (B, T*P, D)
        x = self.backbone.transformer(x)                    # (B, T*P, D)
        return x[:, -self.backbone.num_patches:, :]         # (B, P, D)

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def on_fit_start(self) -> None:
        def _load(path, mean_buf, std_buf, label):
            if path.exists():
                with open(path) as f:
                    s = json.load(f)
                mean_buf.copy_(torch.tensor(s['mean'], device=self.device).view(N_CHANNELS, 1, 1))
                std_buf.copy_( torch.tensor(s['std'],  device=self.device).view(N_CHANNELS, 1, 1))
                c_print(f'Loaded {label}', color='green')
            else:
                c_print(f'Warning: {path.name} not found — using defaults', color='yellow')

        _load(DELTA_STATS_PATH, self.delta_mean, self.delta_std, 'delta stats')
        _load(INPUT_STATS_PATH, self.input_mean, self.input_std, 'input stats')
        self.backbone.eval()

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        window, target = batch
        window_norm = self._normalise_input(window)
        target_norm = (target - self.delta_mean) / self.delta_std

        z    = self._patch_tokens(window_norm)
        loss = self.decoder.compute_loss(z, target_norm)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        window, target = batch
        window_norm = self._normalise_input(window)
        target_norm = (target - self.delta_mean) / self.delta_std
        z   = self._patch_tokens(window_norm)
        loss = self.decoder.compute_loss(z, target_norm)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), lr=self.hparams.lr)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone-ckpt', type=Path, default=Path('checkpoints/last.ckpt'),
                        help='Path to existing backbone Lightning checkpoint')
    parser.add_argument('--resume',        type=Path, default=None,
                        help='Diffusion checkpoint to resume from')
    parser.add_argument('--epochs',        type=int,   default=_HP['epochs'])
    parser.add_argument('--batch-size',    type=int,   default=_HP['batch_size'])
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--num-workers',   type=int,   default=_HP['num_workers'])
    parser.add_argument('--devices',       type=int,   default=_HP['devices'])
    parser.add_argument('--num-nodes',     type=int,   default=_HP['num_nodes'])
    parser.add_argument('--precision',     type=str,   default=_HP['precision'])
    parser.add_argument('--data-dir',      type=Path,  default=DATASET_DIR)
    args = parser.parse_args()

    ckpt_dir = Path(__file__).parent / 'checkpoints_diffusion'
    ckpt_dir.mkdir(exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath   = ckpt_dir,
        filename  = 'diffusion-{epoch:03d}-{val_loss:.5f}',
        save_last = True,
        monitor   = 'val_loss',
        mode      = 'min',
    )

    model = DiffusionStageLightning(backbone_ckpt=str(args.backbone_ckpt), lr=args.lr)

    trainer = L.Trainer(
        max_epochs        = args.epochs,
        devices           = args.devices,
        num_nodes         = args.num_nodes,
        strategy          = 'ddp' if (torch.cuda.is_available() and
                                      (args.devices != 1 or args.num_nodes > 1)) else 'auto',
        precision         = args.precision,
        callbacks         = [checkpoint_cb],
        log_every_n_steps = 10,
    )

    torch.set_float32_matmul_precision('high')
    trainer.fit(model, datamodule=FVMDataModule(args.data_dir, args.batch_size, args.num_workers),
                ckpt_path=args.resume)
    c_print(f'Best checkpoint: {checkpoint_cb.best_model_path}', color='bright_magenta')


if __name__ == '__main__':
    main()
