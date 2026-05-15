"""
Training script for the Multi-Level Fluid Model.

Usage:
  python -m fvm_latent.train --epochs 30

Or run directly:
  cd fvm_model && python fvm_latent/train.py
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from cprint import c_print

_ROOT     = Path(__file__).resolve().parents[2]
_FOUND    = _ROOT / 'fvm_model' / 'fvm_foundation'
_THIS_DIR = Path(__file__).resolve().parent

for _p in (_FOUND, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import FVMDataModule                       # type: ignore[import]  noqa: E402
from model import MultiLevelFluidModel               # type: ignore[import]  noqa: E402
from lightning_model import (                        # type: ignore[import]  noqa: E402
    Phase1LightningModel,
    CurriculumCallback,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HP_PATH = _THIS_DIR / 'hyperparams.json'
with open(_HP_PATH) as _f:
    _HP = json.load(_f)

DATASET_DIR = _ROOT / 'data' / 'fvm_gen_datasets'


def build_model(hp: dict) -> MultiLevelFluidModel:
    return MultiLevelFluidModel(
        img_size             = hp['img_size'],
        patch_size           = hp['patch_size'],
        in_channels          = hp['in_channels'],
        window_size          = hp.get('window_size', 1),
        n_levels             = hp['n_levels'],
        d_model              = hp['d_model'],
        n_heads              = hp['n_heads'],
        n_transformer_layers = hp['n_transformer_layers'],
        gate_threshold       = hp.get('gate_threshold', 0.5),
        gate_budget          = hp.get('gate_budget', 0.4),
        dropout              = hp.get('dropout', 0.0),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Train MultiLevelFluidModel')
    parser.add_argument('--data-dir',        type=Path,  default=DATASET_DIR)
    parser.add_argument('--epochs',          type=int,   default=_HP['epochs'])
    parser.add_argument('--batch-size',      type=int,   default=_HP['batch_size'])
    parser.add_argument('--lr',              type=float, default=_HP['lr'])
    parser.add_argument('--num-workers',     type=int,   default=_HP['num_workers'])
    parser.add_argument('--devices',         type=int,   default=_HP['devices'])
    parser.add_argument('--num-nodes',       type=int,   default=_HP['num_nodes'])
    parser.add_argument('--precision',       type=str,   default=_HP['precision'])
    parser.add_argument('--noise-std',       type=float, default=_HP.get('noise_std', 0.02))
    parser.add_argument('--sparsity-weight', type=float, default=_HP.get('sparsity_weight', 0.1))
    parser.add_argument('--aux-weight',      type=float, default=_HP.get('aux_weight', 0.3),
                        help='Weight for auxiliary lower-level reconstruction losses')
    parser.add_argument('--steps-per-stage', type=int,   default=_HP.get('steps_per_stage', 1000),
                        help='Optimiser steps between curriculum level expansions')
    parser.add_argument('--resume',          type=Path,  default=None,
                        help='Lightning .ckpt to resume training from')
    args = parser.parse_args()

    ckpt_dir = _THIS_DIR / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model   = build_model(_HP)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    P       = model.n_patches
    c_print(
        f'Model: {n_param:.1f}M params | {model.n_levels} levels × {P} patches = '
        f'{model.n_levels * P} tokens max',
        color='cyan',
    )

    img_size: int = _HP['img_size']

    # ------------------------------------------------------------------
    # Lightning model + callbacks
    # ------------------------------------------------------------------
    lit = Phase1LightningModel(
        model,
        lr              = args.lr,
        noise_std       = args.noise_std,
        sparsity_weight = args.sparsity_weight,
        aux_weight      = args.aux_weight,
        img_size        = img_size,
        window_size     = _HP.get('window_size', 1),
    )
    callbacks: list[Callback] = [
        CurriculumCallback(steps_per_stage=args.steps_per_stage),
        ModelCheckpoint(
            dirpath   = ckpt_dir,
            filename  = 'model-{epoch:03d}-{val/loss:.5f}',
            save_last = True,
            monitor   = 'val/loss',
            mode      = 'min',
        ),
    ]
    c_print(
        f'Curriculum: 1 → {model.n_levels} levels, +1 every {args.steps_per_stage} steps',
        color='bright_green',
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    datamodule = FVMDataModule(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    use_ddp = torch.cuda.is_available() and (args.devices != 1 or args.num_nodes > 1)
    trainer = L.Trainer(
        max_epochs        = args.epochs,
        devices           = args.devices,
        num_nodes         = args.num_nodes,
        strategy          = 'ddp' if use_ddp else 'auto',
        precision         = args.precision,
        callbacks         = callbacks,
        log_every_n_steps = 10,
    )

    torch.set_float32_matmul_precision('high')
    trainer.fit(lit, datamodule=datamodule, ckpt_path=args.resume)

    best = callbacks[-1].best_model_path  # type: ignore[union-attr]
    c_print(f'\nBest checkpoint: {best}', color='bright_magenta')


if __name__ == '__main__':
    main()
