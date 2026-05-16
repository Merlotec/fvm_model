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

_FVM_MODEL = Path(__file__).resolve().parents[1]   # fvm_model/
_FOUND     = _FVM_MODEL / 'fvm_foundation'
_THIS_DIR  = Path(__file__).resolve().parent

# Add fvm_model/ so fvm_latent is importable as a package (not a bare module),
# and fvm_foundation/ for the data loader. Inserting fvm_model/ first ensures
# `from fvm_latent.x import` always resolves to our files regardless of what
# else (e.g. fvm_foundation) is on PYTHONPATH.
for _p in (_FOUND, _FVM_MODEL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import FVMDataModule                              # type: ignore[import]  noqa: E402
from fvm_latent.model import MultiLevelFluidModel           # type: ignore[import]  noqa: E402
from fvm_latent.lightning_model import (                    # type: ignore[import]  noqa: E402
    Phase1LightningModel,
    CurriculumCallback,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HP_PATH = _THIS_DIR / 'hyperparams.json'
with open(_HP_PATH) as _f:
    _HP = json.load(_f)

DATASET_DIR = _FVM_MODEL / 'data' / 'fvm_gen_datasets'


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
    parser.add_argument('--aux-weight',      type=float, default=_HP.get('aux_weight', 0.3),
                        help='Weight for auxiliary lower-level reconstruction losses')
    parser.add_argument('--steps-per-stage', type=int,   default=_HP.get('steps_per_stage', 1000),
                        help='Optimiser steps between curriculum level expansions')
    parser.add_argument('--first-frame',     type=int,   default=_HP.get('first_frame', 20),
                        help='Index of first frame to include in training (skips transient startup)')
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
        lr         = args.lr,
        noise_std  = args.noise_std,
        aux_weight = args.aux_weight,
        img_size   = img_size,
        window_size= _HP.get('window_size', 1),
    )
    callbacks: list[Callback] = [
        CurriculumCallback(steps_per_stage=args.steps_per_stage),
        ModelCheckpoint(
            dirpath              = ckpt_dir,
            filename             = 'model-{epoch:03d}-{step:06d}',
            save_last            = True,
            every_n_train_steps  = 500,
            save_top_k           = -1,
        ),
    ]
    c_print(f'Checkpoints → {ckpt_dir}', color='cyan')
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
        first_frame = args.first_frame,
        window_size = _HP.get('window_size', 5),
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
