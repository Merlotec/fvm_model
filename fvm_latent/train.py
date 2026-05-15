"""
Training script for the Hierarchical Latent Fluid Model.

Usage:
  # Phase 1 — curriculum training from scratch
  python -m fvm_latent.train --phase 1 --epochs 30

  # Phase 2 — value learning, load Phase-1 checkpoint
  python -m fvm_latent.train --phase 2 --resume checkpoints/phase1_last.ckpt --epochs 5

Or run directly (script adds needed paths automatically):
  cd fvm_model && python fvm_latent/train.py --phase 1
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from cprint import c_print

# Allow imports from the sibling fvm_foundation package for data loading
_ROOT     = Path(__file__).resolve().parents[2]
_FOUND    = _ROOT / 'fvm_model' / 'fvm_foundation'
_THIS_DIR = Path(__file__).resolve().parent

for _p in (_FOUND, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import FVMDataModule                       # type: ignore[import]  noqa: E402
from model import HierarchicalLatentModel            # type: ignore[import]  noqa: E402
from lightning_model import (                        # type: ignore[import]  noqa: E402
    Phase1LightningModel,
    Phase2LightningModel,
    CurriculumCallback,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HP_PATH = _THIS_DIR / 'hyperparams.json'
with open(_HP_PATH) as _f:
    _HP = json.load(_f)

DATASET_DIR = _ROOT / 'data' / 'fvm_gen_datasets'


def build_model(hp: dict) -> HierarchicalLatentModel:
    return HierarchicalLatentModel(
        img_size             = hp['img_size'],
        patch_size           = hp['patch_size'],
        in_channels          = hp['in_channels'],
        window_size          = hp.get('window_size', 1),
        n_base               = hp['n_base'],
        step_size            = hp['step_size'],
        n_steps              = hp['n_steps'],
        d_model              = hp['d_model'],
        n_heads              = hp['n_heads'],
        n_encoder_layers     = hp['n_encoder_layers'],
        n_transformer_layers = hp['n_transformer_layers'],
        dropout              = hp.get('dropout', 0.0),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Train HierarchicalLatentModel')
    parser.add_argument('--phase',           type=int,   default=1,         choices=[1, 2])
    parser.add_argument('--data-dir',        type=Path,  default=DATASET_DIR)
    parser.add_argument('--epochs',          type=int,   default=_HP['epochs'])
    parser.add_argument('--batch-size',      type=int,   default=_HP['batch_size'])
    parser.add_argument('--lr',              type=float, default=_HP['lr'])
    parser.add_argument('--num-workers',     type=int,   default=_HP['num_workers'])
    parser.add_argument('--devices',         type=int,   default=_HP['devices'])
    parser.add_argument('--num-nodes',       type=int,   default=_HP['num_nodes'])
    parser.add_argument('--precision',       type=str,   default=_HP['precision'])
    parser.add_argument('--noise-std',       type=float, default=_HP.get('noise_std', 0.02))
    parser.add_argument('--steps-per-stage', type=int,   default=_HP.get('steps_per_stage', 1000),
                        help='Phase-1: optimiser steps between curriculum expansions')
    parser.add_argument('--resume',          type=Path,  default=None,
                        help='Lightning .ckpt to resume (Phase 1) or load backbone from (Phase 2)')
    args = parser.parse_args()

    ckpt_dir = _THIS_DIR / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(_HP)
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    c_print(f'Model: {total_p:.1f}M params | tokens: {model.total_tokens} '
            f'(base={model.n_base}, step={model.step_size}, stages={model.n_steps})', color='cyan')

    img_size: int = _HP['img_size']

    # ------------------------------------------------------------------
    # Lightning model + callbacks
    # ------------------------------------------------------------------
    callbacks: list[Callback]

    if args.phase == 1:
        lit = Phase1LightningModel(
            model, lr=args.lr, noise_std=args.noise_std,
            img_size=img_size, window_size=_HP.get('window_size', 1),
        )
        callbacks = [
            CurriculumCallback(steps_per_stage=args.steps_per_stage),
            ModelCheckpoint(
                dirpath   = ckpt_dir,
                filename  = 'phase1-{epoch:03d}-{val/loss:.5f}',
                save_last = True,
                monitor   = 'val/loss',
                mode      = 'min',
            ),
        ]
        c_print(f'Phase 1 — curriculum, starting at {model.n_base} tokens, '
                f'+{model.step_size} every {args.steps_per_stage} steps', color='bright_green')

    else:
        if args.resume is None:
            raise ValueError('--resume is required for Phase 2 (path to a Phase-1 checkpoint)')
        lit_p1   = Phase1LightningModel.load_from_checkpoint(args.resume, model=model)
        lit      = Phase2LightningModel(
            lit_p1.model, lr=args.lr,
            img_size=img_size, window_size=_HP.get('window_size', 1),
        )
        callbacks = [
            ModelCheckpoint(
                dirpath   = ckpt_dir,
                filename  = 'phase2-{epoch:03d}-{val/value_loss:.5f}',
                save_last = True,
                monitor   = 'val/value_loss',
                mode      = 'min',
            ),
        ]
        c_print('Phase 2 — value learning (backbone frozen)', color='bright_yellow')

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

    # For Phase 1, --resume means "continue training from checkpoint"
    ckpt_path = args.resume if args.phase == 1 else None
    trainer.fit(lit, datamodule=datamodule, ckpt_path=ckpt_path)

    best = callbacks[-1].best_model_path  # type: ignore[union-attr]
    c_print(f'\nBest checkpoint: {best}', color='bright_magenta')


if __name__ == '__main__':
    main()
