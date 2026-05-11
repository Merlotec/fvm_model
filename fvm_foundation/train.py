import sys
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from cprint import c_print

from helper import _HP, DATASET_DIR
from data import FVMDataModule
from lightning_model import FVMLightningModel


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train FluidVisionModel with PyTorch Lightning')
    parser.add_argument('--data-dir',    type=Path, default=DATASET_DIR)
    parser.add_argument('--epochs',      type=int,   default=_HP['epochs'])
    parser.add_argument('--batch-size',  type=int,   default=_HP['batch_size'])
    parser.add_argument('--lr',          type=float, default=_HP['lr'])
    parser.add_argument('--num-workers', type=int,   default=_HP['num_workers'])
    parser.add_argument('--devices',     type=int,   default=_HP['devices'],
                        help='Number of GPUs per node (-1 = all available)')
    parser.add_argument('--num-nodes',   type=int,   default=_HP['num_nodes'])
    parser.add_argument('--precision',   type=str,   default=_HP['precision'],
                        help='Training precision: 32, 16-mixed, bf16-mixed')
    parser.add_argument('--resume',      type=Path,  default=None,
                        help='Path to a Lightning checkpoint to resume from')
    args = parser.parse_args()

    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath   = checkpoint_dir,
        filename  = 'model-{epoch:03d}-{train_loss:.5f}',
        save_last = True,
        monitor   = 'train_loss',
        mode      = 'min',
    )

    datamodule      = FVMDataModule(data_dir=args.data_dir, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
    lightning_model = FVMLightningModel(lr=args.lr)

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
    trainer.fit(lightning_model, datamodule=datamodule, ckpt_path=args.resume)
    c_print(f'\nBest checkpoint: {checkpoint_cb.best_model_path}', color='bright_magenta')


if __name__ == '__main__':
    main()
