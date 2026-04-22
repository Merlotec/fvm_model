import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from cprint import c_print

# fvm_solver must be on sys.path to unpickle FVMMesh from shared_mesh.pkl
_SOLVER_DIR = Path(__file__).resolve().parents[2] / 'fvm_solver'
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from model import FluidVisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fvm_gen'))
from renderer import MeshRenderer

DATASET_DIR = Path(__file__).resolve().parents[2] / 'data' / 'fvm_gen_datasets'
RESOLUTION  = (224, 224)
PATCH_SIZE  = 16
EMB_DIM     = 768
N_CHANNELS  = 4
WINDOW_SIZE = 10


def build_renderer(dataset_dir: Path, resolution: tuple[int, int], device: str) -> MeshRenderer:
    """Load or build a MeshRenderer from the shared mesh, caching to disk."""
    H, W = resolution
    cache_path = dataset_dir / f'renderer_cache_{H}x{W}.pt'

    if cache_path.exists():
        c_print(f'Loading renderer cache from {cache_path}', color='green')
        return MeshRenderer.from_cache(str(cache_path), device=device)

    mesh_pkl = dataset_dir / 'shared_mesh.pkl'
    if not mesh_pkl.exists():
        raise FileNotFoundError(f'shared_mesh.pkl not found in {dataset_dir}')

    with open(mesh_pkl, 'rb') as f:
        mesh_dict = pickle.load(f)
    fvm_mesh = mesh_dict['mesh']

    vertices  = fvm_mesh.vertices.cpu().numpy()
    triangles = fvm_mesh.triangles.cpu().numpy()

    c_print('Building renderer (trifinder precomputation)...', color='yellow')
    renderer = MeshRenderer(vertices, triangles, resolution=resolution, device=device)
    renderer.save_cache(str(cache_path))
    c_print(f'Renderer cache saved to {cache_path}', color='green')
    return renderer


class RenderedFVMDataset(Dataset):
    """
    Rolling-window samples from one simulation run, rendered to pixel grids.

    Each sample is (window, target):
        window : (WINDOW_SIZE * N_channels, H, W)
        target : (N_channels, H, W)
    """

    def __init__(self, sim_dir: Path, renderer: MeshRenderer, window_size: int):
        files = sorted(
            [f for f in os.listdir(sim_dir) if f.startswith('t_') and f.endswith('.npz')],
            key=lambda f: float(f[2:-4]),
        )
        self.paths       = [sim_dir / f for f in files]
        self.renderer    = renderer
        self.window_size = window_size

    def __len__(self) -> int:
        return max(0, len(self.paths) - self.window_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.cat(
            [self._render(self.paths[idx + i]) for i in range(self.window_size)],
            dim=0,
        )
        target = self._render(self.paths[idx + self.window_size])
        return window, target

    def _render(self, path: Path) -> torch.Tensor:
        d      = np.load(path)
        values = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        return self.renderer.render_cell_smooth(values)


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

class FVMDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir:    Path,
        window_size: int  = WINDOW_SIZE,
        batch_size:  int  = 4,
        num_workers: int  = 4,
    ):
        super().__init__()
        self.data_dir    = Path(data_dir)
        self.window_size = window_size
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self._renderer: MeshRenderer | None = None

    def setup(self, stage: str | None = None):
        # Renderer is built on CPU here; Lightning moves the model to the right
        # device automatically. The renderer tensors stay on CPU and are moved
        # to device inside each worker via pin_memory + non_blocking transfer.
        self._renderer = build_renderer(self.data_dir, RESOLUTION, device='cpu')

        subdirs = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        datasets = [
            RenderedFVMDataset(d, self._renderer, self.window_size)
            for d in subdirs
        ]
        datasets = [ds for ds in datasets if len(ds) > 0]
        if not datasets:
            raise RuntimeError(f'No usable simulation directories found in {self.data_dir}')
        self._dataset = ConcatDataset(datasets)
        c_print(f'Dataset: {len(self._dataset)} samples across {len(datasets)} runs', color='bright_green')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True,
            persistent_workers = self.num_workers > 0,
        )


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

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
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        window, target = batch
        pred = self(window)
        loss = self.criterion(pred, target)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train FluidVisionModel with PyTorch Lightning')
    parser.add_argument('--data-dir',    type=Path, default=DATASET_DIR,
                        help=f'Dataset root directory (default: {DATASET_DIR})')
    parser.add_argument('--epochs',      type=int,  default=20)
    parser.add_argument('--batch-size',  type=int,  default=4)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int,  default=4)
    parser.add_argument('--devices',     type=int,  default=-1,
                        help='Number of GPUs per node (-1 = all available)')
    parser.add_argument('--num-nodes',   type=int,  default=1)
    parser.add_argument('--precision',   type=str,  default='32',
                        help='Training precision: 32, 16-mixed, bf16-mixed')
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

    datamodule = FVMDataModule(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    lightning_model = FVMLightningModel(lr=args.lr)

    trainer = L.Trainer(
        max_epochs    = args.epochs,
        devices       = args.devices,
        num_nodes     = args.num_nodes,
        strategy      = 'ddp' if (args.devices != 1 or args.num_nodes > 1) else 'auto',
        precision     = args.precision,
        callbacks     = [checkpoint_cb],
        log_every_n_steps = 10,
    )

    torch.set_float32_matmul_precision('high')
    trainer.fit(lightning_model, datamodule=datamodule)
    c_print(f'\nBest checkpoint: {checkpoint_cb.best_model_path}', color='bright_magenta')


if __name__ == '__main__':
    main()
