import json
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

with open(Path(__file__).resolve().parent / 'hyperparams.json') as _f:
    _HP = json.load(_f)

DATASET_DIR = Path(__file__).resolve().parents[2] / 'data' / 'fvm_gen_datasets'
RESOLUTION  = tuple(_HP['resolution'])
PATCH_SIZE  = _HP['patch_size']
EMB_DIM     = _HP['emb_dim']
N_CHANNELS  = _HP['n_channels']
WINDOW_SIZE = _HP['window_size']
NUM_LAYERS  = _HP['num_layers']

DELTA_STATS_PATH = Path(__file__).resolve().parent / 'delta_stats.json'
INPUT_STATS_PATH = Path(__file__).resolve().parent / 'input_stats.json'
PIXEL_MASK_PATH  = Path(__file__).resolve().parent / 'pixel_mask.pt'


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


def _compute_delta_stats(sim_dirs: list[Path], renderer, n_samples: int = 200):
    """Sample consecutive frame pairs to estimate per-channel delta mean and std."""
    all_pairs = []
    for d in sim_dirs:
        files = sorted(
            [f for f in d.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
            key=lambda f: float(f.stem[2:]),
        )
        for i in range(len(files) - 1):
            all_pairs.append((files[i], files[i + 1]))

    n = min(n_samples, len(all_pairs))
    indices = torch.randperm(len(all_pairs))[:n].tolist()

    def _render(p):
        d = np.load(p)
        vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        return renderer.render_cell_smooth(vals)

    deltas = []
    for i in indices:
        path_a, path_b = all_pairs[i]
        deltas.append(_render(path_b) - _render(path_a))

    deltas = torch.stack(deltas)          # (n, C, H, W)
    # nan-safe stats — renderer may fill background pixels with NaN
    mask   = torch.isfinite(deltas)
    safe   = deltas.nan_to_num(0.0)
    count  = mask.sum(dim=(0, 2, 3)).float()
    mean   = (safe * mask).sum(dim=(0, 2, 3)) / count
    var    = ((safe - mean.view(1, -1, 1, 1)) ** 2 * mask).sum(dim=(0, 2, 3)) / count
    std    = var.sqrt().clamp(min=1e-6)
    return mean, std


def _compute_input_stats(sim_dirs: list[Path], renderer, n_samples: int = 200):
    """Sample individual frames to estimate per-channel input mean and std."""
    all_files = []
    for d in sim_dirs:
        all_files.extend(sorted(
            [f for f in d.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')]
        ))

    n       = min(n_samples, len(all_files))
    indices = torch.randperm(len(all_files))[:n].tolist()

    sum1  = torch.zeros(N_CHANNELS)
    sum2  = torch.zeros(N_CHANNELS)
    count = torch.zeros(N_CHANNELS)

    for i in indices:
        d    = np.load(all_files[i])
        vals = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        rendered = renderer.render_cell_smooth(vals)   # (C, H, W)
        for c in range(N_CHANNELS):
            pixels = rendered[c]
            finite = pixels[torch.isfinite(pixels)]
            sum1[c]  += finite.sum()
            sum2[c]  += (finite ** 2).sum()
            count[c] += finite.numel()

    mean = sum1 / count
    std  = ((sum2 / count) - mean ** 2).clamp(min=0).sqrt().clamp(min=1e-6)
    return mean, std


def build_pixel_mask(renderer, resolution: tuple[int, int]) -> torch.Tensor:
    """Boolean (1, 1, H, W) mask — True for pixels inside the fluid mesh, False elsewhere."""
    H, W = resolution
    mask = torch.zeros(H * W, dtype=torch.bool)
    mask[renderer._interior_idx] = True
    return mask.view(1, 1, H, W)


def _print_histogram(values: torch.Tensor, title: str, bins: int = 25, width: int = 50) -> None:
    n = values.numel()
    if n == 0:
        return
    v = values.float().cpu()
    # torch.quantile is limited to 2^24 elements — subsample for statistics only
    max_q = 1 << 24
    v_q = v[torch.randperm(n)[:max_q]] if n > max_q else v
    lo  = torch.quantile(v_q, 0.01).item()
    hi  = torch.quantile(v_q, 0.99).item()
    if lo >= hi:
        hi = lo + 1e-6
    counts = torch.histc(v, bins=bins, min=lo, max=hi)
    edges  = torch.linspace(lo, hi, bins + 1).tolist()
    peak   = counts.max().item()
    print(f'\n{title}  n={n:,}  mean={v.mean():.4f}  std={v.std():.4f}  '
          f'p1={lo:.4f}  p99={hi:.4f}')
    for i in range(bins):
        bar = '█' * int(counts[i].item() / peak * width if peak > 0 else 0)
        print(f'  [{edges[i]:+8.4f}, {edges[i+1]:+8.4f})  {bar}')
    print()


class RenderedFVMDataset(Dataset):
    """
    Rolling-window samples from one simulation run, rendered to pixel grids.

    Each sample is (window, target):
        window : (WINDOW_SIZE * N_channels, H, W)
        target : (N_channels, H, W)
    """

    def __init__(self, sim_dir: Path, renderer: MeshRenderer, window_size: int,
                 skip_initial: int = 20):
        files = sorted(
            [f for f in os.listdir(sim_dir) if f.startswith('t_') and f.endswith('.npz')],
            key=lambda f: float(f[2:-4]),
        )
        # Drop early transient frames — deltas are 10-100× larger than settled state
        self.paths       = [sim_dir / f for f in files][skip_initial:]
        self.renderer    = renderer
        self.window_size = window_size

    def __len__(self) -> int:
        return max(0, len(self.paths) - self.window_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = [self._render(self.paths[idx + i]) for i in range(self.window_size)]
        window     = torch.cat(frames, dim=0)   # (W*C, H, W) — NaN for background
        target     = self._render(self.paths[idx + self.window_size])
        last_frame = frames[-1]
        return window, target - last_frame

    def _render(self, path: Path) -> torch.Tensor:
        d      = np.load(path)
        values = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        return self.renderer.render_cell_smooth(values)



class FVMDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir:     Path,
        window_size:  int = WINDOW_SIZE,
        batch_size:   int = 4,
        num_workers:  int = 4,
        skip_initial: int = 20,
    ):
        super().__init__()
        self.data_dir     = Path(data_dir)
        self.window_size  = window_size
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.skip_initial = skip_initial
        self._renderer: MeshRenderer | None = None

    def setup(self, stage: str | None = None):
        c_print('Building renderer...', color='yellow')
        self._renderer = build_renderer(self.data_dir, RESOLUTION, device='cpu')

        c_print('Scanning simulation directories...', color='yellow')
        subdirs = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        datasets = [
            RenderedFVMDataset(d, self._renderer, self.window_size,
                               skip_initial=self.skip_initial)
            for d in subdirs
        ]
        datasets = [ds for ds in datasets if len(ds) > 0]
        if not datasets:
            raise RuntimeError(f'No usable simulation directories found in {self.data_dir}')
        self._dataset = ConcatDataset(datasets)
        c_print(f'Dataset: {len(self._dataset)} samples across {len(datasets)} runs', color='bright_green')

        pixel_mask = build_pixel_mask(self._renderer, RESOLUTION)
        torch.save(pixel_mask, PIXEL_MASK_PATH)
        c_print(f'Pixel mask saved — {pixel_mask.sum().item()} fluid pixels of {pixel_mask.numel()}', color='green')

        c_print('Computing delta statistics (sampling 200 frame pairs)...', color='yellow')
        mean, std = _compute_delta_stats(subdirs, self._renderer)
        with open(DELTA_STATS_PATH, 'w') as f:
            json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)
        c_print(f'Delta stats saved — mean={[f"{v:.4f}" for v in mean.tolist()]}  std={[f"{v:.4f}" for v in std.tolist()]}', color='green')

        c_print('Computing input statistics (sampling frames)...', color='yellow')
        inp_mean, inp_std = _compute_input_stats(subdirs, self._renderer)
        with open(INPUT_STATS_PATH, 'w') as f:
            json.dump({'mean': inp_mean.tolist(), 'std': inp_std.tolist()}, f)
        c_print(f'Input stats saved — mean={[f"{v:.4f}" for v in inp_mean.tolist()]}  std={[f"{v:.4f}" for v in inp_std.tolist()]}', color='green')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True,
            persistent_workers = self.num_workers > 0,
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
        H, W = RESOLUTION
        self.register_buffer('delta_mean',  torch.zeros(N_CHANNELS, 1, 1))
        self.register_buffer('delta_std',   torch.ones( N_CHANNELS, 1, 1))
        self.register_buffer('input_mean',  torch.zeros(N_CHANNELS, 1, 1))
        self.register_buffer('input_std',   torch.ones( N_CHANNELS, 1, 1))
        self.register_buffer('pixel_mask',  torch.ones(1, 1, H, W, dtype=torch.bool))

    def _normalise_window(self, window: torch.Tensor) -> torch.Tensor:
        """Normalise input per channel then zero background (NaN) pixels."""
        # window: (B, W*C, H, W) — each frame block has N_CHANNELS channels
        nm = self.input_mean.repeat(WINDOW_SIZE, 1, 1).unsqueeze(0)  # (1, W*C, 1, 1)
        ns = self.input_std.repeat( WINDOW_SIZE, 1, 1).unsqueeze(0)
        return ((window - nm) / ns).nan_to_num(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model(self._normalise_window(x))
        return pred * self.pixel_mask  # zero boundary/background pixels

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
        window, target = batch
        pred        = self(window)
        target_norm = (target - self.delta_mean) / self.delta_std
        valid       = self.pixel_mask.expand_as(target_norm)  # fluid-only pixels
        err  = (pred - target_norm)[valid]
        loss = err.pow(2).mean() + err.abs().mean()

        pred_denorm = pred * self.delta_std + self.delta_mean
        rel_err = ((target - pred_denorm).abs()[valid] /
                   target.abs()[valid].clamp(min=1e-6)).mean()

        self._epoch_errs.append(err.detach().cpu())

        self.log('train_loss', loss,    on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('rel_err',    rel_err, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.global_rank == 0 and self._epoch_errs:
            all_errs = torch.cat(self._epoch_errs)
            _print_histogram(all_errs, title=f'Epoch {self.current_epoch} error distribution')
        self._epoch_errs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train FluidVisionModel with PyTorch Lightning')
    parser.add_argument('--data-dir',    type=Path, default=DATASET_DIR,
                        help=f'Dataset root directory (default: {DATASET_DIR})')
    parser.add_argument('--epochs',      type=int,   default=_HP['epochs'])
    parser.add_argument('--batch-size',  type=int,   default=_HP['batch_size'])
    parser.add_argument('--lr',          type=float, default=_HP['lr'])
    parser.add_argument('--num-workers', type=int,   default=_HP['num_workers'])
    parser.add_argument('--devices',     type=int,   default=_HP['devices'],
                        help='Number of GPUs per node (-1 = all available)')
    parser.add_argument('--num-nodes',   type=int,   default=_HP['num_nodes'])
    parser.add_argument('--precision',   type=str,   default=_HP['precision'],
                        help='Training precision: 32, 16-mixed, bf16-mixed')
    parser.add_argument('--resume',      type=Path, default=None,
                        help='Path to a Lightning checkpoint to resume from (e.g. checkpoints/last.ckpt)')
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
        strategy      = 'ddp' if (torch.cuda.is_available() and (args.devices != 1 or args.num_nodes > 1)) else 'auto',
        precision     = args.precision,
        callbacks     = [checkpoint_cb],
        log_every_n_steps = 10,
    )

    torch.set_float32_matmul_precision('high')
    trainer.fit(lightning_model, datamodule=datamodule, ckpt_path=args.resume)
    c_print(f'\nBest checkpoint: {checkpoint_cb.best_model_path}', color='bright_magenta')


if __name__ == '__main__':
    main()
