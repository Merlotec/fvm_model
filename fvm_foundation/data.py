import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning as L
from cprint import c_print

from helper import (
    build_renderer, build_pixel_mask, compute_delta_stats, compute_input_stats,
    RESOLUTION, WINDOW_SIZE, FIRST_FRAME,
    DELTA_STATS_PATH, INPUT_STATS_PATH, PIXEL_MASK_PATH,
)


class RenderedFVMDataset(Dataset):
    """
    Rolling-window samples from one simulation run, rendered to pixel grids.

    Each sample is (window, target):
        window : (WINDOW_SIZE * N_channels, H, W)
        target : (N_channels, H, W)  — delta from last window frame to next frame
    """

    def __init__(self, sim_dir: Path, renderer, window_size: int,
                 first_frame: int = FIRST_FRAME):
        files = sorted(
            [f for f in os.listdir(sim_dir) if f.startswith('t_') and f.endswith('.npz')],
            key=lambda f: float(f[2:-4]),
        )
        # Start from first_frame so the first window covers [first_frame, first_frame+window_size)
        # and the first prediction target is frame first_frame+window_size
        self.paths       = [sim_dir / f for f in files][first_frame:]
        self.renderer    = renderer
        self.window_size = window_size

    def __len__(self) -> int:
        return max(0, len(self.paths) - self.window_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames     = [self._render(self.paths[idx + i]) for i in range(self.window_size)]
        window     = torch.cat(frames, dim=0)
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
        data_dir:    Path,
        window_size: int = WINDOW_SIZE,
        batch_size:  int = 4,
        num_workers: int = 4,
        first_frame: int = FIRST_FRAME,
    ):
        super().__init__()
        self.data_dir    = Path(data_dir)
        self.window_size = window_size
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.first_frame = first_frame
        self._renderer   = None

    def setup(self, stage: str | None = None):
        c_print('Building renderer...', color='yellow')
        self._renderer = build_renderer(self.data_dir, RESOLUTION, device='cpu')

        c_print('Scanning simulation directories...', color='yellow')
        subdirs  = sorted([p for p in self.data_dir.iterdir() if p.is_dir()])
        datasets = [
            RenderedFVMDataset(d, self._renderer, self.window_size,
                               first_frame=self.first_frame)
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
        mean, std = compute_delta_stats(subdirs, self._renderer)
        with open(DELTA_STATS_PATH, 'w') as f:
            json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)
        c_print(f'Delta stats saved — mean={[f"{v:.4f}" for v in mean.tolist()]}  std={[f"{v:.4f}" for v in std.tolist()]}', color='green')

        c_print('Computing input statistics (sampling frames)...', color='yellow')
        inp_mean, inp_std = compute_input_stats(subdirs, self._renderer)
        with open(INPUT_STATS_PATH, 'w') as f:
            json.dump({'mean': inp_mean.tolist(), 'std': inp_std.tolist()}, f)
        c_print(f'Input stats saved — mean={[f"{v:.4f}" for v in inp_mean.tolist()]}  std={[f"{v:.4f}" for v in inp_std.tolist()]}', color='green')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size         = self.batch_size,
            shuffle            = True,
            num_workers        = self.num_workers,
            pin_memory         = True,
            persistent_workers = self.num_workers > 0,
        )
