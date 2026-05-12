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
    RESOLUTION, FIRST_FRAME, SEQ_LEN,
    DELTA_STATS_PATH, INPUT_STATS_PATH, PIXEL_MASK_PATH,
)


class RenderedFVMDataset(Dataset):
    """
    Full-sequence chunks from one simulation run, rendered to pixel grids.

    Each sample is (frames, targets):
        frames  : (seq_len, N_channels, H, W)  — input frames
        targets : (seq_len, N_channels, H, W)  — per-step delta to the next frame
    The model predicts all seq_len deltas in one forward pass via causal attention.
    """

    def __init__(self, sim_dir: Path, renderer, seq_len: int = SEQ_LEN,
                 first_frame: int = FIRST_FRAME):
        files = sorted(
            [f for f in os.listdir(sim_dir) if f.startswith('t_') and f.endswith('.npz')],
            key=lambda f: float(f[2:-4]),
        )
        self.paths    = [sim_dir / f for f in files][first_frame:]
        self.renderer = renderer
        self.seq_len  = seq_len

    def __len__(self) -> int:
        # Need seq_len + 1 consecutive frames: seq_len inputs + 1 lookahead for the last target
        return max(0, len(self.paths) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        all_frames = [self._render(self.paths[idx + i]) for i in range(self.seq_len + 1)]
        frames  = torch.stack(all_frames[:-1])                                     # (T, C, H, W)
        targets = torch.stack([all_frames[i + 1] - all_frames[i]
                               for i in range(self.seq_len)])                      # (T, C, H, W)
        return frames, targets

    def _render(self, path: Path) -> torch.Tensor:
        d      = np.load(path)
        values = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        return self.renderer.render_cell_smooth(values)


class FVMDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir:    Path,
        seq_len:     int = SEQ_LEN,
        batch_size:  int = 4,
        num_workers: int = 4,
        first_frame: int = FIRST_FRAME,
    ):
        super().__init__()
        self.data_dir    = Path(data_dir)
        self.seq_len     = seq_len
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
            RenderedFVMDataset(d, self._renderer, seq_len=self.seq_len,
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
