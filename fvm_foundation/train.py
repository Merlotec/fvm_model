import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
N_CHANNELS  = 4   # Vx, Vy, rho, T
WINDOW_SIZE = 10   # number of input frames per sample

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

    vertices  = fvm_mesh.vertices.cpu().numpy()   # (N, 2)
    triangles = fvm_mesh.triangles.cpu().numpy()  # (M, 3)

    c_print('Building renderer (trifinder precomputation)...', color='yellow')
    renderer = MeshRenderer(vertices, triangles, resolution=resolution, device=device)
    renderer.save_cache(str(cache_path))
    c_print(f'Renderer cache saved to {cache_path}', color='green')
    return renderer




class RenderedFVMDataset(Dataset):
    """
    Rolling-window samples from one simulation run, rendered to pixel grids.

    Each sample is (window, target):
        window : (NUM_OBS * N_channels, H, W) — window_size frames stacked as channels
        target : (N_channels, H, W)           — the frame immediately after the window

    Values are denormalised to physical units before rendering so the model
    sees consistent scales across timesteps.
    """

    def __init__(self, sim_dir: Path, renderer: MeshRenderer, window_size: int):
        files = sorted(
            [f for f in os.listdir(sim_dir) if f.startswith('t_') and f.endswith('.npz')],
            key=lambda f: float(f[2:-4]),
        )
        self.paths    = [sim_dir / f for f in files]
        self.renderer = renderer
        self.window_size = window_size

    def __len__(self) -> int:
        return max(0, len(self.paths) - self.window_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.cat(
            [self._render(self.paths[idx + i]) for i in range(self.window_size)],
            dim=0,
        )  # (WINDOW_SIZE * N_channels, H, W)
        target = self._render(self.paths[idx + self.window_size])  # (N_channels, H, W)
        return window, target

    def _render(self, path: Path) -> torch.Tensor:
        d      = np.load(path)
        values = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
        return self.renderer.render_cell(values)   # (N_channels, H, W)


def train_on_dir(
    sim_dir: Path,
    renderer: MeshRenderer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    window_size: int,
) -> float:
    dataset = RenderedFVMDataset(sim_dir, renderer, window_size)
    if len(dataset) == 0:
        c_print(f'  Skipping {sim_dir.name} — fewer than 2 timesteps', color='yellow')
        return float('nan')

    loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=(device == 'cuda'))
    model.train()

    total_loss = 0.0
    for grid_t, grid_t1 in loader:
        grid_t  = grid_t.to(device)   # (B, N_channels, H, W)
        grid_t1 = grid_t1.to(device)  # (B, N_channels, H, W)

        optimizer.zero_grad()
        pred = model(grid_t)
        loss = criterion(pred, grid_t1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main(data_dir: Path = DATASET_DIR):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    n_epochs  = 20
    lr        = 1e-4
    save_path = Path(__file__).parent / 'checkpoints' / 'model.pt'
    save_path.parent.mkdir(exist_ok=True)

    data_dir = Path(data_dir)
    subdirs  = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not subdirs:
        raise RuntimeError(f'No simulation directories found in {data_dir}')
    c_print(f'Found {len(subdirs)} simulation directories in {data_dir}', color='bright_green')

    renderer = build_renderer(data_dir, RESOLUTION, device)

    H, W        = RESOLUTION
    num_patches = (H // PATCH_SIZE) * (W // PATCH_SIZE)
    model = FluidVisionModel(
        num_obs      = WINDOW_SIZE,
        num_patches  = num_patches,
        patch_size   = PATCH_SIZE,
        emb_dim      = EMB_DIM,
        num_channels = N_CHANNELS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        c_print(f'\nEpoch {epoch + 1}/{n_epochs}', color='bright_cyan')
        losses = []

        for subdir in subdirs:
            loss = train_on_dir(subdir, renderer, model, optimizer, criterion, device, WINDOW_SIZE)
            c_print(f'  {subdir.name}: loss={loss:.5f}', color='bright_green')
            if not np.isnan(loss):
                losses.append(loss)

        mean_loss = float(np.mean(losses)) if losses else float('nan')
        c_print(f'  epoch mean loss: {mean_loss:.5f}', color='bright_yellow')

    torch.save(model.state_dict(), save_path)
    c_print(f'\nSaved model to {save_path}', color='bright_magenta')


if __name__ == '__main__':
    main()
