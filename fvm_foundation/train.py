import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from cprint import c_print

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from model import FluidVisionModel

# Path to generated simulation datasets
DATASET_DIR = Path(__file__).resolve().parents[2] / 'fvm_solver' / 'artefacts' / 'fvm_gen_datasets'


class FVMTrajectoryDataset(Dataset):
    """Consecutive timestep pairs from a single FVM simulation run.

    Each sample is (cells_t, bc_t, cells_t1):
        cells_t   : cell primitives at time t,   shape (N_cells, 4), fp32
        bc_t      : BC-edge primitives at time t, shape (N_bc,    4), fp32
        cells_t1  : cell primitives at time t+1,  shape (N_cells, 4), fp32  -- target

    Values are stored normalised (zero-mean, unit-std per component) so they
    are used directly; no further normalisation is needed.
    """

    def __init__(self, sim_dir: str | Path):
        sim_dir = Path(sim_dir)
        files = sorted(
            [f for f in os.listdir(sim_dir) if f.startswith('t_') and f.endswith('.npz')],
            key=lambda f: float(f[2:-4]),
        )
        self.paths = [sim_dir / f for f in files]

    def __len__(self):
        return max(0, len(self.paths) - 1)

    def __getitem__(self, idx):
        def _load(path):
            d = np.load(path)
            cells = d['cell_primatives'].astype(np.float32)   # (N_cells, 4)
            bc    = d['bc_primatives'].astype(np.float32)     # (N_bc, 4)
            return cells, bc

        cells_t,  bc_t = _load(self.paths[idx])
        cells_t1, _    = _load(self.paths[idx + 1])

        return (
            torch.from_numpy(cells_t),    # input state
            torch.from_numpy(bc_t),       # boundary conditions at t
            torch.from_numpy(cells_t1),   # target state
        )


def train_on_dir(
    sim_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """One training pass over all consecutive-step pairs in sim_dir.

    Returns the mean loss over the pass, or nan if the directory had too few steps.
    """
    dataset = FVMTrajectoryDataset(sim_dir)
    if len(dataset) == 0:
        c_print(f'  Skipping {Path(sim_dir).name} — fewer than 2 timesteps', color='yellow')
        return float('nan')

    loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=(device != 'cpu'))
    model.train()

    total_loss = 0.0
    for cells_t, bc_t, target in loader:
        cells_t = cells_t.to(device)   # (1, N_cells, 4)
        bc_t    = bc_t.to(device)      # (1, N_bc,    4)
        target  = target.to(device)    # (1, N_cells, 4)

        optimizer.zero_grad()
        pred = model(cells_t, bc_t)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main(data_dir: str | Path = DATASET_DIR):
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = 20
    lr       = 1e-4
    save_path = Path(__file__).parent / 'checkpoints' / 'model.pt'
    save_path.parent.mkdir(exist_ok=True)

    # Collect simulation subdirectories (one per mu_b value).
    subdirs = sorted([
        p for p in Path(data_dir).iterdir()
        if p.is_dir()
    ])
    if not subdirs:
        raise RuntimeError(f'No simulation directories found in {data_dir}')
    c_print(f'Found {len(subdirs)} simulation directories in {data_dir}', color='bright_green')

    # TODO: update FluidVisionModel to accept (cells, bc) inputs of shapes
    #       (batch, N_cells, 4) and (batch, N_bc, 4) and output (batch, N_cells, 4).
    model     = FluidVisionModel(...).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        c_print(f'\nEpoch {epoch + 1}/{n_epochs}', color='bright_cyan')
        losses = []

        for subdir in subdirs:
            loss = train_on_dir(subdir, model, optimizer, criterion, device)
            c_print(f'  {subdir.name}: loss={loss:.5f}', color='bright_green')
            if not np.isnan(loss):
                losses.append(loss)

        mean_loss = np.mean(losses) if losses else float('nan')
        c_print(f'  epoch mean loss: {mean_loss:.5f}', color='bright_yellow')

    torch.save(model.state_dict(), save_path)
    c_print(f'\nSaved model to {save_path}', color='bright_magenta')


if __name__ == '__main__':
    main()
