"""
Run autoregressive inference with a trained FluidVisionModel.

The first WINDOW_SIZE frames from the input sim directory are used as the
seed window and copied to the output directory marked as seed frames.
The model then rolls out autoregressively, predicting one frame at a time
and feeding each prediction back as input.

Usage
-----
    python infer.py path/to/sim_dir/ path/to/checkpoints/model.pt path/to/output_dir/

    # Predict a specific number of steps (default: all remaining frames in sim_dir):
    python infer.py path/to/sim_dir/ path/to/model.pt path/to/out/ --steps 50

Output directory structure mirrors the input:
    output_dir/
        t_0.0000.npz   # seed  (is_seed=True,  grid=(4,H,W))
        t_0.1000.npz   # seed
        ...
        t_0.9000.npz   # seed  (last seed frame)
        t_1.0000.npz   # pred  (is_seed=False, grid=(4,H,W))
        t_1.1000.npz   # pred
        ...

Each .npz contains:
    grid      : (4, H, W) float32 rendered field
    t         : scalar float timestamp
    is_seed   : bool
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from cprint import c_print

# Path setup mirrors train.py
_SOLVER_DIR = Path(__file__).resolve().parents[2] / 'fvm_solver'
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from model import FluidVisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fvm_gen'))
from renderer import MeshRenderer

from train import (
    DATASET_DIR,
    RESOLUTION,
    PATCH_SIZE,
    EMB_DIM,
    N_CHANNELS,
    WINDOW_SIZE,
    build_renderer,
)

FIELD_NAMES = ["Vx", "Vy", "rho", "T"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _find_timestep_files(sim_dir: Path) -> list[Path]:
    files = sorted(
        [f for f in sim_dir.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
        key=lambda f: float(f.stem[2:]),
    )
    return files


def _load_and_render(path: Path, renderer: MeshRenderer) -> torch.Tensor:
    """Load a raw timestep file and return a rendered (N_CHANNELS, H, W) tensor."""
    d      = np.load(path)
    values = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
    return renderer.render_cell_smooth(values)   # (N_CHANNELS, H, W)


def _t_of(path: Path) -> float:
    return float(path.stem[2:])


def _save_frame(out_dir: Path, t: float, grid: np.ndarray, is_seed: bool) -> None:
    fname = f"t_{t:.4g}.npz"
    np.savez_compressed(
        out_dir / fname,
        grid    = grid,
        t       = np.float32(t),
        is_seed = np.bool_(is_seed),
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    sim_dir:    Path,
    checkpoint: Path,
    out_dir:    Path,
    n_steps:    int | None = None,
    data_dir:   Path = DATASET_DIR,
) -> None:
    device = _select_device()
    c_print(f'Device: {device}', color='cyan')

    # ---- renderer ----
    renderer = build_renderer(data_dir, RESOLUTION, device)

    # ---- model ----
    H, W        = RESOLUTION
    num_patches = (H // PATCH_SIZE) * (W // PATCH_SIZE)
    model = FluidVisionModel(
        num_obs      = WINDOW_SIZE,
        num_patches  = num_patches,
        patch_size   = PATCH_SIZE,
        emb_dim      = EMB_DIM,
        num_channels = N_CHANNELS,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    c_print(f'Loaded checkpoint: {checkpoint}', color='green')

    # ---- input files ----
    all_files = _find_timestep_files(sim_dir)
    if len(all_files) < WINDOW_SIZE:
        raise RuntimeError(
            f'{sim_dir} has only {len(all_files)} timesteps; need at least {WINDOW_SIZE} for seed.'
        )

    if n_steps is None:
        n_steps = max(1, len(all_files) - WINDOW_SIZE)

    out_dir.mkdir(parents=True, exist_ok=True)
    c_print(f'Output: {out_dir}', color='cyan')
    c_print(f'Seed frames: {WINDOW_SIZE}  |  Prediction steps: {n_steps}', color='cyan')

    # ---- seed window ----
    # Render the first WINDOW_SIZE frames and save them as seed frames.
    seed_files = all_files[:WINDOW_SIZE]
    window: list[torch.Tensor] = []

    c_print('Rendering seed frames...', color='yellow')
    for path in seed_files:
        grid = _load_and_render(path, renderer)   # (N_CHANNELS, H, W) on device
        window.append(grid)
        _save_frame(out_dir, _t_of(path), grid.cpu().numpy(), is_seed=True)
        c_print(f'  seed  t={_t_of(path):.4g}', color='bright_black')

    # Infer dt from the seed files for extrapolating prediction timestamps.
    if len(seed_files) >= 2:
        dt = _t_of(seed_files[-1]) - _t_of(seed_files[-2])
    else:
        dt = 0.1
    t_next = _t_of(seed_files[-1]) + dt

    # ---- autoregressive rollout ----
    c_print('Running inference...', color='yellow')
    with torch.no_grad():
        for step in range(n_steps):
            # Stack window into (1, WINDOW_SIZE * N_CHANNELS, H, W)
            inp = torch.cat(window, dim=0).unsqueeze(0)   # (1, W*C, H, W)

            pred = model(inp).squeeze(0)                  # (N_CHANNELS, H, W)

            _save_frame(out_dir, t_next, pred.cpu().numpy(), is_seed=False)
            c_print(f'  pred  t={t_next:.4g}  [{step + 1}/{n_steps}]', color='bright_green')

            # Slide window: drop oldest frame, append prediction
            window.pop(0)
            window.append(pred)
            t_next += dt

    c_print(f'\nDone. {n_steps} frames written to {out_dir}', color='bright_magenta')


def run_inference_random(
    data_dir:   Path,
    checkpoint: Path,
    out_root:   Path,
    n_runs:     int,
    n_steps:    int | None = None,
    seed:       int        = 0,
) -> None:
    """
    Run inference on a random subset of simulation directories under data_dir.

    Each run is written to out_root/<sim_name>/ mirroring the input layout.
    """
    import random
    random.seed(seed)

    sim_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()
                       and any(f.name.startswith('t_') for f in p.iterdir())])
    if not sim_dirs:
        raise RuntimeError(f'No simulation directories found in {data_dir}')

    n_runs   = min(n_runs, len(sim_dirs))
    selected = random.sample(sim_dirs, n_runs)
    c_print(f'Selected {n_runs}/{len(sim_dirs)} runs from {data_dir}', color='cyan')

    for i, sim_dir in enumerate(selected):
        c_print(f'\n[{i + 1}/{n_runs}]  {sim_dir.name}', color='bright_cyan')
        run_inference(
            sim_dir    = sim_dir,
            checkpoint = checkpoint,
            out_dir    = out_root / sim_dir.name,
            n_steps    = n_steps,
            data_dir   = data_dir,
        )


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help='Path to model .pt checkpoint')
    parser.add_argument('out_dir',    type=Path, help='Output directory')
    parser.add_argument('--steps',    type=int,  default=None,
                        help='Number of frames to predict (default: len(sim_dir) - WINDOW_SIZE)')
    parser.add_argument('--data-dir', type=Path, default=DATASET_DIR,
                        help='Dataset root containing shared_mesh.pkl / renderer cache')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('sim_dir', type=Path, nargs='?',
                      help='Single simulation directory to run inference on')
    mode.add_argument('-r', '--random', type=int, metavar='N',
                      help='Run inference on N randomly selected simulation directories '
                           'from --data-dir, writing results to out_dir/<sim_name>/')

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for -r selection (default: 0)')
    args = parser.parse_args()

    if args.random is not None:
        run_inference_random(
            data_dir   = args.data_dir,
            checkpoint = args.checkpoint,
            out_root   = args.out_dir,
            n_runs     = args.random,
            n_steps    = args.steps,
            seed       = args.seed,
        )
    else:
        run_inference(
            sim_dir    = args.sim_dir,
            checkpoint = args.checkpoint,
            out_dir    = args.out_dir,
            n_steps    = args.steps,
            data_dir   = args.data_dir,
        )


if __name__ == '__main__':
    main()
