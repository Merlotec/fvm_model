"""
Run autoregressive inference with a trained FluidVisionModel.

The seed window is taken from [first_frame, first_frame + WINDOW_SIZE) in the
simulation directory, matching the training regime.  The model then rolls out
autoregressively, or optionally with teacher forcing (real frames used as
window input at every step).

Usage
-----
    python infer.py <checkpoint> <out_dir> <sim_dir> [options]
    python infer.py <checkpoint> <out_dir> -r N      [options]

Output directory structure:
    output_dir/
        t_<t>.npz   # is_seed=True  for seed frames
        t_<t>.npz   # is_seed=False for predicted frames
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from cprint import c_print

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from helper import (
    DATASET_DIR, RESOLUTION, WINDOW_SIZE, FIRST_FRAME,
    build_renderer,
)
from lightning_model import FVMLightningModel


def _select_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _find_timestep_files(sim_dir: Path) -> list[Path]:
    return sorted(
        [f for f in sim_dir.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
        key=lambda f: float(f.stem[2:]),
    )


def _load_and_render(path: Path, renderer) -> torch.Tensor:
    """Load a raw timestep file and return a rendered (N_CHANNELS, H, W) tensor."""
    d      = np.load(path)
    values = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
    return renderer.render_cell_smooth(values).nan_to_num(0.0)


def _t_of(path: Path) -> float:
    return float(path.stem[2:])


def _save_frame(out_dir: Path, t: float, grid: np.ndarray, is_seed: bool) -> None:
    np.savez_compressed(
        out_dir / f't_{t:.4g}.npz',
        grid    = grid,
        t       = np.float32(t),
        is_seed = np.bool_(is_seed),
    )


def run_inference(
    sim_dir:         Path,
    checkpoint:      Path,
    out_dir:         Path,
    n_steps:         int | None = None,
    data_dir:        Path = DATASET_DIR,
    teacher_forcing: bool = False,
) -> None:
    device = _select_device()
    c_print(f'Device: {device}', color='cyan')

    renderer = build_renderer(data_dir, RESOLUTION, device)

    model = FVMLightningModel.load_from_checkpoint(str(checkpoint), map_location=device)
    model.eval()
    c_print(f'Loaded checkpoint: {checkpoint}', color='green')

    # ---- input files ----
    all_files = _find_timestep_files(sim_dir)
    needed = FIRST_FRAME + WINDOW_SIZE
    if len(all_files) < needed:
        c_print(f'Skipping {sim_dir} — only {len(all_files)} timesteps (need {needed}).', color='yellow')
        return

    if n_steps is None:
        n_steps = max(1, len(all_files) - needed)

    out_dir.mkdir(parents=True, exist_ok=True)
    mode_label = 'teacher-forcing' if teacher_forcing else 'autoregressive'
    c_print(f'Output: {out_dir}', color='cyan')
    c_print(f'first_frame: {FIRST_FRAME}  |  seed: {WINDOW_SIZE}  |  steps: {n_steps}  |  mode: {mode_label}', color='cyan')

    # ---- seed window ----
    seed_files = all_files[FIRST_FRAME:FIRST_FRAME + WINDOW_SIZE]
    window: list[torch.Tensor] = []

    c_print('Rendering seed frames...', color='yellow')
    for path in seed_files:
        grid = _load_and_render(path, renderer)
        window.append(grid)
        _save_frame(out_dir, _t_of(path), grid.cpu().numpy(), is_seed=True)
        c_print(f'  seed  t={_t_of(path):.4g}', color='bright_black')

    dt     = _t_of(seed_files[-1]) - _t_of(seed_files[-2]) if len(seed_files) >= 2 else 0.1
    t_next = _t_of(seed_files[-1]) + dt

    # ---- rollout ----
    c_print('Running inference...', color='yellow')
    with torch.no_grad():
        for step in range(n_steps):
            inp       = torch.cat(window, dim=0).unsqueeze(0).to(device)  # (1, W*C, H, W)
            pred_norm = model(inp).squeeze(0)                              # (C, H, W) normalised
            delta     = model.denormalise(pred_norm)                       # (C, H, W) physical
            pred      = window[-1] + delta

            _save_frame(out_dir, t_next, pred.cpu().numpy(), is_seed=False)
            c_print(f'  pred  t={t_next:.4g}  [{step+1}/{n_steps}]  '
                    f'delta={delta.abs().mean():.4f}', color='bright_green')

            next_gt_idx = FIRST_FRAME + WINDOW_SIZE + step
            if teacher_forcing and next_gt_idx < len(all_files):
                next_frame = _load_and_render(all_files[next_gt_idx], renderer)
            else:
                next_frame = pred
            window.pop(0)
            window.append(next_frame)
            t_next += dt

    c_print(f'\nDone. {n_steps} frames written to {out_dir}', color='bright_magenta')


def run_inference_random(
    data_dir:        Path,
    checkpoint:      Path,
    out_root:        Path,
    n_runs:          int,
    n_steps:         int | None = None,
    seed:            int        = 0,
    teacher_forcing: bool       = False,
) -> None:
    """Run inference on a random subset of simulation directories under data_dir."""
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
        c_print(f'\n[{i+1}/{n_runs}]  {sim_dir.name}', color='bright_cyan')
        run_inference(
            sim_dir         = sim_dir,
            checkpoint      = checkpoint,
            out_dir         = out_root / sim_dir.name,
            n_steps         = n_steps,
            data_dir        = data_dir,
            teacher_forcing = teacher_forcing,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help='Path to Lightning checkpoint')
    parser.add_argument('out_dir',    type=Path, help='Output directory')
    parser.add_argument('--steps',    type=int,  default=None)
    parser.add_argument('--data-dir', type=Path, default=DATASET_DIR,
                        help='Dataset root containing shared_mesh.pkl / renderer cache')
    parser.add_argument('--teacher-forcing', action='store_true',
                        help='Use ground-truth frames as window input instead of model predictions')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for -r selection (default: 0)')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('sim_dir', type=Path, nargs='?',
                      help='Single simulation directory')
    mode.add_argument('-r', '--random', type=int, metavar='N',
                      help='Run on N randomly selected simulation directories')

    args = parser.parse_args()

    if args.random is not None:
        run_inference_random(
            data_dir        = args.data_dir,
            checkpoint      = args.checkpoint,
            out_root        = args.out_dir,
            n_runs          = args.random,
            n_steps         = args.steps,
            seed            = args.seed,
            teacher_forcing = args.teacher_forcing,
        )
    else:
        run_inference(
            sim_dir         = args.sim_dir,
            checkpoint      = args.checkpoint,
            out_dir         = args.out_dir,
            n_steps         = args.steps,
            data_dir        = args.data_dir,
            teacher_forcing = args.teacher_forcing,
        )


if __name__ == '__main__':
    main()
