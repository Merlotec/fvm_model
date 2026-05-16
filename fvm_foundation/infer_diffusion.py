"""
Autoregressive inference using a trained DiffusionDecoder (stage-2 checkpoint).

Usage
-----
    # Single sim directory:
    python infer_diffusion.py path/to/sim_dir/ checkpoints_diffusion/last.ckpt path/to/out/

    # Random N sims from the dataset:
    python infer_diffusion.py -r 5 checkpoints_diffusion/last.ckpt path/to/out/

    # Faster inference (fewer DDIM steps, lower quality):
    python infer_diffusion.py sim_dir/ ckpt.ckpt out/ --ddim-steps 10

Options
-------
    --ddim-steps N      DDIM denoising steps (default 50; 10–20 for fast preview)
    --steps N           Number of frames to predict (default: all remaining)
    --data-dir PATH     Dataset root (for mesh renderer)
    --teacher-forcing   Feed ground-truth frames into window instead of predictions
    -r N / --random N   Run on N randomly selected sim dirs under --data-dir
    --seed INT          RNG seed for -r selection
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from cprint import c_print

_SOLVER_DIR = Path(__file__).resolve().parents[2] / 'fvm_solver'
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fvm_gen'))

from renderer import MeshRenderer
from helper import DATASET_DIR, build_renderer, PIXEL_MASK_PATH
from train_diffusion import DiffusionStageLightning

with open(Path(__file__).resolve().parent / 'hyperparams.json') as _f:
    _HP = json.load(_f)

WINDOW_SIZE = _HP['window_size']
FIRST_FRAME = _HP['first_frame']


# ── Helpers shared with infer.py ──────────────────────────────────────────────

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


def _load_and_render(path: Path, renderer: MeshRenderer) -> torch.Tensor:
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


# ── Core inference ────────────────────────────────────────────────────────────

def run_inference(
    sim_dir:         Path,
    checkpoint:      Path,
    out_dir:         Path,
    ddim_steps:      int       = 50,
    n_steps:         int | None = None,
    data_dir:        Path       = DATASET_DIR,
    teacher_forcing: bool       = False,
) -> None:
    device = _select_device()
    c_print(f'Device: {device}', color='cyan')

    renderer = build_renderer(data_dir, tuple(_HP['resolution']), device)

    # ── Load model ────────────────────────────────────────────────────────────
    lightning_model = DiffusionStageLightning.load_from_checkpoint(
        checkpoint, map_location=device
    )
    backbone = lightning_model.backbone.to(device).eval()
    decoder  = lightning_model.decoder.to(device).eval()

    # Normalisation stats
    delta_mean = lightning_model.delta_mean.to(device)
    delta_std  = lightning_model.delta_std.to(device)
    input_mean = lightning_model.input_mean.to(device)
    input_std  = lightning_model.input_std.to(device)

    # Load normalisation from JSON if they were never set (all-zero/one defaults)
    _DELTA = Path(__file__).parent / 'delta_stats.json'
    _INPUT = Path(__file__).parent / 'input_stats.json'
    N = _HP['n_channels']
    if delta_mean.abs().max() < 1e-9 and _DELTA.exists():
        with open(_DELTA) as f:
            s = json.load(f)
        delta_mean = torch.tensor(s['mean'], device=device).view(N, 1, 1)
        delta_std  = torch.tensor(s['std'],  device=device).view(N, 1, 1)
        c_print('Delta stats loaded from JSON (fallback)', color='yellow')

    if input_mean.abs().max() < 1e-9 and _INPUT.exists():
        with open(_INPUT) as f:
            s = json.load(f)
        input_mean = torch.tensor(s['mean'], device=device).view(N, 1, 1)
        input_std  = torch.tensor(s['std'],  device=device).view(N, 1, 1)
        c_print('Input stats loaded from JSON (fallback)', color='yellow')

    pixel_mask = None
    if PIXEL_MASK_PATH.exists():
        pixel_mask = torch.load(PIXEL_MASK_PATH, map_location=device)   # (1,1,H,W)
        c_print(f'Pixel mask: {pixel_mask.sum().item()} fluid pixels', color='green')

    c_print(f'Loaded: {checkpoint.name}  |  DDIM steps: {ddim_steps}', color='green')

    # ── Seed window ───────────────────────────────────────────────────────────
    all_files = _find_timestep_files(sim_dir)
    needed    = FIRST_FRAME + WINDOW_SIZE
    if len(all_files) < needed:
        c_print(f'Skipping {sim_dir} — only {len(all_files)} timesteps (need {needed}).', color='yellow')
        return

    if n_steps is None:
        n_steps = max(1, len(all_files) - needed)

    out_dir.mkdir(parents=True, exist_ok=True)
    mode = 'teacher-forcing' if teacher_forcing else 'autoregressive'
    c_print(f'Seed: {WINDOW_SIZE} frames  |  Steps: {n_steps}  |  Mode: {mode}', color='cyan')

    seed_files = all_files[FIRST_FRAME : FIRST_FRAME + WINDOW_SIZE]
    window: list[torch.Tensor] = []
    c_print('Rendering seed frames...', color='yellow')
    for path in seed_files:
        grid = _load_and_render(path, renderer)
        window.append(grid)
        _save_frame(out_dir, _t_of(path), grid.cpu().numpy(), is_seed=True)

    dt     = _t_of(seed_files[-1]) - _t_of(seed_files[-2]) if len(seed_files) >= 2 else 0.1
    t_next = _t_of(seed_files[-1]) + dt

    # ── Rollout ───────────────────────────────────────────────────────────────
    c_print('Running inference...', color='yellow')

    def _normalise(frames: list[torch.Tensor]) -> torch.Tensor:
        inp = torch.stack(frames).unsqueeze(0)                  # (1, T, C, H, W)
        nm  = input_mean.unsqueeze(0).unsqueeze(0)              # (1, 1, C, 1, 1)
        ns  = input_std.unsqueeze(0).unsqueeze(0)
        return ((inp - nm) / ns).nan_to_num(0.0)

    with torch.no_grad():
        for step in range(n_steps):
            inp = _normalise(window)

            z          = backbone.patch_embed(inp)                  # (1, T*P, D)
            z          = backbone.transformer(z)                    # (1, T*P, D)
            z          = z[:, -backbone.num_patches:, :]            # (1, P, D)
            pred_norm  = decoder(z, n_steps=ddim_steps).squeeze(0) # (C, H, W)

            delta = pred_norm * delta_std + delta_mean
            if pixel_mask is not None:
                delta = delta * pixel_mask.squeeze(0)
            pred = window[-1] + delta
            if pixel_mask is not None:
                pred = pred * pixel_mask.squeeze(0)

            _save_frame(out_dir, t_next, pred.cpu().numpy(), is_seed=False)
            c_print(
                f'  pred  t={t_next:.4g}  [{step + 1}/{n_steps}]'
                f'  delta_mean={delta.abs().mean():.4f}',
                color='bright_green',
            )

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
    ddim_steps:      int        = 50,
    n_steps:         int | None = None,
    seed:            int        = 0,
    teacher_forcing: bool       = False,
) -> None:
    import random
    random.seed(seed)
    sim_dirs = sorted([
        p for p in data_dir.iterdir()
        if p.is_dir() and any(f.name.startswith('t_') for f in p.iterdir())
    ])
    if not sim_dirs:
        raise RuntimeError(f'No simulation directories found in {data_dir}')
    selected = random.sample(sim_dirs, min(n_runs, len(sim_dirs)))
    c_print(f'Selected {len(selected)}/{len(sim_dirs)} runs', color='cyan')
    for i, sim_dir in enumerate(selected):
        c_print(f'\n[{i + 1}/{len(selected)}]  {sim_dir.name}', color='bright_cyan')
        run_inference(sim_dir, checkpoint, out_root / sim_dir.name,
                      ddim_steps=ddim_steps, n_steps=n_steps,
                      data_dir=data_dir, teacher_forcing=teacher_forcing)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint',    type=Path, help='Diffusion stage checkpoint')
    parser.add_argument('out_dir',       type=Path, help='Output directory')
    parser.add_argument('--ddim-steps',  type=int,  default=50,
                        help='DDIM denoising steps (default 50; use 10–20 for fast preview)')
    parser.add_argument('--steps',       type=int,  default=None,
                        help='Frames to predict (default: all remaining in sim_dir)')
    parser.add_argument('--data-dir',    type=Path, default=DATASET_DIR)
    parser.add_argument('--teacher-forcing', action='store_true')
    parser.add_argument('--seed',        type=int,  default=0)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('sim_dir', type=Path, nargs='?')
    mode.add_argument('-r', '--random', type=int, metavar='N')

    args = parser.parse_args()

    if args.random is not None:
        run_inference_random(
            data_dir        = args.data_dir,
            checkpoint      = args.checkpoint,
            out_root        = args.out_dir,
            n_runs          = args.random,
            ddim_steps      = args.ddim_steps,
            n_steps         = args.steps,
            seed            = args.seed,
            teacher_forcing = args.teacher_forcing,
        )
    else:
        run_inference(
            sim_dir         = args.sim_dir,
            checkpoint      = args.checkpoint,
            out_dir         = args.out_dir,
            ddim_steps      = args.ddim_steps,
            n_steps         = args.steps,
            data_dir        = args.data_dir,
            teacher_forcing = args.teacher_forcing,
        )


if __name__ == '__main__':
    main()
