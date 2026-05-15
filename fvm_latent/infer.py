"""
Autoregressive inference for MultiLevelFluidModel.

The first WINDOW_SIZE frames from the input sim directory seed the window.
The model then rolls out autoregressively, predicting one delta per step and
feeding each prediction back as the next frame.

Usage (run from fvm_model/):
    # Single sim dir
    python fvm_latent/infer.py <checkpoint> <out_dir> <sim_dir>

    # Random N sims from data dir
    python fvm_latent/infer.py <checkpoint> <out_dir> -r N --data-dir data/fvm_gen_datasets

Output mirrors fvm_foundation/infer.py format:
    out_dir/
        t_0.0000.npz   # seed  (is_seed=True,  grid=(4,H,W))
        ...
        t_1.0000.npz   # pred  (is_seed=False, grid=(4,H,W))
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from cprint import c_print

_FVM_MODEL = Path(__file__).resolve().parents[1]
_FOUND     = _FVM_MODEL / 'fvm_foundation'

for _p in (_FOUND, _FVM_MODEL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from helper import build_renderer, DATASET_DIR, PIXEL_MASK_PATH  # type: ignore[import]
from fvm_latent.model import MultiLevelFluidModel                 # type: ignore[import]

_THIS_DIR = Path(__file__).resolve().parent

with open(_THIS_DIR / 'hyperparams.json') as _f:
    _HP = json.load(_f)

IMG_SIZE    = _HP['img_size']
RESOLUTION  = (IMG_SIZE, IMG_SIZE)
IN_CHANNELS = _HP['in_channels']
WINDOW_SIZE = _HP['window_size']
FIRST_FRAME = _HP.get('first_frame', 0)

_DELTA_STATS_PATH = _FOUND / 'delta_stats.json'
_INPUT_STATS_PATH = _FOUND / 'input_stats.json'


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
    return sorted(
        [f for f in sim_dir.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
        key=lambda f: float(f.stem[2:]),
    )


def _load_and_render(path: Path, renderer) -> torch.Tensor:
    d      = np.load(path)
    values = d['cell_primatives'].astype(np.float32) * d['prim_std'] + d['prim_mean']
    return renderer.render_cell_smooth(values).nan_to_num(0.0)  # (C, H, W)


def _t_of(path: Path) -> float:
    return float(path.stem[2:])


def _save_frame(out_dir: Path, t: float, grid: np.ndarray, is_seed: bool) -> None:
    np.savez_compressed(
        out_dir / f't_{t:.4g}.npz',
        grid    = grid,
        t       = np.float32(t),
        is_seed = np.bool_(is_seed),
    )


def _load_model(checkpoint: Path, device: str) -> tuple[MultiLevelFluidModel, dict]:
    """Load MultiLevelFluidModel from a Phase1LightningModel checkpoint."""
    model = MultiLevelFluidModel(
        img_size             = _HP['img_size'],
        patch_size           = _HP['patch_size'],
        in_channels          = _HP['in_channels'],
        window_size          = _HP.get('window_size', 1),
        n_levels             = _HP['n_levels'],
        d_model              = _HP['d_model'],
        n_heads              = _HP['n_heads'],
        n_transformer_layers = _HP['n_transformer_layers'],
        gate_threshold       = _HP.get('gate_threshold', 0.5),
        gate_budget          = _HP.get('gate_budget', 0.4),
        dropout              = 0.0,
    ).to(device)

    ckpt  = torch.load(checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}

    norm_bufs = {}
    for key in ('delta_mean', 'delta_std', 'input_mean', 'input_std', 'pixel_mask'):
        norm_bufs[key] = state.pop(key, None)

    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        c_print(f'Warning: missing keys: {result.missing_keys}', color='yellow')
    if result.unexpected_keys:
        c_print(f'Warning: unexpected keys (not loaded): {result.unexpected_keys}', color='yellow')

    model.eval()
    c_print(f'Loaded: {checkpoint.name}', color='green')
    return model, norm_bufs


def _load_norm(norm_bufs: dict, device: str) -> tuple:
    """Return (delta_mean, delta_std, input_mean, input_std, pixel_mask) tensors."""
    def _pair(ckpt_m, ckpt_s, json_path, label):
        if ckpt_m is not None and ckpt_s is not None:
            c_print(f'Loaded {label} from checkpoint', color='green')
            return ckpt_m.to(device).view(-1, 1, 1), ckpt_s.to(device).view(-1, 1, 1)
        if json_path.exists():
            with open(json_path) as f:
                s = json.load(f)
            c_print(f'Loaded {label} from {json_path.name} (fallback)', color='yellow')
            return (torch.tensor(s['mean'], device=device).view(-1, 1, 1),
                    torch.tensor(s['std'],  device=device).view(-1, 1, 1))
        c_print(f'Warning: no {label} found — using identity', color='yellow')
        return None, None

    dm, ds = _pair(norm_bufs['delta_mean'], norm_bufs['delta_std'], _DELTA_STATS_PATH, 'delta stats')
    im, is_ = _pair(norm_bufs['input_mean'], norm_bufs['input_std'], _INPUT_STATS_PATH, 'input stats')

    pm = norm_bufs.get('pixel_mask')
    if pm is not None:
        pixel_mask = pm.to(device)
        c_print(f'Loaded pixel mask from checkpoint — {pixel_mask.sum().item():.0f} fluid pixels', color='green')
    elif PIXEL_MASK_PATH.exists():
        pixel_mask = torch.load(PIXEL_MASK_PATH, map_location=device)
        c_print(f'Loaded pixel mask from {PIXEL_MASK_PATH.name} (fallback)', color='yellow')
    else:
        pixel_mask = None
        c_print('Warning: no pixel mask found', color='yellow')

    return dm, ds, im, is_, pixel_mask


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    sim_dir:         Path,
    checkpoint:      Path,
    out_dir:         Path,
    n_steps:         int | None = None,
    data_dir:        Path       = DATASET_DIR,
    teacher_forcing: bool       = False,
) -> None:
    device = _select_device()
    c_print(f'Device: {device}', color='cyan')

    renderer              = build_renderer(data_dir, RESOLUTION, device)
    model, norm_bufs      = _load_model(checkpoint, device)
    dm, ds, im, is_, pmask = _load_norm(norm_bufs, device)

    all_files = _find_timestep_files(sim_dir)
    needed    = FIRST_FRAME + WINDOW_SIZE
    if len(all_files) < needed:
        c_print(f'Skipping {sim_dir} — only {len(all_files)} timesteps (need {needed})', color='yellow')
        return

    if n_steps is None:
        n_steps = max(1, len(all_files) - needed)

    out_dir.mkdir(parents=True, exist_ok=True)
    mode = 'teacher-forcing' if teacher_forcing else 'autoregressive'
    c_print(f'Output: {out_dir}', color='cyan')
    c_print(f'first_frame={FIRST_FRAME}  seed={WINDOW_SIZE}  steps={n_steps}  mode={mode}', color='cyan')

    # Seed window
    seed_files = all_files[FIRST_FRAME : FIRST_FRAME + WINDOW_SIZE]
    window: list[torch.Tensor] = []

    c_print('Rendering seed frames...', color='yellow')
    for path in seed_files:
        grid = _load_and_render(path, renderer)
        window.append(grid)
        _save_frame(out_dir, _t_of(path), grid.cpu().numpy(), is_seed=True)
        c_print(f'  seed  t={_t_of(path):.4g}', color='bright_black')

    dt     = _t_of(seed_files[-1]) - _t_of(seed_files[-2]) if len(seed_files) >= 2 else 0.1
    t_next = _t_of(seed_files[-1]) + dt

    def _norm_window(frames: list[torch.Tensor]) -> torch.Tensor:
        inp = torch.stack(frames).unsqueeze(0)       # (1, T, C, H, W)
        if im is not None and is_ is not None:
            m = im.unsqueeze(0).unsqueeze(0)         # (1, 1, C, 1, 1)
            s = is_.unsqueeze(0).unsqueeze(0)
            inp = ((inp - m) / s).nan_to_num(0.0)
        return inp.nan_to_num(0.0)

    c_print('Running inference...', color='yellow')
    with torch.no_grad():
        for step in range(n_steps):
            inp = _norm_window(window)
            out = model(inp)

            raw_delta = out['pred'].squeeze(0)       # (C, H, W)
            if pmask is not None:
                raw_delta = raw_delta * pmask.squeeze(0)
            delta = raw_delta * ds + dm if dm is not None else raw_delta
            pred  = window[-1] + delta
            if pmask is not None:
                pred = pred * pmask.squeeze(0)

            _save_frame(out_dir, t_next, pred.cpu().numpy(), is_seed=False)
            c_print(
                f'  pred  t={t_next:.4g}  [{step+1}/{n_steps}]'
                f'  delta={delta.abs().mean():.4f}'
                f'  sparsity={out["sparsity"]:.3f}'
                f'  active={out["active_frac"]:.3f}',
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

    c_print(f'\nDone. {n_steps} frames → {out_dir}', color='bright_magenta')


def run_inference_random(
    data_dir:        Path,
    checkpoint:      Path,
    out_root:        Path,
    n_runs:          int,
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
    c_print(f'Selected {len(selected)}/{len(sim_dirs)} runs from {data_dir}', color='cyan')

    for i, sim_dir in enumerate(selected):
        c_print(f'\n[{i+1}/{len(selected)}]  {sim_dir.name}', color='bright_cyan')
        run_inference(
            sim_dir         = sim_dir,
            checkpoint      = checkpoint,
            out_dir         = out_root / sim_dir.name,
            n_steps         = n_steps,
            data_dir        = data_dir,
            teacher_forcing = teacher_forcing,
        )


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out_dir',    type=Path)
    parser.add_argument('--steps',    type=int, default=None)
    parser.add_argument('--data-dir', type=Path, default=DATASET_DIR)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('sim_dir', type=Path, nargs='?',
                      help='Single simulation directory')
    mode.add_argument('-r', '--random', type=int, metavar='N',
                      help='Run on N randomly selected sim dirs from --data-dir')

    parser.add_argument('--seed',            type=int,  default=0)
    parser.add_argument('--teacher-forcing', action='store_true')
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
