"""
Training diagnostics for FluidVisionModel.

Three checks, run in order:

  1. DATA    -- inspect what targets actually look like (distribution, sparsity,
                variance across the batch). If targets are near-zero or degenerate
                the model has nothing to learn.

  2. GRADIENTS -- after one backward pass, print the mean absolute gradient for
                  every parameter group. Vanishing or missing gradients show up here.

  3. OVERFIT  -- try to drive a single fixed batch to near-zero loss with many
                 gradient steps. If the model cannot memorise one batch, there is a
                 fundamental capacity or optimisation problem.

Usage:
    python debug_training.py                 # all three checks, uses real data
    python debug_training.py --check data
    python debug_training.py --check grads
    python debug_training.py --check overfit
    python debug_training.py --check overfit --overfit-steps 500 --lr 1e-3
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from model import FluidVisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fvm_gen'))
from renderer import MeshRenderer

from helper import (
    DATASET_DIR, RESOLUTION, PATCH_SIZE, EMB_DIM, N_CHANNELS, WINDOW_SIZE,
    DELTA_STATS_PATH, build_renderer,
)
from data import FVMDataModule

H, W        = RESOLUTION
NUM_PATCHES = (H // PATCH_SIZE) * (W // PATCH_SIZE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_delta_stats(device):
    if DELTA_STATS_PATH.exists():
        with open(DELTA_STATS_PATH) as f:
            s = json.load(f)
        mean = torch.tensor(s['mean'], device=device).view(N_CHANNELS, 1, 1)
        std  = torch.tensor(s['std'],  device=device).view(N_CHANNELS, 1, 1)
        print(f'  delta_mean: {[f"{v:.4f}" for v in mean.flatten().tolist()]}')
        print(f'  delta_std:  {[f"{v:.4f}" for v in std.flatten().tolist()]}')
        return mean, std
    print('  WARNING: delta_stats.json missing — using mean=0, std=1')
    return (torch.zeros(N_CHANNELS, 1, 1, device=device),
            torch.ones( N_CHANNELS, 1, 1, device=device))


def fresh_model(device):
    return FluidVisionModel(
        num_obs      = WINDOW_SIZE,
        num_patches  = NUM_PATCHES,
        patch_size   = PATCH_SIZE,
        emb_dim      = EMB_DIM,
        num_channels = N_CHANNELS,
    ).to(device)


def get_batch(data_dir, device, batch_size=4):
    dm = FVMDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=0)
    dm.setup()
    loader = dm.train_dataloader()
    window, target = next(iter(loader))
    return window.to(device), target.to(device)


# ---------------------------------------------------------------------------
# Check 1: Data
# ---------------------------------------------------------------------------

def check_data(data_dir, device):
    print('\n' + '='*60)
    print('CHECK 1: DATA')
    print('='*60)

    print('\nLoading delta normalisation stats:')
    mean, std = load_delta_stats(device)

    if not mean.isfinite().all() or not std.isfinite().all():
        print('  FAIL: delta stats contain NaN/Inf — run training to recompute')
        return

    print('\nLoading one batch...')
    window, target = get_batch(data_dir, device)

    print(f'\nWindow  shape: {tuple(window.shape)}')
    print(f'Target  shape: {tuple(target.shape)}')

    finite_mask = torch.isfinite(target)
    fluid_frac  = finite_mask.float().mean().item()
    print(f'\nFluid pixels (finite target): {fluid_frac*100:.1f}%  '
          f'({finite_mask.sum().item()} / {target.numel()})')

    if fluid_frac < 0.05:
        print('  WARNING: less than 5% of pixels are in-mesh — '
              'check renderer or mesh scale')

    t_fluid = target[finite_mask]
    print(f'\nRaw delta stats over fluid pixels:')
    print(f'  mean={t_fluid.mean():.4f}  std={t_fluid.std():.4f}  '
          f'min={t_fluid.min():.4f}  max={t_fluid.max():.4f}')

    target_norm = (target - mean) / std
    tn_fluid    = target_norm[finite_mask]
    print(f'\nNormalised delta stats over fluid pixels:')
    print(f'  mean={tn_fluid.mean():.4f}  std={tn_fluid.std():.4f}  '
          f'min={tn_fluid.min():.4f}  max={tn_fluid.max():.4f}')

    if tn_fluid.std() < 0.1:
        print('  WARNING: normalised targets are very small — '
              'model is being asked to predict near-zero deltas')
    if tn_fluid.abs().max() > 100:
        print('  WARNING: normalised target has very large outliers — '
              'consider clipping or recomputing stats')

    # Per-channel breakdown
    print('\nPer-channel normalised target std:')
    for c in range(N_CHANNELS):
        ch_mask = finite_mask[:, c]
        ch_std  = target_norm[:, c][ch_mask].std().item()
        print(f'  channel {c}: std={ch_std:.4f}')

    window_finite = window[torch.isfinite(window)]
    print(f'\nWindow input range: '
          f'min={window_finite.min():.4f}  max={window_finite.max():.4f}  '
          f'std={window_finite.std():.4f}')


# ---------------------------------------------------------------------------
# Check 2: Gradients
# ---------------------------------------------------------------------------

def check_gradients(data_dir, device):
    print('\n' + '='*60)
    print('CHECK 2: GRADIENT FLOW')
    print('='*60)

    mean, std = load_delta_stats(device)
    window, target = get_batch(data_dir, device)
    model = fresh_model(device)
    model.train()

    pred        = model(window)
    target_norm = (target - mean) / std
    valid       = torch.isfinite(target_norm)
    loss        = (pred - target_norm).abs()[valid].mean()
    print(f'\nInitial loss: {loss.item():.4f}')
    loss.backward()

    print(f'\n{"Parameter":<55} {"grad mean abs":>14}  {"grad max abs":>12}  {"param std":>10}')
    print('-' * 95)

    dead_layers = []
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f'{name:<55} {"NO GRADIENT":>14}')
            dead_layers.append(name)
            continue
        g_mean = p.grad.abs().mean().item()
        g_max  = p.grad.abs().max().item()
        p_std  = p.data.std().item()
        flag   = ''
        if g_mean < 1e-8:
            flag = '  <-- VANISHING'
            dead_layers.append(name)
        elif g_mean > 1.0:
            flag = '  <-- LARGE'
        print(f'{name:<55} {g_mean:>14.3e}  {g_max:>12.3e}  {p_std:>10.4f}{flag}')

    if dead_layers:
        print(f'\nWARNING: {len(dead_layers)} parameters with vanishing/missing gradients')
    else:
        print('\nAll parameters have gradients ✓')


# ---------------------------------------------------------------------------
# Check 3: Overfit one batch
# ---------------------------------------------------------------------------

def check_overfit(data_dir, device, steps=300, lr=1e-3):
    print('\n' + '='*60)
    print(f'CHECK 3: OVERFIT SINGLE BATCH  (steps={steps}, lr={lr})')
    print('='*60)

    mean, std   = load_delta_stats(device)
    window, target = get_batch(data_dir, device)
    target_norm = (target - mean) / std
    valid       = torch.isfinite(target_norm)

    if not valid.any():
        print('FAIL: no valid (finite) pixels in this batch — check data loading')
        return

    model = fresh_model(device)
    model.train()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    print(f'\nValid pixels: {valid.sum().item()} / {target_norm.numel()}')
    print(f'\n{"Step":>6}  {"Loss":>10}  {"Pred std":>10}  {"Target std":>12}')
    print('-' * 44)

    prev_loss = None
    for step in range(1, steps + 1):
        opt.zero_grad()
        pred = model(window)
        loss = (pred - target_norm).abs()[valid].mean()
        loss.backward()
        # Clip to catch any remaining explosion
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in (1, 5, 10, 25, 50, 100, 200, 300, steps):
            pred_std   = pred[valid].std().item()
            target_std = target_norm[valid].std().item()
            print(f'{step:>6}  {loss.item():>10.4f}  {pred_std:>10.4f}  {target_std:>12.4f}')
            prev_loss  = loss.item()

    print()
    if prev_loss is not None and prev_loss < 0.1:
        print('Model can overfit a single batch ✓')
    elif prev_loss is not None and prev_loss < 0.5:
        print('Loss decreasing but not fully converged — '
              'try more steps or higher lr')
    else:
        print('FAIL: model cannot overfit a single batch — '
              'architecture or optimisation is broken')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir',      type=Path, default=DATASET_DIR)
    parser.add_argument('--check',         choices=['data', 'grads', 'overfit', 'all'],
                        default='all')
    parser.add_argument('--overfit-steps', type=int,   default=300)
    parser.add_argument('--lr',            type=float, default=1e-3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else \
             'mps'  if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}')

    run_all = args.check == 'all'

    if run_all or args.check == 'data':
        check_data(args.data_dir, device)

    if run_all or args.check == 'grads':
        check_gradients(args.data_dir, device)

    if run_all or args.check == 'overfit':
        check_overfit(args.data_dir, device, steps=args.overfit_steps, lr=args.lr)


if __name__ == '__main__':
    main()
