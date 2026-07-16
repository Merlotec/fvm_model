from typing import Optional, Union
"""
NaN diagnostic for FluidVisionModel.

Registers forward hooks on every sub-module and runs a single forward pass
with a synthetic input. Prints a table of every layer's output statistics
and marks the first layer that produces NaN.

Usage:
    python debug_nan.py                        # random input, best checkpoint
    python debug_nan.py --ckpt checkpoints/last-v1.ckpt
    python debug_nan.py --random-input         # skip checkpoint, use fresh weights
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from model import FluidVisionModel

with open(Path(__file__).resolve().parent / 'hyperparams.json') as f:
    HP = json.load(f)

H, W         = HP['resolution']
PATCH_SIZE   = HP['patch_size']
EMB_DIM      = HP['emb_dim']
N_CHANNELS   = HP['n_channels']
WINDOW_SIZE  = HP['window_size']
NUM_PATCHES  = (H // PATCH_SIZE) * (W // PATCH_SIZE)


def build_model(ckpt_path: Optional[Path]) -> nn.Module:
    model = FluidVisionModel(
        num_obs      = WINDOW_SIZE,
        num_patches  = NUM_PATCHES,
        patch_size   = PATCH_SIZE,
        emb_dim      = EMB_DIM,
        num_channels = N_CHANNELS,
    )
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        sd = ckpt.get('state_dict', ckpt)
        # strip Lightning 'model.' prefix and drop delta_mean/std buffers
        sd = {k.removeprefix('model.'): v for k, v in sd.items()
              if not k.startswith('delta_')}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f'[warn] missing keys: {missing}')
        print(f'Loaded {ckpt_path}')
    return model.eval()


def check_weights(model: nn.Module) -> bool:
    nan_params = [(n, p) for n, p in model.named_parameters() if p.isnan().any()]
    if nan_params:
        print(f'\n*** {len(nan_params)} NaN weight tensors ***')
        for n, p in nan_params[:5]:
            print(f'  {n}  shape={tuple(p.shape)}')
        return False
    print('Weights: all finite ✓')
    return True


def run_nan_trace(model: nn.Module, x: torch.Tensor) -> None:
    records: list[tuple[str, str, bool, str]] = []
    hooks = []
    first_nan: list[str] = []

    def make_hook(name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor):
                return
            has_nan = t.isnan().any().item()
            has_inf = t.isinf().any().item()
            mn  = t[t.isfinite()].min().item() if t.isfinite().any() else float('nan')
            mx  = t[t.isfinite()].max().item() if t.isfinite().any() else float('nan')
            std = t[t.isfinite()].std().item()  if t.isfinite().any() else float('nan')
            records.append((name, str(tuple(t.shape)), has_nan or has_inf,
                            f'min={mn:+.3e}  max={mx:+.3e}  std={std:.3e}'
                            + ('  *** NaN ***' if has_nan else '')
                            + ('  *** Inf ***' if has_inf else '')))
            if (has_nan or has_inf) and not first_nan:
                first_nan.append(name)
        return hook

    for name, module in model.named_modules():
        if name == '':
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        try:
            out = model(x)
        except Exception as e:
            print(f'Forward pass raised: {e}')

    for h in hooks:
        h.remove()

    print(f'\n{"Layer":<55} {"Shape":<25} {"Stats"}')
    print('-' * 110)
    for name, shape, bad, stats in records:
        marker = ' <-- FIRST NaN' if name == (first_nan[0] if first_nan else None) else ''
        print(f'{"[BAD] " + name if bad else name:<55} {shape:<25} {stats}{marker}')

    print()
    if first_nan:
        print(f'*** NaN first appears at: {first_nan[0]} ***')
    else:
        out_nan = out.isnan().any().item()
        print(f'Final output NaN: {out_nan}')
        print('No NaN detected anywhere in forward pass.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=Path,
                        default=Path('checkpoints/last-v1.ckpt'))
    parser.add_argument('--random-input', action='store_true',
                        help='Use a random N(0,1) input instead of zeros')
    parser.add_argument('--no-ckpt', action='store_true',
                        help='Use freshly initialised weights (no checkpoint)')
    args = parser.parse_args()

    ckpt = None if args.no_ckpt else args.ckpt
    model = build_model(ckpt)

    ok = check_weights(model)
    if not ok:
        print('Fix NaN weights before tracing forward pass.')
        return

    if args.random_input:
        x = torch.randn(1, WINDOW_SIZE * N_CHANNELS, H, W)
        print(f'Input: random N(0,1)  shape={tuple(x.shape)}')
    else:
        x = torch.zeros(1, WINDOW_SIZE * N_CHANNELS, H, W)
        print(f'Input: all-zeros  shape={tuple(x.shape)}')

    run_nan_trace(model, x)


if __name__ == '__main__':
    main()
