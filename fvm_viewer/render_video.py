"""
Render a video from an inference output directory (t_*.npz files).

Each frame is a 2×2 grid of the four fluid fields (Vx, Vy, rho, T).
Colour limits are computed per-channel from the full video (p1–p99),
so the scale is consistent across all frames.
Seed frames are tinted with a blue border; predicted frames are neutral.

Usage
-----
    python render_video.py path/to/infer_out/ output.mp4
    python render_video.py path/to/infer_out/ output.mp4 --fps 15 --dpi 100
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio

FIELD_NAMES = ["Vx", "Vy", "rho", "T"]


def _find_frames(infer_dir: Path) -> list[Path]:
    return sorted(
        [f for f in infer_dir.iterdir() if f.name.startswith('t_') and f.name.endswith('.npz')],
        key=lambda f: float(f.stem[2:]),
    )


def _load(path: Path) -> tuple[float, np.ndarray, bool]:
    d = np.load(path)
    return float(d['t']), d['grid'].astype(np.float32), bool(d['is_seed'])


def _channel_limits(grids: list[np.ndarray], pct: float = 1.0) -> list[tuple[float, float]]:
    limits = []
    for c in range(4):
        vals = np.concatenate([g[c][np.isfinite(g[c])].ravel() for g in grids])
        lo, hi = np.percentile(vals, [pct, 100 - pct])
        limits.append((float(lo), float(hi)))
    return limits


def _draw_frame(fig: plt.Figure, axes, grid: np.ndarray, t: float,
                is_seed: bool, limits: list[tuple[float, float]]) -> np.ndarray:
    for ax in axes.flat:
        ax.cla()

    for c, ax in enumerate(axes.flat):
        vmin, vmax = limits[c]
        ax.imshow(grid[c], vmin=vmin, vmax=vmax, cmap='viridis',
                  aspect='equal', interpolation='nearest', origin='upper')
        ax.set_title(FIELD_NAMES[c], fontsize=9, pad=3, color='white')
        ax.axis('off')

    label     = f't = {t:.4g}'
    tag_color = '#4fa3e0' if is_seed else '#a0d080'
    tag_text  = 'seed' if is_seed else 'pred'
    fig.text(0.5, 0.015, label, ha='center', va='bottom', fontsize=9,
             color='#cccccc', transform=fig.transFigure)
    fig.text(0.97, 0.015, tag_text, ha='right', va='bottom', fontsize=9,
             color=tag_color, transform=fig.transFigure, fontweight='bold')

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infer_dir', type=Path, help='Inference output directory')
    parser.add_argument('output',    type=Path, help='Output video path (e.g. out.mp4)')
    parser.add_argument('--fps',     type=int,  default=10,  help='Frames per second (default: 10)')
    parser.add_argument('--dpi',     type=int,  default=120, help='Render DPI (default: 120)')
    args = parser.parse_args()

    paths = _find_frames(args.infer_dir)
    if not paths:
        raise RuntimeError(f'No t_*.npz frames found in {args.infer_dir}')
    print(f'Found {len(paths)} frames in {args.infer_dir}')

    print('Loading frames...')
    records = [_load(p) for p in paths]
    grids   = [g for _, g, _ in records]

    print('Computing colour limits...')
    limits = _channel_limits(grids)
    for name, (lo, hi) in zip(FIELD_NAMES, limits):
        print(f'  {name}: [{lo:.4g}, {hi:.4g}]')

    fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=args.dpi,
                             facecolor='#111111')
    fig.subplots_adjust(hspace=0.08, wspace=0.04,
                        top=0.96, bottom=0.05, left=0.02, right=0.98)
    for ax in axes.flat:
        ax.set_facecolor('#111111')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Rendering {len(records)} frames → {args.output}  (fps={args.fps})')

    with imageio.get_writer(str(args.output), fps=args.fps, format='FFMPEG',
                            codec='libx264', quality=8, macro_block_size=None) as writer:
        for i, (t, grid, is_seed) in enumerate(records):
            img = _draw_frame(fig, axes, grid, t, is_seed, limits)
            writer.append_data(img)
            if (i + 1) % 20 == 0 or i == len(records) - 1:
                print(f'  {i + 1}/{len(records)}')

    plt.close(fig)
    print(f'Done → {args.output}')


if __name__ == '__main__':
    main()
