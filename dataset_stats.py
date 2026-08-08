#!/usr/bin/env python3
"""
Dataset inventory: size on disk, run/branch counts, and total frames.

    python fvm_model/dataset_stats.py ../data/fvm_gen_v2
    python fvm_model/dataset_stats.py ../data/fvm_gen_v2 --per-mesh
    python fvm_model/dataset_stats.py ../data/*            # several at once

Layout it understands (both are handled):
    <data>/mesh_<uid>/run_<n>_<uid>/t_*.npz     nested, one dir per geometry
    <data>/run_<n>_<uid>/t_*.npz                flat, single geometry

A run is COMPLETE when it holds at least --min-frames frames (default 2, the
minimum that yields a usable transition).  Runs with no frames are typically
solver jobs that were queued but never ran, or that crashed on the first step.

A run is a BRANCH when it carries branch_meta.json, i.e. it was produced by
fvm_gen/branch_gen.py by restarting a source frame under different physics,
rather than by run_gen.py as an independent trajectory.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path


def human(n_bytes: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n_bytes < 1024 or unit == 'TB':
            return f'{n_bytes:,.1f} {unit}'
        n_bytes /= 1024
    return f'{n_bytes:,.1f} TB'


def scan_run(run_dir: Path) -> dict:
    """Frame count, byte size and branch status for one run directory."""
    n_frames = 0
    n_bytes = 0
    for entry in os.scandir(run_dir):
        if not entry.is_file():
            continue
        n_bytes += entry.stat().st_size
        if entry.name.startswith('t_') and entry.name.endswith('.npz'):
            n_frames += 1
    return {
        'frames': n_frames,
        'bytes': n_bytes,
        'is_branch': (run_dir / 'branch_meta.json').exists(),
    }


def find_mesh_dirs(data_dir: Path) -> list[Path]:
    """Geometry directories; falls back to the data dir itself when runs sit flat."""
    meshes = sorted(p for p in data_dir.iterdir()
                    if p.is_dir() and any(c.name.startswith('run_')
                                          for c in os.scandir(p) if c.is_dir()))
    if meshes:
        return meshes
    if any(p.name.startswith('run_') for p in data_dir.iterdir() if p.is_dir()):
        return [data_dir]
    return []


def scan_dataset(data_dir: Path, min_frames: int) -> dict:
    meshes = find_mesh_dirs(data_dir)
    per_mesh, aux_bytes = [], 0

    # Loose files at the dataset root (stats json, renderer/mask caches).
    for entry in os.scandir(data_dir):
        if entry.is_file():
            aux_bytes += entry.stat().st_size

    for mesh in meshes:
        runs = sorted(p for p in mesh.iterdir() if p.is_dir() and p.name.startswith('run_'))
        info = {'name': mesh.name, 'complete': 0, 'empty': 0, 'short': 0,
                'branches': 0, 'base': 0, 'frames': 0, 'bytes': 0,
                'frames_per_run': []}
        if mesh != data_dir:                      # per-mesh shared files (shared_mesh.pkl …)
            for entry in os.scandir(mesh):
                if entry.is_file():
                    info['bytes'] += entry.stat().st_size
        for run in runs:
            r = scan_run(run)
            info['bytes'] += r['bytes']
            info['frames'] += r['frames']
            if r['frames'] == 0:
                info['empty'] += 1
            elif r['frames'] < min_frames:
                info['short'] += 1
            else:
                info['complete'] += 1
                info['frames_per_run'].append(r['frames'])
                if r['is_branch']:
                    info['branches'] += 1
                else:
                    info['base'] += 1
        per_mesh.append(info)

    return {'per_mesh': per_mesh, 'aux_bytes': aux_bytes, 'n_meshes': len(meshes)}


def report(data_dir: Path, min_frames: int, per_mesh: bool) -> None:
    if not data_dir.is_dir():
        print(f'{data_dir}: not a directory', file=sys.stderr)
        return

    st = scan_dataset(data_dir, min_frames)
    m = st['per_mesh']
    if not m:
        print(f'{data_dir}: no run_* directories found')
        return

    tot = Counter()
    for i in m:
        for k in ('complete', 'empty', 'short', 'branches', 'base', 'frames', 'bytes'):
            tot[k] += i[k]
    tot['bytes'] += st['aux_bytes']
    fpr = [f for i in m for f in i['frames_per_run']]
    total_runs = tot['complete'] + tot['empty'] + tot['short']

    print(f'\n{data_dir}')
    print('=' * max(len(str(data_dir)), 62))
    print(f'  total size on disk       {human(tot["bytes"])}')
    print(f'  geometries (meshes)      {st["n_meshes"]:,}')
    print()
    print(f'  run directories          {total_runs:,}')
    print(f'    complete (>= {min_frames} frames) {tot["complete"]:,}'
          f'   ({100 * tot["complete"] / max(total_runs, 1):.1f}%)')
    if tot['short']:
        print(f'    short (1..{min_frames - 1} frames)     {tot["short"]:,}')
    if tot['empty']:
        print(f'    empty (no frames)      {tot["empty"]:,}'
              f'   ({100 * tot["empty"] / max(total_runs, 1):.1f}%)')
    print()
    print(f'  of the complete runs:')
    print(f'    base trajectories      {tot["base"]:,}')
    print(f'    branches               {tot["branches"]:,}'
          f'   ({100 * tot["branches"] / max(tot["complete"], 1):.1f}%)')
    print()
    print(f'  total frames             {tot["frames"]:,}')
    if fpr:
        fpr.sort()
        print(f'    per complete run       min {fpr[0]}  median {fpr[len(fpr) // 2]}  '
              f'mean {sum(fpr) / len(fpr):.1f}  max {fpr[-1]}')
    if tot['frames']:
        print(f'    mean frame size        {human(tot["bytes"] / tot["frames"])}')

    if per_mesh:
        print(f'\n  {"mesh":<24}{"runs":>7}{"cmplt":>7}{"branch":>8}'
              f'{"frames":>9}{"size":>12}')
        for i in sorted(m, key=lambda x: -x['frames']):
            runs_i = i['complete'] + i['empty'] + i['short']
            print(f'  {i["name"][:23]:<24}{runs_i:>7,}{i["complete"]:>7,}'
                  f'{i["branches"]:>8,}{i["frames"]:>9,}{human(i["bytes"]):>12}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data_dirs', nargs='+', type=Path)
    ap.add_argument('--min-frames', type=int, default=2,
                    help='frames needed for a run to count as complete (default 2)')
    ap.add_argument('--per-mesh', action='store_true', help='per-geometry breakdown')
    args = ap.parse_args()
    for d in args.data_dirs:
        report(d, args.min_frames, args.per_mesh)
    print()


if __name__ == '__main__':
    main()
