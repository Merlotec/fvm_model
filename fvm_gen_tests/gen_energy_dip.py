"""Energy-dip probe dataset: does the context encoder carry the macro-state?

Copies ONE existing run directory and artificially lowers the energy of a single
frame -- the frame the model will step FROM (the autoregressive seed).  The
context frames before it and the ground truth after it are left untouched, so the
resulting rollout separates two hypotheses that ordinary evaluation cannot:

  * the context encoder genuinely encodes macro variables (energy level,
    freestream state): the rollout is pulled back UP toward the context's energy
    within a few steps, because the conditioning says "this flow is hotter/faster
    than your input frame";
  * the model just propagates its input frame and the context is decorative: the
    rollout stays at (or drifts from) the dipped level, since nothing in the input
    frame says the energy is wrong.

This is the drift question inverted: instead of waiting for autoregressive
error to accumulate, we inject a known macro error at step 0 and watch whether
the conditioning corrects it.

The dip (factor f < 1, per --mode):
    total    v *= sqrt(f),  T *= f   ->  kinetic AND internal energy density * f
                                         (rho untouched: mass is conserved, so the
                                         perturbation is purely energetic)
    kinetic  v *= sqrt(f)            ->  kinetic energy density * f
    thermal  T *= f                  ->  internal energy density * f

The dipped copy lands as a SIBLING of the source run (same mesh dir), so
shared_mesh.pkl and the renderer caches keep working and the run shows up in the
viewer and the report pipeline like any other.  Frame files are copied verbatim
except the dipped one; the pack files (frames_fp16.npy / frames_meta.npz), which
hold pre-packed UNDIPPED frame data, are deliberately not copied.

The dip frame index must equal the model's n_context_frames (default 5), because
the report pipeline pins `--seq-start 0` for dip runs: frames 0..n_context-1 are
then the context and frame n_context is the seed.  Regenerate with --frame-idx
if you evaluate a model with a different context length.

NOTE if your pipeline config passes --val-only in infer_args: the dipped run's
name hashes to its own train/val bucket, so it may be filtered out.  Drop
--val-only for dip rollouts (they are synthetic probes, not held-out data).

Usage:
    python fvm_gen_tests/gen_energy_dip.py /path/data/mesh_x/run_a000_s01_ab12cd34 \
        --dip 0.6 --mode total
    # -> /path/data/mesh_x/run_a000_s01_ab12cd34_edip060

Then point a report_pipeline rollout at the new directory with
"include_energy": true to get the energy-vs-step panel.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Solver primitive channel order (fvm_solver ds_saving/saving.py: [Vs, rho, T]).
PRIM_ORDER = ('Vx', 'Vy', 'rho', 'T')

# Pre-packed frame blobs (hfm/data.py PACK_FILE / PACK_META).  They duplicate the
# t_*.npz contents, so a copied pack would silently serve UNDIPPED frames to any
# consumer that prefers it over the npz files.
_SKIP_FILES = {'frames_fp16.npy', 'frames_meta.npz'}


def _frame_files(run_dir: Path) -> "list[Path]":
    files = [f for f in run_dir.iterdir()
             if f.name.startswith('t_') and f.name.endswith('.npz')]
    return sorted(files, key=lambda f: float(f.stem[2:]))


def _load_cv(run_dir: Path) -> Tuple[float, bool]:
    """C_v from params.json/ic.json (merged), or (1.0, False) when absent --
    only used for the human-readable energy printout, never for the dip itself."""
    params = {}
    for name in ('params.json', 'ic.json'):
        p = run_dir / name
        if p.exists():
            try:
                params.update(json.loads(p.read_text()))
            except Exception:
                pass
    try:
        return float(params['C_v']), True
    except (KeyError, TypeError, ValueError):
        return 1.0, False


def _energies(cells: np.ndarray, cv: float) -> Tuple[float, float]:
    """(mean kinetic, mean internal) energy density over cells [n, 4] physical."""
    vx, vy, rho, T = cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3]
    ke = 0.5 * rho * (vx * vx + vy * vy)
    ie = rho * cv * T
    return float(ke.mean()), float(ie.mean())


def dip_frame_npz(src: Path, dst: Path, factor: float, mode: str,
                  cv: float) -> "dict[str, float]":
    """Write `dst` = `src` with the energy dip applied; returns before/after stats.

    Mirrors the solver's Saver normalisation exactly: physical values are
    recovered with the stored per-frame mean/std, dipped, then re-z-scored under
    FRESH statistics computed from the dipped cells (std stored raw, clamped only
    for the division), and stored fp16.  bc_primatives keep their PHYSICAL values
    (the true boundary forcing is not part of the perturbation) but are re-scored
    under the new cell statistics, as the Saver would.
    """
    d = np.load(src)
    mean = d['prim_mean'].astype(np.float64)                     # [1, 4]
    std_safe = np.maximum(d['prim_std'].astype(np.float64), 1e-8)
    cells = d['cell_primatives'].astype(np.float64) * std_safe + mean
    bc = d['bc_primatives'].astype(np.float64) * std_safe + mean

    ke0, ie0 = _energies(cells, cv)
    if mode in ('total', 'kinetic'):
        cells[:, 0:2] *= np.sqrt(factor)
    if mode in ('total', 'thermal'):
        cells[:, 3] *= factor
    ke1, ie1 = _energies(cells, cv)

    new_mean = cells.mean(axis=0, keepdims=True)
    new_std = cells.std(axis=0, keepdims=True)
    new_std_safe = np.maximum(new_std, 1e-8)
    np.savez_compressed(
        dst,
        t=d['t'],
        prim_mean=new_mean.astype(np.float32),
        prim_std=new_std.astype(np.float32),
        cell_primatives=((cells - new_mean) / new_std_safe).astype(np.float16),
        bc_primatives=((bc - new_mean) / new_std_safe).astype(np.float16),
    )
    return {'ke_before': ke0, 'ie_before': ie0, 'ke_after': ke1, 'ie_after': ie1}


def make_dip_run(src_run: Path, factor: float, mode: str, frame_idx: int,
                 name: Optional[str] = None, force: bool = False) -> Path:
    """Copy `src_run` to a dipped sibling directory; returns the new path."""
    src_run = src_run.resolve()
    files = _frame_files(src_run)
    if not files:
        raise SystemExit(f'{src_run} holds no t_*.npz frames')
    if not (0 < factor):
        raise SystemExit(f'--dip must be positive (got {factor})')
    if frame_idx >= len(files):
        raise SystemExit(f'--frame-idx {frame_idx} out of range: run has only '
                         f'{len(files)} frames')

    if name is None:
        suffix = {'total': '', 'kinetic': '-kin', 'thermal': '-thm'}[mode]
        name = f'{src_run.name}_edip{int(round(factor * 100)):03d}{suffix}'
    dst_run = src_run.parent / name
    if dst_run.exists():
        if not force:
            raise SystemExit(f'{dst_run} already exists (use --force to replace)')
        shutil.rmtree(dst_run)
    dst_run.mkdir(parents=True)

    skipped = []
    for f in sorted(src_run.iterdir()):
        if not f.is_file():
            skipped.append(f.name + '/')
            continue
        if f.name in _SKIP_FILES:
            skipped.append(f.name)
            continue
        shutil.copy2(f, dst_run / f.name)
    if skipped:
        print(f'  not copied (stale for a dipped run): {", ".join(skipped)}')

    target = files[frame_idx]
    cv, cv_known = _load_cv(src_run)
    stats = dip_frame_npz(target, dst_run / target.name, factor, mode, cv)

    # Record the perturbation in params.json.  The report pipeline reads this to
    # pin --seq-start 0 and to state the dip in figure captions; the key is in its
    # _NON_FLUID_KEYS so it never leaks into the fluid-parameter caption text.
    params_path = dst_run / 'params.json'
    try:
        params = json.loads(params_path.read_text())
    except Exception:
        params = {}
    params['energy_dip'] = {
        'factor': factor,
        'mode': mode,
        'frame_index': frame_idx,
        'frame': target.name,
        'source_run': src_run.name,
    }
    params_path.write_text(json.dumps(params, indent=2))

    tot0 = stats['ke_before'] + stats['ie_before']
    tot1 = stats['ke_after'] + stats['ie_after']
    cv_note = '' if cv_known else '  (C_v not in params: printout assumes C_v=1)'
    print(f'  dipped {target.name} (frame {frame_idx}, mode={mode}, '
          f'factor={factor}){cv_note}')
    print(f'    kinetic  energy: {stats["ke_before"]:.6g} -> {stats["ke_after"]:.6g} '
          f'({stats["ke_after"] / max(stats["ke_before"], 1e-300):.4f}x)')
    print(f'    internal energy: {stats["ie_before"]:.6g} -> {stats["ie_after"]:.6g} '
          f'({stats["ie_after"] / max(stats["ie_before"], 1e-300):.4f}x)')
    print(f'    total    energy: {tot0:.6g} -> {tot1:.6g} '
          f'({tot1 / max(tot0, 1e-300):.4f}x)')
    print(f'  -> {dst_run}')
    return dst_run


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('src_run', type=Path,
                    help='existing run directory (holds t_*.npz frames)')
    ap.add_argument('--dip', type=float, default=0.6,
                    help='energy factor for the seed frame, e.g. 0.6 = dipped to '
                         '60%% of the true level (default 0.6)')
    ap.add_argument('--mode', choices=('total', 'kinetic', 'thermal'),
                    default='total',
                    help='which energy to dip: total (v*=sqrt(f), T*=f), '
                         'kinetic (v only) or thermal (T only); default total')
    ap.add_argument('--frame-idx', type=int, default=5,
                    help='0-based frame to dip; MUST equal the model\'s '
                         'n_context_frames so the dipped frame is the AR seed '
                         'under the pipeline\'s --seq-start 0 (default 5)')
    ap.add_argument('--name', default=None,
                    help='output directory name (default: '
                         '<src>_edip<pct>[-kin|-thm])')
    ap.add_argument('--force', action='store_true',
                    help='replace the output directory if it exists')
    args = ap.parse_args()

    if not args.src_run.is_dir():
        raise SystemExit(f'{args.src_run} is not a directory')
    if args.dip >= 1.0:
        print(f'  [note] --dip {args.dip} >= 1: this is an energy BOOST, not a '
              f'dip; the test works the same way, in the other direction')
    make_dip_run(args.src_run, args.dip, args.mode, args.frame_idx,
                 name=args.name, force=args.force)


if __name__ == '__main__':
    main()
