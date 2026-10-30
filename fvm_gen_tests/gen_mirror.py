"""Mirror an existing corpus with one context parameter changed.

Re-runs every trajectory of a source dataset on the SAME mesh, from the SAME
initial condition, at the SAME save_t, with the same physics -- except for the
fields given on the command line.  The result is a matched pair of datasets that
differ in exactly one controlled way, which is what a comparison needs and what
resampling a fresh corpus can never give you (different meshes, different
freestreams, different frame rates all move at once).

The mesh is UNPICKLED from the source's shared_mesh.pkl rather than regenerated,
so the geometry -- and therefore the pixel mask the model sees -- is identical
bit for bit.  Output run directories keep the source's names, so run X in the
mirror is the counterpart of run X in the source and the two can be paired by
name alone (this is what infer.py --context-run relies on).

Trajectory chains are replayed in segment order, carrying the final primitives
across each switch exactly as gen_alternating does.  The mirrored physics makes
the trajectories diverge after segment 0 -- that is the point, not a bug: only
segment 0 starts from an identical state.

    # the held-out non-monotonic set, mirrored onto its nearest trained neighbour
    python fvm_gen_tests/gen_mirror.py \\
        /path/data/fvm_val_nonmonotonic \\
        --out /path/data/fvm_val_carreau \\
        --set visc_model=Carreau

    # also match the zero-shear viscosity (see --match-zero-shear)
    python fvm_gen_tests/gen_mirror.py <src> --out <dst> \\
        --set visc_model=Carreau --match-zero-shear
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'fvm_gen'))

import run_gen as RG                                       # noqa: E402

# Zero-shear value of `factor` for each model, i.e. mu_effective(0) / mu.  Used
# only by --match-zero-shear.  NonMonotonic is 1 by construction; the trained
# non-Newtonian models are not normalised that way, so a byte-identical `viscosity`
# would leave the mirror 1.4-1.9x more viscous at low shear.
_F_AT_ZERO_SHEAR = {
    'Newtonian':       lambda p: 1.0,
    'Carreau':         lambda p: 2.0 - p['visc_min_factor'],
    'HerschelBulkley': lambda p: 2.5,
    'PowerLaw':        lambda p: 1.0 / p['visc_min_factor'],
    'NonMonotonic':    lambda p: 1.0,
}


def _read(run: Path) -> "tuple[dict, dict]":
    return (json.loads((run / 'params.json').read_text()),
            json.loads((run / 'ic.json').read_text()))


def _trajectories(mesh_dir: Path) -> "list[list[Path]]":
    """Runs grouped into trajectories, each ordered by segment_id.

    Grouping is on traj_uid from ic.json rather than the directory name, so a
    corpus whose names were rewritten still chains correctly.
    """
    runs = [d for d in sorted(mesh_dir.iterdir())
            if d.is_dir() and (d / 'params.json').exists()
            and any(d.glob('t_*.npz'))]
    by_traj: dict = {}
    for d in runs:
        _, ic = _read(d)
        by_traj.setdefault(ic.get('traj_uid', d.name), []).append(d)
    for v in by_traj.values():
        v.sort(key=lambda d: _read(d)[1].get('segment_id', 0))
    return [by_traj[k] for k in sorted(by_traj)]


def _parse_set(items: "list[str]") -> dict:
    """--set k=v pairs; values are JSON when parseable, else plain strings."""
    out = {}
    for it in items:
        if '=' not in it:
            raise SystemExit(f'--set expects key=value, got {it!r}')
        k, v = it.split('=', 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def mirror(src: Path, dst: Path, overrides: dict, match_zero_shear: bool,
           n_iter: int, print_i: int, compile_: bool, dry_run: bool) -> None:
    RG._import_solver()
    mesh_dirs = [d for d in sorted(src.iterdir())
                 if d.is_dir() and (d / 'shared_mesh.pkl').exists()]
    if not mesh_dirs:
        raise SystemExit(f'{src}: no mesh dir with shared_mesh.pkl found')
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {'source': str(src), 'overrides': overrides,
                'match_zero_shear': match_zero_shear, 'runs': []}
    n_done = 0

    for mi, mesh_dir in enumerate(mesh_dirs):
        trajs = _trajectories(mesh_dir)
        if not trajs:
            print(f'[mesh {mi + 1}/{len(mesh_dirs)}] {mesh_dir.name}: no runs with '
                  f'frames, skipped')
            continue
        out_mesh = dst / mesh_dir.name
        out_mesh.mkdir(parents=True, exist_ok=True)
        # Reuse the geometry verbatim: same mesh object, same pixel mask, so any
        # difference downstream is the physics and nothing else.
        shutil.copy(mesh_dir / 'shared_mesh.pkl', out_mesh / 'shared_mesh.pkl')
        with open(mesh_dir / 'shared_mesh.pkl', 'rb') as f:
            md = pickle.load(f)
        mesh, edge_tag, bound_edgs = md['mesh'], md['edge_tag'], md['bound_edgs']
        print(f'[mesh {mi + 1}/{len(mesh_dirs)}] {mesh_dir.name}: '
              f'{len(trajs)} trajector(ies), mesh reused')

        for ti, segs in enumerate(trajs):
            prims_prev = None
            for si, run in enumerate(segs):
                params, ic = _read(run)
                new = {**params, **overrides}
                if match_zero_shear and 'visc_model' in overrides:
                    f_old = _F_AT_ZERO_SHEAR[params['visc_model']](params)
                    f_new = _F_AT_ZERO_SHEAR[new['visc_model']](new)
                    new['viscosity'] = params['viscosity'] * f_old / f_new
                out_run = out_mesh / run.name
                out_run.mkdir(parents=True, exist_ok=True)
                (out_run / 'params.json').write_text(json.dumps(new, indent=2))
                (out_run / 'ic.json').write_text(json.dumps(ic, indent=2))
                manifest['runs'].append({'mesh': mesh_dir.name, 'run': run.name,
                                         **new, **ic})

                changed = ', '.join(
                    f'{k}: {params.get(k)} -> {new[k]}' for k in sorted(new)
                    if params.get(k) != new[k])
                print(f'  [traj {ti + 1}/{len(trajs)} | seg {si + 1}/{len(segs)}] '
                      f'{run.name}  save_t={params["save_t"]:.3f}  {changed}')
                if dry_run:
                    continue

                # Everything the solver needs comes from the source run's own
                # records -- nothing is resampled.
                cfg_over = {k: v for k, v in new.items()
                            if k not in ('problem', 'mesh_uid', 'n_colliders')}
                base = RG.apply_overrides(
                    RG._make_base_cfg(params['problem']),
                    {'n_colliders': params.get('n_colliders', 1)})
                cfg = RG.apply_overrides(base, {
                    **cfg_over,
                    **{k: v for k, v in ic.items()
                       if k not in ('traj_id', 'traj_uid', 'segment_id',
                                    't_offset', 'duration')},
                    'plot': False, 'exact_interval': True,
                    'save_t': params['save_t'], 'end_t': ic['duration'],
                    'n_iter': n_iter, 'print_i': print_i, 'compile': compile_,
                    'save_dir': str(out_run)})
                phy = RG._S['FluidConstitution2D'](cfg, dim=2)
                bc_tags, us_default = RG._init_conds(cfg, mesh, edge_tag,
                                                     bound_edgs, phy)
                # Segment 0 starts from the identical IC; later segments continue
                # from THIS chain's own handoff, so the mirror diverges from the
                # source exactly as a different fluid should.
                if prims_prev is None:
                    us_init = us_default
                else:
                    V, rho, T = (prims_prev[:, :2], prims_prev[:, 2:3],
                                 prims_prev[:, 3:])
                    mom, rho, Q = phy.primatives_to_state(V, rho, T)
                    import torch
                    us_init = torch.cat([mom, rho, Q], dim=-1)
                solver = RG._S['FVMEquation'](cfg, phy, mesh, cfg.N_comp, bc_tags,
                                              us_init=us_init)
                try:
                    solver.solve()
                except Exception as e:
                    print(f'    !! segment failed ({type(e).__name__}: {e}) — '
                          f'aborting this trajectory')
                    break
                n_have = len(list(out_run.glob('t_*.npz')))
                n_want = len(list(run.glob('t_*.npz')))
                if n_have < n_want:
                    print(f'    !! {n_have}/{n_want} frames (n_iter cap?) — '
                          f'aborting this trajectory')
                    break
                n_done += 1
                prims_prev = solver.cells.get_values()[0].detach().clone()

    (dst / 'mirror_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\nDone. {n_done} segment(s) written to {dst}\n'
          f'Manifest: {dst / "mirror_manifest.json"}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('src', type=Path, help='source dataset root (holds mesh_*/ dirs)')
    ap.add_argument('--out', type=Path, required=True, help='destination root')
    ap.add_argument('--set', dest='sets', action='append', default=[],
                    metavar='KEY=VALUE',
                    help='override a params.json field (repeatable), e.g. '
                         '--set visc_model=Carreau')
    ap.add_argument('--match-zero-shear', action='store_true',
                    help='rescale `viscosity` so the mirror has the SAME zero-shear '
                         'effective viscosity as the source. The trained models are '
                         'not normalised to factor(0)=1 (Carreau gives 2-f_min, '
                         'HerschelBulkley 2.5), so without this the mirror is '
                         'uniformly more viscous and that confounds the comparison. '
                         'With it, params.json differs in viscosity too — state '
                         'which you used.')
    ap.add_argument('--n-iter', type=int, default=200000)
    ap.add_argument('--print-i', type=int, default=500)
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='plan only: write params/ic and print, run no solver')
    args = ap.parse_args()

    overrides = _parse_set(args.sets)
    if not overrides:
        raise SystemExit('nothing to change: pass at least one --set KEY=VALUE')
    if args.match_zero_shear and 'visc_model' in overrides:
        unknown = {overrides['visc_model']} - set(_F_AT_ZERO_SHEAR)
        if unknown:
            raise SystemExit(f'--match-zero-shear: no zero-shear factor known for '
                             f'{unknown}; add it to _F_AT_ZERO_SHEAR')
    mirror(args.src.resolve(), args.out.resolve(), overrides,
           args.match_zero_shear, args.n_iter, args.print_i,
           not args.no_compile, args.dry_run)


if __name__ == '__main__':
    main()
