"""gen_pair.py — a matched switching / non-switching corpus pair.

The comparison this exists to make fair
---------------------------------------
"Does resampling the hidden physics mid-trajectory help?" is only answerable if
the two corpora differ in THAT and nothing else.  Generating each arm with its
own sampler does not give you that: contexts are drawn from continuous specs, so
every draw is unique, and a switching run with S segments touches S contexts
while a frozen one touches 1.  Matching trajectory counts then compares
"switching" against "S times fewer distinct systems", and the result is
uninterpretable.

So everything is drawn ONCE, up front, into a plan:

    * K contexts (the fluid-parameter pool), shared by both arms verbatim;
    * one freestream/IC per trajectory, shared by both arms;
    * the save_t schedule, shared by both arms;
    * the meshes themselves, generated once and reused by both arms.

Why dt is FIXED by default
-------------------------
A varying frame interval cannot be read off a single frame -- it is recoverable
only from how far the field moved between frames.  So a corpus with variable dt
*forces* the encoder to read dynamics, in BOTH arms, whatever else it learns.
That silently answers part of the question being asked: a frozen-arm encoder
that scores well might be inferring physics from dynamics, or it might be
keying on static appearance and reading dynamics only for dt.  The two are no
longer distinguishable.

Holding dt constant across the whole pair removes that mandatory dynamic
signal, and the arms separate cleanly:

  switching  mesh and freestream are constant within a trajectory while the
             context changes, so appearance cannot identify the context.
             Dynamics are the ONLY route.
  frozen     (mesh, freestream, context) is a fixed triple per trajectory, so
             the static route is fully available and nothing forces the
             dynamic one.

Constant dt is therefore the condition most favourable to the static-feature
shortcut, which is what makes a frozen-arm failure evidence rather than noise.

Note the stride does not smuggle dt variance back in: data.py gives stride
augmentation only to legacy corpora without a per-segment save_t
(dt_stride_max = 1 if has_save_t), and these runs record save_t, so they train
at stride 1 and dt == save_t exactly.  One consequence: ContextProbe's log_dt
head gets a zero-variance target, so it learns the constant immediately,
applies no gradient pressure and its R^2 is undefined.  Ignore that head here
(or zero its weight) and read the ctx / visc heads, which is the measurement
this pair exists to make.

Both arms then DEAL from that plan rather than sampling:

    frozen     trajectory t uses context p[t] for all of its segments
    switching  the T*S segment slots are dealt from a deck holding each context
               exactly T*S/K times, shuffled, and never repeated within one
               trajectory

With the default K = n_trajectories this gives both arms the same K contexts and
the same S segments (hence the same frame count) per context.  The only
remaining difference is the one under test:

    are a context's segments consecutive within one trajectory, entered from its
    own history -- or scattered across several, entered from states evolved
    under other physics?

Two differences are NOT removed, because they cannot be at fixed chain length,
and the paper should name them rather than hide them:

  * a context is observed under 1 freestream in the frozen arm and up to S in
    the switching arm.  Matching that would need S times more trajectories per
    context, which breaks the frame match.  (Distinct STARTING STATES per
    context are S in both arms -- it is the boundary condition that differs.)
  * frozen segments start on-manifold for their own physics; switching segments
    start off-manifold.  That is the treatment.

Solver failures
---------------
Some drawn contexts are too stiff for some initial conditions and the run
diverges.  This does not land symmetrically: segment 0 is identical in both
arms, but a later segment can blow up in one arm and not the other (and the
switching arm meets each context under S initial conditions rather than 1, so it
gets more chances to find a bad pairing).  Left alone that silently unmatches the
corpora, so `reconcile` drops any trajectory that did not complete in BOTH arms
from BOTH arms, and records it so a later `generate` does not resurrect it.
Plan ~5% more trajectories than you need.

Dropping costs the exact per-context balance -- a dropped trajectory removes S
uses of one context in the frozen arm but one use of each of S contexts in the
switching arm.  Totals stay equal and `verify` reports the residual imbalance
with numbers, so it can be stated rather than assumed away.

Parallelism and robustness
--------------------------
The plan phase does every random draw and every gmsh call; the generate phase
does none, so shards never coordinate and the corpus does not depend on how it
was split.  Shards are by trajectory, which is the unit of chaining.  A
trajectory is marked complete only once all its segments are on disk; a partial
one is wiped and redone, because a chain cannot resume mid-way without the
in-memory handoff state.  Both arms, and any number of shards, can run at once.

    # 1. plan once (draws, meshes) -- cheap, single process
    python fvm_gen_tests/gen_pair.py plan --config fvm_gen/gen_alternating.json \
        --out /path/data/pair_A --trajectories 128 --segments 4

    # 2. generate, in as many processes as you like
    python fvm_gen_tests/gen_pair.py generate --out /path/data/pair_A \
        --arm switching --shard 0 --num-shards 8
    python fvm_gen_tests/gen_pair.py generate --out /path/data/pair_A \
        --arm frozen --shard 0 --num-shards 8

    # 3. drop trajectories that did not survive in both arms
    python fvm_gen_tests/gen_pair.py reconcile --out /path/data/pair_A

    # 4. prove the pair is matched before training on it
    python fvm_gen_tests/gen_pair.py verify --out /path/data/pair_A
"""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'fvm_gen'))
import run_gen as RG                                        # noqa: E402

PLAN_VERSION = 1
ARMS = ('switching', 'frozen')
_DONE = '.trajectory_complete'


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _spec_centre(spec: dict) -> float:
    """The middle of a save_t spec, on the spec's own scale.

    Used as the fixed dt when none is given, so the constant-dt pair sits at the
    centre of the interval the training corpus samples rather than at an end of
    it.
    """
    if 'dist' not in spec and ('values' in spec or 'choices' in spec):
        raise SystemExit('save_t_spec is categorical; pass --save-t explicitly')
    d = spec.get('dist', 'lognormal')
    if d == 'loguniform':
        return float(np.sqrt(spec['low'] * spec['high']))
    if d == 'uniform':
        return float(0.5 * (spec['low'] + spec['high']))
    if d == 'lognormal':
        return float(np.exp(spec['mean']))
    raise SystemExit(f"cannot centre save_t_spec dist '{d}'; pass --save-t")


def _deal_frozen(T: int, K: int, rng) -> np.ndarray:
    """One context per trajectory, each context used T/K times exactly.

    Dealt from a balanced deck rather than sampled i.i.d.: with T=2048, K=64,
    i.i.d. sampling leaves a per-context count with sd ~5.6 around 32, i.e. the
    arms would differ by up to ~17% in frames per context for no reason.
    """
    deck = np.repeat(np.arange(K), T // K)
    rng.shuffle(deck)
    return deck


def _deal_switching(T: int, S: int, K: int, rng,
                    max_passes: int = 200, max_probe: int = 400) -> np.ndarray:
    """[T, S] context ids: each context used exactly T*S/K times, and never
    twice within one trajectory.

    Distinctness is enforced, not just hoped for.  A context appearing twice in
    a trajectory is a segment boundary with no switch (if adjacent) or a second
    look at the same physics under the same freestream (if not) -- either way it
    weakens the arm that is supposed to embody switching, and a plain shuffle
    produces them often (~2 trajectories in 8 at S=4, K=8).

    Repair is by SWAPPING slots, which preserves the multiset, so the exact
    per-context counts hold however many swaps it takes.  The resulting design
    is clean enough to state in one line: each of the K contexts appears in
    T*S/K different trajectories, at most once in each.
    """
    deck = np.repeat(np.arange(K), T * S // K)
    rng.shuffle(deck)
    a = deck.reshape(T, S)
    if S < 2:
        return a
    rows = [set(r) for r in a.tolist()]        # kept in step with `a`
    for _ in range(max_passes):
        dupes = [(t, s) for t in range(T) for s in range(S)
                 if a[t, s] in a[t, :s]]
        if not dupes:
            return a
        for t, s in dupes:
            c = int(a[t, s])
            for _ in range(max_probe):
                u, v = int(rng.integers(T)), int(rng.integers(S))
                d = int(a[u, v])
                if u == t or d == c:
                    continue
                # The swap must not create a duplicate at either end.
                if d in rows[t] or c in rows[u]:
                    continue
                a[t, s], a[u, v] = d, c
                rows[t] = set(a[t].tolist())
                rows[u] = set(a[u].tolist())
                break
    raise SystemExit(
        f'could not make every trajectory use {S} distinct contexts after '
        f'{max_passes} passes. The pool is too small for the chain length: '
        f'--contexts {K} with {T * S // K} uses each cannot be spread over '
        f'{T} trajectories without collisions. Raise --contexts.')


def _gen_mesh(cfg_json: dict, problem: str, n_colliders: int, mesh_over: dict):
    """One geometry, with the same retry policy as the other generators."""
    over = dict(mesh_over)
    if problem == 'ellipse':
        over['n_colliders'] = n_colliders
    for attempt in range(RG.MAX_MESH_TRIES):
        base = RG.apply_overrides(RG._make_base_cfg(problem), over)
        try:
            Xs, tri, ae, bm, etag, be = RG._S['generate_mesh'](base)
            mesh = RG._S['FVMMesh2D'](Xs, tri, ae, bm, device=base.device)
            return mesh, etag, be
        except Exception as e:
            print(f'    !! mesh gen failed ({attempt + 1}/{RG.MAX_MESH_TRIES}): '
                  f'{type(e).__name__}: {e}')
    return None, None, None


def cmd_plan(args) -> None:
    cfg = json.loads(Path(args.config).read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    T, S, K = args.trajectories, args.segments, args.contexts or args.trajectories
    if T % K:
        raise SystemExit(f'--contexts {K} must divide --trajectories {T} so the '
                         f'frozen arm can give each context a whole number of '
                         f'trajectories')
    if (T * S) % K:
        raise SystemExit(f'--contexts {K} must divide trajectories*segments '
                         f'({T * S}) so the switching deck is balanced')
    # Each context needs T*S/K slots in DIFFERENT trajectories, so T*S/K <= T,
    # i.e. S <= K.  Checked here so an impossible design fails instantly with
    # the arithmetic rather than after the repair loop gives up.
    if S > K:
        raise SystemExit(f'--contexts {K} cannot be smaller than --segments {S}: '
                         f'each context needs {T * S // K} slots in distinct '
                         f'trajectories (only {T} exist) and no trajectory may '
                         f'use a context twice')
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f'{out} is not empty; pass --force to replace the plan '
                         f'(this invalidates any data already generated under it)')
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'big')
    rng = np.random.default_rng(seed)
    print(f'[plan] seed = {seed}   trajectories={T} segments={S} contexts={K}')
    print(f'       each context: {T * S // K} segment(s) in BOTH arms')

    problems = cfg['problem'] if isinstance(cfg['problem'], list) else [cfg['problem']]
    ctx_specs, ic_specs = cfg['context_param_specs'], cfg['ic_param_specs']
    if not ctx_specs or not ic_specs:
        raise SystemExit('--config needs both context_param_specs and ic_param_specs')

    # Every draw happens here and only here.
    contexts = [RG._sample_specs(ctx_specs, rng) for _ in range(K)]
    n_mesh = args.meshes
    mesh_meta = [{'mesh': m, 'problem': str(rng.choice(problems)),
                  'n_colliders': int(rng.integers(cfg.get('min_colliders', 1),
                                                  cfg.get('max_colliders', 4) + 1))}
                 for m in range(n_mesh)]
    save_t_spec = cfg['save_t_spec']
    fixed_dt = (args.save_t if args.save_t is not None
                else _spec_centre(save_t_spec)) if args.save_t_scope == 'fixed' \
        else None
    if fixed_dt is not None:
        print(f'       dt FIXED at {fixed_dt:.6g} for the whole pair '
              f'(no dt variance in either arm; ignore the probe log_dt head)')
    trajs = []
    frozen_ids = _deal_frozen(T, K, rng)
    switch_ids = _deal_switching(T, S, K, rng)
    for t in range(T):
        # save_t is shared by both arms whatever the scope.  Fixed by default,
        # so no mandatory dynamic signal masks the static-vs-dynamic question
        # (see "Why dt is FIXED by default").
        if fixed_dt is not None:
            st = [fixed_dt] * S
        elif args.save_t_scope == 'trajectory':
            st = [float(RG._sample_one('save_t', save_t_spec, rng))] * S
        else:
            st = [float(RG._sample_one('save_t', save_t_spec, rng)) for _ in range(S)]
        trajs.append({
            'traj': t,
            'mesh': t % n_mesh,              # round robin: equal meshes per arm
            'ic': RG._sample_specs(ic_specs, rng),
            'save_t': st,
            'ctx_switching': [int(c) for c in switch_ids[t]],
            'ctx_frozen': [int(frozen_ids[t])] * S,
        })

    out.mkdir(parents=True, exist_ok=True)
    mesh_root = out / 'meshes'
    mesh_root.mkdir(exist_ok=True)
    if not args.no_mesh:
        RG._import_solver()
        mesh_over = {k: cfg[k] for k in ('min_A', 'max_A', 'lnscale') if cfg.get(k)}
        dev = os.environ.get('FVM_DEVICE')
        if dev:
            mesh_over['device'] = dev
        for md in mesh_meta:
            mesh, etag, be = _gen_mesh(cfg, md['problem'], md['n_colliders'], mesh_over)
            if mesh is None:
                raise SystemExit(f'mesh {md["mesh"]} failed every attempt; rerun '
                                 f'the plan (geometry draws are random)')
            d = mesh_root / f'mesh_{md["mesh"]:03d}'
            d.mkdir(exist_ok=True)
            with open(d / 'shared_mesh.pkl', 'wb') as f:
                pickle.dump({'mesh': mesh, 'edge_tag': etag, 'bound_edgs': be}, f)
            md['sha256'] = _sha(d / 'shared_mesh.pkl')
            print(f'  [mesh {md["mesh"] + 1}/{n_mesh}] {md["problem"]}, '
                  f'{md["n_colliders"]} collider(s)')

    plan = {'version': PLAN_VERSION, 'seed': seed, 'source_config': str(Path(args.config).resolve()),
            'n_trajectories': T, 'n_segments': S, 'n_contexts': K,
            'steps_per_segment': cfg['steps_per_segment'],
            'save_t_scope': args.save_t_scope, 'save_t_spec': save_t_spec,
            'save_t_fixed': fixed_dt,
            'n_iter': cfg.get('n_iter', 200000), 'print_i': cfg.get('print_i', 500),
            'compile': cfg.get('compile', True),
            'meshes': mesh_meta, 'contexts': contexts, 'trajectories': trajs}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'\n[plan] -> {out / "plan.json"}\n'
          f'       {T * S} segments per arm, {T * S * (plan["steps_per_segment"] + 1)} '
          f'frames per arm')


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

def _run_dir(out: Path, arm: str, tr: dict, seg: int) -> Path:
    return out / arm / f'mesh_{tr["mesh"]:03d}' / f'run_t{tr["traj"]:05d}_s{seg:02d}'


def _complete(out: Path, arm: str, tr: dict, S: int, steps: int) -> bool:
    """Judged from the frames on disk, not from the marker file: a marker can
    outlive the data it vouches for (interrupted rmtree, copied tree, full
    disk), and every later stage trusts this answer."""
    return all(len(list(_run_dir(out, arm, tr, s).glob('t_*.npz'))) == steps + 1
               for s in range(S))


def _dropped(out: Path) -> set:
    f = out / 'dropped.json'
    if not f.exists():
        return set()
    return set(json.loads(f.read_text())['trajectories'])


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def _copy_atomic(src: Path, dst: Path) -> None:
    """Copy via a unique temp name + rename, so concurrent shards racing to
    stage the same mesh cannot leave a half-written pickle behind."""
    if dst.exists():
        return
    tmp = dst.with_suffix(f'.tmp{os.getpid()}')
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def cmd_generate(args) -> None:
    out = Path(args.out).resolve()
    plan = json.loads((out / 'plan.json').read_text())
    if plan['version'] != PLAN_VERSION:
        raise SystemExit(f'plan version {plan["version"]} != {PLAN_VERSION}')
    arm = args.arm
    S, steps = plan['n_segments'], plan['steps_per_segment']
    dropped = _dropped(out)
    mine = [t for t in plan['trajectories']
            if t['traj'] % args.num_shards == args.shard
            and t['traj'] not in dropped]
    print(f'[{arm}] shard {args.shard}/{args.num_shards}: '
          f'{len(mine)}/{len(plan["trajectories"])} trajectories'
          + (f' ({len(dropped)} dropped by reconcile, skipped)' if dropped else ''))
    if not args.dry_run:
        RG._import_solver()
    import torch                                            # noqa: E402

    n_ok = n_skip = n_fail = 0
    for ti, tr in enumerate(mine):
        md = plan['meshes'][tr['mesh']]
        mesh_dir = out / arm / f'mesh_{tr["mesh"]:03d}'
        run_dirs = [mesh_dir / f'run_t{tr["traj"]:05d}_s{s:02d}' for s in range(S)]
        # The marker alone is not enough: it can outlive the frames it vouches
        # for, and a silently short run would then be skipped forever and fail
        # verify instead.  Data wins over the marker here, as in reconcile.
        if (mesh_dir / f'.done_t{tr["traj"]:05d}').exists():
            if _complete(out, arm, tr, S, steps):
                n_skip += 1
                continue
            print(f'  marker for traj {tr["traj"]:05d} is stale (frames missing) '
                  f'— regenerating')
            (mesh_dir / f'.done_t{tr["traj"]:05d}').unlink()
        # A chain cannot resume mid-way (the handoff state is in memory), so a
        # partial trajectory is wiped rather than continued.
        for d in run_dirs:
            if d.exists():
                shutil.rmtree(d)
        ctx_ids = tr[f'ctx_{arm}']
        print(f'  [{arm} {ti + 1}/{len(mine)}] traj {tr["traj"]:05d} mesh {tr["mesh"]} '
              f'contexts {ctx_ids} save_t {[round(x, 4) for x in tr["save_t"]]}')
        if args.dry_run:
            n_ok += 1
            continue

        mesh_dir.mkdir(parents=True, exist_ok=True)
        _copy_atomic(out / 'meshes' / f'mesh_{tr["mesh"]:03d}' / 'shared_mesh.pkl',
                     mesh_dir / 'shared_mesh.pkl')
        with open(mesh_dir / 'shared_mesh.pkl', 'rb') as f:
            m = pickle.load(f)
        mesh, etag, be = m['mesh'], m['edge_tag'], m['bound_edgs']

        prims_prev, t_offset, failed = None, 0.0, False
        for s, run_dir in enumerate(run_dirs):
            ctx = plan['contexts'][ctx_ids[s]]
            save_t = tr['save_t'][s]
            dur = steps * save_t
            run_dir.mkdir(parents=True, exist_ok=True)
            # params.json = the system: physics + frame interval + geometry meta.
            # context_id is constant within a context, so it never splits a group.
            (run_dir / 'params.json').write_text(json.dumps(
                {**ctx, 'save_t': save_t, 'problem': md['problem'],
                 'mesh_uid': f'{tr["mesh"]:03d}', 'n_colliders': md['n_colliders'],
                 'context_id': ctx_ids[s]}, indent=2))
            (run_dir / 'ic.json').write_text(json.dumps(
                {**tr['ic'], 'traj_id': tr['traj'], 'traj_uid': f't{tr["traj"]:05d}',
                 'segment_id': s, 't_offset': round(t_offset, 6),
                 'duration': round(dur, 6), 'arm': arm}, indent=2))
            t_offset += dur

            base = RG.apply_overrides(RG._make_base_cfg(md['problem']),
                                      {'n_colliders': md['n_colliders']})
            cfg = RG.apply_overrides(base, {
                **ctx, **tr['ic'], 'plot': False, 'exact_interval': True,
                'save_t': save_t, 'end_t': dur, 'n_iter': plan['n_iter'],
                'print_i': plan['print_i'], 'compile': plan['compile'],
                'save_dir': str(run_dir)})
            phy = RG._S['FluidConstitution2D'](cfg, dim=2)
            bc_tags, us_default = RG._init_conds(cfg, mesh, etag, be, phy)
            if prims_prev is None:
                us_init = us_default
            else:
                # Continuity is enforced in PRIMITIVES: the observable fields must
                # not jump at a boundary, or the switch would be visible in a
                # single frame and nothing would need inferring.
                V, rho, T_ = prims_prev[:, :2], prims_prev[:, 2:3], prims_prev[:, 3:]
                mom, rho, Q = phy.primatives_to_state(V, rho, T_)
                us_init = torch.cat([mom, rho, Q], dim=-1)
            solver = RG._S['FVMEquation'](cfg, phy, mesh, cfg.N_comp, bc_tags,
                                          us_init=us_init)
            try:
                solver.solve()
            except Exception as e:
                print(f'    !! segment {s} failed ({type(e).__name__}: {e}) — '
                      f'abandoning trajectory {tr["traj"]:05d}')
                failed = True
                break
            n_have = len(list(run_dir.glob('t_*.npz')))
            if n_have != steps + 1:
                print(f'    !! segment {s}: {n_have}/{steps + 1} frames — '
                      f'abandoning trajectory {tr["traj"]:05d}')
                failed = True
                break
            prims_prev = solver.cells.get_values()[0].detach().clone()

        if failed:
            # Leave nothing partial: a half trajectory would be an unmatched run
            # in one arm, which is exactly what verify exists to catch.
            for d in run_dirs:
                shutil.rmtree(d, ignore_errors=True)
            n_fail += 1
            continue
        (mesh_dir / f'.done_t{tr["traj"]:05d}').write_text('')
        n_ok += 1

    print(f'\n[{arm}] shard {args.shard}: {n_ok} done, {n_skip} already present, '
          f'{n_fail} failed')
    if n_fail:
        print('  Re-run this shard to retry the failures (completed trajectories '
              'are skipped).')
        sys.exit(1)


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------

def cmd_reconcile(args) -> None:
    out = Path(args.out).resolve()
    plan = json.loads((out / 'plan.json').read_text())
    S, steps = plan['n_segments'], plan['steps_per_segment']
    trajs = plan['trajectories']
    already = _dropped(out)

    done = {arm: {t['traj'] for t in trajs if _complete(out, arm, t, S, steps)}
            for arm in ARMS}
    keep = done['switching'] & done['frozen']
    missing = {t['traj'] for t in trajs} - keep - already
    one_sided = (done['switching'] ^ done['frozen'])
    print(f'complete: switching {len(done["switching"])}, frozen {len(done["frozen"])}, '
          f'both {len(keep)} / {len(trajs)} planned')
    print(f'  one arm only : {len(one_sided)}  {sorted(one_sided)[:12]}'
          f'{" ..." if len(one_sided) > 12 else ""}')
    print(f'  neither arm  : {len(missing - one_sided)}')
    if not missing:
        print('\nNothing to reconcile.')
        return

    for t in (t for t in trajs if t['traj'] in missing):
        for arm in ARMS:
            for s in range(S):
                d = _run_dir(out, arm, t, s)
                if d.exists():
                    print(f'  drop {arm}/{d.parent.name}/{d.name}')
                    if not args.dry_run:
                        shutil.rmtree(d, ignore_errors=True)
            m = out / arm / f'mesh_{t["mesh"]:03d}' / f'.done_t{t["traj"]:05d}'
            if m.exists() and not args.dry_run:
                m.unlink()

    if args.dry_run:
        print(f'\n[dry run] would drop {len(missing)} trajector(ies) from both arms.')
        return
    (out / 'dropped.json').write_text(json.dumps(
        {'trajectories': sorted(already | missing),
         'note': 'did not complete in both arms; skipped by generate and absent '
                 'from both corpora'}, indent=2) + '\n')
    rate = len(already | missing) / len(trajs)
    print(f'\nDropped {len(missing)} trajector(ies); {len(keep)} matched pairs remain '
          f'({rate:.1%} of the plan lost).')
    if rate > 0.05:
        print('  NOTE: >5% lost. The surviving contexts are biased toward the ones '
              'that are numerically stable, and the per-context balance has moved. '
              'Consider re-planning with narrower context specs.')
    print('  Run `verify` next.')


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(args) -> None:
    out = Path(args.out).resolve()
    plan = json.loads((out / 'plan.json').read_text())
    S, steps, K = plan['n_segments'], plan['steps_per_segment'], plan['n_contexts']
    T = plan['n_trajectories']
    dropped = _dropped(out)
    want = (T - len(dropped)) * S
    problems, ok = [], True

    def chk(cond: bool, msg: str) -> None:
        nonlocal ok
        print(f'  [{"ok " if cond else "FAIL"}] {msg}')
        if not cond:
            ok = False
            problems.append(msg)

    runs = {arm: ({f'{d.parent.name}/{d.name}': d
                   for d in sorted((out / arm).glob('mesh_*/run_*'))
                   if any(d.glob('t_*.npz'))} if (out / arm).exists() else {})
            for arm in ARMS}
    print(f'plan: {T} trajectories x {S} segments, {K} contexts'
          + (f'; {len(dropped)} trajector(ies) dropped -> {want} segments expected'
             if dropped else f' -> {want} segments expected') + '\n')

    for arm in ARMS:
        chk(len(runs[arm]) == want,
            f'{arm} has all {want} expected segments (found {len(runs[arm])})')
    chk(set(runs['switching']) == set(runs['frozen']),
        'the two arms contain exactly the same run paths')
    shared = sorted(set(runs['switching']) & set(runs['frozen']))
    chk(len(shared) > 0, 'the pair is non-empty')
    if not shared:
        print('\nnothing to compare.')
        sys.exit(1)

    frames = {arm: {len(list(runs[arm][k].glob('t_*.npz'))) for k in shared}
              for arm in ARMS}
    chk(frames['switching'] == frames['frozen'] == {steps + 1},
        f'every run has exactly {steps + 1} frames (switching '
        f'{sorted(frames["switching"])}, frozen {sorted(frames["frozen"])})')

    # Geometry: the arms must share the mesh bit for bit, or the pixel mask the
    # model sees differs and nothing downstream is comparable.
    mesh_ok = True
    for md in plan['meshes']:
        ref = md.get('sha256')
        for arm in ARMS:
            f = out / arm / f'mesh_{md["mesh"]:03d}' / 'shared_mesh.pkl'
            if not f.exists():          # mesh unused after dropping
                continue
            mesh_ok &= (_sha(f) == ref) if ref else True
    chk(mesh_ok, 'shared_mesh.pkl identical to the plan in both arms')

    # Nothing was resampled: the physics on disk must BE the planned context,
    # value for value.  Checked against the pool rather than via context_id, so
    # a mislabelled or hand-edited run cannot slip through.
    ctx_exact, ctx_in_pool, use = True, True, {arm: {} for arm in ARMS}
    for arm in ARMS:
        for d in runs[arm].values():
            pj = json.loads((d / 'params.json').read_text())
            cid = pj.get('context_id')
            use[arm][cid] = use[arm].get(cid, 0) + 1
            if not isinstance(cid, int) or not 0 <= cid < K:
                ctx_in_pool = False
                continue
            want_ctx = plan['contexts'][cid]
            ctx_exact &= all(pj.get(k) == v for k, v in want_ctx.items())
    chk(ctx_in_pool, 'every run names a context from the planned pool')
    chk(ctx_exact, 'every run\'s physics equals the planned context exactly '
                   '(nothing resampled)')

    ic_same = st_same = True
    for k in shared:
        js = {arm: (json.loads((runs[arm][k] / 'params.json').read_text()),
                    json.loads((runs[arm][k] / 'ic.json').read_text()))
              for arm in ARMS}
        a, b = js['switching'], js['frozen']
        ic_same &= {x: y for x, y in a[1].items() if x != 'arm'} == \
                   {x: y for x, y in b[1].items() if x != 'arm'}
        st_same &= a[0]['save_t'] == b[0]['save_t']
    chk(ic_same, 'freestream/IC and lineage identical for every matched run')
    chk(st_same, 'save_t identical for every matched run')
    if plan.get('save_t_fixed') is not None:
        seen = {json.loads((d / 'params.json').read_text())['save_t']
                for arm in ARMS for d in runs[arm].values()}
        chk(seen == {plan['save_t_fixed']},
            f'dt is the single planned constant {plan["save_t_fixed"]:.6g} across '
            f'both arms (found {len(seen)} distinct value(s))')
    chk(sum(use['switching'].values()) == sum(use['frozen'].values()),
        f'same total segments per arm ({sum(use["switching"].values())} vs '
        f'{sum(use["frozen"].values())})')

    # The treatment, and only the treatment.
    per = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for d in runs[arm].values():
            ic = json.loads((d / 'ic.json').read_text())
            cid = json.loads((d / 'params.json').read_text())['context_id']
            per[arm].setdefault(ic['traj_id'], {})[ic['segment_id']] = cid
    multi = sum(1 for v in per['switching'].values() if len(set(v.values())) == S)
    chk(multi == len(per['switching']),
        f'every switching trajectory uses {S} distinct contexts '
        f'({multi}/{len(per["switching"])})')
    chk(all(len(set(v.values())) == 1 for v in per['frozen'].values()),
        'every frozen trajectory uses exactly 1 context')

    # Reported, not asserted.  Both are exactly balanced as planned and drift
    # only where reconcile dropped trajectories: a drop costs the frozen arm S
    # uses of one context and the switching arm one use of each of S.  The
    # distinct-context counts differ for the same reason and cannot be equal
    # once anything is dropped, so stating the residual is the honest move.
    dev = {c: use['switching'].get(c, 0) - use['frozen'].get(c, 0)
           for c in set(use['switching']) | set(use['frozen'])}
    worst = max((abs(v) for v in dev.values()), default=0)
    if worst == 0 and set(use['switching']) == set(use['frozen']):
        chk(True, 'segments-per-context identical between arms (exactly balanced)')
    else:
        print(f'  [note] contexts touched: switching {len(use["switching"])}, '
              f'frozen {len(use["frozen"])} of {K}; segments-per-context differs '
              f'for {sum(1 for v in dev.values() if v)}/{len(dev)} contexts by at '
              f'most {worst}. Caused by the {len(dropped)} dropped trajector(ies); '
              f'totals are equal. State it in the paper.')

    print()
    if ok:
        print('WATERTIGHT: same meshes, same ICs, same save_t, same context pool, '
              'same frame counts.\n'
              'The arms differ only in how contexts are assigned to segments.')
    else:
        print(f'{len(problems)} problem(s); the pair is NOT matched.')
        if any('expected segments' in m or 'same run paths' in m for m in problems):
            print('  If trajectories failed in the solver, run `reconcile` to drop '
                  'them from both arms, then verify again.')
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('plan', help='draw everything once (and the meshes)')
    p.add_argument('--config', required=True,
                   help='gen_alternating-style JSON supplying the DISTRIBUTIONS '
                        '(context_param_specs, ic_param_specs, save_t_spec, '
                        'steps_per_segment, problem, collider range)')
    p.add_argument('--out', required=True)
    p.add_argument('--trajectories', type=int, required=True)
    p.add_argument('--segments', type=int, default=4)
    p.add_argument('--contexts', type=int, default=None,
                   help='size of the shared context pool (default: = '
                        '--trajectories, which gives both arms the same pool AND '
                        'the same segments per context). Must divide '
                        'trajectories and trajectories*segments.')
    p.add_argument('--meshes', type=int, default=8)
    p.add_argument('--save-t-scope', choices=('fixed', 'trajectory', 'segment'),
                   default='fixed',
                   help="fixed (default): ONE dt for the whole pair, so no "
                        "mandatory dynamic signal masks whether the frozen arm "
                        "leans on static features (see the module docstring). "
                        "trajectory: one interval per trajectory in both arms. "
                        "segment: the interval alternates in both arms, isolating "
                        "physics switching alone. All three give the two arms the "
                        "same intervals.")
    p.add_argument('--save-t', type=float, default=None,
                   help='the fixed dt for --save-t-scope fixed; defaults to the '
                        "centre of the config's save_t_spec")
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--no-mesh', action='store_true',
                   help='plan without generating geometry (inspection only)')
    p.add_argument('--force', action='store_true')
    p.set_defaults(func=cmd_plan)

    g = sub.add_parser('generate', help='run one arm, one shard (no sampling)')
    g.add_argument('--out', required=True)
    g.add_argument('--arm', choices=ARMS, required=True)
    g.add_argument('--shard', type=int, default=0)
    g.add_argument('--num-shards', type=int, default=1)
    g.add_argument('--dry-run', action='store_true')
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser('reconcile',
                       help='drop trajectories that did not complete in both arms')
    r.add_argument('--out', required=True)
    r.add_argument('--dry-run', action='store_true')
    r.set_defaults(func=cmd_reconcile)

    v = sub.add_parser('verify', help='prove the finished pair is matched')
    v.add_argument('--out', required=True)
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    if args.cmd == 'generate' and not (0 <= args.shard < args.num_shards):
        raise SystemExit(f'--shard must be in [0, {args.num_shards})')
    args.func(args)


if __name__ == '__main__':
    main()
