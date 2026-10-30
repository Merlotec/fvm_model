"""Wall-clock: FVM solver vs HFM model (with/without refiner) for one Delta_t.

The comparison the paper needs is "how long to advance the state by Delta_t":

  * the SOLVER integrates with an ERROR-CONTROLLED adaptive step of order
    1e-4 (cfg.dt is only the initial value), so a Delta_t of 0.1 costs
    ~10^3 steps -- its cost scales LINEARLY with Delta_t, and the step
    count is a property of the flow and tolerances, not a setting;
  * the MODEL emits the whole interval in ONE forward pass (the timestep is
    implicit in the context-frame spacing), so its cost is CONSTANT in Delta_t.

That asymmetry is the result; the speedup ratio is therefore itself a function
of Delta_t and is meaningless quoted without it.

Fairness rules applied here:
  * both sides run on the SAME device (--device), because a GPU model vs a CPU
    solver measures the hardware, not the method;
  * one-off setup is EXCLUDED from both (mesh generation + flux-matrix assembly
    for the solver, checkpoint load + context encode for the model): each is
    paid once per trajectory and amortises away over a rollout.  Both are
    reported separately;
  * solver cost is measured by integrating a real sim-time interval and scaling,
    never by assuming a nominal dt.

    python scripts/bench_solver_vs_model.py --checkpoint ../hfm/checkpoints/dynamics_a.ckpt \\
        --dt 0.1 --refiner ../hfm/checkpoints/refiner_a.pt
"""

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / 'fvm_gen'))
sys.path.insert(0, str(_HERE.parents[1] / 'hfm'))


def auto_device() -> torch.device:
    """Best device BOTH sides can actually use.

    MPS is deliberately excluded: the solver's sparse operators have no MPS
    kernels, so it would silently fall back and the two sides would no longer be
    measured on the same hardware -- which is the whole point of the comparison.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    return torch.device('cpu')


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def bench_solver(dt_target: float, probe_t: float, device: str, n_colliders: int):
    """Returns (setup_s, seconds_per_unit_simtime, n_cells, steps_per_unit, dt_stats).

    The integrator is ERROR-CONTROLLED (Butcher_adapt), so the step size is chosen
    per step, not configured: cfg.dt is only the initial value.  dt_stats reports
    the realised (min, mean, max) so the step count is never mistaken for a fixed
    setting -- and because the count is a property of the FLOW and the tolerances,
    the speedup ratio it implies is specific to this problem, not universal.
    """
    import run_gen
    run_gen._import_solver()
    S = run_gen._S

    t0 = time.perf_counter()
    base = run_gen.apply_overrides(run_gen._make_base_cfg('ellipse'),
                                   {'n_colliders': n_colliders, 'device': device})
    Xs, tri, ae, bm, etag, be = S['generate_mesh'](base)
    mesh = S['FVMMesh2D'](Xs, tri, ae, bm, device=base.device)
    cfg = run_gen.apply_overrides(base, {
        'plot': False, 'exact_interval': False, 'compile': False,
        'save_t': 1e9, 'end_t': probe_t, 'n_iter': 10 ** 9,
        'print_i': 10 ** 9, 'save_dir': None})
    phy = S['FluidConstitution2D'](cfg, dim=2)
    bc, us = S['init_conds_ellipses'](mesh, etag, be, phy, cfg)
    eq = S['FVMEquation'](cfg, phy, mesh, cfg.N_comp, bc, us_init=us)
    setup = time.perf_counter() - t0

    # Count steps so the measurement reports the real integration cost, not a
    # nominal dt: wrap the stepping function.
    dts = []
    inner = eq.t_solver._solve_step
    def counted(*a, **k):
        dts.append(float(eq.t_solver.dt))
        return inner(*a, **k)
    eq.t_solver._solve_step = counted

    dev = torch.device(device)
    _sync(dev)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        eq.solve()
    _sync(dev)                       # GPU work is async; time the completed run
    elapsed = time.perf_counter() - t0
    d = np.array(dts)
    dt_stats = (float(d.min()), float(d.mean()), float(d.max()))
    return setup, elapsed / probe_t, int(mesh.n_cells), len(d) / probe_t, dt_stats


def bench_model(ckpt_path: str, refiner_path, device: torch.device,
                refine_steps: int, reps: int):
    """Returns (load_s, ctx_encode_s, forward_s, refine_s|None, img)."""
    from hfm import build_model, ContextEncoder

    t0 = time.perf_counter()
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    model = build_model(cfg)
    model.load_state_dict(ck['model'], strict=False)
    ce = ContextEncoder(cfg)
    if 'context_encoder' in ck:
        ce.load_state_dict(ck['context_encoder'], strict=False)
    model, ce = model.to(device).eval(), ce.to(device).eval()
    load_s = time.perf_counter() - t0

    C, img = cfg.in_channels, cfg.img_size
    x = torch.randn(1, C, img, img, device=device)
    ctx_frames = [torch.randn(1, C, img, img, device=device)
                  for _ in range(cfg.n_context_frames)]
    mask = torch.ones(1, 1, img, img, device=device)

    def sync():
        _sync(device)

    with torch.no_grad():
        # context encode: ONCE per rollout, not per step
        ce(ctx_frames, pixel_mask=mask); sync()
        t0 = time.perf_counter()
        context = ce(ctx_frames, pixel_mask=mask); sync()
        ctx_s = time.perf_counter() - t0

        for _ in range(2):                       # warmup
            model(x, context, pixel_mask=mask)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            pred = model(x, context, pixel_mask=mask)
        sync()
        fwd_s = (time.perf_counter() - t0) / reps

        refine_s = None
        if refiner_path:
            from hfm.refiner import RefinerUNet, sample_detail
            rck = torch.load(refiner_path, map_location='cpu', weights_only=False)
            rcfg = rck['refiner_cfg']
            ref = RefinerUNet(rcfg).to(device).eval()
            ref.load_state_dict(rck.get('refiner_ema', rck['refiner']))
            sigma = rck['sigma_d'].to(device)
            full = vars(rcfg).get('n_ctx_tokens', 0) > 0
            cvec = context.float() if full else context.float().mean(dim=1)
            sample_detail(ref, pred, x, mask, cvec, sigma, n_steps=refine_steps)
            sync()
            t0 = time.perf_counter()
            for _ in range(reps):
                sample_detail(ref, pred, x, mask, cvec, sigma, n_steps=refine_steps)
            sync()
            refine_s = (time.perf_counter() - t0) / reps
    return load_s, ctx_s, fwd_s, refine_s, img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--refiner', default=None)
    ap.add_argument('--refine-steps', type=int, default=6)
    ap.add_argument('--dt', type=float, nargs='+', default=[0.1],
                    help='Delta_t values to report (space separated)')
    ap.add_argument('--device', default=None,
                    help='torch device for BOTH sides, e.g. cuda / xpu / cpu '
                         '(default: auto — cuda, else xpu, else cpu; mps is '
                         'never auto-picked, see auto_device)')
    ap.add_argument('--probe-t', type=float, default=0.02,
                    help='sim-time actually integrated, then scaled (default 0.02)')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--n-colliders', type=int, default=3)
    args = ap.parse_args()

    dev = torch.device(args.device) if args.device else auto_device()
    print(f'device (both sides): {dev}\n')

    print('== solver ==')
    setup_s, sec_per_t, n_cells, steps_per_t, dt_s = bench_solver(
        args.probe_t, args.probe_t, str(dev), args.n_colliders)
    print(f'  mesh {n_cells} cells | setup (once) {setup_s:.1f}s')
    print(f'  {sec_per_t:.2f} s per unit sim-time, {steps_per_t:.0f} steps per unit')
    print(f'  adaptive dt: min {dt_s[0]:.3g}  mean {dt_s[1]:.3g}  max {dt_s[2]:.3g} '
          f'({dt_s[2] / dt_s[0]:.1f}x spread) -- error-controlled, not fixed\n')

    print('== model ==')
    load_s, ctx_s, fwd_s, refine_s, img = bench_model(
        args.checkpoint, args.refiner, dev, args.refine_steps, args.reps)
    print(f'  grid {img}x{img} | load (once) {load_s:.1f}s | '
          f'context encode (once/rollout) {ctx_s * 1e3:.0f}ms')
    print(f'  forward {fwd_s * 1e3:.0f}ms/step'
          + (f' | refiner (+{args.refine_steps} NFE) {refine_s * 1e3:.0f}ms'
             if refine_s else ''))

    print(f'\n{"Delta_t":>8s} {"solver":>12s} {"steps":>8s} {"model":>10s} '
          f'{"speedup":>9s}' + (f' {"model+ref":>10s} {"speedup":>9s}' if refine_s else ''))
    for dt in args.dt:
        soln = sec_per_t * dt
        nst = steps_per_t * dt
        row = (f'{dt:8.3g} {soln:11.2f}s {nst:8.0f} {fwd_s * 1e3:9.0f}ms '
               f'{soln / fwd_s:8.0f}x')
        if refine_s:
            tot = fwd_s + refine_s
            row += f' {tot * 1e3:9.0f}ms {soln / tot:8.0f}x'
        print(row)
    print('\nSolver cost scales with Delta_t (its error-controlled step is set by '
          'the flow, not by Delta_t); the model emits any Delta_t in one forward, '
          'so its column is flat and the speedup is proportional to Delta_t.')


if __name__ == '__main__':
    main()
