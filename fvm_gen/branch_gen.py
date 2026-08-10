"""
branch_gen.py — counterfactual context-branching augmentation for MFM.

The problem this solves
-----------------------
In the base grid (run_gen.py) every trajectory is a unique (freestream, physics)
draw with its own initial condition, and trajectories under different physics
*diverge*.  So there are almost no frame pairs that share a starting state but
differ in context — which means the model can, in principle, infer the context
from what a single frame *looks like* rather than from the dynamics, and there is
no clean "same state, different operator" signal to force it to use context.

What this does
--------------
For a state q taken from a base trajectory, it restarts the numerical solver from
q under SEVERAL different physics contexts, holding the freestream / BCs fixed:

    q --(physics B)--> q'_B
    q --(physics C)--> q'_C      (identical frame 0 = q)
    q --(physics D)--> q'_D

The solver saves t=0, so the shared starting frame q is preserved byte-for-byte
across contexts.  Two properties result, everywhere in state space (not just at
t=0, where the base grid already shares the freestream-fill IC):

  * anti-shortcut       — the same q maps to different next-frames per context,
                          so a context-blind model provably cannot fit the data.
  * demonstration regime — many source frames from one run share a freestream, so
                          branching them under one physics yields many
                          independent-IC trajectories of a single *system*, i.e.
                          the sibling trajectories the context encoder infers from.

The restart uses the exact cell-level solver state from the source `t_*.npz`
(NOT the lossy pixel render), rebuilt into conserved variables under the target
physics' EOS.  The freestream is held at the source run's, so the boundary
conditions stay consistent with the state (no spurious restart transient), and
the branch is a physically valid trajectory of the target system.  Off-manifold
states (a high-viscosity state evolved under low viscosity) are a *feature*: they
broaden the state distribution the operator must act on, doubling as the
off-manifold data the design's stage-4 fine-tune wants.

Output layout
-------------
Each branch is written as a run dir under its source mesh (so shared_mesh.pkl is
already there), tagged with

    params.json      = {target physics} ∪ {source freestream} ∪ {geometry meta}
    branch_meta.json = provenance (source run, source frame, physics before/after)

Because mfm/data.py groups runs by identical params.json, all branches sharing a
(source-freestream, target-physics) pair form ONE system with several ICs — the
demonstration regime — while branches of the same frame under DIFFERENT physics
land in different systems that share frame 0.  Branches under the source's own
physics group straight into the base run's system, upgrading it from a singleton
to a demonstration system.

Usage
-----
    python fvm_model/fvm_gen/branch_gen.py --data ../../data/fvm_gen_v2 \
        --branch-points 4 --contexts 4 --branch-frames 12

    # preview the plan (how many solves) without running the solver:
    python fvm_model/fvm_gen/branch_gen.py --data ../../data/fvm_gen_v2 --dry-run
"""

import argparse
import json
import os
import pickle
import secrets
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_gen as rg   # reuse the solver plumbing (_import_solver, apply_overrides, ...)


# Params split.  The context to infer is the PHYSICS; the freestream is the BC/IC
# that must stay consistent with the branched state; geometry meta is fixed by the
# mesh.  Anything not freestream/meta is treated as physics.
FREESTREAM_KEYS = ("rho_inf", "T_inf", "v_n_inf", "v_t_inf", "aoa_deg")
GEOM_META_KEYS  = ("problem", "mesh_uid", "n_colliders")
# Provenance/bookkeeping written by newer generators.  These are NOT solver physics —
# passing them to the config raises AttributeError — and a branch must not inherit
# them either: `context_id` labels the SOURCE run's system, but a branch is evolved
# under different physics and so belongs to a different one.
BOOKKEEPING_KEYS = ("context_id", "ic_id")

# Unknown-config-key names already reported, so the warning prints once per key.
_WARNED_UNKNOWN_KEYS: set[str] = set()


def load_run_params(run_dir: str) -> dict:
    """A run's full parameter record: params.json ∪ ic.json.

    Older generators put the freestream (rho_inf/T_inf/v_n_inf/aoa_deg) in
    params.json.  Newer ones write params.json = the CONTEXT (physics) only — so it is
    identical across runs of one system — and split the per-run freestream/IC draw into
    a separate `ic.json`.  Reading params.json alone therefore yields an EMPTY
    freestream on new-schema data, and the branch silently restarts under the solver's
    DEFAULT boundary conditions while its interior state came from a run with a very
    different freestream.  That inlet/interior mismatch is a violent transient — the
    solver NaNs on the first step (t=nan, avg_dt=nan) and then spins to the n_iter cap.
    Merging both files keeps the freestream consistent with the state, which is exactly
    what the branch design requires.
    """
    out: dict = {}
    for name in ("params.json", "ic.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                out.update(json.load(f))
    return out


def split_params(params: dict) -> tuple[dict, dict, dict]:
    """(physics, freestream, geom_meta) from a run's parameter record."""
    freestream = {k: params[k] for k in FREESTREAM_KEYS if k in params}
    meta       = {k: params[k] for k in GEOM_META_KEYS if k in params}
    physics    = {k: v for k, v in params.items()
                  if k not in FREESTREAM_KEYS and k not in GEOM_META_KEYS
                  and k not in BOOKKEEPING_KEYS}
    return physics, freestream, meta


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def find_mesh_dirs(root: str) -> list[str]:
    """Dirs holding a shared_mesh.pkl (flat single-mesh root or multi-mesh)."""
    if os.path.exists(os.path.join(root, "shared_mesh.pkl")):
        return [root]
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "shared_mesh.pkl")):
            out.append(d)
    return out


def base_run_dirs(mesh_dir: str) -> list[str]:
    """Base run dirs (exclude branches we may have written earlier)."""
    out = []
    for name in sorted(os.listdir(mesh_dir)):
        d = os.path.join(mesh_dir, name)
        if (os.path.isdir(d) and name.startswith("run")
                and not name.startswith("run_br")
                and os.path.exists(os.path.join(d, "params.json"))):
            out.append(d)
    return out


def frame_files(run_dir: str, first_frame: int) -> list[str]:
    fs = [f for f in os.listdir(run_dir)
          if f.startswith("t_") and f.endswith(".npz")]
    fs.sort(key=lambda f: float(f[2:-4]))
    return [os.path.join(run_dir, f) for f in fs[first_frame:]]


def infer_save_t(files: list[str]) -> float:
    """Frame interval from the first two saved timestamps."""
    if len(files) < 2:
        return 0.01
    t0 = float(os.path.basename(files[0])[2:-4])
    t1 = float(os.path.basename(files[1])[2:-4])
    return round(t1 - t0, 8)


# ---------------------------------------------------------------------------
# State restart
# ---------------------------------------------------------------------------

def frame_to_us_init(frame_path: str, phy, device: str) -> torch.Tensor:
    """Cell-level saved primitives -> conserved us_init under the TARGET physics.

    The npz stores per-cell primitives (Vx, Vy, rho, T) — the exact solver state
    before the lossy pixel render.  Energy is re-derived with the target physics'
    C_v (primatives_to_state), so the conserved state is consistent with the EOS
    the branch will evolve under.
    """
    d = np.load(frame_path)
    prim = d["cell_primatives"].astype(np.float32) * d["prim_std"] + d["prim_mean"]
    prim = torch.from_numpy(prim).float().to(device)            # [n_cells, 4]
    V, rho, T = prim[:, :2], prim[:, 2:3], prim[:, 3:4]
    momentum, rho_, Q = phy.primatives_to_state(V, rho, T)
    return torch.cat([momentum, rho_, Q], dim=-1)


def run_one_branch(mesh_dict, problem, physics, freestream, frame_path,
                   save_dir, save_t, branch_frames, device, compile_step):
    """Restart from `frame_path` under (physics, freestream) and evolve a short
    trajectory into `save_dir`.  Mirrors run_gen's solver call, but supplies the
    initial state from a saved frame instead of the freestream-fill IC."""
    mesh       = mesh_dict["mesh"]
    edge_tag   = mesh_dict["edge_tag"]
    bound_edgs = mesh_dict["bound_edgs"]

    cfg = rg._make_base_cfg(problem)

    # Defensive: params.json schemas drift between generator versions (newer ones add
    # fields the solver config has never heard of).  apply_overrides sets attributes
    # directly, so a single unknown key raises AttributeError and kills EVERY branch.
    # Keep only real config fields, and say once which keys were dropped.
    unknown = sorted(k for k in physics if not hasattr(cfg, k))
    if unknown:
        new = [k for k in unknown if k not in _WARNED_UNKNOWN_KEYS]
        if new:
            print(f"    [warn] params.json key(s) not on the solver config, ignoring: {new}")
            _WARNED_UNKNOWN_KEYS.update(new)
        physics = {k: v for k, v in physics.items() if k not in unknown}

    overrides = {
        **physics, **freestream,
        "plot": False, "exact_interval": True, "compile": compile_step,
        "save_t": save_t,
        # end_t bounds the branch length precisely; n_iter is only a safety cap
        # (dt is adaptive, so step count per frame is not fixed).
        "end_t": (branch_frames - 1) * save_t,
        "n_iter": 1_000_000,
        "print_i": 1_000_000,
        "save_dir": save_dir,
        "device": device,
    }
    cfg = rg.apply_overrides(cfg, overrides)

    phy = rg._S["FluidConstitution2D"](cfg, dim=2)
    # bc_tags come from the freestream + edge tags; discard the freestream-fill
    # us_init it also returns — we restart from the saved state instead.
    bc_tags, _ = rg._init_conds(cfg, mesh, edge_tag, bound_edgs, phy)
    us_init = frame_to_us_init(frame_path, phy, device)

    solver = rg._S["FVMEquation"](cfg, phy, mesh, cfg.N_comp, bc_tags, us_init=us_init)
    solver.solve()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Counterfactual context-branching augmentation for MFM.")
    p.add_argument("--data", required=True,
                   help="base dataset root (flat or multi-mesh)")
    p.add_argument("--branch-points", type=int, default=4,
                   help="source frames sampled per source run")
    p.add_argument("--contexts", type=int, default=4,
                   help="target physics contexts per branch (incl. the source's own)")
    p.add_argument("--branch-frames", type=int, default=24,
                   help="frames saved per branch.  Must exceed the training config's "
                        "episode span so a branch is usable: mfm/data.py needs "
                        "h_history + 1 + rollout_horizon + ctx_len + 2 (~20 for the "
                        "defaults).  Longer = more solve time, so keep it just above that.")
    p.add_argument("--source-runs", type=int, default=None,
                   help="cap on source runs per mesh (default: all)")
    p.add_argument("--first-frame", type=int, default=20,
                   help="skip this many initial (transient) frames when sampling states")
    p.add_argument("--seed", type=int, default=None,
                   help="None (default) draws fresh entropy per invocation, so "
                        "parallel/repeated branch jobs sample DIFFERENT source "
                        "frames and physics. Pass an int only to replay.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the solver step (off by default — branches are "
                        "short, so compile overhead dominates)")
    p.add_argument("--order", choices=("interleave", "shuffle", "mesh"),
                   default="interleave",
                   help="job ordering across meshes.  'interleave' (default) "
                        "round-robins so every mesh gains branches at the same "
                        "rate and an interrupted sweep is still balanced; "
                        "'shuffle' is fully random; 'mesh' is the old "
                        "one-mesh-at-a-time behaviour")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many branch solves (time-boxed sweeps)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan (solve count per mesh) without solving")
    args = p.parse_args()

    root = os.path.abspath(args.data)
    seed = args.seed if args.seed is not None else __import__("secrets").randbits(32)
    print(f"[branch_gen] seed = {seed}")
    rng = np.random.default_rng(seed)
    mesh_dirs = find_mesh_dirs(root)
    if not mesh_dirs:
        raise SystemExit(f"No shared_mesh.pkl found under {root}")

    device = os.environ.get("FVM_DEVICE", "cpu")
    if not args.dry_run:
        rg._import_solver()            # populates rg._S and injects solver paths

    print(f"Branching: {len(mesh_dirs)} mesh(es) | branch_points={args.branch_points} "
          f"contexts={args.contexts} branch_frames={args.branch_frames} "
          f"device={device} order={args.order} dry_run={args.dry_run}\n")

    # ------------------------------------------------------------------
    # PLAN.  Enumerate every branch job first, WITHOUT touching the solver or
    # loading any mesh pickle.  Planning is cheap (params.json + a directory
    # listing per run), and doing it up front is what lets the jobs be ordered
    # across meshes rather than draining one mesh at a time.
    # ------------------------------------------------------------------
    jobs_by_mesh: "dict[str, list[dict]]" = {}
    for mesh_dir in mesh_dirs:
        runs = base_run_dirs(mesh_dir)
        if not runs:
            print(f"[{os.path.basename(mesh_dir)}] no base runs — skipped")
            continue

        # physics palette on this mesh: one physics dict per base run (dedup)
        run_params, physics_palette = {}, []
        for rd in runs:
            pj = load_run_params(rd)
            run_params[rd] = pj
            physics_palette.append(split_params(pj)[0])

        src_runs = runs
        if args.source_runs is not None and args.source_runs < len(runs):
            idx = rng.choice(len(runs), args.source_runs, replace=False)
            src_runs = [runs[i] for i in sorted(idx)]

        mesh_jobs: "list[dict]" = []
        for rd in src_runs:
            src_params = run_params[rd]
            src_physics, freestream, meta = split_params(src_params)
            problem = str(src_params.get("problem", "ellipse"))

            files = frame_files(rd, args.first_frame)
            usable = list(files)
            if len(usable) < 1:
                continue
            save_t = infer_save_t(files)

            # branch points: distinct source frames sharing this run's freestream
            n_bp = min(args.branch_points, len(usable))
            bp_idx = rng.choice(len(usable), n_bp, replace=False)
            branch_frames_paths = [usable[i] for i in sorted(bp_idx)]

            # target physics: the source's own + (contexts-1) other runs' physics
            others = [ph for ph in physics_palette if ph != src_physics]
            n_other = min(args.contexts - 1, len(others))
            chosen = []
            if n_other > 0:
                oidx = rng.choice(len(others), n_other, replace=False)
                chosen = [others[i] for i in oidx]
            targets = [src_physics] + chosen        # index 0 = self (native demo)

            # Mirror the source layout.  New-schema runs keep params.json = CONTEXT
            # (physics) only and put the freestream in ic.json, which is what makes
            # runs sharing a physics group into ONE system with several ICs.  Folding
            # the freestream into a branch's params.json would split that system per
            # freestream and destroy the demonstration regime, so only do it for
            # old-schema sources, which grouped on the combined record.
            src_has_ic = os.path.exists(os.path.join(rd, "ic.json"))

            for ci, target_physics in enumerate(targets):
                branch_params = ({**target_physics, **meta} if src_has_ic
                                 else {**target_physics, **freestream, **meta})
                for bi, frame_path in enumerate(branch_frames_paths):
                    mesh_jobs.append(dict(
                        mesh_dir=mesh_dir, rd=rd, problem=problem,
                        src_physics=src_physics, target_physics=target_physics,
                        freestream=freestream, branch_params=branch_params,
                        frame_path=frame_path, save_t=save_t,
                        ci=ci, bi=bi, is_self=(ci == 0), src_has_ic=src_has_ic,
                    ))
        if mesh_jobs:
            jobs_by_mesh[mesh_dir] = mesh_jobs
            print(f"[{os.path.basename(mesh_dir)}] {len(set(j['rd'] for j in mesh_jobs))} "
                  f"source run(s) -> {len(mesh_jobs)} branch job(s)")

    if not jobs_by_mesh:
        raise SystemExit("No branch jobs planned — check --data and --first-frame.")

    # ------------------------------------------------------------------
    # ORDER.  The generator used to run every job for mesh 1, then mesh 2, ...
    # Each job is a full solver run, so a sweep that is interrupted (or simply
    # slower than expected) produced branches for the first mesh ONLY, leaving
    # the rest of the geometries with none.  Interleaving round-robin means the
    # branches are spread evenly over meshes at every point in time, so stopping
    # early still yields a balanced dataset.
    # ------------------------------------------------------------------
    for mj in jobs_by_mesh.values():
        rng.shuffle(mj)                                  # random within a mesh

    order = args.order
    if order == "mesh":
        jobs = [j for mj in jobs_by_mesh.values() for j in mj]
    elif order == "shuffle":
        jobs = [j for mj in jobs_by_mesh.values() for j in mj]
        rng.shuffle(jobs)
    else:                                                # "interleave" (default)
        buckets = list(jobs_by_mesh.values())
        rng.shuffle(buckets)                             # no fixed mesh precedence
        jobs = []
        for i in range(max(len(b) for b in buckets)):
            for b in buckets:
                if i < len(b):
                    jobs.append(b[i])

    if args.limit is not None and args.limit < len(jobs):
        jobs = jobs[:args.limit]
        print(f"\n--limit {args.limit}: truncating to the first {len(jobs)} job(s)")

    n_mesh = len(jobs_by_mesh)
    print(f"\nPlanned {len(jobs)} branch job(s) over {n_mesh} mesh(es) "
          f"in '{order}' order.")
    if args.dry_run:
        per = {}
        for j in jobs:
            per[os.path.basename(j["mesh_dir"])] = per.get(
                os.path.basename(j["mesh_dir"]), 0) + 1
        for k in sorted(per):
            print(f"    {k:<28}{per[k]:>6}")
        print(f"\nPlanned {len(jobs)} branch solve(s).")
        return

    # ------------------------------------------------------------------
    # EXECUTE.  Mesh pickles are cached, so interleaving costs nothing beyond
    # the first load of each mesh (the solver is rebuilt per branch regardless).
    # ------------------------------------------------------------------
    mesh_cache: "dict[str, tuple]" = {}

    def get_mesh(mesh_dir):
        if mesh_dir not in mesh_cache:
            with open(os.path.join(mesh_dir, "shared_mesh.pkl"), "rb") as f:
                md = pickle.load(f)
            # branches must run on the device the mesh tensors live on
            mesh_cache[mesh_dir] = (md, str(md["mesh"].vertices.device))
        return mesh_cache[mesh_dir]

    total_branches = 0
    done_per_mesh: "dict[str, int]" = {}
    for n, j in enumerate(jobs, 1):
        mesh_dir = j["mesh_dir"]
        mesh_dict, dev = get_mesh(mesh_dir)
        mesh_name = os.path.basename(mesh_dir)
        src_tag = os.path.basename(j["rd"]).replace("run_", "")[:8]
        uid = secrets.token_hex(3)
        run_name = f"run_br_{src_tag}_c{j['ci']}_b{j['bi']}_{uid}"
        save_dir = os.path.join(mesh_dir, run_name)
        src_frame_t = float(os.path.basename(j["frame_path"])[2:-4])

        print(f"[{n}/{len(jobs)}] {mesh_name} {run_name}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "params.json"), "w") as f:
            json.dump(j["branch_params"], f, indent=2)
        # New-schema layout: the freestream lives beside params.json so the
        # branch is a fresh IC of the target-physics system, not its own.
        if j["src_has_ic"] and j["freestream"]:
            with open(os.path.join(save_dir, "ic.json"), "w") as f:
                json.dump(j["freestream"], f, indent=2)
        with open(os.path.join(save_dir, "branch_meta.json"), "w") as f:
            json.dump({
                "source_run": os.path.basename(j["rd"]),
                "source_frame_t": src_frame_t,
                "source_physics": j["src_physics"],
                "target_physics": j["target_physics"],
                "freestream": j["freestream"],
                "is_self_physics": j["is_self"],
            }, f, indent=2)
        try:
            run_one_branch(mesh_dict, j["problem"], j["target_physics"],
                           j["freestream"], j["frame_path"], save_dir,
                           j["save_t"], args.branch_frames, dev, args.compile)
            total_branches += 1
            done_per_mesh[mesh_name] = done_per_mesh.get(mesh_name, 0) + 1
        except Exception as e:      # one bad branch must not kill the sweep
            print(f"    !! branch failed ({type(e).__name__}: {e}) — "
                  f"removing {run_name}")
            import shutil
            shutil.rmtree(save_dir, ignore_errors=True)

    print(f"\nDone. Wrote {total_branches} branch trajectories "
          f"over {len(done_per_mesh)} mesh(es):")
    for k in sorted(done_per_mesh):
        print(f"    {k:<28}{done_per_mesh[k]:>6}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)   # type: ignore[union-attr]
    main()
