"""
General fluid-dataset generator for the FVM solver.

Uses the `fvm_solver` fork, which carries the non-Newtonian viscosity physics
cherry-picked from upstream (Maxzhu123) plus the fork's mesh-generation and
save-dir fixes.  Override the solver with FVM_SOLVER_DIR if needed.

Generates a broad spectrum of 2D compressible-flow trajectories for training a
general fluid model, sweeping over:

  * NONLINEARITY   — viscosity model {Newtonian, PowerLaw, Carreau, HerschelBulkley}
                     + its shape params (visc_n power-law index, visc_min_factor,
                     visc_gamma_scale).
  * PHYSICS        — viscosity, bulk viscosity, thermal conductivity, C_v, gamma.
  * BOUNDARY COND. — inflow speed v_n_inf, T_inf, rho_inf (set on the BC configs).
  * COLLIDERS      — a fresh random mesh (ellipse obstacles) per `n_meshes`, so the
                     geometry varies across the dataset, not just the physics.

Output (renderer / nomad-dfm compatible): for each mesh a directory holding a
`shared_mesh.pkl` and one `run_XXXX_<uid>/` per parameter draw, each containing the
solver's `t_*.npz` frames (cell_primatives / prim_mean / prim_std) + a `params.json`.

    FVM_DEVICE=cuda python fvm_model/fvm_gen/run_gen.py fvm_model/fvm_gen/gen.json

    # sample the sweep + write params.json/manifest WITHOUT running the solver:
    python fvm_model/fvm_gen/run_gen.py fvm_model/fvm_gen/gen.json --dry-run
"""

import argparse
import json
import os
import pickle
import secrets
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "data"))

_S: dict[str, Any] = {}     # lazily-populated handle on solver symbols


def _import_solver():
    """Add the FVM solver to sys.path and import what we need.  Imports are lazy so
    --dry-run needs only numpy.

    Default: the `fvm_solver` fork beside this repo (which carries the cherry-picked
    non-Newtonian physics and imports/runs cleanly).  Override with FVM_SOLVER_DIR.
    """
    if _S:
        return
    default_solver = os.path.abspath(os.path.join(_HERE, "..", "..", "fvm_solver"))
    solver_dir = os.environ.get("FVM_SOLVER_DIR", default_solver)
    if not os.path.isdir(solver_dir):
        raise SystemExit(f"FVM solver not found at {solver_dir}; set FVM_SOLVER_DIR.")
    for p in (solver_dir, os.path.join(solver_dir, "time_fvm")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from time_fvm.config_fvm import ConfigEllipse, ConfigNozzle, ViscosityModel
    from time_fvm.fvm_equation import FVMEquation, FluidConstitution2D
    from time_fvm.mesh_utils.fvm_mesh import FVMMesh2D
    from run_fvm import generate_mesh, init_conds_ellipses, init_conds_nozzle

    _S.update(ConfigEllipse=ConfigEllipse, ConfigNozzle=ConfigNozzle,
              ViscosityModel=ViscosityModel, FVMEquation=FVMEquation,
              FluidConstitution2D=FluidConstitution2D, FVMMesh2D=FVMMesh2D,
              generate_mesh=generate_mesh, init_conds_ellipses=init_conds_ellipses,
              init_conds_nozzle=init_conds_nozzle)


# ---------------------------------------------------------------------------
# Sweep configuration + sampling
# ---------------------------------------------------------------------------

_BC_PARAMS = {"rho_inf", "T_inf", "v_n_inf", "v_t_inf"}


def _sample_one(name: str, spec: dict, rng: np.random.Generator):
    """Draw a single value from a sampling spec (continuous or categorical)."""
    if "choices" in spec:
        return str(rng.choice(spec["choices"]))
    if "values" in spec:
        return float(rng.choice(spec["values"]))
    dist = spec.get("dist", "lognormal")
    if dist == "lognormal":
        return float(np.exp(rng.normal(spec["mean"], spec["std"])))
    if dist == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    if dist == "loguniform":
        return float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
    raise ValueError(f"param '{name}': unknown dist '{dist}'")


@dataclass
class GenConfig:
    problem: "Union[str, list]" = "ellipse"  # "ellipse" | "nozzle" | a list to mix per mesh
    n_meshes: int = 8                 # distinct collider geometries
    runs_per_mesh: int = 16           # LEGACY mode only: parameter draws per geometry
    seed: int = 42

    # --- in-context grid mode (preferred) --------------------------------------
    # The model must INFER the context (hidden physics) from observed transitions,
    # not memorise "what a frame under context X looks like".  That shortcut is
    # only foreclosed if the SAME starting frame appears under MANY contexts, so
    # its next frame is genuinely ambiguous without the context.
    #
    # The FVM initial condition depends only on the freestream/BC params
    # (rho_inf, T_inf, v_n_inf, aoa) and C_v — NOT on viscosity/gamma/thermal_cond/
    # bulk-viscosity/visc-model.  So we can hold the IC fixed and vary only the
    # hidden physics.  Per mesh we sample:
    #   * n_context physics draws  -> each is one SYSTEM (the thing to infer)
    #   * n_ic freestream draws    -> shared initial conditions
    # and run the full n_context x n_ic grid:
    #   - a column (fixed physics, varying IC) = one system with many ICs
    #     -> the demonstration regime (context from sibling trajectories)
    #   - a row (fixed IC, varying physics) = the SAME start frame under many
    #     contexts -> forces the model to actually use the context.
    #
    # params.json carries ONLY the context (physics) params, so it is identical
    # across a column and mfm/data.py groups the column into one multi-IC system
    # with no consumer-side change.  The per-run freestream/IC draw is written to
    # ic.json (which grouping ignores).  Leave BOTH of these empty to fall back to
    # the legacy per-run `param_specs` sampling.
    n_context: int = 12               # physics-parameter sets (systems) per mesh
    n_ic: int = 6                     # shared initial-condition seeds per mesh
    context_param_specs: dict[str, dict] = field(default_factory=dict)
    ic_param_specs: dict[str, dict] = field(default_factory=dict)

    # Solver run controls (applied to every run).
    save_t: float = 0.01              # sim-time between saved frames
    n_iter: int = 10000               # max solver iterations
    end_t: Optional[float] = None        # max sim time (optional; None => run n_iter steps)
    print_i: int = 500
    compile: bool = True              # torch.compile the step (fast on GPU)

    # Mesh sizing (obstacle/cell scale); passed straight to the config.
    min_A: Optional[float] = None
    max_A: Optional[float] = None
    lnscale: Optional[float] = None

    # Collider count is geometry, so it is drawn per mesh (inclusive range), not per run.
    min_colliders: int = 1
    max_colliders: int = 4

    # Per-run sampling.  Categorical specs use {"choices": [...]}, continuous use
    # {"dist": "lognormal"|"uniform"|"loguniform", ...} or {"values": [...]}.
    param_specs: dict[str, dict] = field(default_factory=dict)
    phys_overrides: dict[str, Any] = field(default_factory=dict)

    output_subdir: str = "fvm_gen_v2"


def _make_base_cfg(problem: str):
    if problem == "ellipse":
        return _S["ConfigEllipse"]()
    if problem == "nozzle":
        return _S["ConfigNozzle"]()
    raise ValueError(f"Unknown problem '{problem}' (use 'ellipse' or 'nozzle').")


def apply_overrides(cfg, overrides: dict[str, Any]):
    """Deep-copy cfg and apply field overrides.  visc_model strings are mapped to
    the ViscosityModel enum; BC params propagate to inlet_cfg + exit_cfg.

    `aoa_deg` is consumed here rather than set on the config: together with the speed
    (v_n_inf) it becomes the freestream vector `v_inf`, which the solver projects onto
    each boundary face.  That is what lets the stream arrive at an arbitrary angle
    instead of only along +x.
    """
    cfg = deepcopy(cfg)
    aoa_deg = overrides.get("aoa_deg")
    for key, value in overrides.items():
        if key == "aoa_deg":
            continue
        if key == "visc_model" and isinstance(value, str):
            value = _S["ViscosityModel"][value]
        if key in _BC_PARAMS:
            for bc in (cfg.inlet_cfg, cfg.exit_cfg):
                if bc is not None and hasattr(bc, key):
                    setattr(bc, key, value)
        else:
            if not hasattr(cfg, key):
                raise AttributeError(f"ConfigFVM has no field '{key}'")
            setattr(cfg, key, value)

    if aoa_deg is not None:
        # Speed from the inlet config (v_n_inf is inward-positive there); direction from
        # the sampled angle.  Both BC configs get the same vector — with one freestream
        # around the whole boundary there is no longer a separate inlet/exit direction.
        theta = np.deg2rad(float(aoa_deg))
        speed = abs(float(cfg.inlet_cfg.v_n_inf))
        v_inf = (speed * float(np.cos(theta)), speed * float(np.sin(theta)))
        for bc in (cfg.inlet_cfg, cfg.exit_cfg):
            # hasattr guards the nozzle, whose BC configs do not declare v_inf: its flow
            # is pressure-driven (v_n_inf = 0) and must not be given a freestream.
            if bc is not None and hasattr(bc, "v_inf"):
                bc.v_inf = v_inf
    return cfg


def _init_conds(cfg, mesh, edge_tag, bound_edgs, phy):
    if cfg.problem_setup == "ellipse":
        return _S["init_conds_ellipses"](mesh, edge_tag, bound_edgs, phy, cfg)
    if cfg.problem_setup == "nozzle":
        return _S["init_conds_nozzle"](mesh, edge_tag, bound_edgs, phy, cfg)
    raise ValueError(f"Unknown problem_setup '{cfg.problem_setup}'")


def _sample_specs(specs: dict[str, dict], rng: np.random.Generator) -> dict[str, Any]:
    """Draw one value per named spec (categorical + continuous)."""
    return {name: _sample_one(name, spec, rng) for name, spec in specs.items()}


def _sample_run(gen: GenConfig, rng: np.random.Generator) -> dict[str, Any]:
    """One run's parameter draw (categorical + continuous)."""
    return _sample_specs(gen.param_specs, rng)


def run_gen(gen: GenConfig, dry_run: bool = False, out_dir: Optional[str] = None):
    # Grid mode is selected by providing context_param_specs and/or ic_param_specs.
    grid_mode = bool(gen.context_param_specs or gen.ic_param_specs)

    if grid_mode:
        # The shared-IC grid only foreclosures the shortcut if there is more than
        # one context to disambiguate AND more than one IC per system (so the
        # demonstration regime has sibling trajectories).  n_ic == 1 would give
        # single-trajectory systems; n_context == 1 would give one context.
        if gen.n_context < 2 or gen.n_ic < 2:
            raise SystemExit(
                f"grid mode needs n_context>=2 and n_ic>=2 to force context use "
                f"(got n_context={gen.n_context}, n_ic={gen.n_ic}).\n"
                f"  n_context<2 -> nothing to infer; n_ic<2 -> no sibling "
                f"trajectories for the demonstration regime."
            )
    elif not gen.param_specs and gen.runs_per_mesh > 1:
        # Legacy mode: without param_specs there is nothing to vary per run, so
        # every run on a mesh comes out identical (the solver is deterministic).
        raise SystemExit(
            f"param_specs is empty but runs_per_mesh={gen.runs_per_mesh}: every run on a "
            f"mesh would be identical.\n"
            f"  Use grid mode (context_param_specs + ic_param_specs), pass a sweep "
            f"config, or set runs_per_mesh=1 for one run per geometry."
        )

    rng = np.random.default_rng(gen.seed)
    # Output dir precedence: explicit --out-dir (used as the root directly) >
    # data/<output_subdir> default.  Lets a caller drop the dataset on a specific
    # filesystem (e.g. Dawn's /rds scratch) without editing the config.
    out_root = os.path.abspath(out_dir) if out_dir else os.path.join(
        _DEFAULT_DATA_DIR, gen.output_subdir)
    # `problem` may be a single setup or a list to mix (e.g. ["ellipse","nozzle"]),
    # sampled per mesh so the dataset spans different BC topologies.
    problems = gen.problem if isinstance(gen.problem, list) else [gen.problem]

    device = mesh_over = None
    if not dry_run:
        _import_solver()
        np.random.seed(gen.seed)
        torch.manual_seed(gen.seed)
        device = os.environ.get("FVM_DEVICE")
        mesh_over = {k: v for k, v in {"min_A": gen.min_A, "max_A": gen.max_A,
                                       "lnscale": gen.lnscale, "device": device}.items()
                     if v is not None}

    os.makedirs(out_root, exist_ok=True)
    print(f"Output root: {out_root}   device: {device or '-'}   "
          f"problems: {problems}   dry_run: {dry_run}")

    manifest = {"problems": problems, "n_meshes": gen.n_meshes,
                "mode": "grid" if grid_mode else "legacy",
                "n_context": gen.n_context if grid_mode else None,
                "n_ic": gen.n_ic if grid_mode else None,
                "runs_per_mesh": None if grid_mode else gen.runs_per_mesh, "runs": []}

    for m in range(gen.n_meshes):
        problem = str(rng.choice(problems))
        mesh_uid = secrets.token_hex(4)
        # Collider count is geometry, so it varies per mesh rather than per run.
        n_colliders = int(rng.integers(gen.min_colliders, gen.max_colliders + 1))
        # one dataset dir per collider geometry (shared_mesh.pkl + its runs), so each
        # is directly consumable by FVMDataModule / the renderer.  n_meshes==1 keeps
        # a flat single-mesh dataset at the root for the legacy pipeline.
        mesh_dir = out_root if gen.n_meshes == 1 else os.path.join(out_root, f"mesh_{mesh_uid}")
        os.makedirs(mesh_dir, exist_ok=True)

        base_cfg = mesh = edge_tag = bound_edgs = None
        if not dry_run:
            mesh_cfg_over = dict(mesh_over or {})
            if problem == "ellipse":
                mesh_cfg_over["n_colliders"] = n_colliders
            base_cfg = apply_overrides(_make_base_cfg(problem), mesh_cfg_over)
            print(f"[mesh {m + 1}/{gen.n_meshes}] problem={problem} — collider geometry "
                  f"({mesh_uid}, n_colliders={n_colliders})")
            Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = _S["generate_mesh"](base_cfg)
            mesh = _S["FVMMesh2D"](Xs, tri_idx, all_edgs, bc_edge_mask, device=base_cfg.device)
            with open(os.path.join(mesh_dir, "shared_mesh.pkl"), "wb") as f:
                pickle.dump({"mesh": mesh, "edge_tag": edge_tag, "bound_edgs": bound_edgs}, f)
        else:
            print(f"[mesh {m + 1}/{gen.n_meshes}] problem={problem} ({mesh_uid})")

        # ---- build this mesh's run plan -------------------------------------
        # Each entry: (label, context_params, ic_params, context_id, ic_id).
        if grid_mode:
            # Sample the axes ONCE per mesh, then take the full cartesian product,
            # so the same n_ic freestream draws are reused across every context —
            # that reuse is what makes a starting frame recur under many contexts.
            contexts = [_sample_specs(gen.context_param_specs, rng) for _ in range(gen.n_context)]
            ics = [_sample_specs(gen.ic_param_specs, rng) for _ in range(gen.n_ic)]
            run_plan = [(f"c{j:03d}_i{i:03d}", ctx, ic, j, i)
                        for j, ctx in enumerate(contexts)
                        for i, ic in enumerate(ics)]
        else:
            run_plan = [(f"{r:04d}", _sample_run(gen, rng), {}, None, None)
                        for r in range(gen.runs_per_mesh)]

        n_runs_mesh = len(run_plan)
        for idx, (label, context_params, ic_params, cid, iid) in enumerate(run_plan):
            run_uid = secrets.token_hex(4)
            run_dir = os.path.join(mesh_dir, f"run_{label}_{run_uid}")
            os.makedirs(run_dir, exist_ok=True)

            # params.json carries ONLY the context (physics) params — identical
            # across a column of shared ICs — so mfm/data.py groups the column
            # into one multi-IC system with no consumer-side change.  (problem/
            # mesh_uid/n_colliders/context_id are constant within a column too, so
            # they do not split the grouping.)  C_v lives here, not in ic.json,
            # because the data loader reads it from params.json to convert
            # primitives -> conserved.
            record = {**context_params, "problem": problem, "mesh_uid": mesh_uid,
                      "n_colliders": n_colliders}
            if cid is not None:
                record["context_id"] = cid
            with open(os.path.join(run_dir, "params.json"), "w") as f:
                json.dump(record, f, indent=2)
            # ic.json = the per-run freestream/IC draw.  It varies within a system
            # and grouping ignores it, so it never splits a system.
            if grid_mode:
                with open(os.path.join(run_dir, "ic.json"), "w") as f:
                    json.dump({**ic_params, "ic_id": iid}, f, indent=2)

            all_params = {**context_params, **ic_params}
            manifest["runs"].append({"mesh": mesh_uid, "problem": problem,
                                     "run": run_uid, "n_colliders": n_colliders,
                                     "context_id": cid, "ic_id": iid, **all_params})

            desc = ", ".join(f"{k}={v}" if isinstance(v, str) else f"{k}={v:.3g}"
                             for k, v in all_params.items())
            print(f"  [mesh {m + 1}/{gen.n_meshes} | run {idx + 1}/{n_runs_mesh}] "
                  f"{label}: {desc}")
            if dry_run:
                continue

            # Both context and IC params drive the solver (the IC params set the
            # freestream/initial condition; the context params set the physics).
            overrides = {**all_params, **gen.phys_overrides,
                         "plot": False, "exact_interval": True,   # headless, land on save_t
                         "save_t": gen.save_t, "n_iter": gen.n_iter,
                         "print_i": gen.print_i, "compile": gen.compile,
                         "save_dir": run_dir}                     # fork saves here directly
            if gen.end_t is not None:
                overrides["end_t"] = gen.end_t
            cfg = apply_overrides(base_cfg, overrides)
            phy = _S["FluidConstitution2D"](cfg, dim=2)
            bc_tags, us_init = _init_conds(cfg, mesh, edge_tag, bound_edgs, phy)
            solver = _S["FVMEquation"](cfg, phy, mesh, cfg.N_comp, bc_tags, us_init=us_init)
            try:
                solver.solve()
            except Exception as e:                    # one bad draw shouldn't kill the sweep
                print(f"    !! run failed ({type(e).__name__}: {e}) — skipping")

    with open(os.path.join(out_root, "sweep_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. {len(manifest['runs'])} run(s) planned across {gen.n_meshes} mesh(es). "
          f"Manifest: {os.path.join(out_root, 'sweep_manifest.json')}")


def gen_from_file(path: str) -> GenConfig:
    with open(path) as f:
        data = json.load(f)
    # Drop underscore-prefixed keys so a config can carry _comment/_note fields.
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    return GenConfig(**data)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="General FVM dataset generator (upstream solver)")
    p.add_argument("config", nargs="?", help="path to a gen.json sweep config")
    p.add_argument("--dry-run", action="store_true",
                   help="sample the sweep and write params.json/manifest without running the solver")
    p.add_argument("-o", "--out-dir", default=None,
                   help="output directory (used as the dataset root directly); "
                        "overrides the data/<output_subdir> default")
    args = p.parse_args()
    cfg = gen_from_file(args.config) if args.config else GenConfig()
    run_gen(cfg, dry_run=args.dry_run, out_dir=args.out_dir)
