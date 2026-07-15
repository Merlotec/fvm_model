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
from typing import Any

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
    problem: "str | list" = "ellipse"  # "ellipse" | "nozzle" | a list to mix per mesh
    n_meshes: int = 8                 # distinct collider geometries
    runs_per_mesh: int = 16           # parameter draws per geometry
    seed: int = 42

    # Solver run controls (applied to every run).
    save_t: float = 0.01              # sim-time between saved frames
    n_iter: int = 10000               # max solver iterations
    end_t: float | None = None        # max sim time (optional; None => run n_iter steps)
    print_i: int = 500
    compile: bool = True              # torch.compile the step (fast on GPU)

    # Mesh sizing (obstacle/cell scale); passed straight to the config.
    min_A: float | None = None
    max_A: float | None = None
    lnscale: float | None = None

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
    the ViscosityModel enum; BC params propagate to inlet_cfg + exit_cfg."""
    cfg = deepcopy(cfg)
    for key, value in overrides.items():
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
    return cfg


def _init_conds(cfg, mesh, edge_tag, bound_edgs, phy):
    if cfg.problem_setup == "ellipse":
        return _S["init_conds_ellipses"](mesh, edge_tag, bound_edgs, phy, cfg)
    if cfg.problem_setup == "nozzle":
        return _S["init_conds_nozzle"](mesh, edge_tag, bound_edgs, phy, cfg)
    raise ValueError(f"Unknown problem_setup '{cfg.problem_setup}'")


def _sample_run(gen: GenConfig, rng: np.random.Generator) -> dict[str, Any]:
    """One run's parameter draw (categorical + continuous)."""
    return {name: _sample_one(name, spec, rng) for name, spec in gen.param_specs.items()}


def run_gen(gen: GenConfig, dry_run: bool = False):
    rng = np.random.default_rng(gen.seed)
    out_root = os.path.join(_DEFAULT_DATA_DIR, gen.output_subdir)
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
                "runs_per_mesh": gen.runs_per_mesh, "runs": []}

    for m in range(gen.n_meshes):
        problem = str(rng.choice(problems))
        mesh_uid = secrets.token_hex(4)
        # one dataset dir per collider geometry (shared_mesh.pkl + its runs), so each
        # is directly consumable by FVMDataModule / the renderer.  n_meshes==1 keeps
        # a flat single-mesh dataset at the root for the legacy pipeline.
        mesh_dir = out_root if gen.n_meshes == 1 else os.path.join(out_root, f"mesh_{mesh_uid}")
        os.makedirs(mesh_dir, exist_ok=True)

        base_cfg = mesh = edge_tag = bound_edgs = None
        if not dry_run:
            base_cfg = apply_overrides(_make_base_cfg(problem), mesh_over or {})
            print(f"[mesh {m + 1}/{gen.n_meshes}] problem={problem} — collider geometry ({mesh_uid})")
            Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = _S["generate_mesh"](base_cfg)
            mesh = _S["FVMMesh2D"](Xs, tri_idx, all_edgs, bc_edge_mask, device=base_cfg.device)
            with open(os.path.join(mesh_dir, "shared_mesh.pkl"), "wb") as f:
                pickle.dump({"mesh": mesh, "edge_tag": edge_tag, "bound_edgs": bound_edgs}, f)
        else:
            print(f"[mesh {m + 1}/{gen.n_meshes}] problem={problem} ({mesh_uid})")

        for r in range(gen.runs_per_mesh):
            params = _sample_run(gen, rng)
            run_uid = secrets.token_hex(4)
            run_dir = os.path.join(mesh_dir, f"run_{r:04d}_{run_uid}")
            os.makedirs(run_dir, exist_ok=True)
            record = {**params, "problem": problem, "mesh_uid": mesh_uid}
            with open(os.path.join(run_dir, "params.json"), "w") as f:
                json.dump(record, f, indent=2)
            manifest["runs"].append({"mesh": mesh_uid, "problem": problem,
                                     "run": run_uid, **params})

            desc = ", ".join(f"{k}={v}" if isinstance(v, str) else f"{k}={v:.3g}"
                             for k, v in params.items())
            print(f"  [mesh {m + 1}/{gen.n_meshes} | run {r + 1}/{gen.runs_per_mesh}] {desc}")
            if dry_run:
                continue

            overrides = {**params, **gen.phys_overrides,
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
        return GenConfig(**json.load(f))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="General FVM dataset generator (upstream solver)")
    p.add_argument("config", nargs="?", help="path to a gen.json sweep config")
    p.add_argument("--dry-run", action="store_true",
                   help="sample the sweep and write params.json/manifest without running the solver")
    args = p.parse_args()
    cfg = gen_from_file(args.config) if args.config else GenConfig()
    run_gen(cfg, dry_run=args.dry_run)
