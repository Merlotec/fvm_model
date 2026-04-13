"""
Entry point for fvm_gen.

Runs time_fvm for each mu_b value (and any other phys_overrides) defined in
SweepConfig, saving each run into its own subdirectory.

Usage
-----
    python fvm_model/fvm_gen/run_sweep.py fvm_model/fvm_gen/sweep.json

You can also import and call `run_sweep(cfg)` or `run_sweep_from_file(path)` directly.
"""

import json
import os
import sys
import pickle
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from cprint import c_print

# ---------------------------------------------------------------------------
# Make fvm_solver importable regardless of where the script is invoked from.
#
# The modules inside time_fvm/ use bare imports (e.g. `from sparse_utils import
# ...`) so both fvm_solver/ and fvm_solver/time_fvm/ must be on sys.path.
# ---------------------------------------------------------------------------
_SOLVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fvm_solver"))
_TIME_FVM_DIR = os.path.join(_SOLVER_DIR, "time_fvm")
for _p in (_SOLVER_DIR, _TIME_FVM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from base_cfg import ARTEFACT_DIR
from time_fvm.fvm_equation import FVMEquation, PhysicalSetup
from time_fvm.fvm_mesh import FVMMesh
from time_fvm.config_fvm import ConfigFVM, ConfigEllipse, ConfigNozzle
from run_fvm import generate_mesh, init_conds_ellipses, init_conds_nozzle

from gen_cfg import SweepConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(problem: str) -> ConfigFVM:
    if problem == "ellipse":
        return ConfigEllipse()
    elif problem == "nozzle":
        return ConfigNozzle()
    else:
        raise ValueError(f"Unknown problem '{problem}'. Choose 'ellipse' or 'nozzle'.")


def apply_overrides(cfg: ConfigFVM, overrides: dict[str, Any]) -> ConfigFVM:
    """
    Return a *copy* of cfg with the given field overrides applied.

    Any field on ConfigFVM (or its subclass) can be overridden.
    Raises AttributeError if a key does not correspond to a known field.
    """
    cfg = deepcopy(cfg)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(
                f"ConfigFVM has no field '{key}'. "
                f"Valid physical fields: visc_bulk, viscosity, thermal_cond, "
                f"T_0, S_const, gamma, C_v, v_factor, lim_p, lim_K"
            )
        setattr(cfg, key, value)
    return cfg


def _init_conds(cfg, mesh, edge_tag, bound_edgs, phy_setup):
    if cfg.problem_setup == "ellipse":
        return init_conds_ellipses(mesh, edge_tag, bound_edgs, phy_setup, cfg)
    elif cfg.problem_setup == "nozzle":
        return init_conds_nozzle(mesh, edge_tag, bound_edgs, phy_setup, cfg)
    else:
        raise ValueError(f"Unknown problem_setup '{cfg.problem_setup}'")


# ---------------------------------------------------------------------------
# Core sweep runner
# ---------------------------------------------------------------------------

def sweep_cfg_from_file(path: str) -> SweepConfig:
    """Load a SweepConfig from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return SweepConfig(**data)


def run_sweep(sweep_cfg: SweepConfig | None = None):
    if sweep_cfg is None:
        sweep_cfg = SweepConfig()

    np.random.seed(42)
    torch.manual_seed(42)

    base_cfg = _make_cfg(sweep_cfg.problem)

    # Output root
    out_root = os.path.join(ARTEFACT_DIR, sweep_cfg.output_subdir)
    os.makedirs(out_root, exist_ok=True)
    c_print(f"Output root: {out_root}", "cyan")

    # ---- optionally generate mesh once and reuse it ----
    mesh_cache_path = os.path.join(out_root, "shared_mesh.pkl")

    if sweep_cfg.reuse_mesh and os.path.exists(mesh_cache_path):
        c_print("Loading cached shared mesh...", "green")
        mesh_dict = pickle.load(open(mesh_cache_path, "rb"))
    else:
        c_print("Generating mesh...", "green")
        prob_def = generate_mesh(base_cfg)
        Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = prob_def
        mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask, device=base_cfg.device)
        mesh_dict = {"mesh": mesh, "edge_tag": edge_tag, "bound_edgs": bound_edgs}
        if sweep_cfg.reuse_mesh:
            pickle.dump(mesh_dict, open(mesh_cache_path, "wb"))
            c_print(f"Mesh saved to {mesh_cache_path}", "green")

    mesh: FVMMesh = mesh_dict["mesh"]
    edge_tag = mesh_dict["edge_tag"]
    bound_edgs = mesh_dict["bound_edgs"]

    # ---- sweep over mu_b values ----
    n_runs = len(sweep_cfg.mu_b_values)
    c_print(f"\nStarting sweep: {n_runs} mu_b value(s)\n", "cyan")

    for run_idx, mu_b in enumerate(sweep_cfg.mu_b_values):
        c_print(f"[{run_idx + 1}/{n_runs}]  mu_b = {mu_b:.4e}", "yellow")

        # Per-run save directory
        run_name = f"mu_b_{mu_b:.4e}".replace("+", "").replace("-0", "-")
        run_save_dir = os.path.join(out_root, run_name)
        os.makedirs(run_save_dir, exist_ok=True)

        # Build per-run overrides: mu_b first, then any extra user overrides
        # (extra overrides can themselves override mu_b if desired)
        overrides = {"visc_bulk": mu_b, "plot": False, "exact_interval": True, "save_t": 0.1,
                     "save_dir": run_save_dir, **sweep_cfg.phys_overrides}
        cfg = apply_overrides(base_cfg, overrides)

        phy_setup = PhysicalSetup(cfg)
        bc_tags, us_init = _init_conds(cfg, mesh, edge_tag, bound_edgs, phy_setup)

        solver = FVMEquation(cfg, phy_setup, mesh, cfg.N_comp, bc_tags, us_init=us_init)

        c_print(f"  Saving to: {run_save_dir}", "green")
        solver.solve()

        c_print(f"  Run {run_idx + 1} complete.\n", "green")

    c_print("Sweep finished.", "cyan")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_sweep(sweep_cfg_from_file(sys.argv[1]))
    else:
        run_sweep()
