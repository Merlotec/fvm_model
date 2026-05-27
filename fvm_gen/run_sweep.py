"""
Entry point for fvm_gen.

Runs time_fvm for each sampled parameter combination defined in SweepConfig,
saving each run into its own numbered subdirectory alongside a params.json
recording the exact values used.

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
# ---------------------------------------------------------------------------
_SOLVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fvm_solver"))
_TIME_FVM_DIR = os.path.join(_SOLVER_DIR, "time_fvm")
for _p in (_SOLVER_DIR, _TIME_FVM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

from time_fvm.fvm_equation import FVMEquation, FluidConstitution2D
from time_fvm.mesh_utils.fvm_mesh import FVMMesh2D
from time_fvm.config_fvm import ConfigFVM, ConfigEllipse, ConfigNozzle
from run_fvm import generate_mesh, init_conds_ellipses, init_conds_nozzle

from gen_cfg import SweepConfig


def _make_cfg(problem: str) -> ConfigFVM:
    if problem == "ellipse":
        return ConfigEllipse()
    elif problem == "nozzle":
        return ConfigNozzle()
    else:
        raise ValueError(f"Unknown problem '{problem}'. Choose 'ellipse' or 'nozzle'.")


# Parameters that live on BC sub-configs rather than directly on ConfigFVM.
_BC_PARAMS = {"rho_inf", "T_inf", "v_n_inf", "v_t_inf"}


def apply_overrides(cfg: ConfigFVM, overrides: dict[str, Any]) -> ConfigFVM:
    """Return a deep copy of cfg with the given field overrides applied.

    Direct ConfigFVM fields are set normally. The following BC parameters
    are propagated to both inlet_cfg and exit_cfg when present:
        rho_inf, T_inf, v_n_inf, v_t_inf
    """
    cfg = deepcopy(cfg)
    for key, value in overrides.items():
        if key in _BC_PARAMS:
            for bc_cfg in (cfg.inlet_cfg, cfg.exit_cfg):
                if bc_cfg is not None and hasattr(bc_cfg, key):
                    setattr(bc_cfg, key, value)
        else:
            if not hasattr(cfg, key):
                raise AttributeError(
                    f"ConfigFVM has no field '{key}'. "
                    f"Valid physical fields: visc_bulk, viscosity, thermal_cond, "
                    f"C_v, gamma, T_0, v_factor, lim_p, lim_K, rho_inf"
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

    device = os.environ.get("FVM_DEVICE")
    if device is not None:
        base_cfg = apply_overrides(base_cfg, {"device": device})
        c_print(f"Device (FVM_DEVICE): {device}", "cyan")
    else:
        c_print(f"Device (FVM_DEVICE not set): {base_cfg.device}", "cyan")

    out_root = os.path.join(_DEFAULT_DATA_DIR, sweep_cfg.output_subdir)
    os.makedirs(out_root, exist_ok=True)
    c_print(f"Output root: {out_root}", "cyan")

    # ---- optionally generate mesh once and reuse it ----
    mesh_cache_path = os.path.join(out_root, "shared_mesh.pkl")

    mesh_dict = None
    if sweep_cfg.reuse_mesh and os.path.exists(mesh_cache_path):
        c_print("Loading cached shared mesh...", "green")
        try:
            mesh_dict = pickle.load(open(mesh_cache_path, "rb"))
        except Exception as e:
            c_print(f"Cache load failed ({e}), regenerating mesh...", "yellow")
            mesh_dict = None

    if mesh_dict is None:
        c_print("Generating mesh...", "green")
        prob_def = generate_mesh(base_cfg)
        Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = prob_def
        mesh = FVMMesh2D(Xs, tri_idx, all_edgs, bc_edge_mask, device=base_cfg.device)
        mesh_dict = {"mesh": mesh, "edge_tag": edge_tag, "bound_edgs": bound_edgs}
        if sweep_cfg.reuse_mesh:
            pickle.dump(mesh_dict, open(mesh_cache_path, "wb"))
            c_print(f"Mesh saved to {mesh_cache_path}", "green")

    mesh: FVMMesh2D = mesh_dict["mesh"]
    edge_tag = mesh_dict["edge_tag"]
    bound_edgs = mesh_dict["bound_edgs"]

    # ---- sweep over sampled parameter combinations ----
    param_samples = sweep_cfg.param_samples
    n_runs = len(param_samples)
    c_print(f"\nStarting sweep: {n_runs} run(s), params: {list(sweep_cfg.param_specs)}\n", "cyan")

    for run_idx, run_params in enumerate(param_samples):
        param_str = ", ".join(f"{k}={v:.3g}" for k, v in run_params.items())
        c_print(f"[{run_idx + 1}/{n_runs}]  {param_str}", "yellow")

        run_save_dir = os.path.join(out_root, f"run_{run_idx:04d}")
        os.makedirs(run_save_dir, exist_ok=True)

        # Save the exact parameter values for this run.
        with open(os.path.join(run_save_dir, "params.json"), "w") as f:
            json.dump(run_params, f, indent=2)

        # Per-run overrides: sampled params, then fixed overrides, then housekeeping.
        overrides = {
            **run_params,
            **sweep_cfg.phys_overrides,
            "plot": False,
            "exact_interval": True,
            "save_t": sweep_cfg.save_t,
            "n_iter": sweep_cfg.n_iter,
            "end_t": sweep_cfg.end_t,
            "save_dir": run_save_dir,
        }
        cfg = apply_overrides(base_cfg, overrides)

        phy_setup = FluidConstitution2D(cfg, dim=2)
        bc_tags, us_init = _init_conds(cfg, mesh, edge_tag, bound_edgs, phy_setup)

        solver = FVMEquation(cfg, phy_setup, mesh, cfg.N_comp, bc_tags, us_init=us_init)

        c_print(f"  Saving to: {run_save_dir}", "green")
        solver.solve()

        c_print(f"  Run {run_idx + 1} complete.\n", "green")

    c_print("Sweep finished.", "cyan")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_sweep(sweep_cfg_from_file(sys.argv[1]))
    else:
        run_sweep()
