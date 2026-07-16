"""
Configuration for fvm_gen dataset generation sweeps.

Each run independently samples from `param_specs` — a dict mapping
parameter names to a sampling specification:

  {"dist": "lognormal", "mean": <float>, "std": <float>}
  {"dist": "uniform",   "low":  <float>, "high": <float>}
  {"values": [<float>, ...]}   # explicit list; must have exactly n_samples entries

Directly settable ConfigFVM fields:
  visc_bulk, viscosity, thermal_cond, C_v, gamma, T_0, v_factor, lim_K, ...

Special parameter handled via BC configs:
  rho_inf  — applied to both inlet_cfg.rho_inf and exit_cfg.rho_inf
"""
from dataclasses import dataclass, field
from typing import Union, Any, Optional

import numpy as np


def _sample_param(name: str, spec: dict, n: int, rng: np.random.Generator) -> list[float]:
    if "values" in spec:
        vals = list(spec["values"])
        if len(vals) != n:
            raise ValueError(
                f"param '{name}': values list has {len(vals)} entries but n_samples={n}"
            )
        return vals
    dist = spec.get("dist", "lognormal")
    if dist == "lognormal":
        return np.exp(rng.normal(spec["mean"], spec["std"], size=n)).tolist()
    elif dist == "uniform":
        return rng.uniform(spec["low"], spec["high"], size=n).tolist()
    else:
        raise ValueError(f"param '{name}': unknown dist '{dist}'. Use 'lognormal' or 'uniform'.")


@dataclass
class SweepConfig:
    problem: str = "ellipse"   # "ellipse" | "nozzle"
    n_samples: int = 1         # number of simulation runs

    # Maps parameter name → sampling spec.
    param_specs: dict[str, dict] = field(default_factory=dict)

    save_t: float = 0.1
    n_iter: Optional[int] = 5000
    end_t: Optional[float] = None

    # Fixed overrides applied to every run after per-run sampling.
    phys_overrides: dict[str, Any] = field(default_factory=dict)

    output_subdir: str = "fvm_gen_datasets"
    reuse_mesh: bool = True

    # Populated by __post_init__: one dict per run with sampled values.
    param_samples: Optional[list[dict[str, float]]] = None

    def __post_init__(self):
        rng = np.random.default_rng()
        per_param = {
            name: _sample_param(name, spec, self.n_samples, rng)
            for name, spec in self.param_specs.items()
        }
        self.param_samples = [
            {name: vals[i] for name, vals in per_param.items()}
            for i in range(self.n_samples)
        ]
