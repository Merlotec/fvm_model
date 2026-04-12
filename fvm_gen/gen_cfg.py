"""
Configuration for fvm_gen dataset generation sweeps.

PhysicalSetup can be overridden by passing `phys_overrides` — a dict of
ConfigFVM physical-parameter field names to new values.  Any field on
ConfigFVM is valid (e.g. visc_bulk, viscosity, gamma, …).
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SweepConfig:
    problem: str = "ellipse"   # "ellipse" | "nozzle"

    mu_b_values: list[float] = field(default_factory=lambda: list(
        np.logspace(-3, -1, 5)   # 5 values from 1e-3 to 1e-1
    ))

    # e.g. {"viscosity": 2e-3, "gamma": 1.4}
    phys_overrides: dict[str, Any] = field(default_factory=dict)

    output_subdir: str = "fvm_gen_datasets"

    reuse_mesh: bool = True

