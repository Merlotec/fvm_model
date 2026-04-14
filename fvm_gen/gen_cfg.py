"""
Configuration for fvm_gen dataset generation sweeps.

PhysicalSetup can be overridden by passing `phys_overrides` — a dict of
ConfigFVM physical-parameter field names to new values.  Any field on
ConfigFVM is valid (e.g. visc_bulk, viscosity, gamma, …).

mu_b sourcing (mutually exclusive):
  - Set mu_b_values explicitly, OR
  - Set mu_b_values=null and provide mu_b_gen_count, mu_b_gen_mean, mu_b_gen_stdev
    to sample from a log-normal distribution:
        log(mu_b) ~ Normal(mu_b_gen_mean, mu_b_gen_stdev)
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SweepConfig:
    problem: str = "ellipse"   # "ellipse" | "nozzle"

    # Log-normal sampling params — used only when mu_b_values is None.
    mu_b_gen_count: int | None = None
    mu_b_gen_mean: float | None = None
    mu_b_gen_stdev: float | None = None

    # Explicit list of mu_b values. Set to null in JSON to use random sampling.
    mu_b_values: list[float] | None = None

    # e.g. {"viscosity": 2e-3, "gamma": 1.4}
    phys_overrides: dict[str, Any] = field(default_factory=dict)

    output_subdir: str = "fvm_gen_datasets"

    reuse_mesh: bool = True

    def __post_init__(self):
        if self.mu_b_values is None:
            missing = [
                name for name, val in [
                    ("mu_b_gen_count",  self.mu_b_gen_count),
                    ("mu_b_gen_mean",   self.mu_b_gen_mean),
                    ("mu_b_gen_stdev",  self.mu_b_gen_stdev),
                ]
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"mu_b_values is None but required gen params are missing: {missing}"
                )
            log_samples = np.random.normal(
                self.mu_b_gen_mean, self.mu_b_gen_stdev, size=self.mu_b_gen_count
            )
            self.mu_b_values = np.exp(log_samples).tolist()

