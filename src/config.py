"""Shared configuration for the AI inference energy simulation.

Single source of truth for power, cluster, and burst-overlay parameters.
Both ``src/Simulation.py`` and the notebook import from here so the two
cannot drift.

Anything you want to tweak across runs goes here as a dataclass field
or option-list constant - don't put numeric constants back in the
notebook. --E
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerParams:
    """Power and facility parameters used by the energy model."""

    p_idle: float = 140.0
    p_execution_idle: float = 220.0
    p_active_avg: float = 450.0
    pue: float = 1.08
    execution_idle_fraction: float = 0.197


@dataclass(frozen=True)
class ClusterParams:
    """Cluster, queueing, and time-discretisation parameters."""

    total_gpus: int = 256
    mu: float = 0.2                       # service rate per GPU (req/s)
    pod_size: int = 8                     # GPUs per node (tensor-parallel shard)
    c_peak: int = 200                     # GPUs active at peak hour (sets lambda_peak)

    # Provisioning thresholds (target effective utilisation).
    # Conservative leaves ~22% headroom to absorb bursts/cold-start, Aggressive
    # leaves only ~5%. These values are used by Block 6/7 of the notebook. --E
    target_rho_conservative: float = 0.78
    target_rho_aggressive: float = 0.95

    # Time discretisation. Drop step_minutes to expose finer-grain spikes
    # (energy total stays invariant; max_rho and SLA-violation count grow
    # because shorter bursts stop being averaged out). --E
    T_day_hours: float = 24.0
    step_minutes: float = 15.0

    @property
    def step_hours(self) -> float:
        return self.step_minutes / 60.0

    @property
    def num_steps(self) -> int:
        return int(round(self.T_day_hours / self.step_hours))

    @property
    def lambda_peak(self) -> float:
        """Peak NHPP arrival rate implied by ``c_peak`` and the conservative target."""
        return self.target_rho_conservative * self.mu * self.c_peak


@dataclass(frozen=True)
class BurstParams:
    """Burst-overlay parameters (used when arrival_mode == "hybrid").

    ``distribution`` selects which distribution the per-burst SIZE multiplier
    is drawn from - this is the wrong place to model BurstGPT-style
    inter-arrival heaviness; that lives in ArrivalParams + the
    ``gamma_renewal`` mode. Keeping this overlay around as a stress-test
    knob even after gamma_renewal exists. --E
    """

    distribution: str = "pareto"
    pareto_shape: float = 1.2
    gamma_shape: float = 0.5
    gamma_scale: float = 2.0
    burst_rate_per_hour: float = 3.0
    burst_window_hours: float = 0.01
    size_multiplier_x_lambda_peak: float = 10.0


@dataclass(frozen=True)
class ArrivalParams:
    """Parameters for non-Poisson base arrival processes.

    Used when ``arrival_mode == "gamma_renewal"``. The renewal process
    samples inter-arrival times from Gamma(shape, scale), normalised so
    the local mean equals 1/lambda(t). The shape parameter controls the
    coefficient of variation: ``CV = 1/sqrt(shape)``. ``shape=1`` recovers
    the standard NHPP (CV=1); ``shape=0.5`` matches the BurstGPT fit
    (CV ~= 1.41) reported in Wang et al., arXiv:2401.17644, 2024.

    Scale is irrelevant to CV (only sets the absolute mean, which we
    re-normalise away). Keeping the BurstGPT scale=2 default for
    documentation. --E
    """

    gamma_renewal_shape: float = 0.5
    gamma_renewal_scale: float = 2.0


DIURNAL_SHAPE = np.array(
    [
        0.20, 0.15, 0.10, 0.09, 0.09, 0.13, 0.22, 0.38,
        0.55, 0.70, 0.82, 0.90, 0.95, 1.00, 0.98, 0.92,
        0.86, 0.84, 0.79, 0.69, 0.56, 0.45, 0.35, 0.26,
    ],
    dtype=float,
)

# Option lists. Note: gamma_renewal still works at runtime even if it's
# not listed in ARRIVAL_MODES - the tuple only constrains the @param
# dropdown comment, not the actual function dispatch in build_lambda_15min. --E
STRATEGIES = ("Static", "Conservative", "Aggressive")
ARRIVAL_MODES = ("nhpp_only", "hybrid")
BURST_DISTRIBUTIONS = ("pareto", "gamma")
