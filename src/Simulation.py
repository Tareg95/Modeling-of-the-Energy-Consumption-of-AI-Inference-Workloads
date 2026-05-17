"""
Core simulation functions for the AI inference energy model.

This file contains reusable calculation logic only.  plotting,
markdown explanations, and thesis-specific discussion in the notebook.
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np

from src.config import ArrivalParams, BurstParams, ClusterParams, PowerParams

__all__ = [
    "ArrivalParams",
    "BurstParams",
    "ClusterParams",
    "PowerParams",
    "scale_in_pods",
    "build_diurnal_lambda",
    "daily_intensity_from_hourly",
    "simulate_nhpp",
    "simulate_pareto_bursts",
    "simulate_gamma_bursts",
    "simulate_bursts",
    "simulate_gamma_renewal_nhpp",
    "bin_to_rate",
    "bin_to_15min_rate",
    "build_lambda_15min",
    "provisioning_arrays",
    "apply_cold_start_lag",
    "utilisation_arrays",
    "gpu_state_arrays",
    "energy_arrays",
    "run_one",
    "run_strategy_comparison",
]


def scale_in_pods(c_target: np.ndarray | float, pod_size: int = 8) -> np.ndarray:
    """Round a target GPU count up to the nearest pod size."""

    return np.ceil(np.asarray(c_target, dtype=float) / pod_size) * pod_size


def build_diurnal_lambda(
    shape: Iterable[float],
    lambda_peak: float,
    step_hours: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create hourly and 15-minute demand curves from a normalized daily shape.

    Returns
    -------
    lambda_hourly:
        24 hourly request-rate values in requests/second.
    lambda_15min:
        Interpolated request-rate values at the chosen step resolution.
    hours_96:
        Time axis corresponding to lambda_15min.
    """

    shape_arr = np.asarray(shape, dtype=float)
    if shape_arr.size != 24:
        raise ValueError("shape must contain exactly 24 hourly values")

    lambda_hourly = shape_arr * lambda_peak
    hours_24 = np.arange(24)
    hours_96 = np.arange(0, 24, step_hours)
    lambda_15min = np.interp(hours_96, hours_24, lambda_hourly)

    return lambda_hourly, lambda_15min, hours_96


def daily_intensity_from_hourly(lambda_hourly: np.ndarray) -> Callable[[float], float]:
    """Return a function lambda(t) that linearly interpolates between hours.

    Was a piecewise-constant hourly step before; the step pattern then
    propagated through the binned demand and made every downstream plot
    look chunky. The hourly anchors are still the only data we have, but
    interpolating between them at least lets the NHPP thinning see a
    continuous rate. --E
    """

    hours = np.arange(24)

    def lambda_func(t: float) -> float:
        return float(np.interp(t % 24, hours, lambda_hourly))

    return lambda_func


def simulate_nhpp(
    lambda_func: Callable[[float], float],
    max_lambda: float,
    T: float,
) -> list[float]:
    """Simulate an NHPP over [0, T] hours using thinning.

    lambda_func and max_lambda are expressed in requests/second.
    Returned event times are expressed in hours.
    """

    if max_lambda <= 0:
        return []

    times: list[float] = []
    t = 0.0

    while t < T:
        t += np.random.exponential(1 / (max_lambda * 3600))
        if t < T and np.random.rand() <= lambda_func(t) / max_lambda:
            times.append(t)

    return times


def simulate_pareto_bursts(
    shape: float,
    T: float,
    lambda_peak: float,
    burst_rate_per_hour: float = 3.0,
    burst_window_hours: float = 0.01,
    size_multiplier_x_lambda_peak: float = 10.0,
) -> tuple[list[float], list[float], list[int]]:
    """Generate Pareto-distributed burst arrivals over a T-hour window.

    Burst events occur as a Poisson process with rate
    ``burst_rate_per_hour``. Each event spawns
    ``int((Pareto(shape) + 1) * int(lambda_peak * size_multiplier_x_lambda_peak))``
    requests, all uniformly placed inside a ``burst_window_hours`` window.

    Returns ``(arrival_times, burst_times, burst_sizes)`` with times in hours.
    """

    size_scale = int(lambda_peak * size_multiplier_x_lambda_peak)
    times: list[float] = []
    burst_times: list[float] = []
    burst_sizes: list[int] = []
    t = 0.0

    while t < T:
        t += np.random.exponential(1.0 / burst_rate_per_hour)
        if t < T:
            burst_size = int((np.random.pareto(shape) + 1) * size_scale)
            burst_times.append(t)
            burst_sizes.append(burst_size)

            for _ in range(burst_size):
                times.append(t + np.random.uniform(0, burst_window_hours))

    return times, burst_times, burst_sizes


def simulate_gamma_bursts(
    shape: float,
    scale: float,
    T: float,
    lambda_peak: float,
    burst_rate_per_hour: float = 3.0,
    burst_window_hours: float = 0.01,
    size_multiplier_x_lambda_peak: float = 10.0,
) -> tuple[list[float], list[float], list[int]]:
    """Generate Gamma-distributed burst arrivals over a T-hour window.

    Same structure as ``simulate_pareto_bursts`` but draws the multiplicative
    factor from a Gamma(``shape``, ``scale``) distribution. The default
    Gamma parameters in :class:`BurstParams` (shape=0.5, scale=2.0) match
    the fit reported by BurstGPT (Wang et al., arXiv:2401.17644, 2024).
    """

    size_scale = int(lambda_peak * size_multiplier_x_lambda_peak)
    times: list[float] = []
    burst_times: list[float] = []
    burst_sizes: list[int] = []
    t = 0.0

    while t < T:
        t += np.random.exponential(1.0 / burst_rate_per_hour)
        if t < T:
            burst_size = int((np.random.gamma(shape, scale) + 1) * size_scale)
            burst_times.append(t)
            burst_sizes.append(burst_size)

            for _ in range(burst_size):
                times.append(t + np.random.uniform(0, burst_window_hours))

    return times, burst_times, burst_sizes


def simulate_bursts(
    burst_params: BurstParams,
    T: float,
    lambda_peak: float,
) -> tuple[list[float], list[float], list[int]]:
    """Dispatch to the burst-size distribution selected in ``burst_params``."""

    if burst_params.distribution == "pareto":
        return simulate_pareto_bursts(
            shape=burst_params.pareto_shape,
            T=T,
            lambda_peak=lambda_peak,
            burst_rate_per_hour=burst_params.burst_rate_per_hour,
            burst_window_hours=burst_params.burst_window_hours,
            size_multiplier_x_lambda_peak=burst_params.size_multiplier_x_lambda_peak,
        )
    if burst_params.distribution == "gamma":
        return simulate_gamma_bursts(
            shape=burst_params.gamma_shape,
            scale=burst_params.gamma_scale,
            T=T,
            lambda_peak=lambda_peak,
            burst_rate_per_hour=burst_params.burst_rate_per_hour,
            burst_window_hours=burst_params.burst_window_hours,
            size_multiplier_x_lambda_peak=burst_params.size_multiplier_x_lambda_peak,
        )
    raise ValueError(
        f"Unknown burst distribution {burst_params.distribution!r}; "
        "expected 'pareto' or 'gamma'."
    )


def simulate_gamma_renewal_nhpp(
    shape: float,
    scale: float,
    lambda_func: Callable[[float], float],
    T: float,
) -> list[float]:
    """Non-homogeneous Gamma renewal process over [0, T] hours.

    Inter-arrival times are drawn from Gamma(``shape``, ``scale``) and
    rescaled so the local mean equals ``1 / lambda(t)`` (matching the
    Poisson-rate mean at time ``t``). The coefficient of variation is
    ``1 / sqrt(shape)`` and is independent of ``scale``:

    - ``shape=1.0`` recovers a standard NHPP (CV = 1).
    - ``shape=0.5`` matches the BurstGPT fit (Wang et al., 2024); CV ~= 1.41.

    ``lambda_func`` returns the local rate in requests per second, given a
    time ``t`` in hours.

    Uses rate-at-start-of-interval as the local rate; fine because
    lambda(t) only changes hourly while inter-arrivals are sub-second. --E
    """

    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be positive")

    times: list[float] = []
    t = 0.0
    raw_mean = shape * scale  # mean of the unscaled Gamma draw

    while t < T:
        rate_per_s = max(float(lambda_func(t)), 1e-12)
        raw = np.random.gamma(shape, scale)
        # Rescale: dt has mean 1 / (rate_per_s * 3600) hours.
        dt_hours = raw / (raw_mean * rate_per_s * 3600.0)
        t += dt_hours
        if t < T:
            times.append(t)

    return times


def bin_to_rate(
    arrivals: Iterable[float],
    num_steps: int = 96,
    step_hours: float = 0.25,
) -> np.ndarray:
    """Convert event times in hours to a binned request-rate array in req/s."""

    binned = np.zeros(num_steps)
    seconds_per_step = step_hours * 3600

    for t_arr in arrivals:
        bin_idx = int(t_arr / step_hours)
        if 0 <= bin_idx < num_steps:
            binned[bin_idx] += 1.0 / seconds_per_step

    return binned


# Backward-compatible name used in the notebook.
bin_to_15min_rate = bin_to_rate


def build_lambda_15min(
    arrival_mode: str,
    burst_params: BurstParams,
    lambda_hourly: np.ndarray,
    lambda_peak: float,
    lambda_15min_nhpp: np.ndarray,
    T: float = 24.0,
    num_steps: int = 96,
    step_hours: float = 0.25,
    seed: int | None = 42,
    arrival_params: ArrivalParams = ArrivalParams(),
) -> np.ndarray:
    """Build the lambda(t) array for a given arrival-mode configuration.

    - ``"nhpp_only"`` returns the pre-computed NHPP base.
    - ``"hybrid"`` re-runs the NHPP thinning and adds the ``burst_params``
      overlay on top.
    - ``"gamma_renewal"`` replaces the NHPP base with a non-homogeneous
      Gamma renewal process driven by ``arrival_params`` (no burst overlay).

    For "hybrid" and "gamma_renewal" modes ``np.random`` is re-seeded first
    so each combo is independently reproducible.

    Block 7 of the notebook calls this once per scenario; Block 1 calls it
    indirectly via the inline arrival-mode branches. --E
    """

    if arrival_mode == "nhpp_only":
        return lambda_15min_nhpp.copy()

    if seed is not None:
        np.random.seed(seed)

    lambda_func = daily_intensity_from_hourly(lambda_hourly)

    if arrival_mode == "hybrid":
        max_lam = float(lambda_hourly.max())
        base = simulate_nhpp(lambda_func, max_lam, T)
        bursts, _, _ = simulate_bursts(burst_params, T=T, lambda_peak=lambda_peak)
        all_arrivals = sorted(base + bursts)
    elif arrival_mode == "gamma_renewal":
        all_arrivals = simulate_gamma_renewal_nhpp(
            arrival_params.gamma_renewal_shape,
            arrival_params.gamma_renewal_scale,
            lambda_func,
            T,
        )
    else:
        raise ValueError(
            f"Unknown arrival_mode {arrival_mode!r}; "
            "expected 'nhpp_only', 'hybrid', or 'gamma_renewal'."
        )

    return bin_to_rate(all_arrivals, num_steps=num_steps, step_hours=step_hours)


def provisioning_arrays(
    lambda_arr: np.ndarray,
    mu: float,
    total_gpus: int,
    pod_size: int = 8,
) -> dict[str, np.ndarray]:
    """Create static + per-rho-target provisioning arrays.

    Strategy ordering is Conservative -> Moderate -> Tight -> Aggressive
    (decreasing headroom: 22 / 15 / 10 / 5 %). --E
    """

    targets = {
        "Conservative": 0.78,
        "Moderate":     0.85,
        "Tight":        0.90,
        "Aggressive":   0.95,
    }
    out = {"Static": np.full(len(lambda_arr), total_gpus, dtype=float)}
    for name, rho in targets.items():
        out[name] = np.minimum(
            scale_in_pods(lambda_arr / (rho * mu), pod_size), total_gpus
        )
    return out


def apply_cold_start_lag(c_active: np.ndarray) -> np.ndarray:
    """Apply one-step scale-up lag and immediate scale-down.

    If target capacity increases, ready capacity stays at the previous value
    for one step. If target capacity decreases, capacity drops immediately.
    """

    c_ready = np.zeros_like(c_active, dtype=float)
    c_ready[0] = c_active[0]

    for t in range(1, len(c_active)):
        c_ready[t] = np.minimum(c_active[t], c_active[t - 1])

    return c_ready


def utilisation_arrays(
    lambda_arr: np.ndarray,
    c_active: np.ndarray,
    c_ready: np.ndarray,
    mu: float,
    total_gpus: int,
) -> dict[str, np.ndarray]:
    """Compute capacity and utilisation arrays."""

    capacity_target_qps = c_active * mu
    capacity_ready_qps = c_ready * mu

    rho_theoretical = np.where(
        capacity_target_qps > 0,
        lambda_arr / np.maximum(capacity_target_qps, 1e-9),
        0.0,
    )
    rho_effective = np.where(
        capacity_ready_qps > 0,
        lambda_arr / np.maximum(capacity_ready_qps, 1e-9),
        0.0,
    )
    global_rho = lambda_arr / (total_gpus * mu)

    return {
        "capacity_target_qps": capacity_target_qps,
        "capacity_ready_qps": capacity_ready_qps,
        "demand_qps": lambda_arr,
        "rho_theoretical": rho_theoretical,
        "rho_effective": rho_effective,
        "global_rho": global_rho,
    }


def gpu_state_arrays(
    c_ready: np.ndarray,
    rho_effective: np.ndarray,
    total_gpus: int,
    execution_idle_fraction: float = 0.197,
) -> dict[str, np.ndarray]:
    """Partition GPUs into active, execution-idle, and deep-idle states."""

    rho_state = np.minimum(rho_effective, 1.0)

    gpus_active = c_ready * rho_state * (1 - execution_idle_fraction)
    gpus_execution_idle = c_ready - gpus_active
    gpus_deep_idle = total_gpus - c_ready

    return {
        "rho_state": rho_state,
        "gpus_active": gpus_active,
        "gpus_execution_idle": gpus_execution_idle,
        "gpus_deep_idle": gpus_deep_idle,
        "active_ratio": gpus_active / total_gpus,
        "execution_idle_ratio": gpus_execution_idle / total_gpus,
        "deep_idle_ratio": gpus_deep_idle / total_gpus,
    }


def energy_arrays(
    c_ready: np.ndarray,
    rho_effective: np.ndarray,
    gpus_active: np.ndarray,
    gpus_deep_idle: np.ndarray,
    power: PowerParams = PowerParams(),
    step_hours: float = 0.25,
) -> dict[str, np.ndarray | float]:
    """Compute energy use per step and daily totals.

    ``step_hours`` is the duration of one simulation step in hours
    (e.g. 0.25 for 15-minute steps, 0.05 for 3-minute steps). The total
    daily kWh is invariant under this choice; only the per-step values
    differ.

    step_hours used to be a hardcoded 0.25 here and in 4 other places -
    if it ever stops being a kwarg, the daily kWh will silently scale
    wrong when step_minutes != 15. --E
    """

    rho_state = np.minimum(rho_effective, 1.0)

    pure_active_power = gpus_active * power.p_active_avg
    active_overhead_power = (
        c_ready * rho_state * power.execution_idle_fraction * power.p_execution_idle
    )
    inactive_idle_power = c_ready * (1 - rho_state) * power.p_execution_idle
    deep_idle_power = gpus_deep_idle * power.p_idle

    energy_pure_work = (pure_active_power * step_hours) / 1000
    energy_active_overhead = (active_overhead_power * step_hours) / 1000
    energy_inactive_idle = (inactive_idle_power * step_hours) / 1000
    energy_deep_idle = (deep_idle_power * step_hours) / 1000

    energy_total = (
        energy_pure_work
        + energy_active_overhead
        + energy_inactive_idle
        + energy_deep_idle
    )
    energy_ideal = (pure_active_power * step_hours) / 1000

    return {
        "energy_pure_work": energy_pure_work,
        "energy_active_overhead": energy_active_overhead,
        "energy_inactive_idle": energy_inactive_idle,
        "energy_deep_idle": energy_deep_idle,
        "energy_total": energy_total,
        "energy_ideal": energy_ideal,
        "total_kwh": float(np.sum(energy_total)),
        "work_kwh": float(np.sum(energy_pure_work)),
        "overhead_kwh": float(np.sum(energy_active_overhead)),
        "idle_kwh": float(np.sum(energy_inactive_idle)),
        "deep_kwh": float(np.sum(energy_deep_idle)),
        "ideal_total_kwh": float(np.sum(energy_ideal)),
    }


def run_one(
    lambda_arr: np.ndarray,
    c_active_arr: np.ndarray,
    mu: float = 0.2,
    total_gpus: int = 256,
    power: PowerParams = PowerParams(),
    step_hours: float = 0.25,
) -> dict[str, float | int]:
    """Run one strategy/arrival combination and return summary metrics."""

    c_ready = apply_cold_start_lag(c_active_arr)
    util = utilisation_arrays(lambda_arr, c_active_arr, c_ready, mu, total_gpus)
    states = gpu_state_arrays(
        c_ready,
        util["rho_effective"],
        total_gpus,
        power.execution_idle_fraction,
    )
    energy = energy_arrays(
        c_ready,
        util["rho_effective"],
        states["gpus_active"],
        states["gpus_deep_idle"],
        power,
        step_hours=step_hours,
    )

    return {
        "total": energy["total_kwh"],
        "pure": energy["work_kwh"],
        "oh": energy["overhead_kwh"],
        "inact": energy["idle_kwh"],
        "deep": energy["deep_kwh"],
        "mean_rho": float(np.mean(util["rho_effective"])),
        "max_rho": float(np.max(util["rho_effective"])),
        "sla_viol": int(np.sum(util["rho_effective"] > 1.0)),
    }


def run_strategy_comparison(
    lambda_sets: dict[str, np.ndarray],
    mu: float = 0.2,
    total_gpus: int = 256,
    pod_size: int = 8,
    power: PowerParams = PowerParams(),
    step_hours: float = 0.25,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Run Static/Conservative/Aggressive for each arrival process."""

    results: dict[str, dict[str, dict[str, float | int]]] = {}

    for arrival_name, lambda_arr in lambda_sets.items():
        results[arrival_name] = {}
        provisioning = provisioning_arrays(lambda_arr, mu, total_gpus, pod_size)

        for strategy_name, c_active in provisioning.items():
            results[arrival_name][strategy_name] = run_one(
                lambda_arr=lambda_arr,
                c_active_arr=c_active,
                mu=mu,
                total_gpus=total_gpus,
                power=power,
                step_hours=step_hours,
            )

    return results
