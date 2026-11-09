from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np


def sample_two_point_delays(
    horizon: int,
    small_delay: int,
    large_delay: int,
    large_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if small_delay < 0 or large_delay <= small_delay:
        raise ValueError("two-point delays require 0 <= d_small < d_large")
    if not 0.0 <= large_probability <= 1.0:
        raise ValueError("large_probability must lie in [0, 1]")
    number_large = int(round(large_probability * horizon))
    is_large = np.zeros(horizon, dtype=bool)
    if number_large:
        is_large[rng.choice(horizon, size=number_large, replace=False)] = True
    return np.where(is_large, large_delay, small_delay).astype(np.int64)


def sample_lomax_delays(
    horizon: int,
    shape: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if shape <= 1.0:
        raise ValueError("shape must exceed 1 for a finite mean")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    uniforms = rng.random(horizon)
    raw = scale * ((1.0 - uniforms) ** (-1.0 / shape) - 1.0)
    delays = 1 + np.floor(raw).astype(np.int64)
    remaining = np.arange(horizon - 1, -1, -1, dtype=np.int64)
    return np.minimum(delays, remaining)


def calibrate_lomax_scale(
    target_mean: float,
    shape: float,
    horizon: int,
    calibration_uniforms: np.ndarray,
    calibration_rounds: np.ndarray,
    relative_tolerance: float = 5e-3,
) -> float:
    if target_mean <= 0.0:
        raise ValueError("target_mean must be positive")
    if shape <= 1.0:
        raise ValueError("shape must exceed one")
    if horizon <= 1:
        raise ValueError("horizon must exceed one")
    if calibration_uniforms.shape != calibration_rounds.shape:
        raise ValueError("calibration arrays must have equal shape")
    if len(calibration_uniforms) == 0:
        raise ValueError("calibration arrays may not be empty")
    if np.any((calibration_uniforms < 0.0) | (calibration_uniforms >= 1.0)):
        raise ValueError("calibration uniforms must lie in [0, 1)")
    if np.any((calibration_rounds < 0) | (calibration_rounds >= horizon)):
        raise ValueError("calibration rounds lie outside the horizon")

    lomax_factors = (1.0 - calibration_uniforms) ** (-1.0 / shape) - 1.0
    remaining = horizon - 1 - calibration_rounds.astype(np.int64, copy=False)

    def realized_mean(scale: float) -> float:
        delays = 1.0 + np.floor(scale * lomax_factors)
        return float(np.minimum(delays, remaining).mean())

    low = 0.0
    high = max(1.0, target_mean * (shape - 1.0) * 2.0)
    while realized_mean(high) < target_mean:
        high *= 2.0
        if not np.isfinite(high):
            raise FloatingPointError("failed to bracket Lomax scale")

    best = high
    best_error = float("inf")
    for _ in range(64):
        middle = 0.5 * (low + high)
        mean = realized_mean(middle)
        error = abs(mean - target_mean) / target_mean
        if error < best_error:
            best, best_error = middle, error
        if error <= relative_tolerance:
            return middle
        if mean < target_mean:
            low = middle
        else:
            high = middle
    if best_error > relative_tolerance:
        raise RuntimeError(
            f"Lomax calibration missed tolerance: relative error {best_error:.6g}"
        )
    return best


def calibration_arrays(
    horizon: int, samples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    return rng.random(samples), rng.integers(0, horizon, size=samples, dtype=np.int64)


def calibrate_target_scales(
    targets: Iterable[float],
    shape: float,
    horizon: int,
    samples: int,
    seed: int,
    relative_tolerance: float = 5e-3,
) -> Dict[float, float]:
    uniforms, rounds = calibration_arrays(horizon, samples, seed)
    return {
        float(target): calibrate_lomax_scale(
            float(target), shape, horizon, uniforms, rounds, relative_tolerance
        )
        for target in targets
    }


def delay_scales_for_config(config: Any) -> Dict[float, float]:

    if config.actual_delay_distribution in {"two_point", "binary"}:
        return {float(target): float(target) for target in config.target_mean_delays}
    return calibrate_target_scales(
        config.target_mean_delays,
        config.actual_delay_shape,
        config.horizon,
        config.calibration_samples,
        config.calibration_seed,
        config.calibration_relative_tolerance,
    )


def compute_backlog(delays: np.ndarray) -> np.ndarray:
    delays = np.asarray(delays, dtype=np.int64)
    horizon = len(delays)
    if np.any(delays < 0):
        raise ValueError("delays must be nonnegative")
    diff = np.zeros(horizon + 1, dtype=np.int64)
    positive = delays > 0
    diff[:horizon] = positive.astype(np.int64)
    ends = np.minimum(
        np.arange(horizon, dtype=np.int64)[positive] + delays[positive],
        horizon,
    )
    np.add.at(diff, ends, -1)
    return np.cumsum(diff[:-1])


def delay_diagnostics(delays: np.ndarray, target: float, capacity: int) -> Dict[str, float]:
    backlog = compute_backlog(delays)
    sigma_max = int(backlog.max(initial=0))
    return {
        "target_mean_delay": float(target),
        "realized_mean_delay": float(np.mean(delays)),
        "median_delay": float(np.percentile(delays, 50)),
        "90th_delay_percentile": float(np.percentile(delays, 90)),
        "99th_delay_percentile": float(np.percentile(delays, 99)),
        "maximum_delay": int(np.max(delays, initial=0)),
        "mean_backlog": float(np.mean(backlog)),
        "sigma_max": sigma_max,
        "sigma_max_over_capacity": sigma_max / capacity,
    }
