from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class ProxyDraw:
    lifetime: Optional[int]


class ProxyDelayGenerator(Protocol):
    def sample(self, round_index: int, rng: np.random.Generator) -> ProxyDraw: ...
    def survival_probability(self, actual_delay: int) -> float: ...
    def importance_weight(self, actual_delay: int) -> float: ...


def _discrete_lomax_mean(scale: float, shape: float, terms: int = 200_000) -> float:

    k = np.arange(1, terms + 1, dtype=np.float64)
    values = (1.0 + k / scale) ** (-shape)
    total = float(values.sum())

    x = terms + 1.0
    tail = scale / (shape - 1.0) * (1.0 + x / scale) ** (1.0 - shape)
    tail += 0.5 * (1.0 + x / scale) ** (-shape)
    return total + tail


@lru_cache(maxsize=None)
def calibrate_pareto_proxy_scale(
    capacity: int,
    shape: float,
    occupancy_fraction: float,
    relative_tolerance: float = 1e-6,
) -> float:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if shape <= 1.0:
        raise ValueError("proxy shape must exceed one")
    if not 0.0 < occupancy_fraction <= 1.0:
        raise ValueError("occupancy_fraction must lie in (0, 1]")
    target = occupancy_fraction * capacity
    low, high = np.finfo(float).tiny, max(1.0, target * (shape - 1.0) * 2.0)
    while _discrete_lomax_mean(high, shape) < target:
        high *= 2.0
    best = high
    for _ in range(60):
        middle = 0.5 * (low + high)
        mean = _discrete_lomax_mean(middle, shape)
        best = middle
        if abs(mean - target) <= relative_tolerance * target:
            break
        if mean < target:
            low = middle
        else:
            high = middle
    return best


class ParetoProxyGenerator:


    def __init__(self, shape: float, scale: float) -> None:
        if shape <= 1.0 or scale <= 0.0:
            raise ValueError("invalid Pareto proxy parameters")
        self.shape = float(shape)
        self.scale = float(scale)

    def sample(self, round_index: int, rng: np.random.Generator) -> ProxyDraw:
        del round_index
        raw = self.scale * ((1.0 - rng.random()) ** (-1.0 / self.shape) - 1.0)
        return ProxyDraw(lifetime=int(np.floor(raw)))

    def sample_lifetimes(
        self, size: int, rng: np.random.Generator, horizon_cap: Optional[int] = None
    ) -> np.ndarray:
        raw = self.scale * ((1.0 - rng.random(size)) ** (-1.0 / self.shape) - 1.0)
        if horizon_cap is not None:
            raw = np.minimum(raw, float(horizon_cap))
        return np.floor(raw).astype(np.int64)

    def survival_probability(self, actual_delay: int) -> float:
        if actual_delay < 0:
            raise ValueError("actual_delay must be nonnegative")
        return float((1.0 + actual_delay / self.scale) ** (-self.shape))

    def importance_weight(self, actual_delay: int) -> float:
        q = self.survival_probability(actual_delay)
        if q <= 0.0:
            raise FloatingPointError("nonpositive survival probability")
        return 1.0 / q


class UniformProxyGenerator:
    def __init__(self, capacity: int, sigma_hat: int, occupancy_fraction: float) -> None:
        if capacity <= 0 or sigma_hat < 0:
            raise ValueError("invalid capacity or backlog bound")
        if not 0.0 < occupancy_fraction <= 1.0:
            raise ValueError("occupancy_fraction must lie in (0, 1]")
        self.p = min(1.0, occupancy_fraction * capacity / max(1, sigma_hat))

    def sample(self, round_index: int, rng: np.random.Generator) -> ProxyDraw:
        del round_index
        selected = bool(rng.random() < self.p)
        return ProxyDraw(lifetime=None if selected else 0)

    def sample_selected(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.random(size) < self.p

    def survival_probability(self, actual_delay: int) -> float:
        if actual_delay < 0:
            raise ValueError("actual_delay must be nonnegative")
        return 1.0 if actual_delay == 0 else self.p

    def importance_weight(self, actual_delay: int) -> float:
        return 1.0 / self.survival_probability(actual_delay)
