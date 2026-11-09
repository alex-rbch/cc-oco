from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def project_euclidean_ball(x: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= radius or norm == 0.0:
        return x
    return x * (radius / norm)


@dataclass
class ProjectedOGDState:
    dimension: int
    radius: float
    learning_rate: float

    def __post_init__(self) -> None:
        if self.radius < 0.0 or self.learning_rate <= 0.0:
            raise ValueError("invalid radius or learning rate")
        self.point = np.zeros(self.dimension, dtype=np.float64)

    def action(self) -> np.ndarray:
        return self.point.copy()

    def observe(self, weighted_gradient: np.ndarray) -> float:
        norm = float(np.linalg.norm(weighted_gradient))
        if not np.isfinite(norm) or not np.all(np.isfinite(weighted_gradient)):
            raise FloatingPointError("nonfinite weighted gradient")
        self.point = project_euclidean_ball(
            self.point - self.learning_rate * weighted_gradient, self.radius
        )
        return norm


class OCOLearner:
    def __init__(self, dimension: int, radius: float, learning_rate: float) -> None:
        self.state = ProjectedOGDState(dimension, radius, learning_rate)

    def action(self) -> np.ndarray:
        return self.state.action()

    def observe(self, gradient: np.ndarray, importance_weight: float) -> float:
        return self.state.observe(importance_weight * gradient)


class BCOLearner:
    def __init__(
        self,
        dimension: int,
        radius: float,
        delta: float,
        learning_rate: float,
    ) -> None:
        if not 0.0 < delta <= radius:
            raise ValueError("delta must lie in (0, radius]")
        self.dimension = dimension
        self.delta = float(delta)
        self.state = ProjectedOGDState(dimension, radius - delta, learning_rate)

    def center(self) -> np.ndarray:
        return self.state.action()

    def action(self, direction: np.ndarray) -> np.ndarray:
        return self.center() + self.delta * direction

    def observe(
        self, scalar_loss: float, direction: np.ndarray, importance_weight: float
    ) -> float:
        estimator = (
            (self.dimension / self.delta)
            * scalar_loss
            * np.asarray(direction, dtype=np.float64)
        )
        return self.state.observe(importance_weight * estimator)


@dataclass(frozen=True)
class FixedLearningParameters:
    learning_rate: float
    smoothing_radius: float
    backlog_factor: float


def fixed_learning_parameters(
    problem_type: str,
    scheduler: str,
    horizon: int,
    realized_mean_delay: float,
    capacity: int,
    dimension: int,
    radius: float,
    oco_constant: float,
    bco_constant: float,
    smoothing_constant: float,
) -> FixedLearningParameters:

    if horizon <= 0 or realized_mean_delay < 0.0 or capacity <= 0:
        raise ValueError("invalid horizon, mean delay, or capacity")
    if problem_type == "oco":
        eta = oco_constant * 2.0 / np.sqrt(horizon * (1.0 + realized_mean_delay))
        return FixedLearningParameters(float(eta), 0.0, 1.0)
    if problem_type != "bco":
        raise ValueError("problem_type must be 'oco' or 'bco'")
    backlog_factor = (
        1.0
        if scheduler == "full"
        else 1.0 + realized_mean_delay / capacity
    )
    normalized_delta = min(
        1.0,
        smoothing_constant
        * np.sqrt(float(dimension))
        * backlog_factor ** 0.25
        / horizon ** 0.25,
    )
    delta = radius * normalized_delta
    denominator = np.sqrt(horizon) * np.sqrt(
        realized_mean_delay
        + (float(dimension) / normalized_delta) ** 2 * backlog_factor
    )
    eta = bco_constant * 2.0 / denominator
    return FixedLearningParameters(float(eta), float(delta), float(backlog_factor))


def sample_sphere_directions(
    horizon: int,
    dimension: int,
    rng: np.random.Generator,
    dtype: np.dtype = np.float32,
    chunk_size: int = 100_000,
) -> np.ndarray:
    if horizon <= 0 or dimension < 2:
        raise ValueError("invalid direction array shape")
    result = np.empty((horizon, dimension), dtype=dtype)
    for start in range(0, horizon, chunk_size):
        stop = min(horizon, start + chunk_size)
        block = rng.normal(size=(stop - start, dimension))
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        result[start:stop] = block
    return result
