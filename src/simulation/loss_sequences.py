from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np


def quantized_delay_threshold(
    delays: np.ndarray, quantile: float, step: int
) -> int:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    if step <= 0:
        raise ValueError("step must be positive")
    raw = float(np.quantile(np.asarray(delays, dtype=np.int64), quantile))
    return int(step * np.ceil(raw / step))


def lomax_tail_threshold(scale: float, shape: float, tail_fraction: float) -> int:
    if scale <= 0.0 or shape <= 0.0:
        raise ValueError("scale and shape must be positive")
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie in (0, 1)")
    raw = scale * (tail_fraction ** (-1.0 / shape) - 1.0)
    return max(1, int(np.ceil(raw)))


def generate_delay_correlated_gradients(
    delays: np.ndarray,
    dimension: int,
    threshold: int,
    tail_fraction: float,
    short_mass_balance: float,
    long_magnitude: float,
    angle_degrees: float,
) -> np.ndarray:
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if long_magnitude <= 0.0 or long_magnitude > 1.0:
        raise ValueError("long_magnitude must lie in (0, 1]")
    theta = np.deg2rad(angle_degrees)
    short_direction = np.zeros(dimension, dtype=np.float64)
    short_direction[0] = 1.0
    long_direction = np.zeros(dimension, dtype=np.float64)
    long_direction[0] = np.cos(theta)
    long_direction[1] = np.sin(theta)
    short_magnitude = min(1.0, short_mass_balance * tail_fraction / (1.0 - tail_fraction))
    is_long = np.asarray(delays) >= threshold
    gradients = np.empty((len(delays), dimension), dtype=np.float64)
    gradients[~is_long] = short_magnitude * short_direction
    gradients[is_long] = long_magnitude * long_direction
    norms = np.linalg.norm(gradients, axis=1)
    if np.any(norms > 1.0 + 1e-12):
        raise AssertionError("gradient norm exceeds one")
    return gradients


@dataclass
class GradientSequence:
    dimension: int
    is_long: np.ndarray
    short_vector: np.ndarray
    long_vector: np.ndarray
    explicit: Optional[np.ndarray] = None
    constant_offset: float = 0.0
    quadratic_coefficient: float = 0.0

    def __len__(self) -> int:
        return len(self.is_long)

    def vector(self, index: int) -> np.ndarray:
        if self.explicit is not None:
            return self.explicit[index]
        return self.long_vector if self.is_long[index] else self.short_vector

    def materialize(self) -> np.ndarray:
        if self.explicit is not None:
            return self.explicit.copy()
        result = np.empty((len(self), self.dimension), dtype=np.float64)
        result[~self.is_long] = self.short_vector
        result[self.is_long] = self.long_vector
        return result

    def total_sum(self) -> np.ndarray:
        if self.explicit is not None:
            return self.explicit.sum(axis=0, dtype=np.float64)
        n_long = int(np.count_nonzero(self.is_long))
        n_short = len(self) - n_long
        return n_short * self.short_vector + n_long * self.long_vector

    def loss(self, index: int, action: np.ndarray) -> float:
        value = self.constant_offset + float(np.dot(self.vector(index), action))
        if self.quadratic_coefficient:
            value += self.quadratic_coefficient * float(np.dot(action, action))
        return value

    def gradient(self, index: int, action: np.ndarray) -> np.ndarray:
        return self.vector(index) + 2.0 * self.quadratic_coefficient * action

    def with_quadratic_mixture(self, mixture: float) -> "GradientSequence":

        if not 0.0 <= mixture <= 1.0:
            raise ValueError("mixture must lie in [0, 1]")
        return replace(
            self,
            short_vector=(1.0 - mixture) * self.short_vector,
            long_vector=(1.0 - mixture) * self.long_vector,
            explicit=None if self.explicit is None else (1.0 - mixture) * self.explicit,
            constant_offset=(1.0 - mixture) * self.constant_offset,
            quadratic_coefficient=(1.0 - mixture) * self.quadratic_coefficient + mixture,
        )

    def static_comparator(self, radius: float) -> np.ndarray:
        total = self.total_sum()
        norm = float(np.linalg.norm(total))
        if norm == 0.0:
            return np.zeros(self.dimension, dtype=np.float64)
        if self.quadratic_coefficient:
            denominator = max(2.0 * self.quadratic_coefficient * len(self), norm / radius)
            return -total / denominator
        return -radius * total / norm

    def total_offset(self) -> float:
        return self.constant_offset * len(self)

    def group_sums(self) -> tuple[np.ndarray, np.ndarray]:
        if self.explicit is not None:
            short_sum = self.explicit[~self.is_long].sum(axis=0, dtype=np.float64)
            long_sum = self.explicit[self.is_long].sum(axis=0, dtype=np.float64)
            return short_sum, long_sum
        n_long = int(np.count_nonzero(self.is_long))
        return (len(self) - n_long) * self.short_vector, n_long * self.long_vector


def _add_direction_noise(
    gradients: np.ndarray,
    magnitude: np.ndarray,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise == 0.0:
        return gradients
    directions = rng.normal(size=gradients.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    mixed = (1.0 - noise) * gradients + noise * magnitude[:, None] * directions
    norms = np.linalg.norm(mixed, axis=1)
    too_large = norms > 1.0
    mixed[too_large] /= norms[too_large, None]
    return mixed


def build_gradient_sequence(
    delays: np.ndarray,
    dimension: int,
    threshold: int,
    tail_fraction: float,
    short_mass_balance: float,
    long_magnitude: float,
    angle_degrees: float,
    loss_coupling: str = "threshold",
    smooth_exponent: float = 1.0,
    minimum_gradient_magnitude: float = 0.1,
    maximum_direction_angle_degrees: float = 90.0,
    direction_noise: float = 0.0,
    noise_rng: Optional[np.random.Generator] = None,
    changing_direction_bias: float = 1.0,
) -> GradientSequence:
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    is_long = np.asarray(delays, dtype=np.int64) >= threshold
    if loss_coupling in {"delayed_indicator", "delayed_indicator_changing"}:
        short_vector = np.zeros(dimension, dtype=np.float64)
        long_vector = np.full(
            dimension, 0.5 / np.sqrt(float(dimension)), dtype=np.float64
        )
        if loss_coupling == "delayed_indicator_changing":
            if not 0.0 <= changing_direction_bias <= 1.0:
                raise ValueError("changing direction bias must lie in [0, 1]")
            if noise_rng is None:
                raise ValueError("noise_rng is required for changing directions")
            g_star = np.full(
                dimension, 1.0 / np.sqrt(float(dimension)), dtype=np.float64
            )
            orthogonal = noise_rng.normal(size=(len(delays), dimension))


            with np.errstate(all="ignore"):
                projections = orthogonal @ g_star
            orthogonal -= projections[:, None] * g_star
            if not np.all(np.isfinite(orthogonal)):
                raise FloatingPointError("nonfinite changing loss direction")
            norms = np.linalg.norm(orthogonal, axis=1)
            degenerate = norms <= np.finfo(np.float64).eps
            if np.any(degenerate):
                fallback = np.zeros(dimension, dtype=np.float64)
                fallback[0] = 1.0
                fallback -= float(np.dot(fallback, g_star)) * g_star
                fallback /= np.linalg.norm(fallback)
                orthogonal[degenerate] = fallback
                norms[degenerate] = 1.0
            orthogonal /= norms[:, None]
            directions = (
                changing_direction_bias * g_star
                + np.sqrt(1.0 - changing_direction_bias**2) * orthogonal
            )
            explicit = np.zeros((len(delays), dimension), dtype=np.float64)
            explicit[is_long] = 0.5 * directions[is_long]
            return GradientSequence(
                dimension,
                is_long,
                short_vector,
                long_vector,
                explicit=explicit,
                constant_offset=0.5,
            )
        return GradientSequence(
            dimension,
            is_long,
            short_vector,
            long_vector,
            explicit=None,
            constant_offset=0.5,
        )
    if loss_coupling == "threshold":
        short_magnitude = min(1.0, short_mass_balance * tail_fraction / (1.0 - tail_fraction))
        theta = np.deg2rad(angle_degrees)
        short_vector = np.zeros(dimension, dtype=np.float64)
        short_vector[0] = short_magnitude
        long_vector = np.zeros(dimension, dtype=np.float64)
        long_vector[0] = long_magnitude * np.cos(theta)
        long_vector[1] = long_magnitude * np.sin(theta)
        sequence = GradientSequence(dimension, is_long, short_vector, long_vector)
        if direction_noise > 0.0:
            if noise_rng is None:
                raise ValueError("noise_rng is required when direction noise is enabled")
            explicit = sequence.materialize()
            magnitudes = np.linalg.norm(explicit, axis=1)
            explicit = _add_direction_noise(explicit, magnitudes, direction_noise, noise_rng)
            sequence.explicit = explicit
        return sequence
    if loss_coupling != "smooth":
        raise ValueError("unknown loss coupling")
    z = np.minimum(1.0, (np.asarray(delays, dtype=np.float64) / threshold) ** smooth_exponent)
    magnitudes = minimum_gradient_magnitude + (1.0 - minimum_gradient_magnitude) * z
    angles = np.deg2rad(maximum_direction_angle_degrees) * z
    explicit = np.zeros((len(delays), dimension), dtype=np.float64)
    explicit[:, 0] = magnitudes * np.cos(angles)
    explicit[:, 1] = magnitudes * np.sin(angles)
    if direction_noise > 0.0:
        if noise_rng is None:
            raise ValueError("noise_rng is required when direction noise is enabled")
        explicit = _add_direction_noise(explicit, magnitudes, direction_noise, noise_rng)
    return GradientSequence(
        dimension,
        is_long,
        np.zeros(dimension, dtype=np.float64),
        np.zeros(dimension, dtype=np.float64),
        explicit,
    )


def build_iid_control_gradients(
    horizon: int,
    dimension: int,
    is_long: np.ndarray,
    rng: np.random.Generator,
    rho: float = 0.25,
) -> GradientSequence:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    noise = rng.normal(size=(horizon, dimension))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    gradients = np.sqrt(1.0 - rho * rho) * noise
    gradients[:, 0] += rho
    norms = np.linalg.norm(gradients, axis=1)
    gradients /= np.maximum(1.0, norms)[:, None]
    return GradientSequence(
        dimension,
        np.asarray(is_long, dtype=bool).copy(),
        np.zeros(dimension),
        np.zeros(dimension),
        gradients,
    )


def gradient_mass_diagnostics(sequence: GradientSequence) -> Dict[str, float]:
    short_sum, long_sum = sequence.group_sums()
    short_mass = float(np.linalg.norm(short_sum))
    long_mass = float(np.linalg.norm(long_sum))
    denom = short_mass * long_mass
    if denom == 0.0:
        angle = float("nan")
    else:
        cosine = float(np.clip(np.dot(short_sum, long_sum) / denom, -1.0, 1.0))
        angle = float(np.rad2deg(np.arccos(cosine)))
    n_long = int(np.count_nonzero(sequence.is_long))
    return {
        "number_short_rounds": len(sequence) - n_long,
        "number_long_rounds": n_long,
        "realized_long_fraction": n_long / max(1, len(sequence)),
        "short_total_gradient_mass": short_mass,
        "long_total_gradient_mass": long_mass,
        "long_to_short_mass_ratio": long_mass / short_mass if short_mass else float("inf"),
        "angle_between_group_sums": angle,
    }
