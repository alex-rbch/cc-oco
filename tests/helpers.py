from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.simulation.config import ExperimentConfig
from src.simulation.environment import EnvironmentData
from src.simulation.loss_sequences import GradientSequence, gradient_mass_diagnostics


def small_config(horizon: int = 5, capacity: int = 2, dimension: int = 2) -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        horizon=horizon,
        capacity=capacity,
        target_mean_delays=(capacity + 1,),
        actual_delay_distribution="lomax",
        dimension=dimension,
        calibration_samples=20_000,
        final_seeds=(0,),
        bco_smoothing_constant=0.1 / np.sqrt(dimension),
    )


def deterministic_environment(
    delays: np.ndarray,
    gradients: np.ndarray,
    directions: np.ndarray | None = None,
) -> EnvironmentData:
    horizon, dimension = gradients.shape
    if directions is None:
        directions = np.zeros((horizon, dimension), dtype=np.float32)
        directions[:, 0] = 1.0
    is_long = np.zeros(horizon, dtype=bool)
    sequence = GradientSequence(
        dimension,
        is_long,
        np.zeros(dimension),
        np.zeros(dimension),
        gradients.astype(np.float64),
    )
    diagnostics = {
        "target_mean_delay": 3,
        "realized_mean_delay": float(np.mean(delays)),
        "actual_delay_shape": 1.5,
        "delay_scale": 1.0,
        "delay_threshold": 2,
        "realized_long_fraction": 0.0,
        "sigma_max": horizon,
        "sigma_max_over_capacity": 1.0,
    }
    diagnostics.update(gradient_mass_diagnostics(sequence))
    return EnvironmentData(
        seed=0,
        target_mean_delay=3,
        delay_scale=1.0,
        delay_threshold=2,
        delays=np.asarray(delays, dtype=np.int64),
        gradients=sequence,
        bco_directions=np.asarray(directions, dtype=np.float32),
        sigma_max=horizon,
        diagnostics=diagnostics,
    )


class UnitWeightProxy:
    def importance_weight(self, actual_delay: int) -> float:
        return 1.0

    def survival_probability(self, actual_delay: int) -> float:
        return 1.0
