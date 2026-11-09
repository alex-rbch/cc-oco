from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    horizon: int = 100_000
    capacity: int = 20
    target_mean_delays: Tuple[int, ...] = (100, 200, 400, 800)
    dimension: int = 5
    domain_radius: float = 1.0

    actual_delay_distribution: str = "lomax"
    actual_delay_shape: float = 1.9
    small_delay: int = 40
    calibration_seed: int = 9999
    calibration_samples: int = 5_000_000
    calibration_relative_tolerance: float = 5e-3

    long_delay_fraction: float = 0.50
    large_delay_probabilities: Tuple[Tuple[int, float], ...] = ()
    long_gradient_magnitude: float = 1.0
    short_mass_balance: float = 1.02
    direction_angle_degrees: float = 180.0
    direction_noise: float = 0.0
    changing_direction_bias: float = 1.0
    quadratic_mixture: float = 0.0
    loss_coupling: str = "delayed_indicator"
    loss_delay_threshold: int = 150
    loss_delay_thresholds: Tuple[Tuple[int, int], ...] = ()
    loss_delay_quantiles: Tuple[Tuple[int, float], ...] = (
        (100, 0.70),
        (200, 0.50),
        (400, 0.40),
        (800, 0.27),
    )
    loss_threshold_step: int = 25
    smooth_exponent: float = 1.0
    minimum_gradient_magnitude: float = 0.10
    maximum_direction_angle_degrees: float = 90.0

    pareto_proxy_shape: float = 2.5
    proxy_occupancy_fraction: float = 0.75

    oco_learning_rate_constant: float = 1.0
    bco_learning_rate_constant: float = 1.0
    bco_smoothing_constant: float = 1.0

    final_seeds: Tuple[int, ...] = tuple(range(100, 200))
    dtype: str = "float64"

    smoke_horizon: int = 100_000
    smoke_capacity: int = 20
    smoke_mean_delays: Tuple[int, ...] = (100, 200, 400, 800)
    smoke_seeds: Tuple[int, ...] = tuple(range(30))

    def validate(self) -> None:
        import math

        if self.horizon <= 1:
            raise ValueError("horizon must exceed one")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.dimension < 2:
            raise ValueError("dimension must be at least two")
        if not math.log(self.horizon) < self.capacity:
            raise ValueError("the requested regime requires log(T) < C")
        if min(self.target_mean_delays) <= self.capacity:
            raise ValueError("the requested regime requires C < every target mean delay")
        if self.actual_delay_distribution not in {"lomax", "two_point", "binary"}:
            raise ValueError("actual delay distribution must be Lomax or binary")
        if self.actual_delay_distribution == "lomax" and self.actual_delay_shape <= 1.0:
            raise ValueError("actual_delay_shape must exceed one")
        if self.small_delay < 0 or self.small_delay >= min(self.target_mean_delays):
            raise ValueError("small_delay must be nonnegative and below every large delay")
        if not 0.0 < self.long_delay_fraction < 1.0:
            raise ValueError("long_delay_fraction must lie in (0, 1)")
        if any(
            target < 0 or not 0.0 < probability < 1.0
            for target, probability in self.large_delay_probabilities
        ):
            raise ValueError("large-delay probability map is invalid")
        if self.loss_coupling not in {
            "threshold",
            "smooth",
            "delayed_indicator",
            "delayed_indicator_changing",
        }:
            raise ValueError(
                "unknown loss coupling"
            )
        if self.loss_delay_threshold < 0:
            raise ValueError("loss_delay_threshold must be nonnegative")
        if any(target < 0 or threshold < 0 for target, threshold in self.loss_delay_thresholds):
            raise ValueError("loss_delay_thresholds must be nonnegative")
        if any(
            target < 0 or not 0.0 <= quantile <= 1.0
            for target, quantile in self.loss_delay_quantiles
        ):
            raise ValueError("loss-delay quantiles are invalid")
        if self.loss_threshold_step <= 0:
            raise ValueError("loss_threshold_step must be positive")
        if not 0.0 <= self.direction_noise <= 1.0:
            raise ValueError("direction_noise must lie in [0, 1]")
        if not 0.0 <= self.changing_direction_bias <= 1.0:
            raise ValueError("changing_direction_bias must lie in [0, 1]")
        if not 0.0 <= self.quadratic_mixture <= 1.0:
            raise ValueError("quadratic_mixture must lie in [0, 1]")
        if self.quadratic_mixture and (
            self.domain_radius != 1.0
            or self.loss_coupling not in {"delayed_indicator", "delayed_indicator_changing"}
        ):
            raise ValueError("quadratic mixing requires bounded indicator losses on the unit ball")
        if self.oco_learning_rate_constant <= 0.0:
            raise ValueError("oco_learning_rate_constant must be positive")
        if self.bco_learning_rate_constant <= 0.0:
            raise ValueError("bco_learning_rate_constant must be positive")
        if self.bco_smoothing_constant <= 0.0:
            raise ValueError("bco_smoothing_constant must be positive")
        if self.dtype != "float64":
            raise ValueError("learner state must use float64")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("target_mean_delays", "final_seeds", "smoke_mean_delays", "smoke_seeds"):
            result[key] = list(result[key])
        result["loss_delay_thresholds"] = {
            int(target): int(threshold)
            for target, threshold in self.loss_delay_thresholds
        }
        result["large_delay_probabilities"] = {
            int(target): float(probability)
            for target, probability in self.large_delay_probabilities
        }
        result["loss_delay_quantiles"] = {
            int(target): float(quantile)
            for target, quantile in self.loss_delay_quantiles
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        values = dict(data)
        for key in ("target_mean_delays", "final_seeds", "smoke_mean_delays", "smoke_seeds"):
            if key in values:
                values[key] = tuple(values[key])
        if "loss_delay_thresholds" in values:
            mapping = values["loss_delay_thresholds"] or {}
            values["loss_delay_thresholds"] = tuple(
                sorted((int(target), int(threshold)) for target, threshold in mapping.items())
            )
        if "large_delay_probabilities" in values:
            mapping = values["large_delay_probabilities"] or {}
            values["large_delay_probabilities"] = tuple(
                sorted(
                    (int(target), float(probability))
                    for target, probability in mapping.items()
                )
            )
        if "loss_delay_quantiles" in values:
            mapping = values["loss_delay_quantiles"] or {}
            values["loss_delay_quantiles"] = tuple(
                sorted(
                    (int(target), float(quantile))
                    for target, quantile in mapping.items()
                )
            )
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with path.open("r", encoding="utf-8") as handle:
            contents = yaml.safe_load(handle) or {}
        return cls.from_dict(contents)

    def write_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def smoke_config(self) -> "ExperimentConfig":
        return replace(
            self,
            horizon=self.smoke_horizon,
            capacity=self.smoke_capacity,
            target_mean_delays=self.smoke_mean_delays,
            final_seeds=self.smoke_seeds,
            calibration_samples=min(self.calibration_samples, 1_000_000),
        )

    def loss_threshold_for(self, target_mean_delay: int) -> int:
        if self.actual_delay_distribution in {"two_point", "binary"}:
            return int(self.small_delay)
        mapping = dict(self.loss_delay_thresholds)
        return int(mapping.get(int(target_mean_delay), self.loss_delay_threshold))

    def large_delay_probability_for(self, target_mean_delay: int) -> float:
        mapping = dict(self.large_delay_probabilities)
        return float(mapping.get(int(target_mean_delay), self.long_delay_fraction))

    def loss_quantile_for(self, target_mean_delay: int) -> Optional[float]:
        mapping = dict(self.loss_delay_quantiles)
        value = mapping.get(int(target_mean_delay))
        return None if value is None else float(value)


def with_overrides(config: ExperimentConfig, **kwargs: Any) -> ExperimentConfig:
    updated = replace(config, **kwargs)
    updated.validate()
    return updated


def parse_int_list(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())
