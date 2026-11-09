from __future__ import annotations

import heapq
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .actual_delays import (
    compute_backlog,
    delay_diagnostics,
    sample_lomax_delays,
    sample_two_point_delays,
)
from .config import ExperimentConfig
from .learners import (
    BCOLearner,
    OCOLearner,
    FixedLearningParameters,
    fixed_learning_parameters,
    sample_sphere_directions,
)
from .loss_sequences import (
    GradientSequence,
    build_gradient_sequence,
    build_iid_control_gradients,
    gradient_mass_diagnostics,
    lomax_tail_threshold,
    quantized_delay_threshold,
)
from .proxy_delays import (
    ParetoProxyGenerator,
    UniformProxyGenerator,
    calibrate_pareto_proxy_scale,
)


UNASSIGNED = np.uint8(255)
ACTIVE = np.uint8(0)
OBSERVED = np.uint8(1)
PROXY_REJECTED = np.uint8(2)
PROXY_EXPIRED = np.uint8(3)
CAPACITY_REJECTED = np.uint8(4)
HORIZON_UNRESOLVED = np.uint8(5)

STATE_NAMES = {
    int(OBSERVED): "observed",
    int(PROXY_REJECTED): "proxy_rejected",
    int(PROXY_EXPIRED): "proxy_expired",
    int(CAPACITY_REJECTED): "capacity_rejected",
    int(HORIZON_UNRESOLVED): "horizon_unresolved",
}


def rng_stream(seed: int, stream: int, target_mean_delay: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream, target_mean_delay]))


@dataclass
class EnvironmentData:
    seed: int
    target_mean_delay: int
    delay_scale: float
    delay_threshold: int
    delays: np.ndarray
    gradients: GradientSequence
    bco_directions: np.ndarray
    sigma_max: int
    diagnostics: Dict[str, Any]


@dataclass
class SimulationOutcome:
    raw: Dict[str, Any]
    diagnostics: Dict[str, Any]
    terminal_states: np.ndarray
    maximum_active_set_size: int


def build_environment(
    config: ExperimentConfig,
    target_mean_delay: int,
    seed: int,
    delay_scale: float,
    iid_control: bool = False,
) -> EnvironmentData:
    delay_rng = rng_stream(seed, 10, target_mean_delay)
    if config.actual_delay_distribution in {"two_point", "binary"}:
        delays = sample_two_point_delays(
            config.horizon,
            config.small_delay,
            target_mean_delay,
            config.large_delay_probability_for(target_mean_delay),
            delay_rng,
        )
    else:
        delays = sample_lomax_delays(
            config.horizon, config.actual_delay_shape, delay_scale, delay_rng
        )
    configured_loss_quantile = config.loss_quantile_for(target_mean_delay)
    configured_loss_threshold = (
        quantized_delay_threshold(
            delays, configured_loss_quantile, config.loss_threshold_step
        )
        if configured_loss_quantile is not None
        else config.loss_threshold_for(target_mean_delay)
    )
    threshold = (
        configured_loss_threshold + 1
        if config.loss_coupling in {
            "delayed_indicator",
            "delayed_indicator_changing",
        }
        else (
            config.small_delay + 1
            if config.actual_delay_distribution in {"two_point", "binary"}
            else lomax_tail_threshold(
                delay_scale, config.actual_delay_shape, config.long_delay_fraction
            )
        )
    )
    noise_rng = rng_stream(seed, 20, target_mean_delay)
    gradients = build_gradient_sequence(
        delays,
        config.dimension,
        threshold,
        config.long_delay_fraction,
        config.short_mass_balance,
        config.long_gradient_magnitude,
        config.direction_angle_degrees,
        config.loss_coupling,
        config.smooth_exponent,
        config.minimum_gradient_magnitude,
        config.maximum_direction_angle_degrees,
        config.direction_noise,
        noise_rng,
        config.changing_direction_bias,
    )
    if iid_control:
        if config.quadratic_mixture:
            raise ValueError("quadratic mixing is only supported for the bounded delay-dependent losses")
        gradients = build_iid_control_gradients(
            config.horizon,
            config.dimension,
            gradients.is_long,
            noise_rng,
        )
    if config.quadratic_mixture:
        gradients = gradients.with_quadratic_mixture(config.quadratic_mixture)
    direction_rng = rng_stream(seed, 30, target_mean_delay)
    directions = sample_sphere_directions(
        config.horizon, config.dimension, direction_rng, dtype=np.float32
    )
    backlog = compute_backlog(delays)
    sigma_max = int(backlog.max(initial=0))
    diagnostics: Dict[str, Any] = delay_diagnostics(
        delays, target_mean_delay, config.capacity
    )
    diagnostics.update(gradient_mass_diagnostics(gradients))
    diagnostics.update(
        {
            "actual_delay_distribution": config.actual_delay_distribution,
            "actual_delay_shape": config.actual_delay_shape,
            "delay_scale": delay_scale,
            "small_delay": config.small_delay,
            "large_delay": target_mean_delay,
            "large_delay_probability": config.large_delay_probability_for(
                target_mean_delay
            ),
            "delay_threshold": threshold,
            "loss_delay_threshold": configured_loss_threshold,
            "loss_delay_quantile": configured_loss_quantile,
            "loss_threshold_step": config.loss_threshold_step,
            "changing_direction_bias": config.changing_direction_bias,
            "quadratic_mixture": config.quadratic_mixture,
            "strong_convexity": 2.0 * gradients.quadratic_coefficient,
        }
    )
    return EnvironmentData(
        seed,
        target_mean_delay,
        delay_scale,
        threshold,
        delays,
        gradients,
        directions,
        sigma_max,
        diagnostics,
    )


def _proxy_for(
    scheduler: str, config: ExperimentConfig, environment: EnvironmentData
) -> Any:
    if scheduler == "full":
        return None
    if scheduler == "pareto":
        scale = calibrate_pareto_proxy_scale(
            config.capacity,
            config.pareto_proxy_shape,
            config.proxy_occupancy_fraction,
            horizon=config.horizon,
        )
        return ParetoProxyGenerator(config.pareto_proxy_shape, scale)
    if scheduler == "uniform":
        return UniformProxyGenerator(
            config.capacity,
            environment.sigma_max,
            config.proxy_occupancy_fraction,
        )
    raise ValueError(f"unknown scheduler: {scheduler}")


def _proxy_draws(
    scheduler: str,
    proxy: Any,
    environment: EnvironmentData,
    rng: np.random.Generator,
) -> np.ndarray:
    horizon = len(environment.delays)
    if scheduler == "full":
        return np.full(horizon, -1, dtype=np.int64)
    if scheduler == "pareto":
        return proxy.sample_lifetimes(horizon, rng, horizon_cap=horizon)
    selected = proxy.sample_selected(horizon, rng)
    return np.where(selected, -1, 0).astype(np.int64)


def _finalize_outcome(
    environment: EnvironmentData,
    config: ExperimentConfig,
    problem_type: str,
    scheduler: str,
    proxy: Any,
    states: np.ndarray,
    retained_weights: np.ndarray,
    learner_loss: float,
    short_learner_loss: float,
    long_learner_loss: float,
    maximum_weighted_gradient_norm: float,
    number_nonfinite_updates: int,
    maximum_active: int,
    saturated_rounds: int,
    observed_short: int,
    observed_long: int,
    learning_parameters: FixedLearningParameters,
) -> SimulationOutcome:
    horizon = len(environment.delays)
    total_gradient = environment.gradients.total_sum()
    comparator = environment.gradients.static_comparator(config.domain_radius)
    quadratic_at_comparator = environment.gradients.quadratic_coefficient * float(
        np.dot(comparator, comparator)
    )
    if environment.gradients.quadratic_coefficient:
        comparator_loss = (
            environment.gradients.total_offset()
            + float(np.dot(total_gradient, comparator))
            + horizon * quadratic_at_comparator
        )
    else:
        comparator_loss = environment.gradients.total_offset() - config.domain_radius * float(
            np.linalg.norm(total_gradient)
        )
    total_regret = learner_loss - comparator_loss
    normalized_regret = total_regret / horizon
    short_sum, long_sum = environment.gradients.group_sums()
    n_long = int(np.count_nonzero(environment.gradients.is_long))
    n_short = horizon - n_long
    short_comparator_loss = (
        environment.gradients.constant_offset * n_short
        + float(np.dot(short_sum, comparator))
        + n_short * quadratic_at_comparator
    )
    long_comparator_loss = (
        environment.gradients.constant_offset * n_long
        + float(np.dot(long_sum, comparator))
        + n_long * quadratic_at_comparator
    )
    if len(retained_weights):
        weight_stats = {
            "mean_importance_weight": float(np.mean(retained_weights)),
            "median_importance_weight": float(np.percentile(retained_weights, 50)),
            "p95_importance_weight": float(np.percentile(retained_weights, 95)),
            "p99_importance_weight": float(np.percentile(retained_weights, 99)),
            "maximum_importance_weight": float(np.max(retained_weights)),
            "mean_squared_importance_weight": float(
                np.mean(retained_weights * retained_weights)
            ),
        }
    else:
        weight_stats = {
            key: float("nan")
            for key in (
                "mean_importance_weight",
                "median_importance_weight",
                "p95_importance_weight",
                "p99_importance_weight",
                "maximum_importance_weight",
                "mean_squared_importance_weight",
            )
        }
    counts = {
        name: int(np.count_nonzero(states == code))
        for code, name in STATE_NAMES.items()
    }
    if sum(counts.values()) != horizon:
        raise AssertionError("terminal-state count mismatch")
    raw = {
        "problem_type": problem_type,
        "scheduler": scheduler,
        "seed": environment.seed,
        "target_mean_delay": environment.target_mean_delay,
        "normalized_regret": normalized_regret,
        "total_regret": total_regret,
        "learner_loss": learner_loss,
        "comparator_loss": comparator_loss,
    }
    diagnostics: Dict[str, Any] = dict(environment.diagnostics)
    diagnostics.update(raw)
    diagnostics.update(
        {
            "fraction_observed": counts["observed"] / horizon,
            "fraction_proxy_rejected": counts["proxy_rejected"] / horizon,
            "fraction_proxy_expired": counts["proxy_expired"] / horizon,
            "fraction_capacity_rejected": counts["capacity_rejected"] / horizon,
            "fraction_horizon_unresolved": counts["horizon_unresolved"] / horizon,
            "fraction_saturated_rounds": saturated_rounds / horizon,
            "short_round_observation_rate": observed_short / max(1, n_short),
            "long_round_observation_rate": observed_long / max(1, n_long),
            "short_round_regret_contribution": short_learner_loss
            - short_comparator_loss,
            "long_round_regret_contribution": long_learner_loss
            - long_comparator_loss,
            "maximum_weighted_gradient_norm": maximum_weighted_gradient_norm,
            "number_nonfinite_updates": number_nonfinite_updates,
            "number_clipped_updates": 0,
            "maximum_active_set_size": maximum_active,
            "fixed_learning_rate": learning_parameters.learning_rate,
            "fixed_smoothing_radius": learning_parameters.smoothing_radius,
            "bco_backlog_factor": learning_parameters.backlog_factor,
            "oco_learning_rate_constant": config.oco_learning_rate_constant,
            "bco_learning_rate_constant": config.bco_learning_rate_constant,
            "bco_smoothing_constant": config.bco_smoothing_constant,
            "proxy_implementation": "full_tracking"
            if scheduler == "full"
            else "fallback_discrete_lomax"
            if scheduler == "pareto"
            else "fallback_all_or_nothing",
            "proxy_selection_probability": getattr(proxy, "p", float("nan")),
            "proxy_scale": getattr(proxy, "scale", float("nan")),
            "comparator_norm": float(np.linalg.norm(comparator)),
        }
    )
    diagnostics.update(weight_stats)
    return SimulationOutcome(raw, diagnostics, states, maximum_active)


def run_simulation(
    environment: EnvironmentData,
    config: ExperimentConfig,
    problem_type: str,
    scheduler: str,
    proxy_draws: Optional[np.ndarray] = None,
    proxy_generator: Any = None,
) -> SimulationOutcome:
    if problem_type not in {"oco", "bco"}:
        raise ValueError("problem_type must be 'oco' or 'bco'")
    horizon = len(environment.delays)
    if horizon != config.horizon:
        raise ValueError("environment and configuration horizons differ")
    proxy = proxy_generator if proxy_generator is not None else _proxy_for(
        scheduler, config, environment
    )
    if proxy_draws is None:
        stream = 40 if scheduler == "pareto" else 50
        proxy_rng = rng_stream(
            environment.seed, stream, environment.target_mean_delay
        )
        proxy_draws = _proxy_draws(scheduler, proxy, environment, proxy_rng)
    else:
        proxy_draws = np.asarray(proxy_draws, dtype=np.int64)
        if len(proxy_draws) != horizon:
            raise ValueError("proxy draw array has the wrong length")

    learning_parameters = fixed_learning_parameters(
        problem_type,
        scheduler,
        horizon,
        float(np.mean(environment.delays)),
        config.capacity,
        config.dimension,
        config.domain_radius,
        config.oco_learning_rate_constant,
        config.bco_learning_rate_constant,
        config.bco_smoothing_constant,
    )

    if horizon >= 10_000 and os.environ.get("SIMULATION_DISABLE_NATIVE") != "1":
        from .native import run_native_core

        native = run_native_core(
            environment,
            config,
            problem_type,
            scheduler,
            proxy_draws,
            proxy,
            learning_parameters,
        )
        if native is not None:
            return _finalize_outcome(
                environment,
                config,
                problem_type,
                scheduler,
                proxy,
                native["states"],
                native["weights"],
                native["learner_loss"],
                native["short_learner_loss"],
                native["long_learner_loss"],
                native["maximum_weighted_gradient_norm"],
                native["number_nonfinite_updates"],
                native["maximum_active"],
                native["saturated_rounds"],
                native["observed_short"],
                native["observed_long"],
                learning_parameters,
            )

    if problem_type == "oco":
        learner: Any = OCOLearner(
            config.dimension,
            config.domain_radius,
            learning_parameters.learning_rate,
        )
    else:
        learner = BCOLearner(
            config.dimension,
            config.domain_radius,
            learning_parameters.smoothing_radius,
            learning_parameters.learning_rate,
        )

    delays = environment.delays
    feedback_times = np.arange(horizon, dtype=np.int64) + delays
    arrival_order = np.argsort(feedback_times, kind="stable")
    states = np.full(horizon, UNASSIGNED, dtype=np.uint8)
    scalar_losses = (
        np.empty(horizon, dtype=np.float64) if problem_type == "bco" else None
    )
    played_gradients = (
        np.empty((horizon, config.dimension), dtype=np.float64)
        if problem_type == "oco" and environment.gradients.quadratic_coefficient
        else None
    )
    active: Dict[int, int] = {}
    expiry_heap: list[tuple[int, int]] = []
    arrival_pointer = 0
    weights = np.empty(horizon, dtype=np.float64)
    weight_count = 0
    observed_short = 0
    observed_long = 0
    saturated_rounds = 0
    maximum_active = 0
    maximum_weighted_gradient_norm = 0.0
    number_nonfinite_updates = 0
    learner_loss = 0.0
    short_learner_loss = 0.0
    long_learner_loss = 0.0

    def observe_round(round_index: int) -> None:
        nonlocal weight_count, observed_short, observed_long
        nonlocal maximum_weighted_gradient_norm, number_nonfinite_updates
        delay = int(delays[round_index])
        weight = 1.0 if scheduler == "full" else float(proxy.importance_weight(delay))
        weights[weight_count] = weight
        weight_count += 1
        if environment.gradients.is_long[round_index]:
            observed_long += 1
        else:
            observed_short += 1
        try:
            if problem_type == "oco":
                update_norm = learner.observe(
                    environment.gradients.vector(round_index)
                    if played_gradients is None else played_gradients[round_index], weight
                )
            else:
                assert scalar_losses is not None
                update_norm = learner.observe(
                    float(scalar_losses[round_index]),
                    environment.bco_directions[round_index],
                    weight,
                )
            maximum_weighted_gradient_norm = max(
                maximum_weighted_gradient_norm, update_norm
            )
        except FloatingPointError:
            number_nonfinite_updates += 1
        states[round_index] = OBSERVED

    for time in range(horizon):


        while (
            arrival_pointer < horizon
            and feedback_times[arrival_order[arrival_pointer]] == time
        ):
            round_index = int(arrival_order[arrival_pointer])
            arrival_pointer += 1
            if round_index >= time:
                continue
            if round_index in active:
                del active[round_index]
                observe_round(round_index)

        while expiry_heap and expiry_heap[0][0] == time:
            _, round_index = heapq.heappop(expiry_heap)
            if round_index in active:
                del active[round_index]
                states[round_index] = PROXY_EXPIRED

        if problem_type == "oco":
            action = learner.action()
        else:
            action = learner.action(environment.bco_directions[time])
        current_loss = environment.gradients.loss(time, action)
        if environment.gradients.quadratic_coefficient and not -1e-7 <= current_loss <= 1.0 + 1e-7:
            raise AssertionError("quadratic-mixture loss is outside [0, 1]")
        if played_gradients is not None:

            played_gradients[time] = environment.gradients.gradient(time, action)
        learner_loss += current_loss
        if environment.gradients.is_long[time]:
            long_learner_loss += current_loss
        else:
            short_learner_loss += current_loss
        if scalar_losses is not None:
            scalar_losses[time] = current_loss

        if scheduler != "full" and len(active) >= config.capacity:
            saturated_rounds += 1

        delay = int(delays[time])
        if delay == 0:


            observe_round(time)
        else:
            lifetime = int(proxy_draws[time])
            if lifetime == 0:
                states[time] = PROXY_REJECTED
            elif scheduler != "full" and len(active) >= config.capacity:
                states[time] = CAPACITY_REJECTED
            else:
                expiry_time = -1 if lifetime < 0 else time + lifetime
                active[time] = expiry_time
                states[time] = ACTIVE
                if expiry_time >= 0 and expiry_time < horizon:
                    heapq.heappush(expiry_heap, (expiry_time, time))
                maximum_active = max(maximum_active, len(active))
                if scheduler != "full" and len(active) > config.capacity:
                    raise AssertionError("tracking capacity exceeded")

    for round_index in active:
        states[round_index] = HORIZON_UNRESOLVED
    if np.any((states == UNASSIGNED) | (states == ACTIVE)):
        raise AssertionError("terminal states do not partition all rounds")

    return _finalize_outcome(
        environment,
        config,
        problem_type,
        scheduler,
        proxy,
        states,
        weights[:weight_count],
        learner_loss,
        short_learner_loss,
        long_learner_loss,
        maximum_weighted_gradient_norm,
        number_nonfinite_updates,
        maximum_active,
        saturated_rounds,
        observed_short,
        observed_long,
        learning_parameters,
    )
