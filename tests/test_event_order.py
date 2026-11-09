from __future__ import annotations

import unittest

import numpy as np

from src.simulation.environment import OBSERVED, run_simulation
from tests.helpers import UnitWeightProxy, deterministic_environment, small_config


class EventOrderTests(unittest.TestCase):
    def test_arrival_at_exact_expiry_is_observed(self) -> None:
        delays = np.array([2, 1, 1, 1, 0], dtype=np.int64)
        gradients = np.tile([0.1, 0.0], (5, 1))
        environment = deterministic_environment(delays, gradients)
        result = run_simulation(
            environment,
            small_config(horizon=5, capacity=2),
            "oco",
            "pareto",
            proxy_draws=delays,
            proxy_generator=UnitWeightProxy(),
        )
        self.assertTrue(np.all(result.terminal_states == OBSERVED))

    def test_zero_delay_is_processed_after_admission_same_step(self) -> None:
        environment = deterministic_environment(
            np.array([0], dtype=np.int64), np.array([[1.0, 0.0]])
        )
        result = run_simulation(
            environment,
            small_config(horizon=1, capacity=1),
            "oco",
            "pareto",
            proxy_draws=np.array([0]),
            proxy_generator=UnitWeightProxy(),
        )
        self.assertEqual(result.terminal_states[0], OBSERVED)


if __name__ == "__main__":
    unittest.main()
