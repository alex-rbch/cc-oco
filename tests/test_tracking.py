from __future__ import annotations

import unittest

import numpy as np

from src.simulation.environment import (
    CAPACITY_REJECTED,
    OBSERVED,
    run_simulation,
)
from tests.helpers import UnitWeightProxy, deterministic_environment, small_config


class TrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.delays = np.array([4, 3, 2, 1, 0], dtype=np.int64)
        gradients = np.tile([0.2, 0.0], (5, 1))
        self.environment = deterministic_environment(self.delays, gradients)

    def test_capacity_and_terminal_partition(self) -> None:
        config = small_config(horizon=5, capacity=2)
        draws = np.full(5, -1, dtype=np.int64)
        result = run_simulation(
            self.environment,
            config,
            "oco",
            "uniform",
            proxy_draws=draws,
            proxy_generator=UnitWeightProxy(),
        )
        self.assertLessEqual(result.maximum_active_set_size, 2)
        self.assertEqual(len(result.terminal_states), 5)
        self.assertEqual(np.count_nonzero(result.terminal_states == CAPACITY_REJECTED), 2)
        self.assertEqual(np.count_nonzero(result.terminal_states == OBSERVED), 3)

    def test_full_tracking_observes_every_item(self) -> None:
        result = run_simulation(
            self.environment, small_config(), "oco", "full"
        )
        self.assertTrue(np.all(result.terminal_states == OBSERVED))
        self.assertEqual(result.diagnostics["fraction_observed"], 1.0)


if __name__ == "__main__":
    unittest.main()
