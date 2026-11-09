from __future__ import annotations

import unittest

import numpy as np

from src.simulation.actual_delays import calibrate_target_scales
from src.simulation.environment import build_environment
from tests.helpers import small_config


class PairedEnvironmentTests(unittest.TestCase):
    def test_environment_repeats_exactly_for_same_seed(self) -> None:
        config = small_config(horizon=200, capacity=6, dimension=4)
        target = 8
        scale = calibrate_target_scales(
            [target], 1.5, config.horizon, 50_000, 9999
        )[float(target)]
        first = build_environment(config, target, 7, scale)
        second = build_environment(config, target, 7, scale)
        np.testing.assert_array_equal(first.delays, second.delays)
        np.testing.assert_array_equal(first.gradients.is_long, second.gradients.is_long)
        np.testing.assert_array_equal(first.gradients.materialize(), second.gradients.materialize())
        np.testing.assert_array_equal(first.bco_directions, second.bco_directions)
        np.testing.assert_array_equal(first.gradients.total_sum(), second.gradients.total_sum())


if __name__ == "__main__":
    unittest.main()
