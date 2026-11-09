from __future__ import annotations

import unittest

import numpy as np

from src.simulation.proxy_delays import ParetoProxyGenerator, UniformProxyGenerator


class UnbiasednessTests(unittest.TestCase):
    def test_weighted_retention_expectation(self) -> None:
        rng = np.random.default_rng(55)
        pareto = ParetoProxyGenerator(1.5, 5.0)
        uniform = UniformProxyGenerator(20, 200, 0.75)
        for delay in (2, 15, 60):
            draws = pareto.sample_lifetimes(800_000, rng)
            estimate = np.mean((draws >= delay) * pareto.importance_weight(delay))
            self.assertAlmostEqual(float(estimate), 1.0, delta=0.025)
            selected = uniform.sample_selected(800_000, rng)
            uniform_estimate = np.mean(selected * uniform.importance_weight(delay))
            self.assertAlmostEqual(float(uniform_estimate), 1.0, delta=0.025)


if __name__ == "__main__":
    unittest.main()
