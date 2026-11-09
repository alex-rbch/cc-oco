from __future__ import annotations

import unittest

import numpy as np

from src.simulation.proxy_delays import ParetoProxyGenerator, UniformProxyGenerator


class ProxyDelayTests(unittest.TestCase):
    def test_pareto_sampling_matches_survival(self) -> None:
        proxy = ParetoProxyGenerator(shape=1.5, scale=7.0)
        lifetimes = proxy.sample_lifetimes(500_000, np.random.default_rng(2))
        for delay in (1, 5, 20):
            empirical = float(np.mean(lifetimes >= delay))
            self.assertAlmostEqual(empirical, proxy.survival_probability(delay), delta=0.003)

    def test_infinite_capacity_importance_is_unbiased(self) -> None:
        proxy = ParetoProxyGenerator(shape=1.5, scale=7.0)
        delay = 20
        lifetimes = proxy.sample_lifetimes(500_000, np.random.default_rng(3))
        estimate = np.mean((lifetimes >= delay) * proxy.importance_weight(delay))
        self.assertAlmostEqual(float(estimate), 1.0, delta=0.015)

    def test_uniform_positive_delay_weights_are_constant(self) -> None:
        proxy = UniformProxyGenerator(capacity=20, sigma_hat=100, occupancy_fraction=0.75)
        selected = proxy.sample_selected(100_000, np.random.default_rng(4))
        weights = np.full(np.count_nonzero(selected), proxy.importance_weight(9))
        self.assertTrue(np.all(weights == weights[0]))
        self.assertEqual(proxy.survival_probability(0), 1.0)


if __name__ == "__main__":
    unittest.main()
