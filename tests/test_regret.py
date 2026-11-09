from __future__ import annotations

import unittest

import numpy as np

from src.simulation.environment import run_simulation
from src.simulation.learners import fixed_learning_parameters
from src.simulation.loss_sequences import build_gradient_sequence, quantized_delay_threshold
from src.simulation.regret import linear_regret
from tests.helpers import deterministic_environment, small_config


class RegretTests(unittest.TestCase):
    def test_quantile_threshold_rounds_up_in_steps_of_25(self) -> None:
        delays = np.array([10, 50, 80, 120, 180], dtype=np.int64)
        self.assertEqual(quantized_delay_threshold(delays, 0.5, 25), 100)
        self.assertEqual(quantized_delay_threshold(delays, 0.8, 25), 150)

    def test_delayed_indicator_affine_loss_is_bounded(self) -> None:
        sequence = build_gradient_sequence(
            np.array([0, 2], dtype=np.int64),
            dimension=4,
            threshold=2,
            tail_fraction=0.1,
            short_mass_balance=1.0,
            long_magnitude=1.0,
            angle_degrees=90.0,
            loss_coupling="delayed_indicator",
        )
        g_star = np.ones(4) / 2.0
        self.assertAlmostEqual(sequence.loss(0, g_star), 0.5)
        self.assertAlmostEqual(sequence.loss(1, g_star), 1.0)
        self.assertAlmostEqual(sequence.loss(1, -g_star), 0.0)
        self.assertAlmostEqual(np.linalg.norm(sequence.vector(1)), 0.5)
        self.assertEqual(sequence.total_offset(), 1.0)

    def test_changing_indicator_directions_have_fixed_bias_and_norm(self) -> None:
        dimension = 5
        beta = 0.7
        sequence = build_gradient_sequence(
            np.array([1, 3, 5, 7], dtype=np.int64),
            dimension=dimension,
            threshold=3,
            tail_fraction=0.1,
            short_mass_balance=1.0,
            long_magnitude=1.0,
            angle_degrees=90.0,
            loss_coupling="delayed_indicator_changing",
            noise_rng=np.random.default_rng(123),
            changing_direction_bias=beta,
        )
        gradients = sequence.materialize()
        g_star = np.ones(dimension) / np.sqrt(float(dimension))
        np.testing.assert_array_equal(gradients[0], np.zeros(dimension))
        active = gradients[1:]
        np.testing.assert_allclose(np.linalg.norm(active, axis=1), 0.5)
        np.testing.assert_allclose(active @ g_star, 0.5 * beta)
        for gradient in active:
            self.assertGreaterEqual(0.5 + float(np.dot(gradient, -g_star)), 0.0)
            self.assertLessEqual(0.5 + float(np.dot(gradient, g_star)), 1.0)

    def test_fixed_terminal_learning_parameter_formulas(self) -> None:
        oco = fixed_learning_parameters(
            "oco", "uniform", 100_000, 40.0, 20, 20, 1.0, 1.0, 1.0, 1.0
        )
        self.assertAlmostEqual(oco.learning_rate, 2.0 / np.sqrt(100_000 * 41.0))
        bco = fixed_learning_parameters(
            "bco", "pareto", 100_000, 40.0, 20, 20, 1.0, 1.0, 1.0, 1.0
        )
        expected_a = 3.0
        expected_delta = min(
            1.0, np.sqrt(20.0) * expected_a ** 0.25 / 100_000 ** 0.25
        )
        expected_eta = 2.0 / (
            np.sqrt(100_000)
            * np.sqrt(40.0 + (20.0 / expected_delta) ** 2 * expected_a)
        )
        self.assertAlmostEqual(bco.backlog_factor, expected_a)
        self.assertAlmostEqual(bco.smoothing_radius, expected_delta)
        self.assertAlmostEqual(bco.learning_rate, expected_eta)

    def test_small_deterministic_regret(self) -> None:
        gradients = np.array([[1.0, 0.0], [0.0, 1.0]])
        actions = np.zeros_like(gradients)
        self.assertAlmostEqual(linear_regret(actions, gradients, 1.0), np.sqrt(2.0))

    def test_bco_uses_perturbed_played_action(self) -> None:
        environment = deterministic_environment(
            np.array([0], dtype=np.int64),
            np.array([[1.0, 0.0]]),
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        outcome = run_simulation(
            environment, small_config(horizon=1, capacity=1), "bco", "full"
        )
        self.assertAlmostEqual(outcome.raw["learner_loss"], 0.1)
        self.assertAlmostEqual(outcome.raw["total_regret"], 1.1)

    def test_bco_one_point_estimator_uses_raw_affine_loss(self) -> None:
        environment = deterministic_environment(
            np.array([0], dtype=np.int64),
            np.zeros((1, 2), dtype=np.float64),
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        environment.gradients.constant_offset = 0.5
        outcome = run_simulation(
            environment, small_config(horizon=1, capacity=1), "bco", "full"
        )
        self.assertAlmostEqual(outcome.raw["learner_loss"], 0.5)
        self.assertAlmostEqual(
            outcome.diagnostics["maximum_weighted_gradient_norm"], 10.0
        )

    def test_negative_realized_regret_is_preserved(self) -> None:
        gradients = np.array([[1.0], [-1.0]])
        actions = np.array([[-1.0], [1.0]])
        self.assertEqual(linear_regret(actions, gradients, 1.0), -2.0)


if __name__ == "__main__":
    unittest.main()
