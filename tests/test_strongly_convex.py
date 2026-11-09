from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from src.simulation.config import ExperimentConfig
from src.simulation.environment import build_environment, run_simulation
from src.simulation.learners import fixed_learning_parameters
from src.simulation.loss_sequences import GradientSequence, build_gradient_sequence
from src.simulation.native import _load_library
from tests.helpers import deterministic_environment, small_config


class StronglyConvexTests(unittest.TestCase):
    def test_bounded_loss_gradient_and_curvature(self):
        base = build_gradient_sequence(
            np.array([1, 200]), 5, 125, 0.5, 1.02, 1.0, 180.0,
            loss_coupling="delayed_indicator_changing",
            noise_rng=np.random.default_rng(7), changing_direction_bias=0.95,
        )
        loss = base.with_quadratic_mixture(0.25)
        rng = np.random.default_rng(19)
        for t in range(2):
            for _ in range(100):
                x, y = rng.normal(size=(2, 5))
                x /= max(1.0, np.linalg.norm(x))
                y /= max(1.0, np.linalg.norm(y))
                self.assertGreaterEqual(loss.loss(t, x), -1e-14)
                self.assertLessEqual(loss.loss(t, x), 1.0 + 1e-14)
                self.assertAlmostEqual(loss.loss(t, x), 0.75 * base.loss(t, x) + 0.25 * (x @ x))

                self.assertAlmostEqual(
                    loss.loss(t, y) - loss.loss(t, x) - loss.gradient(t, x) @ (y - x),
                    0.25 * np.linalg.norm(y - x) ** 2,
                )
                eps = 1e-6
                numeric = np.array([
                    (loss.loss(t, x + eps * e) - loss.loss(t, x - eps * e)) / (2 * eps)
                    for e in np.eye(5)
                ])
                np.testing.assert_allclose(loss.gradient(t, x), numeric, atol=1e-9)

    def test_exact_comparator_interior_boundary_and_zero(self):
        for coefficient, quadratic in ((0.375, 0.25), (0.45, 0.1), (0.0, 0.25)):
            sequence = GradientSequence(
                2, np.ones(4, dtype=bool), np.zeros(2), np.array([coefficient, 0.0]),
                constant_offset=0.5, quadratic_coefficient=quadratic,
            )
            comparator = sequence.static_comparator(1.0)

            grid = np.linspace(-1.0, 1.0, 20001)
            totals = 4.0 * (0.5 + coefficient * grid + quadratic * grid**2)
            expected = grid[np.argmin(totals)]
            self.assertAlmostEqual(comparator[0], expected, places=4)
            self.assertEqual(comparator[1], 0.0)

    def test_delayed_gradient_uses_original_action_and_quadratic_comparator(self):
        linear = np.array([[0.2, 0.0], [0.0, 0.3], [-0.2, 0.1], [0.1, -0.3]])
        environment = deterministic_environment(np.array([2, 1, 1, 0]), linear)
        environment.gradients.constant_offset = 0.375
        environment.gradients.quadratic_coefficient = 0.25
        config = small_config(horizon=4, capacity=4)
        eta = fixed_learning_parameters("oco", "full", 4, 1.0, 4, 2, 1.0, 1.0, 1.0, 1.0).learning_rate
        actions = np.zeros((4, 2))
        actions[2] = -eta * (linear[0] + linear[1])
        actions[3] = actions[2] - eta * (linear[2] + 0.5 * actions[2])
        expected_loss = np.sum(0.375 + np.sum(linear * actions, axis=1) + 0.25 * np.sum(actions**2, axis=1))
        result = run_simulation(environment, config, "oco", "full")
        self.assertAlmostEqual(result.raw["learner_loss"], expected_loss)
        self.assertAlmostEqual(result.raw["comparator_loss"], 1.495)
        self.assertAlmostEqual(result.raw["total_regret"], expected_loss - 1.495)

    def test_python_native_parity_for_all_feedback_scheduler_pairs(self):
        self.assertIsNotNone(_load_library(), "native backend required for parity validation")
        config = replace(
            ExperimentConfig(), horizon=10_000, capacity=20, target_mean_delays=(400,),
            loss_coupling="delayed_indicator_changing", changing_direction_bias=0.95,
            quadratic_mixture=0.25,
        )
        config.validate()
        environment = build_environment(config, 400, 100, 360.0)
        for problem in ("oco", "bco"):
            for scheduler in ("full", "uniform", "pareto"):
                with self.subTest(problem=problem, scheduler=scheduler):
                    with patch.dict(os.environ, {"SIMULATION_DISABLE_NATIVE": "1"}):
                        python = run_simulation(environment, config, problem, scheduler)
                    with patch.dict(os.environ, {"SIMULATION_DISABLE_NATIVE": "0"}):
                        native = run_simulation(environment, config, problem, scheduler)
                    np.testing.assert_array_equal(python.terminal_states, native.terminal_states)
                    for metric in ("learner_loss", "total_regret", "comparator_loss"):

                        self.assertAlmostEqual(python.raw[metric], native.raw[metric], delta=1e-5)
                    for result in (python, native):
                        self.assertEqual(result.diagnostics["number_nonfinite_updates"], 0)
                        self.assertAlmostEqual(
                            result.raw["total_regret"],
                            result.diagnostics["short_round_regret_contribution"]
                            + result.diagnostics["long_round_regret_contribution"], places=7,
                        )

    def test_invalid_mixture_and_domain_rejected(self):
        for values in ({"quadratic_mixture": -0.1}, {"quadratic_mixture": 1.1},
                       {"quadratic_mixture": 0.25, "domain_radius": 2.0},
                       {"quadratic_mixture": 0.25, "loss_coupling": "smooth"}):
            with self.assertRaises(ValueError):
                replace(ExperimentConfig(), **values).validate()


if __name__ == "__main__":
    unittest.main()
