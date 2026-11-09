from __future__ import annotations

import unittest

import numpy as np

from src.simulation.actual_delays import (
    calibrate_lomax_scale,
    compute_backlog,
    sample_lomax_delays,
    sample_two_point_delays,
)


class ActualDelayTests(unittest.TestCase):
    def test_lomax_calibration_and_horizon_validity(self) -> None:
        horizon = 100_000
        samples = 300_000
        rng = np.random.default_rng(9999)
        uniforms = rng.random(samples)
        rounds = rng.integers(0, horizon, size=samples, dtype=np.int64)
        scale = calibrate_lomax_scale(40.0, 1.5, horizon, uniforms, rounds)
        remaining = horizon - 1 - rounds
        realized = np.minimum(
            1 + np.floor(scale * ((1 - uniforms) ** (-1 / 1.5) - 1)), remaining
        )
        self.assertLess(abs(float(realized.mean()) - 40.0) / 40.0, 0.0051)
        delays = sample_lomax_delays(horizon, 1.5, scale, np.random.default_rng(1))
        self.assertTrue(np.all(delays >= 0))
        self.assertTrue(np.all(delays <= np.arange(horizon - 1, -1, -1)))

    def test_heavier_tail_has_larger_extremes_after_recalibration(self) -> None:


        horizon = 10_000_000
        rng = np.random.default_rng(22)
        uniforms = rng.random(400_000)
        rounds = rng.integers(0, horizon, size=len(uniforms), dtype=np.int64)
        scales = {
            shape: calibrate_lomax_scale(40.0, shape, horizon, uniforms, rounds)
            for shape in (1.2, 1.8)
        }
        samples = {}
        sample_rng = np.random.default_rng(23)
        common = sample_rng.random(200_000)
        sample_rounds = sample_rng.integers(0, horizon, size=len(common))
        remaining = horizon - 1 - sample_rounds
        for shape in scales:
            raw = scales[shape] * ((1 - common) ** (-1 / shape) - 1)
            samples[shape] = np.minimum(1 + np.floor(raw), remaining)
        self.assertGreater(np.percentile(samples[1.2], 99.9), np.percentile(samples[1.8], 99.9))
        self.assertGreater(samples[1.2].max(), samples[1.8].max())

    def test_backlog_event_convention(self) -> None:
        delays = np.array([2, 1, 1, 0], dtype=np.int64)
        np.testing.assert_array_equal(compute_backlog(delays), [1, 2, 1, 0])

    def test_two_point_delays_are_exact_and_allow_post_horizon_feedback(self) -> None:
        delays = sample_two_point_delays(
            10_000, 40, 320, 0.1, np.random.default_rng(123)
        )
        self.assertEqual(set(np.unique(delays)), {40, 320})
        self.assertEqual(int(np.count_nonzero(delays == 320)), 1_000)
        backlog = compute_backlog(delays)
        self.assertEqual(len(backlog), len(delays))
        self.assertTrue(np.all(backlog >= 0))


if __name__ == "__main__":
    unittest.main()
