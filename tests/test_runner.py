import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.experiment import (
    CAPACITIES, FAMILIES, complete_seeds, paper_config, prepare_settings,
    read_rows, run, run_seed, write_rows,
)
from src.plotting import statistics_for


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = paper_config(horizon=10000, seeds=1)
        cls.rows = run_seed(cls.config, 400.0, 100)

    def settings(self, output):
        settings = dict(schema_version=1, base_configuration=self.config.to_dict(),
                        capacities=list(CAPACITIES), loss_families=FAMILIES, delay_scale=400.0)
        (output / "config.json").write_text(json.dumps(settings))

    def test_partial_repetition_is_rerun_and_complete_repetition_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.settings(output)
            write_rows(output / "runs.csv", self.rows[:-1])
            with patch("src.experiment.run_seed", return_value=self.rows) as simulate, patch("src.plotting.plot_results"):
                run(output)
                simulate.assert_called_once()
            restored = read_rows(output / "runs.csv", self.config)
            self.assertEqual(complete_seeds(restored, (100,)), {100})
            with patch("src.experiment.run_seed") as simulate, patch("src.plotting.plot_results"):
                run(output)
                simulate.assert_not_called()

    def test_settings_mismatch_preserves_saved_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.settings(output)
            write_rows(output / "runs.csv", self.rows)
            before = (output / "runs.csv").read_bytes()
            with self.assertRaisesRegex(ValueError, "different settings"):
                prepare_settings(output, 20000, None)
            self.assertEqual(before, (output / "runs.csv").read_bytes())

    def test_paired_losses_and_unconstrained_control_are_reused(self):
        for family in FAMILIES:
            selected = [row for row in self.rows if row["loss_family"] == family]
            self.assertEqual(len({row["comparator_loss"] for row in selected}), 1)
            self.assertEqual(len({row["realized_mean_delay"] for row in selected}), 1)
            for problem in ("oco", "bco"):
                full = [row for row in selected if row["problem_type"] == problem and row["scheduler"] == "full"]
                self.assertEqual(len({row["total_regret"] for row in full}), 1)

    def test_duplicate_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.csv"
            write_rows(path, self.rows + [self.rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate run"):
                read_rows(path, self.config)

    def test_small_sample_uncertainty(self):
        self.assertIsNone(statistics_for([0.1])["ci95_high"])
        result = statistics_for([0.0, 2.0])
        self.assertAlmostEqual(result["mean"], 1.0)
        self.assertAlmostEqual(result["ci95_high"], 13.7062047364, places=8)


if __name__ == "__main__":
    unittest.main()
