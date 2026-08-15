"""Dependency-light tests for checkpoint and reporting metrics."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from metrics import opening_angle_metrics, weighted_mean  # noqa: E402


class OpeningAngleMetricsTest(unittest.TestCase):
    def test_global_and_macro_median_are_distinct_and_correct(self) -> None:
        # First energy bin has three events; second has one.  Global median is
        # event-population weighted, whereas macro median gives each bin one vote.
        angles = torch.tensor([1.0, 2.0, 3.0, 10.0])
        energies = torch.tensor([2.0, 3.0, 4.0, 20.0])
        metrics, rows = opening_angle_metrics(
            angles,
            energies,
            [0.0, 1.0, 2.0],
            minimum_events_per_bin=1,
        )
        self.assertAlmostEqual(metrics["val_global_median_deg"], 2.5)
        self.assertAlmostEqual(metrics["val_macro_median_deg"], 6.0)
        self.assertEqual(len(rows), 2)

    def test_split_prefix_and_weighted_mean(self) -> None:
        metrics, _ = opening_angle_metrics(
            torch.tensor([1.0, 3.0]),
            torch.tensor([2.0, 20.0]),
            [0.0, 1.0, 2.0],
            minimum_events_per_bin=1,
            prefix="test",
        )
        self.assertIn("test_global_median_deg", metrics)
        self.assertNotIn("val_global_median_deg", metrics)
        self.assertAlmostEqual(
            weighted_mean(torch.tensor([1.0, 3.0]), torch.tensor([3.0, 1.0])),
            1.5,
        )


if __name__ == "__main__":
    unittest.main()

