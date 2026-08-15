"""Dependency-light tests for the joint Muon-CC direction core."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from direction_utils import (  # noqa: E402
    opening_angle_degrees,
    unit_vector_to_zenith_azimuth,
    zenith_azimuth_to_unit_vector,
)
from energy_weighting import (  # noqa: E402
    EnergyWeightLookup,
    EnergyWeightManifest,
    fit_energy_weight_manifest,
)
from losses import (  # noqa: E402
    EnergyWeightedDirectionLoss,
    von_mises_fisher_3d_nll,
)


class DirectionUtilityTest(unittest.TestCase):
    def test_canonical_directions_and_round_trip(self) -> None:
        zenith = torch.tensor(
            [0.0, math.pi / 2.0, math.pi / 2.0, math.pi],
            dtype=torch.float64,
        )
        azimuth = torch.tensor(
            [0.0, 0.0, math.pi / 2.0, 1.3], dtype=torch.float64
        )
        vectors = zenith_azimuth_to_unit_vector(zenith, azimuth)
        expected = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(vectors, expected, atol=1.0e-12))

        safe_zenith = torch.tensor([0.2, 1.1, 2.8], dtype=torch.float64)
        safe_azimuth = torch.tensor(
            [0.0, 2.0, 2.0 * math.pi - 0.1], dtype=torch.float64
        )
        round_trip = zenith_azimuth_to_unit_vector(safe_zenith, safe_azimuth)
        zenith_back, azimuth_back = unit_vector_to_zenith_azimuth(round_trip)
        self.assertTrue(torch.allclose(zenith_back, safe_zenith, atol=1.0e-12))
        self.assertTrue(torch.allclose(azimuth_back, safe_azimuth, atol=1.0e-12))

    def test_opening_angle_reference_values(self) -> None:
        prediction = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        target = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        result = opening_angle_degrees(prediction, target)
        self.assertTrue(torch.allclose(result, torch.tensor([0.0, 90.0, 180.0])))


class EnergyWeightingTest(unittest.TestCase):
    def test_alpha_zero_and_inverse_sqrt_ratios(self) -> None:
        energies = torch.tensor([2.0, 2.5, 3.0, 4.0, 20.0], dtype=torch.float64)
        edges = [0.0, 1.0, 2.0]

        flat = fit_energy_weight_manifest(energies, edges, alpha=0.0)
        self.assertEqual(flat.bin_counts, (4, 1))
        self.assertEqual(flat.bin_weights, (1.0, 1.0))

        weighted = fit_energy_weight_manifest(energies, edges, alpha=0.5)
        ratio = weighted.bin_weights[1] / weighted.bin_weights[0]
        self.assertAlmostEqual(ratio, 2.0, places=12)
        self.assertAlmostEqual(weighted.event_weighted_mean, 1.0, places=12)

    def test_clipping_renormalization_and_manifest_round_trip(self) -> None:
        energies = torch.tensor([2.0] * 9 + [20.0], dtype=torch.float64)
        manifest = fit_energy_weight_manifest(
            energies,
            [0.0, 1.0, 2.0],
            alpha=1.0,
            clip_min=0.5,
            clip_max=2.0,
            source_files=["train/truth/part-0.parquet"],
        )
        self.assertAlmostEqual(manifest.event_weighted_mean, 1.0, places=12)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            manifest.save(path)
            loaded = EnergyWeightManifest.load(path)
        self.assertEqual(loaded, manifest)

    def test_lookup_is_one_dimensional_and_rejects_empty_bins(self) -> None:
        lookup = EnergyWeightLookup(
            log10_bin_edges=[0.0, 1.0, 2.0],
            bin_weights=[0.8, 1.2],
            bin_counts=[4, 1],
        )
        result = lookup(torch.tensor([2.0, 20.0]))
        self.assertEqual(result.shape, (2,))
        self.assertTrue(torch.allclose(result, torch.tensor([0.8, 1.2])))

        empty_lookup = EnergyWeightLookup(
            log10_bin_edges=[0.0, 1.0, 2.0, 3.0],
            bin_weights=[1.0, 0.0, 1.0],
            bin_counts=[1, 0, 1],
        )
        with self.assertRaises(ValueError):
            empty_lookup(torch.tensor([20.0]))


class DirectionLossTest(unittest.TestCase):
    @staticmethod
    def _batch() -> tuple[torch.Tensor, torch.Tensor]:
        prediction = torch.tensor(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 3.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor(
            [
                [math.pi / 2.0, 0.0, 2.0],
                [math.pi / 2.0, math.pi / 2.0 + 0.2, 20.0],
            ],
            dtype=torch.float64,
        )
        return prediction, target

    def test_pure_torch_vmf_matches_closed_form(self) -> None:
        direction = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        target = direction.clone()
        kappa = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        result = von_mises_fisher_3d_nll(direction, kappa, target)
        reference = (
            math.log(4.0 * math.pi)
            + torch.log(torch.sinh(kappa) / kappa)
            - kappa
        )
        self.assertTrue(torch.allclose(result, reference, atol=1.0e-12))
        result.sum().backward()
        self.assertTrue(bool(torch.isfinite(kappa.grad).all()))

    def test_train_only_weighting_shape_and_backward(self) -> None:
        prediction, target = self._batch()
        loss = EnergyWeightedDirectionLoss(
            log10_bin_edges=[0.0, 1.0, 2.0],
            bin_weights=[0.5, 2.0],
            bin_counts=[4, 1],
            objective="vmf",
            weighting_mode="train_only",
        )

        loss.train()
        train_components = loss.elementwise_components(prediction, target)
        self.assertEqual(train_components["weighted"].shape, (2,))
        self.assertTrue(
            torch.allclose(
                train_components["weights"],
                torch.tensor([0.5, 2.0], dtype=torch.float64),
            )
        )
        self.assertAlmostEqual(
            float(train_components["normalized_weights"].mean()), 1.0
        )

        scalar = loss(prediction, target)
        self.assertEqual(scalar.ndim, 0)
        scalar.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

        loss.eval()
        eval_components = loss.elementwise_components(prediction.detach(), target)
        self.assertTrue(
            torch.equal(
                eval_components["weights"], torch.ones(2, dtype=torch.float64)
            )
        )
        self.assertTrue(
            torch.allclose(eval_components["weighted"], eval_components["unweighted"])
        )

    def test_batch_reduction_is_sum_w_loss_over_sum_w(self) -> None:
        prediction, target = self._batch()
        loss = EnergyWeightedDirectionLoss(
            log10_bin_edges=[0.0, 1.0, 2.0],
            bin_weights=[0.5, 2.0],
            bin_counts=[4, 1],
            objective="vmf",
            weighting_mode="train_only",
        )
        loss.train()
        components = loss.elementwise_components(prediction, target)
        expected = torch.sum(
            components["weights"] * components["unweighted"]
        ) / torch.sum(components["weights"])
        self.assertTrue(torch.allclose(loss(prediction, target), expected))

    def test_stage_b_hybrid_and_checkpoint_buffers(self) -> None:
        prediction, target = self._batch()
        manifest = fit_energy_weight_manifest(
            target[:, 2], [0.0, 1.0, 2.0], alpha=0.5
        )
        loss = EnergyWeightedDirectionLoss.from_manifest(
            manifest,
            objective="angular_hybrid",
            angular_surrogate="opening_angle",
            vmf_factor=0.05,
        )
        components = loss.elementwise_components(prediction, target)
        expected = components["angular"] + 0.05 * components["vmf"]
        self.assertTrue(torch.allclose(components["unweighted"], expected))
        loss.train()
        loss(prediction, target).backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

        state = loss.state_dict()
        self.assertIn("_weight_lookup.log10_bin_edges", state)
        self.assertIn("_weight_lookup.bin_weights", state)
        restored = EnergyWeightedDirectionLoss.from_manifest(
            manifest, objective="angular_hybrid"
        )
        restored.load_state_dict(state)
        self.assertTrue(
            torch.equal(
                restored.state_dict()["_weight_lookup.bin_weights"],
                state["_weight_lookup.bin_weights"],
            )
        )

    def test_shape_guard_rejects_broadcastable_weight_target(self) -> None:
        prediction, target = self._batch()
        loss = EnergyWeightedDirectionLoss(
            objective="vmf", weighting_mode="disabled"
        )
        with self.assertRaises(ValueError):
            loss(prediction, target.unsqueeze(-1))


if __name__ == "__main__":
    unittest.main()
