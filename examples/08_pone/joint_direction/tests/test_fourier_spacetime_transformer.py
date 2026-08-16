"""Focused tensor-contract tests for the paper-inspired transformer."""

from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Batch, Data

from fourier_spacetime_transformer import (
    FourierSpaceTimeTransformer,
    SinusoidalFourierEmbedding,
)


BASE_FEATURES = ("pmt_x", "pmt_y", "pmt_z", "dom_time", "charge")
PMT_FEATURES = (*BASE_FEATURES, "pmt_dir_x", "pmt_dir_y", "pmt_dir_z")


def small_model(*, use_pmt: bool, train_limit: int = 4, eval_limit: int = 6):
    return FourierSpaceTimeTransformer(
        PMT_FEATURES if use_pmt else BASE_FEATURES,
        dimension=32,
        base_dimension=16,
        depth_relative=1,
        relative_bias_blocks=1,
        depth_transformer=1,
        head_size=8,
        train_max_pulses=train_limit,
        eval_max_pulses=eval_limit,
        train_selection="random",
        eval_selection="uniform_time",
        use_pmt_direction=use_pmt,
    )


def graph(nodes: int, *, use_pmt: bool, offset: float = 0.0) -> Data:
    position = torch.stack(
        (
            torch.linspace(-500.0, 500.0, nodes),
            torch.linspace(100.0, 200.0, nodes),
            torch.linspace(-200.0, 300.0, nodes),
        ),
        dim=1,
    )
    time = torch.linspace(1000.0 + offset, 2000.0 + offset, nodes).unsqueeze(1)
    charge = torch.linspace(0.25, 4.0, nodes).unsqueeze(1)
    x = torch.cat((position, time, charge), dim=1)
    if use_pmt:
        direction = torch.nn.functional.normalize(position + 0.3, dim=1)
        x = torch.cat((x, direction), dim=1)
    return Data(x=x)


class FourierTransformerTest(unittest.TestCase):
    def test_fourier_shape_and_finite(self) -> None:
        embedding = SinusoidalFourierEmbedding(16)
        result = embedding(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
        self.assertEqual(tuple(result.shape), (2, 2, 16))
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_eval_selection_is_deterministic_and_event_relative(self) -> None:
        model = small_model(use_pmt=False)
        model.eval()
        batch_a = Batch.from_data_list([graph(10, use_pmt=False, offset=0.0)])
        batch_b = Batch.from_data_list([graph(10, use_pmt=False, offset=90_000.0)])
        packed_a = model._pack(batch_a)
        packed_b = model._pack(batch_b)
        self.assertEqual(int(packed_a["mask"].sum()), 6)
        self.assertTrue(torch.equal(packed_a["mask"], packed_b["mask"]))
        self.assertTrue(
            torch.allclose(packed_a["time"], packed_b["time"], atol=1.0e-6)
        )
        self.assertTrue(torch.allclose(packed_a["position"], packed_b["position"]))

    def test_batched_forward_base_and_pmt(self) -> None:
        for use_pmt in (False, True):
            model = small_model(use_pmt=use_pmt)
            model.eval()
            batch = Batch.from_data_list(
                [graph(3, use_pmt=use_pmt), graph(8, use_pmt=use_pmt)]
            )
            result = model(batch)
            self.assertEqual(tuple(result.shape), (2, 32))
            self.assertTrue(bool(torch.isfinite(result).all()))

    def test_nonpositive_charge_is_rejected(self) -> None:
        model = small_model(use_pmt=False)
        bad = graph(3, use_pmt=False)
        bad.x[1, 4] = 0.0
        with self.assertRaisesRegex(ValueError, "charge"):
            model(Batch.from_data_list([bad]))


if __name__ == "__main__":
    unittest.main()
