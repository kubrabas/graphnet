"""Focused tests for the optional P-ONE v3 PMT-direction node features."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


JOINT_DIRECTION_DIR = Path(__file__).resolve().parents[1]
if str(JOINT_DIRECTION_DIR) not in sys.path:
    sys.path.insert(0, str(JOINT_DIRECTION_DIR))

from pmt_direction_features import (  # noqa: E402
    BASE_MODEL_FEATURES,
    PMT_DIRECTION_FEATURES,
    PMT_NUMBER_FEATURE,
    PONE_V3_PMT_DIRECTION_CONTRACT,
    PONE_V3_PMT_DIRECTIONS,
    PONEV3PMTDirectionNodes,
)


EXPECTED_DIRECTIONS = torch.tensor(
    [
        [0.848048096156426, 0.0, 0.529919264233205],
        [0.848048096156426, -0.529919264233205, 0.0],
        [0.848048096156426, 0.0, -0.529919264233205],
        [0.848048096156426, 0.529919264233205, 0.0],
        [0.469480513557510, 0.624381012320296, 0.624288714333087],
        [0.469480513557509, -0.624381012320297, 0.624288714333087],
        [0.469480513557509, -0.624381012320297, -0.624288714333087],
        [0.469480513557510, 0.624381012320297, -0.624288714333087],
        [-0.848048096156426, 0.0, 0.529919264233205],
        [-0.848048096156426, 0.529919264233205, 0.0],
        [-0.848048096156426, 0.0, -0.529919264233205],
        [-0.848048096156426, -0.529919264233205, 0.0],
        [-0.469480513557510, -0.624381012320296, 0.624288714333087],
        [-0.469480513557509, 0.624381012320296, 0.624288714333087],
        [-0.469480513557510, 0.624381012320297, -0.624288714333087],
        [-0.469480513557510, -0.624381012320297, -0.624288714333087],
    ],
    dtype=torch.float64,
)


class PMTDirectionFeatureTest(unittest.TestCase):
    """Verify the full auxiliary-column-to-model-feature contract."""

    def test_contract_keeps_auxiliary_id_out_of_model_features(self) -> None:
        contract = PONE_V3_PMT_DIRECTION_CONTRACT
        self.assertEqual(contract.scaled_features, BASE_MODEL_FEATURES)
        self.assertEqual(contract.identity_features, (PMT_NUMBER_FEATURE,))
        self.assertEqual(
            contract.loader_features,
            BASE_MODEL_FEATURES + (PMT_NUMBER_FEATURE,),
        )
        self.assertEqual(
            contract.output_features,
            BASE_MODEL_FEATURES + PMT_DIRECTION_FEATURES,
        )
        self.assertNotIn(PMT_NUMBER_FEATURE, contract.output_features)

    def test_exact_v3_direction_table_and_unit_norms(self) -> None:
        actual = torch.tensor(PONE_V3_PMT_DIRECTIONS, dtype=torch.float64)
        torch.testing.assert_close(
            actual, EXPECTED_DIRECTIONS, rtol=0.0, atol=1e-14
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(actual, dim=1),
            torch.ones(16, dtype=torch.float64),
            rtol=0.0,
            atol=1e-14,
        )

    def test_output_shape_values_and_canonical_order(self) -> None:
        # Deliberately scramble loader columns to prove canonical ordering.
        input_names = [
            PMT_NUMBER_FEATURE,
            "charge",
            "pmt_z",
            "dom_time",
            "pmt_x",
            "pmt_y",
        ]
        nodes = PONEV3PMTDirectionNodes(input_feature_names=input_names)
        x = torch.tensor(
            [
                [1.0, 50.0, 3.0, 40.0, 1.0, 2.0],
                [16.0, 500.0, 30.0, 400.0, 10.0, 20.0],
            ],
            dtype=torch.float32,
        )

        output = nodes(x)

        self.assertEqual(nodes.nb_outputs, 8)
        self.assertEqual(
            nodes._output_feature_names,  # pylint: disable=protected-access
            list(BASE_MODEL_FEATURES + PMT_DIRECTION_FEATURES),
        )
        self.assertEqual(tuple(output.shape), (2, 8))
        torch.testing.assert_close(
            output[:, :5],
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 40.0, 50.0],
                    [10.0, 20.0, 30.0, 400.0, 500.0],
                ]
            ),
        )
        torch.testing.assert_close(
            output[:, 5:], EXPECTED_DIRECTIONS[[0, 15]].to(torch.float32)
        )

    def test_rejects_invalid_pmt_ids(self) -> None:
        names = list(PONE_V3_PMT_DIRECTION_CONTRACT.loader_features)
        for bad_id in (0.0, 17.0, 1.5, float("nan"), float("inf")):
            with self.subTest(bad_id=bad_id):
                nodes = PONEV3PMTDirectionNodes(input_feature_names=names)
                x = torch.zeros((1, len(names)), dtype=torch.float32)
                x[0, names.index(PMT_NUMBER_FEATURE)] = bad_id
                with self.assertRaisesRegex(ValueError, "pmt_number"):
                    nodes(x)

    def test_rejects_missing_or_extra_loader_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "input mismatch"):
            PONEV3PMTDirectionNodes(
                input_feature_names=[*BASE_MODEL_FEATURES, "not_pmt_number"]
            )


if __name__ == "__main__":
    unittest.main()
