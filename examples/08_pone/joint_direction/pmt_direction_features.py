"""Optional node-feature augmentation for P-ONE v3 PMT directions.

This module deliberately keeps the PMT lookup out of the routed data loader.
The loader only has to follow :data:`PONE_V3_PMT_DIRECTION_CONTRACT`:

* read ``pmt_number`` as an auxiliary parquet column;
* pass it through the detector with an identity transform; and
* use :class:`PONEV3PMTDirectionNodes` as the node definition.

The auxiliary identifier is validated and consumed by the node definition.  It
is not present in the final model inputs.  No source parquet or percentile file
is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import torch

from graphnet.models.data_representation import NodeDefinition


BASE_MODEL_FEATURES: Tuple[str, ...] = (
    "pmt_x",
    "pmt_y",
    "pmt_z",
    "dom_time",
    "charge",
)
PMT_NUMBER_FEATURE = "pmt_number"
PMT_DIRECTION_FEATURES: Tuple[str, ...] = (
    "pmt_dir_x",
    "pmt_dir_y",
    "pmt_dir_z",
)

# (zenith, azimuth) in degrees.  This is the exact PMTAcceptance convention
# used by pone_offline v3 and by I3FeatureExtractorPONE for PMTs 1 through 16.
PONE_V3_PMT_ANGLES_DEG: Tuple[Tuple[float, float], ...] = (
    (58.0, 0.0),
    (90.0, 328.0),
    (122.0, 0.0),
    (90.0, 32.0),
    (51.37, 53.06),
    (51.37, 306.94),
    (128.63, 306.94),
    (128.63, 53.06),
    (58.0, 180.0),
    (90.0, 148.0),
    (122.0, 180.0),
    (90.0, 212.0),
    (51.37, 233.06),
    (51.37, 126.94),
    (128.63, 126.94),
    (128.63, 233.06),
)


def _unit_vector(zenith_deg: float, azimuth_deg: float) -> Tuple[float, ...]:
    """Convert the pone_offline spherical convention to Cartesian xyz."""

    zenith = math.radians(zenith_deg)
    azimuth = math.radians(azimuth_deg)
    return (
        math.sin(zenith) * math.cos(azimuth),
        math.sin(zenith) * math.sin(azimuth),
        math.cos(zenith),
    )


PONE_V3_PMT_DIRECTIONS: Tuple[Tuple[float, ...], ...] = tuple(
    _unit_vector(zenith, azimuth)
    for zenith, azimuth in PONE_V3_PMT_ANGLES_DEG
)


@dataclass(frozen=True)
class NodeFeatureAugmentationContract:
    """Describe loader, scaler, and model columns for an augmentation.

    ``scaled_features`` are passed to the normal detector scaler.
    ``identity_features`` are auxiliary raw columns passed through unchanged.
    ``loader_features`` is the parquet query order, while ``output_features``
    is the only feature order visible to the GNN.
    """

    scaled_features: Tuple[str, ...]
    identity_features: Tuple[str, ...]
    output_features: Tuple[str, ...]

    @property
    def loader_features(self) -> Tuple[str, ...]:
        """Return the complete ordered list queried from parquet."""

        return self.scaled_features + self.identity_features


PONE_V3_PMT_DIRECTION_CONTRACT = NodeFeatureAugmentationContract(
    scaled_features=BASE_MODEL_FEATURES,
    identity_features=(PMT_NUMBER_FEATURE,),
    output_features=BASE_MODEL_FEATURES + PMT_DIRECTION_FEATURES,
)


class CategoricalLookupFeatureAppender(NodeDefinition):
    """Replace one auxiliary categorical column with lookup-vector features.

    Input columns may arrive in any order, but must contain exactly the model
    columns and the single auxiliary category column.  Model columns are
    explicitly reordered to ``model_feature_names`` before lookup vectors are
    appended.  Categories are one-based finite integers in ``[1, N]``.

    This general class makes future fixed categorical node features possible
    without adding experiment-specific branches to the loader.
    """

    def __init__(
        self,
        *,
        model_feature_names: Sequence[str],
        category_feature_name: str,
        appended_feature_names: Sequence[str],
        lookup_table: Sequence[Sequence[float]],
        input_feature_names: Sequence[str] | None = None,
    ) -> None:
        self._model_feature_names = tuple(str(x) for x in model_feature_names)
        self._category_feature_name = str(category_feature_name)
        self._appended_feature_names = tuple(
            str(x) for x in appended_feature_names
        )
        self._validate_definition(lookup_table)

        names = None
        if input_feature_names is not None:
            names = [str(x) for x in input_feature_names]
        super().__init__(input_feature_names=names)

        table = torch.as_tensor(lookup_table, dtype=torch.float64)
        self.register_buffer("lookup_table", table, persistent=True)

    def _validate_definition(
        self, lookup_table: Sequence[Sequence[float]]
    ) -> None:
        names = (
            *self._model_feature_names,
            self._category_feature_name,
            *self._appended_feature_names,
        )
        if any(not name for name in names):
            raise ValueError("Feature names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError(f"Feature names must be unique, got {names}")
        if not self._model_feature_names:
            raise ValueError("model_feature_names cannot be empty")
        if not self._appended_feature_names:
            raise ValueError("appended_feature_names cannot be empty")

        table = torch.as_tensor(lookup_table, dtype=torch.float64)
        if table.ndim != 2:
            raise ValueError(
                "lookup_table must have shape [categories, appended_features]; "
                f"got {tuple(table.shape)}"
            )
        expected_width = len(self._appended_feature_names)
        if table.shape[1] != expected_width:
            raise ValueError(
                "lookup_table must have shape [categories, appended_features]; "
                f"got {tuple(table.shape)}"
            )
        if table.shape[0] < 1 or not torch.isfinite(table).all():
            raise ValueError("lookup_table must be non-empty and finite")

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        expected = (*self._model_feature_names, self._category_feature_name)
        if len(input_feature_names) != len(set(input_feature_names)):
            raise ValueError(
                f"Input feature names must be unique: {input_feature_names}"
            )
        if set(input_feature_names) != set(expected):
            missing = sorted(set(expected) - set(input_feature_names))
            unexpected = sorted(set(input_feature_names) - set(expected))
            raise ValueError(
                "Node-feature augmentation input mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

        self._model_indices = tuple(
            input_feature_names.index(name)
            for name in self._model_feature_names
        )
        self._category_index = input_feature_names.index(
            self._category_feature_name
        )
        return [*self._model_feature_names, *self._appended_feature_names]

    def _construct_nodes(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected a rank-2 node tensor, got {x.shape}")
        expected_width = len(self._model_feature_names) + 1
        if x.shape[1] != expected_width:
            raise ValueError(
                f"Expected {expected_width} input columns, got {x.shape[1]}"
            )

        category = x[:, self._category_index]
        if not torch.isfinite(category).all():
            raise ValueError(
                f"{self._category_feature_name} must contain finite integers"
            )
        rounded = torch.round(category)
        if not torch.equal(category, rounded):
            invalid = category[category != rounded][:5].detach().cpu().tolist()
            raise ValueError(
                f"{self._category_feature_name} must contain integers; "
                f"examples={invalid}"
            )
        category_index = rounded.to(torch.long) - 1
        if category_index.numel() and (
            int(category_index.min()) < 0
            or int(category_index.max()) >= self.lookup_table.shape[0]
        ):
            invalid = category[
                (category_index < 0)
                | (category_index >= self.lookup_table.shape[0])
            ][:5].detach().cpu().tolist()
            raise ValueError(
                f"{self._category_feature_name} must be in "
                f"[1, {self.lookup_table.shape[0]}]; examples={invalid}"
            )

        model_features = x[:, self._model_indices]
        appended = self.lookup_table.to(device=x.device, dtype=x.dtype)[
            category_index
        ]
        return torch.cat((model_features, appended), dim=1)


class PONEV3PMTDirectionNodes(CategoricalLookupFeatureAppender):
    """Append exact pone_offline v3 PMT unit directions to pulse nodes."""

    def __init__(
        self, input_feature_names: Sequence[str] | None = None
    ) -> None:
        super().__init__(
            model_feature_names=BASE_MODEL_FEATURES,
            category_feature_name=PMT_NUMBER_FEATURE,
            appended_feature_names=PMT_DIRECTION_FEATURES,
            lookup_table=PONE_V3_PMT_DIRECTIONS,
            input_feature_names=input_feature_names,
        )
