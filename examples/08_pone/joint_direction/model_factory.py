"""Opt-in model/data-representation dispatch for routed joint direction.

The historical DynEdge branch remains the default and calls the established
09 builder and KNN representation unchanged.  Transformer-specific imports and
construction occur only when ``model.name=fourier_spacetime_transformer``.
"""

from __future__ import annotations

from typing import Any, Mapping

from graphnet.models.data_representation import EdgelessGraph, NodesAsPulses
from graphnet.models.detector import PONE

from experiment_config import node_feature_augmentations
from fourier_spacetime_transformer import (
    MODEL_NAME as TRANSFORMER_MODEL_NAME,
    build_fourier_transformer_model,
)
from model import build_joint_direction_model as build_dynedge_direction_model
from pmt_direction_features import (
    PONE_V3_PMT_DIRECTION_CONTRACT,
    PONEV3PMTDirectionNodes,
)
from routed_data import build_data_representation as build_dynedge_data_representation


DYNEDGE_MODEL_NAME = "dynedge"
SUPPORTED_MODEL_NAMES = (DYNEDGE_MODEL_NAME, TRANSFORMER_MODEL_NAME)


def configured_model_name(config: Mapping[str, Any]) -> str:
    name = str(config.get("model", {}).get("name", DYNEDGE_MODEL_NAME))
    if name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model.name={name!r}; expected one of {SUPPORTED_MODEL_NAMES}"
        )
    return name


def build_model_data_representation(
    config: Mapping[str, Any], percentiles_csv: str
):
    """Build KNN DynEdge data or an opt-in raw edgeless pulse sequence."""

    if configured_model_name(config) == DYNEDGE_MODEL_NAME:
        return build_dynedge_data_representation(config, percentiles_csv)

    augmentations = node_feature_augmentations(config)
    base_features = tuple(str(value) for value in config["data"]["features"])
    expected_base = PONE_V3_PMT_DIRECTION_CONTRACT.scaled_features
    if base_features != expected_base:
        raise ValueError(
            f"Transformer base data.features must be {expected_base}, got {base_features}"
        )

    if augmentations:
        if augmentations != ("pmt_direction_v3",):
            raise ValueError(f"Unsupported transformer augmentation: {augmentations}")
        contract = PONE_V3_PMT_DIRECTION_CONTRACT
        detector = PONE(
            percentiles_csv=str(percentiles_csv),
            selected_features=list(contract.scaled_features),
            replace_with_identity=list(contract.loader_features),
        )
        representation = EdgelessGraph(
            detector=detector,
            node_definition=PONEV3PMTDirectionNodes(),
            input_feature_names=list(contract.loader_features),
            sort_by="dom_time",
            add_static_features=False,
        )
        expected_output = contract.output_features
    else:
        detector = PONE(
            percentiles_csv=str(percentiles_csv),
            selected_features=list(base_features),
            replace_with_identity=list(base_features),
        )
        representation = EdgelessGraph(
            detector=detector,
            node_definition=NodesAsPulses(),
            input_feature_names=list(base_features),
            sort_by="dom_time",
            add_static_features=False,
        )
        expected_output = base_features

    if tuple(representation.output_feature_names) != tuple(expected_output):
        raise RuntimeError(
            "Transformer output feature contract drifted: "
            f"expected={expected_output}, got={representation.output_feature_names}"
        )
    return representation


def build_direction_model(
    config: Mapping[str, Any],
    stage_name: str,
    data_representation: Any,
    energy_manifest: Any,
    steps_per_optimizer_epoch: int,
):
    """Build the configured backbone inside the common validation wrapper."""

    if configured_model_name(config) == DYNEDGE_MODEL_NAME:
        return build_dynedge_direction_model(
            config,
            stage_name,
            data_representation,
            energy_manifest,
            steps_per_optimizer_epoch,
        )
    return build_fourier_transformer_model(
        config,
        stage_name,
        data_representation,
        energy_manifest,
        steps_per_optimizer_epoch,
    )


__all__ = [
    "DYNEDGE_MODEL_NAME",
    "SUPPORTED_MODEL_NAMES",
    "build_direction_model",
    "build_model_data_representation",
    "configured_model_name",
]
