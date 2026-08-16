"""Lightweight config contracts for joint-direction ablations.

This module intentionally imports only the Python standard library. Both the
login-node submitter and the container worker use it, so malformed ablations
fail before an output directory is reserved as well as before training.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


SUPPORTED_NODE_FEATURE_AUGMENTATIONS = ("pmt_direction_v3",)
EXPERIMENTAL_BASELINE_FIELDS = {
    "weighting.alpha": 0.5,
    "data.node_feature_augmentations": (),
    "training.seed": 20260202,
    "training.stage_a.base_lr": 1.0e-5,
    "training.stage_a.peak_lr": 2.0e-3,
    "training.stage_a.warmup_fraction": 0.5,
    "training.stage_b.base_lr": 1.0e-6,
    "training.stage_b.peak_lr": 2.0e-4,
    "training.stage_b.warmup_fraction": 0.5,
    "loader.batch_size": 256,
    "loader.val_batch_size": 256,
    "loader.accumulate_grad_batches": 4,
    "model.name": "dynedge",
    "model.transformer": None,
    "model.dynedge_layer_sizes": None,
    "model.post_processing_layer_sizes": None,
    "model.readout_layer_sizes": None,
    "optimizer.name": "adam",
    "optimizer.weight_decay": 0.0,
    "optimizer.eps": None,
    "trainer.precision": "32-true",
    "trainer.gradient_clip_val": 0.0,
}


def node_feature_augmentations(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return validated, explicitly requested node-feature augmentations."""

    raw = config.get("data", {}).get("node_feature_augmentations", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise TypeError("data.node_feature_augmentations must be a list")
    values = tuple(str(value) for value in raw)
    if len(values) != len(set(values)):
        raise ValueError("data.node_feature_augmentations contains duplicates")
    unknown = sorted(set(values) - set(SUPPORTED_NODE_FEATURE_AUGMENTATIONS))
    if unknown:
        raise ValueError(f"Unsupported node feature augmentations: {unknown}")
    if len(values) > 1:
        raise ValueError(
            "Only one node feature augmentation can currently be composed safely"
        )
    return values


def _nested_config_value(
    config: Mapping[str, Any], dotted_name: str, missing_default: Any
) -> Any:
    value: Any = config
    for part in dotted_name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return missing_default
        value = value[part]
    if isinstance(value, list):
        return tuple(
            tuple(item) if isinstance(item, list) else item for item in value
        )
    return value


def validate_experiment_contract(config: Mapping[str, Any]) -> None:
    """Require every controlled scientific deviation to be declared."""

    contract = config.get("experiment_contract", {})
    if contract is None:
        contract = {}
    if not isinstance(contract, Mapping):
        raise TypeError("experiment_contract must be a mapping")
    raw_fields = contract.get("varied_fields", [])
    if not isinstance(raw_fields, list):
        raise TypeError("experiment_contract.varied_fields must be a list")
    declared = {str(value) for value in raw_fields}
    if len(declared) != len(raw_fields):
        raise ValueError("experiment_contract.varied_fields contains duplicates")
    unsupported = sorted(declared - set(EXPERIMENTAL_BASELINE_FIELDS))
    if unsupported:
        raise ValueError(f"Unsupported experimental varied_fields: {unsupported}")

    observed = {
        name
        for name, baseline in EXPERIMENTAL_BASELINE_FIELDS.items()
        if _nested_config_value(config, name, baseline) != baseline
    }
    if observed != declared:
        raise ValueError(
            "Experimental variation declaration does not match the config: "
            f"declared={sorted(declared)}, observed={sorted(observed)}"
        )


def validate_optional_model_sizes(model: Mapping[str, Any]) -> None:
    """Validate optional DynEdge capacity overrides without importing torch."""

    dynedge = model.get("dynedge_layer_sizes")
    if dynedge is not None:
        if not isinstance(dynedge, list) or not dynedge:
            raise ValueError("model.dynedge_layer_sizes must be null or non-empty")
        for index, layer in enumerate(dynedge):
            if (
                not isinstance(layer, list)
                or len(layer) < 2
                or any(int(value) <= 0 for value in layer)
            ):
                raise ValueError(
                    "Each model.dynedge_layer_sizes entry must contain at least "
                    f"two positive integers; bad entry {index}: {layer!r}"
                )
    for name in ("post_processing_layer_sizes", "readout_layer_sizes"):
        sizes = model.get(name)
        if sizes is not None and (
            not isinstance(sizes, list)
            or not sizes
            or any(int(value) <= 0 for value in sizes)
        ):
            raise ValueError(f"model.{name} must be null or positive integers")


def validate_transformer_options(config: Mapping[str, Any]) -> None:
    """Validate the opt-in paper transformer without importing torch."""

    model = config.get("model", {})
    name = str(model.get("name", "dynedge"))
    if name == "dynedge":
        if model.get("transformer") is not None:
            raise ValueError("DynEdge configs cannot define model.transformer")
        return
    if name != "fourier_spacetime_transformer":
        raise ValueError(
            "model.name must be dynedge or fourier_spacetime_transformer"
        )
    transformer = model.get("transformer")
    if not isinstance(transformer, Mapping):
        raise TypeError("Transformer configs require a model.transformer mapping")
    required = {
        "dimension",
        "base_dimension",
        "depth_relative",
        "relative_bias_blocks",
        "depth_transformer",
        "head_size",
        "train_max_pulses",
        "eval_max_pulses",
        "train_selection",
        "eval_selection",
        "use_pmt_direction",
        "position_scale_m",
        "time_scale_ns",
        "charge_log10_scale",
        "position_fourier_scale",
        "time_fourier_scale",
        "charge_fourier_scale",
        "relative_fourier_scale",
        "relative_clip",
        "speed_of_light_m_per_ns",
        "dropout",
    }
    missing = sorted(required - set(transformer))
    unexpected = sorted(set(transformer) - required)
    if missing or unexpected:
        raise ValueError(
            "model.transformer key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    integer_fields = {
        "dimension",
        "base_dimension",
        "depth_relative",
        "relative_bias_blocks",
        "depth_transformer",
        "head_size",
        "train_max_pulses",
        "eval_max_pulses",
    }
    values = {field: int(transformer[field]) for field in integer_fields}
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"Transformer integer fields must be positive: {values}")
    if values["dimension"] % values["head_size"]:
        raise ValueError("Transformer dimension must be divisible by head_size")
    if values["base_dimension"] % 4:
        raise ValueError("Transformer base_dimension must be divisible by four")
    if values["relative_bias_blocks"] > values["depth_relative"]:
        raise ValueError("relative_bias_blocks cannot exceed depth_relative")
    if values["eval_max_pulses"] < values["train_max_pulses"]:
        raise ValueError("eval_max_pulses cannot be smaller than train_max_pulses")
    if transformer["train_selection"] != "random":
        raise ValueError("First transformer experiment requires train_selection=random")
    if transformer["eval_selection"] != "uniform_time":
        raise ValueError(
            "First transformer experiment requires eval_selection=uniform_time"
        )
    numeric_fields = required - integer_fields - {
        "train_selection",
        "eval_selection",
        "use_pmt_direction",
    }
    numeric = {field: float(transformer[field]) for field in numeric_fields}
    if any(not math.isfinite(value) for value in numeric.values()):
        raise ValueError(f"Transformer numeric fields must be finite: {numeric}")
    positive = {key: value for key, value in numeric.items() if key != "dropout"}
    if any(value <= 0.0 for value in positive.values()):
        raise ValueError(f"Transformer scales must be positive: {positive}")
    if not 0.0 <= numeric["dropout"] < 1.0:
        raise ValueError("Transformer dropout must lie in [0,1)")

    augmentations = node_feature_augmentations(config)
    has_pmt_direction = augmentations == ("pmt_direction_v3",)
    if bool(transformer["use_pmt_direction"]) != has_pmt_direction:
        raise ValueError(
            "model.transformer.use_pmt_direction must exactly match "
            "data.node_feature_augmentations=[pmt_direction_v3]"
        )
    if tuple(config.get("data", {}).get("features", [])) != (
        "pmt_x",
        "pmt_y",
        "pmt_z",
        "dom_time",
        "charge",
    ):
        raise ValueError("Transformer requires the canonical five base data.features")

    loader = config.get("loader", {})
    for field in ("batch_size", "val_batch_size", "accumulate_grad_batches"):
        if int(loader.get(field, 0)) <= 0:
            raise ValueError(f"loader.{field} must be positive for transformer")
    optimizer = config.get("optimizer", {})
    if str(optimizer.get("name", "")).lower() != "adamw":
        raise ValueError("Transformer requires optimizer.name=adamw")
    if float(optimizer.get("weight_decay", -1.0)) < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative")
    if float(optimizer.get("eps", 0.0)) <= 0.0:
        raise ValueError("optimizer.eps must be positive")
    if str(config.get("trainer", {}).get("precision")) != "bf16-mixed":
        raise ValueError("First transformer experiment requires bf16-mixed precision")


def validate_experiment_extensions(config: Mapping[str, Any]) -> None:
    """Validate all opt-in feature, capacity, and alpha extensions."""

    node_feature_augmentations(config)
    validate_optional_model_sizes(config.get("model", {}))
    validate_transformer_options(config)
    alpha = float(config.get("weighting", {}).get("alpha", -1.0))
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("weighting.alpha must be finite and non-negative")
    validate_experiment_contract(config)
