"""Shape-safe direction objectives with optional train-only energy weights."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

try:
    from graphnet.training.loss_functions import LossFunction
except ModuleNotFoundError as import_error:  # Lightweight pure-torch tests.
    if import_error.name != "graphnet":
        raise

    class LossFunction(nn.Module):  # type: ignore[no-redef]
        """Minimal fallback matching the GraphNeT LossFunction call contract."""

        def forward(
            self,
            prediction: Tensor,
            target: Tensor,
            weights: Tensor | None = None,
            return_elements: bool = False,
        ) -> Tensor:
            elements = self._forward(prediction, target)
            if weights is not None:
                elements = elements * weights
            return elements if return_elements else torch.mean(elements)

        def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
            raise NotImplementedError

try:
    from .direction_utils import (
        normalize_unit_vectors,
        opening_angle_radians,
        unpack_direction_target,
        zenith_azimuth_to_unit_vector,
    )
    from .energy_weighting import (
        EnergyWeightLookup,
        EnergyWeightManifest,
    )
except ImportError:  # Support direct imports from this numeric example folder.
    from direction_utils import (  # type: ignore[no-redef]
        normalize_unit_vectors,
        opening_angle_radians,
        unpack_direction_target,
        zenith_azimuth_to_unit_vector,
    )
    from energy_weighting import (  # type: ignore[no-redef]
        EnergyWeightLookup,
        EnergyWeightManifest,
    )


def _require_shape(tensor: Tensor, shape_tail: tuple[int, ...], name: str) -> None:
    expected_dimensions = 1 + len(shape_tail)
    if tensor.ndim != expected_dimensions or tuple(tensor.shape[1:]) != shape_tail:
        readable = ", ".join(("N", *(str(value) for value in shape_tail)))
        raise ValueError(
            f"{name} must have shape [{readable}], received {tuple(tensor.shape)}."
        )


def log_sinh_over_x(value: Tensor, series_threshold: float = 1.0e-2) -> Tensor:
    """Calculate ``log(sinh(x) / x)`` stably for non-negative ``x``."""

    if not value.is_floating_point():
        raise TypeError("value must have a floating-point dtype.")
    if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()):
        raise ValueError("value must contain finite non-negative entries.")
    if series_threshold <= 0.0:
        raise ValueError("series_threshold must be positive.")

    squared = value.square()
    series = squared / 6.0 - squared.square() / 180.0
    series = series + squared.pow(3) / 2835.0

    safe_value = value.clamp_min(float(series_threshold))
    regular = (
        safe_value
        + torch.log1p(-torch.exp(-2.0 * safe_value))
        - math.log(2.0)
        - torch.log(safe_value)
    )
    return torch.where(value < series_threshold, series, regular)


def von_mises_fisher_3d_nll(
    predicted_direction: Tensor,
    kappa: Tensor,
    target_direction: Tensor,
) -> Tensor:
    """Pure-Torch elementwise negative log-likelihood on the unit sphere."""

    _require_shape(predicted_direction, (3,), "predicted_direction")
    _require_shape(target_direction, (3,), "target_direction")
    if kappa.ndim != 1:
        raise ValueError(f"kappa must have shape [N], got {tuple(kappa.shape)}.")
    if predicted_direction.shape != target_direction.shape:
        raise ValueError("Predicted and target directions must have equal shapes.")
    if kappa.shape[0] != predicted_direction.shape[0]:
        raise ValueError("kappa must contain one value per event.")
    if not bool(torch.isfinite(kappa).all()) or bool((kappa <= 0.0).any()):
        raise ValueError("kappa must contain finite positive values.")

    predicted_unit = normalize_unit_vectors(predicted_direction)
    target_unit = normalize_unit_vectors(target_direction.to(predicted_direction))
    cosine = torch.sum(predicted_unit * target_unit, dim=-1).clamp(-1.0, 1.0)
    return (
        math.log(4.0 * math.pi)
        + log_sinh_over_x(kappa)
        - kappa * cosine
    )


class EnergyWeightedDirectionLoss(LossFunction):
    """Joint direction loss for target columns ``zenith, azimuth, energy``.

    ``objective='vmf'`` is the stage-A baseline.  Stage B uses
    ``objective='angular_hybrid'`` and minimizes an explicit angular surrogate
    plus ``vmf_factor * vMF``.  Internal weights always have shape ``[N]`` and,
    by default, are active only while the module is in training mode.

    GraphNeT's external ``loss_weight`` mechanism must remain disabled for the
    task using this loss; pass ``loss_weight=None`` to the reconstruction task.
    """

    def __init__(
        self,
        log10_bin_edges: Sequence[float] | None = None,
        bin_weights: Sequence[float] | None = None,
        bin_counts: Sequence[int] | None = None,
        *,
        objective: str = "vmf",
        angular_surrogate: str = "chord",
        vmf_factor: float = 0.05,
        weighting_mode: str = "train_only",
        out_of_range: str = "error",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if objective not in {"vmf", "angular_hybrid"}:
            raise ValueError("objective must be 'vmf' or 'angular_hybrid'.")
        if angular_surrogate not in {"chord", "opening_angle"}:
            raise ValueError(
                "angular_surrogate must be 'chord' or 'opening_angle'."
            )
        if not math.isfinite(vmf_factor) or vmf_factor < 0.0:
            raise ValueError("vmf_factor must be finite and non-negative.")
        if weighting_mode not in {"train_only", "always", "disabled"}:
            raise ValueError(
                "weighting_mode must be 'train_only', 'always', or 'disabled'."
            )

        table_was_supplied = log10_bin_edges is not None or bin_weights is not None
        if (log10_bin_edges is None) != (bin_weights is None):
            raise ValueError(
                "log10_bin_edges and bin_weights must be supplied together."
            )
        if weighting_mode != "disabled" and not table_was_supplied:
            raise ValueError("Enabled weighting requires an energy-weight table.")

        self.objective = objective
        self.angular_surrogate = angular_surrogate
        self.vmf_factor = float(vmf_factor)
        self.weighting_mode = weighting_mode
        self._weight_lookup: EnergyWeightLookup | None
        if table_was_supplied:
            assert log10_bin_edges is not None
            assert bin_weights is not None
            self._weight_lookup = EnergyWeightLookup(
                log10_bin_edges=log10_bin_edges,
                bin_weights=bin_weights,
                bin_counts=bin_counts,
                out_of_range=out_of_range,
            )
        else:
            self._weight_lookup = None

    @classmethod
    def from_manifest(
        cls,
        manifest: EnergyWeightManifest | Mapping[str, Any] | str | Path,
        **kwargs: Any,
    ) -> "EnergyWeightedDirectionLoss":
        """Construct the loss from a manifest object, mapping, or JSON path."""

        if isinstance(manifest, (str, Path)):
            resolved = EnergyWeightManifest.load(manifest)
        elif isinstance(manifest, EnergyWeightManifest):
            resolved = manifest
        else:
            resolved = EnergyWeightManifest.from_dict(manifest)
        return cls(
            log10_bin_edges=list(resolved.log10_bin_edges),
            bin_weights=list(resolved.bin_weights),
            bin_counts=list(resolved.bin_counts),
            out_of_range=resolved.out_of_range,
            **kwargs,
        )

    def _event_weights(self, energy: Tensor) -> Tensor:
        use_weights = self.weighting_mode == "always" or (
            self.weighting_mode == "train_only" and self.training
        )
        if not use_weights:
            return torch.ones_like(energy)
        if self._weight_lookup is None:
            raise RuntimeError("Energy weighting is enabled without a lookup table.")
        weights = self._weight_lookup(energy)
        if weights.ndim != 1 or weights.shape != energy.shape:
            raise RuntimeError(
                "Internal energy weights must have shape [N]; refusing unsafe "
                "broadcasting."
            )
        return weights

    def diagnostic_energy_weights(self, energy: Tensor) -> Tensor:
        """Return the fixed train-derived weights regardless of module mode.

        This is used to aggregate the weighted validation diagnostic globally
        as ``sum(w * loss) / sum(w)``.  It intentionally does not perform
        per-batch normalization.
        """

        if self._weight_lookup is None:
            return torch.ones_like(energy)
        return self._weight_lookup(energy)

    def elementwise_components(
        self, prediction: Tensor, target: Tensor
    ) -> dict[str, Tensor]:
        """Return diagnostic per-event components without reducing the batch."""

        _require_shape(prediction, (4,), "prediction")
        _require_shape(target, (3,), "target")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("prediction and target batch sizes must match.")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise TypeError("prediction and target must be floating-point tensors.")
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("prediction contains NaN or infinite values.")

        target = target.to(device=prediction.device, dtype=prediction.dtype)
        zenith, azimuth, energy = unpack_direction_target(target)
        if bool((energy <= 0.0).any()):
            raise ValueError("Target totalEnergy must be strictly positive.")

        predicted_unit = normalize_unit_vectors(prediction[:, :3])
        target_unit = zenith_azimuth_to_unit_vector(zenith, azimuth)
        kappa = prediction[:, 3]
        vmf = von_mises_fisher_3d_nll(predicted_unit, kappa, target_unit)

        if self.angular_surrogate == "chord":
            angular = torch.linalg.vector_norm(
                predicted_unit - target_unit, dim=-1
            ) / math.sqrt(3.0)
        else:
            angular = opening_angle_radians(predicted_unit, target_unit)

        if self.objective == "vmf":
            unweighted = vmf
        else:
            unweighted = angular + self.vmf_factor * vmf
        weights = self._event_weights(energy)
        # GraphNeT's final LossFunction.forward applies torch.mean to the
        # elements returned here.  Dividing by the batch mean makes that
        # reduction exactly sum(w_i * L_i) / sum(w_i), while keeping all
        # tensors one-dimensional and preventing the native [N] x [N, 1]
        # broadcasting bug.
        weight_sum = weights.sum()
        if not bool(torch.isfinite(weight_sum)) or float(weight_sum) <= 0.0:
            raise ValueError("The batch energy weights must have a positive sum.")
        normalized_weights = weights * (weights.numel() / weight_sum)
        weighted = unweighted * normalized_weights
        expected_shape = (prediction.shape[0],)
        for name, value in {
            "vmf": vmf,
            "angular": angular,
            "unweighted": unweighted,
            "weights": weights,
            "normalized_weights": normalized_weights,
            "weighted": weighted,
        }.items():
            if value.shape != expected_shape:
                raise RuntimeError(
                    f"{name} has shape {tuple(value.shape)}, expected "
                    f"{expected_shape}."
                )
        return {
            "vmf": vmf,
            "angular": angular,
            "unweighted": unweighted,
            "weights": weights,
            "normalized_weights": normalized_weights,
            "weighted": weighted,
        }

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.elementwise_components(prediction, target)["weighted"]
