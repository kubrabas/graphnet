"""Pure-PyTorch helpers for three-dimensional direction reconstruction.

Angles follow the IceCube/GraphNeT convention: zenith is measured from the
positive z-axis and azimuth is measured in the x-y plane.  All public
functions preserve leading dimensions and operate on the final dimension.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor


def _require_floating_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values.")


def normalize_unit_vectors(vectors: Tensor, eps: float | None = None) -> Tensor:
    """Return normalized 3-vectors, failing on invalid or zero-length input."""

    _require_floating_tensor(vectors, "vectors")
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError(
            "vectors must have shape [..., 3], "
            f"received {tuple(vectors.shape)}."
        )

    threshold = (
        float(eps)
        if eps is not None
        else float(torch.finfo(vectors.dtype).eps)
    )
    if threshold <= 0.0 or not math.isfinite(threshold):
        raise ValueError("eps must be a finite positive number.")

    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    if bool((norms <= threshold).any()):
        raise ValueError("Cannot normalize a zero-length direction vector.")
    return vectors / norms


def zenith_azimuth_to_unit_vector(zenith: Tensor, azimuth: Tensor) -> Tensor:
    """Convert zenith and azimuth in radians to Cartesian unit vectors."""

    _require_floating_tensor(zenith, "zenith")
    _require_floating_tensor(azimuth, "azimuth")
    if zenith.shape != azimuth.shape:
        raise ValueError(
            "zenith and azimuth must have identical shapes, received "
            f"{tuple(zenith.shape)} and {tuple(azimuth.shape)}."
        )
    if zenith.device != azimuth.device:
        raise ValueError("zenith and azimuth must be on the same device.")

    azimuth = azimuth.to(dtype=zenith.dtype)
    sin_zenith = torch.sin(zenith)
    return torch.stack(
        (
            sin_zenith * torch.cos(azimuth),
            sin_zenith * torch.sin(azimuth),
            torch.cos(zenith),
        ),
        dim=-1,
    )


def unit_vector_to_zenith_azimuth(vectors: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert Cartesian directions to zenith and azimuth in radians.

    Azimuth is returned in the half-open interval ``[0, 2*pi)``.
    """

    unit_vectors = normalize_unit_vectors(vectors)
    zenith = torch.acos(unit_vectors[..., 2].clamp(-1.0, 1.0))
    azimuth = torch.remainder(
        torch.atan2(unit_vectors[..., 1], unit_vectors[..., 0]),
        unit_vectors.new_tensor(2.0 * math.pi),
    )
    return zenith, azimuth


def opening_angle_radians(prediction: Tensor, target: Tensor) -> Tensor:
    """Calculate the opening angle using a stable ``atan2`` formulation."""

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes, received "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    predicted_unit = normalize_unit_vectors(prediction)
    target_unit = normalize_unit_vectors(target.to(prediction))

    cosine = torch.sum(predicted_unit * target_unit, dim=-1).clamp(-1.0, 1.0)
    sine = torch.linalg.vector_norm(
        torch.linalg.cross(predicted_unit, target_unit, dim=-1), dim=-1
    ).clamp(0.0, 1.0)
    return torch.atan2(sine, cosine)


def opening_angle_degrees(prediction: Tensor, target: Tensor) -> Tensor:
    """Calculate the opening angle in degrees."""

    return torch.rad2deg(opening_angle_radians(prediction, target))


def prediction_to_zenith_azimuth(prediction: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert a ``[N, >=3]`` direction prediction to angular coordinates."""

    _require_floating_tensor(prediction, "prediction")
    if prediction.ndim != 2 or prediction.shape[1] < 3:
        raise ValueError(
            "prediction must have shape [N, >=3], "
            f"received {tuple(prediction.shape)}."
        )
    return unit_vector_to_zenith_azimuth(prediction[:, :3])


def unpack_direction_target(target: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Unpack the pipeline target ``[zenith, azimuth, totalEnergy]``."""

    _require_floating_tensor(target, "target")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(
            "target must have shape [N, 3] with columns "
            "[zenith, azimuth, totalEnergy], "
            f"received {tuple(target.shape)}."
        )
    return target[:, 0], target[:, 1], target[:, 2]
