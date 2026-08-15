"""Opening-angle metrics used for validation, checkpointing, and reports."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from torch import Tensor


ANGLE_QUANTILES: Mapping[str, float] = {
    "median": 0.50,
    "q68": 0.68,
    "q84": 0.84,
    "q90": 0.90,
}


def _finite_vector(values: Tensor, name: str) -> Tensor:
    values = torch.as_tensor(values).detach().to(dtype=torch.float64, device="cpu")
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def _edge_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _angle_summary(angle_deg: Tensor) -> Dict[str, float]:
    result: Dict[str, float] = {
        "count": float(angle_deg.numel()),
        "mean_deg": float(angle_deg.mean()),
    }
    for name, quantile in ANGLE_QUANTILES.items():
        result[f"{name}_deg"] = float(torch.quantile(angle_deg, quantile))
    for threshold in (1.0, 2.0, 5.0, 10.0):
        result[f"fraction_below_{_edge_token(threshold)}deg"] = float(
            (angle_deg < threshold).to(torch.float64).mean()
        )
    return result


def opening_angle_metrics(
    opening_angle_deg: Tensor,
    total_energy: Tensor,
    log10_energy_bin_edges: Sequence[float],
    *,
    minimum_events_per_bin: int = 100,
    require_all_bins: bool = True,
    prefix: str = "val",
) -> Tuple[Dict[str, float], list[Dict[str, float]]]:
    """Compute global, energy-bin, and equal-bin (macro) angle metrics.

    The global median is the ordinary median over validation events.  The
    macro median is the arithmetic mean of the median from each fixed true
    energy bin, so every configured energy interval gets one equal vote.
    """

    angle = _finite_vector(opening_angle_deg, "opening_angle_deg")
    energy = _finite_vector(total_energy, "total_energy")
    if angle.shape != energy.shape:
        raise ValueError("opening angles and energies must have equal shapes")
    if bool((angle < 0.0).any()) or bool((angle > 180.0).any()):
        raise ValueError("opening angles must lie in [0, 180] degrees")
    if bool((energy <= 0.0).any()):
        raise ValueError("total_energy must be strictly positive")
    if minimum_events_per_bin < 1:
        raise ValueError("minimum_events_per_bin must be positive")

    edges = torch.as_tensor(
        list(log10_energy_bin_edges), dtype=torch.float64, device="cpu"
    )
    if edges.ndim != 1 or edges.numel() < 2 or bool((edges[1:] <= edges[:-1]).any()):
        raise ValueError("log10_energy_bin_edges must be strictly increasing")
    log_energy = torch.log10(energy)
    if bool((log_energy < edges[0]).any()) or bool((log_energy > edges[-1]).any()):
        raise ValueError("Validation energy lies outside the configured metric bins")

    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError(f"Invalid metric prefix: {prefix!r}")
    metrics = {
        f"{prefix}_global_{key}": value
        for key, value in _angle_summary(angle).items()
    }
    rows: list[Dict[str, float]] = []
    macro_values: Dict[str, list[float]] = {
        "mean_deg": [],
        "median_deg": [],
        "q68_deg": [],
        "q84_deg": [],
        "q90_deg": [],
        "fraction_below_1deg": [],
        "fraction_below_2deg": [],
        "fraction_below_5deg": [],
        "fraction_below_10deg": [],
    }

    last_bin = edges.numel() - 2
    for index, (left_tensor, right_tensor) in enumerate(zip(edges[:-1], edges[1:])):
        left = float(left_tensor)
        right = float(right_tensor)
        if index == last_bin:
            mask = (log_energy >= left) & (log_energy <= right)
        else:
            mask = (log_energy >= left) & (log_energy < right)
        count = int(mask.sum())
        if count < minimum_events_per_bin:
            if require_all_bins:
                raise ValueError(
                    f"Energy bin [{left}, {right}) has {count} events; "
                    f"minimum is {minimum_events_per_bin}"
                )
            continue

        summary = _angle_summary(angle[mask])
        row: Dict[str, float] = {
            "log10_energy_low": left,
            "log10_energy_high": right,
            **summary,
        }
        rows.append(row)
        token = f"loge_{_edge_token(left)}_{_edge_token(right)}"
        for name, value in summary.items():
            metrics[f"{prefix}_{token}_{name}"] = value
            if name in macro_values:
                macro_values[name].append(value)

    if not rows:
        raise ValueError("No energy bins had enough events for direction metrics")
    for name, values in macro_values.items():
        if values:
            metrics[f"{prefix}_macro_{name}"] = sum(values) / len(values)
    metrics[f"{prefix}_macro_number_of_bins"] = float(len(rows))
    return metrics, rows


def weighted_mean(values: Tensor, weights: Tensor) -> float:
    """Return a globally normalized weighted mean with shape safeguards."""

    values = _finite_vector(values, "values")
    weights = _finite_vector(weights, "weights")
    if values.shape != weights.shape:
        raise ValueError("values and weights must have exactly equal [N] shapes")
    if bool((weights < 0.0).any()):
        raise ValueError("weights must be non-negative")
    denominator = weights.sum()
    if not math.isfinite(float(denominator)) or float(denominator) <= 0.0:
        raise ValueError("weights must have a finite positive sum")
    return float(torch.sum(values * weights) / denominator)
