"""Train-set energy-density weights without modifying source parquet files."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


MANIFEST_VERSION = 1


def _as_float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values.")
    return result


@dataclass(frozen=True)
class EnergyWeightManifest:
    """Serializable description of weights fitted exclusively on training data."""

    log10_bin_edges: tuple[float, ...]
    bin_counts: tuple[int, ...]
    bin_weights: tuple[float, ...]
    alpha: float
    energy_column: str = "totalEnergy"
    clip_min: float | None = None
    clip_max: float | None = None
    out_of_range: str = "error"
    source_files: tuple[str, ...] = ()
    version: int = MANIFEST_VERSION

    def __post_init__(self) -> None:
        edges = _as_float_tuple(self.log10_bin_edges, "log10_bin_edges")
        object.__setattr__(self, "log10_bin_edges", edges)
        object.__setattr__(
            self, "bin_counts", tuple(int(value) for value in self.bin_counts)
        )
        object.__setattr__(
            self,
            "bin_weights",
            tuple(float(value) for value in self.bin_weights),
        )
        object.__setattr__(
            self, "source_files", tuple(str(value) for value in self.source_files)
        )

        n_bins = len(edges) - 1
        if n_bins < 1:
            raise ValueError("At least two log10_bin_edges are required.")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("log10_bin_edges must be strictly increasing.")
        if len(self.bin_counts) != n_bins or len(self.bin_weights) != n_bins:
            raise ValueError(
                "bin_counts and bin_weights must each have len(edges) - 1 "
                "entries."
            )
        if any(count < 0 for count in self.bin_counts):
            raise ValueError("bin_counts cannot contain negative values.")
        if sum(self.bin_counts) <= 0:
            raise ValueError("The manifest must describe at least one event.")
        if any(
            (not math.isfinite(weight)) or weight < 0.0
            for weight in self.bin_weights
        ):
            raise ValueError("bin_weights must be finite and non-negative.")
        for count, weight in zip(self.bin_counts, self.bin_weights):
            if count > 0 and weight <= 0.0:
                raise ValueError("Every occupied bin must have a positive weight.")
            if count == 0 and weight != 0.0:
                raise ValueError("Empty bins must have weight zero.")
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be a finite non-negative number.")
        if self.clip_min is not None and (
            not math.isfinite(self.clip_min) or self.clip_min <= 0.0
        ):
            raise ValueError("clip_min must be finite and positive.")
        if self.clip_max is not None and (
            not math.isfinite(self.clip_max) or self.clip_max <= 0.0
        ):
            raise ValueError("clip_max must be finite and positive.")
        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_min > self.clip_max
        ):
            raise ValueError("clip_min cannot exceed clip_max.")
        if self.out_of_range not in {"error", "clip"}:
            raise ValueError("out_of_range must be 'error' or 'clip'.")
        if self.version != MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported manifest version {self.version}; "
                f"expected {MANIFEST_VERSION}."
            )

    @property
    def number_of_events(self) -> int:
        """Number of train events used to fit the histogram."""

        return sum(self.bin_counts)

    @property
    def event_weighted_mean(self) -> float:
        """Mean weight over fitted train events."""

        weighted_sum = sum(
            count * weight
            for count, weight in zip(self.bin_counts, self.bin_weights)
        )
        return weighted_sum / self.number_of_events

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "version": self.version,
            "energy_column": self.energy_column,
            "space": "log10_energy",
            "log10_bin_edges": list(self.log10_bin_edges),
            "bin_counts": list(self.bin_counts),
            "bin_weights": list(self.bin_weights),
            "alpha": self.alpha,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "out_of_range": self.out_of_range,
            "source_files": list(self.source_files),
            "normalization": "event_mean_one",
            "number_of_events": self.number_of_events,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EnergyWeightManifest":
        """Validate and construct a manifest from parsed JSON."""

        if payload.get("space", "log10_energy") != "log10_energy":
            raise ValueError("Only log10-energy manifests are supported.")
        if payload.get("normalization", "event_mean_one") != "event_mean_one":
            raise ValueError("Only event-mean-one normalization is supported.")
        return cls(
            version=int(payload.get("version", MANIFEST_VERSION)),
            energy_column=str(payload.get("energy_column", "totalEnergy")),
            log10_bin_edges=tuple(payload["log10_bin_edges"]),
            bin_counts=tuple(payload["bin_counts"]),
            bin_weights=tuple(payload["bin_weights"]),
            alpha=float(payload["alpha"]),
            clip_min=(
                None
                if payload.get("clip_min") is None
                else float(payload["clip_min"])
            ),
            clip_max=(
                None
                if payload.get("clip_max") is None
                else float(payload["clip_max"])
            ),
            out_of_range=str(payload.get("out_of_range", "error")),
            source_files=tuple(payload.get("source_files", ())),
        )

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """Save the manifest as JSON, leaving parquet inputs untouched."""

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Manifest already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "EnergyWeightManifest":
        """Load and validate a JSON manifest."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Energy-weight manifest must contain a JSON object.")
        return cls.from_dict(payload)


def _one_dimensional_floating_tensor(
    values: Tensor | Sequence[float], name: str
) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return tensor


def _bin_indices(log10_energy: Tensor, edges: Tensor) -> Tensor:
    """Map values to ``[left, right)`` bins; the last right edge is inclusive."""

    if bool((log10_energy < edges[0]).any()) or bool(
        (log10_energy > edges[-1]).any()
    ):
        raise ValueError("Energy lies outside the configured histogram range.")
    return torch.bucketize(log10_energy, edges[1:-1], right=True)


def fit_energy_weight_manifest(
    energies: Tensor | Sequence[float],
    log10_bin_edges: Tensor | Sequence[float],
    *,
    alpha: float = 0.5,
    clip_min: float | None = None,
    clip_max: float | None = None,
    energy_column: str = "totalEnergy",
    out_of_range: str = "error",
    source_files: Sequence[str | Path] = (),
) -> EnergyWeightManifest:
    """Fit inverse-density weights from train energies.

    For occupied bin ``b``, the unnormalized weight is ``N_b ** -alpha``.
    Weights are normalized, optionally clipped, and normalized again so that
    their event-weighted mean on the fitted train sample is exactly one.
    """

    energy = _one_dimensional_floating_tensor(energies, "energies")
    if bool((energy <= 0.0).any()):
        raise ValueError("energies must be strictly positive.")
    edges = _one_dimensional_floating_tensor(
        log10_bin_edges, "log10_bin_edges"
    )
    if edges.numel() < 2 or bool((edges[1:] <= edges[:-1]).any()):
        raise ValueError("log10_bin_edges must be strictly increasing.")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite non-negative number.")
    if clip_min is not None and (
        not math.isfinite(clip_min) or clip_min <= 0.0
    ):
        raise ValueError("clip_min must be finite and positive.")
    if clip_max is not None and (
        not math.isfinite(clip_max) or clip_max <= 0.0
    ):
        raise ValueError("clip_max must be finite and positive.")
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        raise ValueError("clip_min cannot exceed clip_max.")

    indices = _bin_indices(torch.log10(energy), edges)
    counts = torch.bincount(indices, minlength=edges.numel() - 1)
    occupied = counts > 0
    weights = torch.zeros_like(counts, dtype=torch.float64)
    weights[occupied] = counts[occupied].to(torch.float64).pow(-float(alpha))

    def normalize(current: Tensor) -> Tensor:
        mean = torch.sum(counts.to(current) * current) / counts.sum()
        if not bool(torch.isfinite(mean)) or float(mean) <= 0.0:
            raise ValueError("Could not normalize energy weights.")
        return current / mean

    weights = normalize(weights)
    occupied_weights = weights[occupied]
    if clip_min is not None:
        occupied_weights = occupied_weights.clamp_min(float(clip_min))
    if clip_max is not None:
        occupied_weights = occupied_weights.clamp_max(float(clip_max))
    weights[occupied] = occupied_weights
    weights = normalize(weights)

    return EnergyWeightManifest(
        log10_bin_edges=tuple(edges.tolist()),
        bin_counts=tuple(int(value) for value in counts.tolist()),
        bin_weights=tuple(float(value) for value in weights.tolist()),
        alpha=float(alpha),
        energy_column=energy_column,
        clip_min=clip_min,
        clip_max=clip_max,
        out_of_range=out_of_range,
        source_files=tuple(str(path) for path in source_files),
    )


class EnergyWeightLookup(nn.Module):
    """Device-safe, checkpoint-persistent lookup of fitted energy weights."""

    def __init__(
        self,
        log10_bin_edges: Sequence[float],
        bin_weights: Sequence[float],
        bin_counts: Sequence[int] | None = None,
        out_of_range: str = "error",
    ) -> None:
        super().__init__()
        edges = torch.as_tensor(log10_bin_edges, dtype=torch.float64)
        weights = torch.as_tensor(bin_weights, dtype=torch.float64)
        if bin_counts is None:
            counts = torch.ones_like(weights, dtype=torch.int64)
        else:
            counts = torch.as_tensor(bin_counts, dtype=torch.int64)

        if edges.ndim != 1 or edges.numel() < 2:
            raise ValueError("At least two one-dimensional bin edges are required.")
        if bool((edges[1:] <= edges[:-1]).any()):
            raise ValueError("log10_bin_edges must be strictly increasing.")
        if weights.shape != (edges.numel() - 1,):
            raise ValueError("bin_weights must have len(edges) - 1 entries.")
        if counts.shape != weights.shape or bool((counts < 0).any()):
            raise ValueError("bin_counts must match bin_weights and be non-negative.")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("bin_weights must be finite and non-negative.")
        if bool(((counts > 0) & (weights <= 0)).any()):
            raise ValueError("Occupied bins must have positive weights.")
        if bool(((counts == 0) & (weights != 0)).any()):
            raise ValueError("Empty bins must have weight zero.")
        if out_of_range not in {"error", "clip"}:
            raise ValueError("out_of_range must be 'error' or 'clip'.")

        self.out_of_range = out_of_range
        self.register_buffer("log10_bin_edges", edges)
        self.register_buffer("bin_weights", weights)
        self.register_buffer("bin_counts", counts)

    @classmethod
    def from_manifest(cls, manifest: EnergyWeightManifest) -> "EnergyWeightLookup":
        """Build a lookup module from a validated manifest."""

        return cls(
            log10_bin_edges=manifest.log10_bin_edges,
            bin_weights=manifest.bin_weights,
            bin_counts=manifest.bin_counts,
            out_of_range=manifest.out_of_range,
        )

    def forward(self, energy: Tensor) -> Tensor:
        """Return exactly one scalar weight per event, shape ``[N]``."""

        if not isinstance(energy, Tensor) or not energy.is_floating_point():
            raise TypeError("energy must be a floating-point torch.Tensor.")
        if energy.ndim != 1:
            raise ValueError(
                f"energy must have shape [N], received {tuple(energy.shape)}."
            )
        if not bool(torch.isfinite(energy).all()) or bool((energy <= 0).any()):
            raise ValueError("energy must contain finite positive values.")

        edges = self.log10_bin_edges.to(device=energy.device, dtype=energy.dtype)
        log10_energy = torch.log10(energy)
        below = log10_energy < edges[0]
        above = log10_energy > edges[-1]
        if self.out_of_range == "error" and bool((below | above).any()):
            raise ValueError("Energy lies outside the fitted histogram range.")
        if self.out_of_range == "clip":
            log10_energy = log10_energy.clamp(edges[0], edges[-1])

        indices = torch.bucketize(log10_energy, edges[1:-1], right=True)
        counts = self.bin_counts.to(device=energy.device)[indices]
        if bool((counts == 0).any()):
            raise ValueError("An energy mapped to a bin that was empty during fit.")
        weights = self.bin_weights.to(
            device=energy.device, dtype=energy.dtype
        )[indices]
        if weights.shape != energy.shape:
            raise RuntimeError("Energy-weight lookup produced an invalid shape.")
        return weights
