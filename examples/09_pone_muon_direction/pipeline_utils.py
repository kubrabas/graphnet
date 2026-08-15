"""Shared I/O and configuration helpers for the joint Muon-CC pipeline.

This module deliberately resolves only the canonical ``Muon`` train/val/test
entries from ``Metadata/paths.py``.  It never traverses a router category and
never writes to the source parquet directories.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import yaml


DEFAULT_PATHS_PY = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Results/340StringMC_JointDirection"
)
DEFAULT_GRAPHNET_SOURCE = Path("/project/def-nahee/kbas/graphnet/src")
SUPPORTED_GEOMETRIES = (
    "102_string_emax1e6",
    "160_string_emax1e6",
    "full_geometry_emax1e6",
)
SPLITS = ("train", "val", "test")


def assert_local_graphnet_source(config: Mapping[str, Any]) -> Dict[str, str]:
    """Fail fast unless ``graphnet`` was imported from the user's local tree."""

    import graphnet

    configured = config.get("environment", {}).get(
        "graphnet_source", DEFAULT_GRAPHNET_SOURCE
    )
    source_root = Path(configured).resolve()
    package_root = (source_root / "graphnet").resolve()
    module_file = getattr(graphnet, "__file__", None)
    if module_file is None:
        raise RuntimeError("graphnet.__file__ is unavailable; source cannot be verified")
    imported = Path(module_file).resolve()
    if (
        imported != package_root / "__init__.py"
        and package_root not in imported.parents
    ):
        raise RuntimeError(
            "Wrong GraphNeT source imported. "
            f"Expected a module below {package_root}, got {imported}. "
            "The container dependency environment may be used, but PYTHONPATH must "
            "put the local graphnet/src first."
        )
    return {
        "configured_graphnet_source": str(source_root),
        "imported_graphnet_file": str(imported),
    }


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML mapping and reject an empty/non-mapping document."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def load_paths_module(path: str | Path = DEFAULT_PATHS_PY):
    """Import the applications path registry without modifying it."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"paths.py does not exist: {path}")
    spec = importlib.util.spec_from_file_location("pone_joint_direction_paths", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import paths module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_muon_split_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    """Resolve direct canonical Muon paths, bypassing every router view."""

    data = config["data"]
    geometry = str(data["geometry"])
    if geometry not in SUPPORTED_GEOMETRIES:
        raise ValueError(
            f"Unsupported geometry {geometry!r}; expected one of {SUPPORTED_GEOMETRIES}"
        )
    if str(data.get("flavor", "Muon")) != "Muon":
        raise ValueError("This pipeline is intentionally restricted to flavor=Muon")

    module = load_paths_module(data.get("paths_py", DEFAULT_PATHS_PY))
    table_name = str(data.get("parquet_table", "STRING340MC_PARQUET"))
    table = getattr(module, table_name, None)
    if not isinstance(table, dict):
        raise KeyError(f"{table_name} is missing from paths.py")

    try:
        muon = table[geometry]["Muon"]
    except KeyError as exc:
        raise KeyError(f"Missing canonical path entry: {table_name}.{geometry}.Muon") from exc

    resolved: Dict[str, Path] = {}
    for split in SPLITS:
        raw = muon.get(split)
        if not raw or raw == "does_not_exist":
            raise ValueError(
                f"Missing direct canonical split: {table_name}.{geometry}.Muon.{split}"
            )
        path = Path(raw)
        if not path.is_dir():
            raise FileNotFoundError(f"Muon {split} directory does not exist: {path}")
        resolved[split] = path
    return resolved


def resolve_percentiles_csv(config: Mapping[str, Any]) -> Path:
    """Resolve the explicit, already-produced Muon-CC feature scaler."""

    data = config["data"]
    explicit = data.get("percentiles_csv")
    if not explicit:
        raise ValueError(
            "data.percentiles_csv must be explicit; router keys are not used to "
            "discover scalers in the joint pipeline"
        )
    path = Path(explicit)
    if not path.is_file():
        raise FileNotFoundError(f"Percentiles CSV does not exist: {path}")
    return path


def experiment_dir(config: Mapping[str, Any]) -> Path:
    """Return Results/340StringMC_JointDirection/<geometry>/<experiment>."""

    root = Path(config.get("output", {}).get("root_dir", DEFAULT_OUTPUT_ROOT))
    geometry = str(config["data"]["geometry"])
    experiment = str(config["experiment_name"])
    if not experiment or experiment in {".", ".."} or "/" in experiment:
        raise ValueError(f"Unsafe experiment_name: {experiment!r}")
    return root / geometry / experiment


def prepare_experiment_dir(
    path: Path,
    policy: str,
    *,
    output_prepared: bool = False,
) -> None:
    """Create an output directory without ever deleting prior results."""

    if policy not in {"error", "resume"}:
        raise ValueError("run.existing_output must be 'error' or 'resume'")
    if path.exists() and any(path.iterdir()) and policy == "error" and not output_prepared:
        raise FileExistsError(
            f"Experiment output already exists: {path}. Use a new experiment_name "
            "or set existing_output=resume explicitly. Nothing was deleted."
        )
    path.mkdir(parents=True, exist_ok=True)


def table_files(split_path: str | Path, table: str) -> List[Path]:
    """Return numerically sorted parquet chunks for one GraphNeT table."""

    directory = Path(split_path) / table
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing parquet table directory: {directory}")

    def key(path: Path):
        match = re.search(r"_(\d+)\.parquet$", path.name)
        return (0, int(match.group(1))) if match else (1, path.name)

    files = sorted(directory.glob("*.parquet"), key=key)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    return files


def chunk_ids(files: Iterable[Path]) -> set[str]:
    """Extract the chunk suffix used to pair truth and feature parquet files."""

    result: set[str] = set()
    for path in files:
        match = re.search(r"_(\d+)\.parquet$", path.name)
        if match is None:
            raise ValueError(f"Unexpected parquet filename: {path.name}")
        result.add(match.group(1))
    return result


def audit_parquet_layout(
    split_paths: Mapping[str, Path],
    *,
    truth_columns: Sequence[str],
    feature_columns: Sequence[str],
    pulsemaps: str = "features",
    truth_table: str = "truth",
    minimum_pulses_per_event: int = 2,
) -> Dict[str, Any]:
    """Fast schema/chunk audit using parquet metadata only."""

    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the parquet schema audit") from exc

    report: Dict[str, Any] = {"splits": {}}
    for split, path in split_paths.items():
        truth_files = table_files(path, truth_table)
        feature_files = table_files(path, pulsemaps)
        truth_ids = chunk_ids(truth_files)
        feature_ids = chunk_ids(feature_files)
        if truth_ids != feature_ids:
            raise ValueError(
                f"{split}: truth/features chunk mismatch; "
                f"truth-only={sorted(truth_ids - feature_ids)[:10]}, "
                f"features-only={sorted(feature_ids - truth_ids)[:10]}"
            )

        truth_schema = set(pq.read_schema(truth_files[0]).names)
        feature_schema = set(pq.read_schema(feature_files[0]).names)
        for parquet_path in truth_files:
            missing_truth = sorted(
                set(truth_columns) - set(pq.read_schema(parquet_path).names)
            )
            if missing_truth:
                raise ValueError(
                    f"{split}: missing truth columns in {parquet_path.name}: "
                    f"{missing_truth}"
                )
        for parquet_path in feature_files:
            missing_features = sorted(
                set(feature_columns) - set(pq.read_schema(parquet_path).names)
            )
            if missing_features:
                raise ValueError(
                    f"{split}: missing feature columns in {parquet_path.name}: "
                    f"{missing_features}"
                )

        if minimum_pulses_per_event < 1:
            raise ValueError("minimum_pulses_per_event must be positive")
        truth_by_chunk = {
            next(iter(chunk_ids([parquet_path]))): parquet_path
            for parquet_path in truth_files
        }
        feature_by_chunk = {
            next(iter(chunk_ids([parquet_path]))): parquet_path
            for parquet_path in feature_files
        }
        minimum_pulses = None
        maximum_pulses = 0
        for chunk in sorted(truth_by_chunk, key=int):
            truth_event_no = (
                pq.read_table(truth_by_chunk[chunk], columns=["event_no"])
                .column(0)
                .to_numpy()
            )
            pulse_event_no = (
                pq.read_table(feature_by_chunk[chunk], columns=["event_no"])
                .column(0)
                .to_numpy()
            )
            unique_event_no, pulse_counts = np.unique(
                pulse_event_no, return_counts=True
            )
            if len(unique_event_no) != len(truth_event_no) or not np.array_equal(
                np.sort(unique_event_no), np.sort(truth_event_no)
            ):
                missing = np.setdiff1d(truth_event_no, unique_event_no)
                extra = np.setdiff1d(unique_event_no, truth_event_no)
                raise ValueError(
                    f"{split} chunk {chunk}: truth/feature event coverage mismatch; "
                    f"missing={missing[:10].tolist()}, extra={extra[:10].tolist()}"
                )
            chunk_minimum = int(pulse_counts.min())
            minimum_pulses = (
                chunk_minimum
                if minimum_pulses is None
                else min(minimum_pulses, chunk_minimum)
            )
            maximum_pulses = max(maximum_pulses, int(pulse_counts.max()))
            if chunk_minimum < minimum_pulses_per_event:
                raise ValueError(
                    f"{split} chunk {chunk}: event with only {chunk_minimum} "
                    "pulse(s) would be silently dropped by GraphNeT DataLoader"
                )

        truth_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in truth_files)
        feature_rows = sum(
            pq.ParquetFile(path).metadata.num_rows for path in feature_files
        )
        report["splits"][split] = {
            "path": str(path),
            "chunks": len(truth_files),
            "truth_rows": int(truth_rows),
            "pulse_rows": int(feature_rows),
            "minimum_pulses_per_event": int(minimum_pulses or 0),
            "maximum_pulses_per_event": int(maximum_pulses),
            "truth_schema": sorted(truth_schema),
            "feature_schema": sorted(feature_schema),
        }
    return report


def read_truth_columns(split_path: str | Path, columns: Sequence[str]):
    """Read selected scalar truth columns only, never pulse data."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read truth parquet metadata") from exc

    tables = [pq.read_table(path, columns=list(columns)) for path in table_files(split_path, "truth")]
    if not tables:
        raise ValueError(f"No truth rows found in {split_path}")
    # All chunks are produced by one converter and the schema audit already
    # enforces the requested columns, so no schema promotion is needed.  This
    # call is compatible with the pyarrow version in the GraphNeT 1.8 image.
    return pa.concat_tables(tables).to_pandas()


def extract_field(batch, field: str) -> torch.Tensor:
    """Concatenate an event-level field from a GraphNeT batch."""

    try:
        from torch_geometric.data import Data
    except ImportError:
        Data = ()  # type: ignore[assignment]
    if Data and isinstance(batch, Data):
        return batch[field]
    if isinstance(batch, (list, tuple)):
        return torch.cat([item[field] for item in batch], dim=0)
    raise TypeError(f"Unsupported GraphNeT batch type: {type(batch)}")


def move_batch_to_device(batch, device: torch.device):
    """Move either a PyG Data object or GraphNeT's list batch to a device."""

    if isinstance(batch, (list, tuple)):
        return [item.to(device) for item in batch]
    return batch.to(device)


def atomic_json_dump(payload: Mapping[str, Any], destination: str | Path) -> None:
    """Write JSON atomically so interrupted jobs do not leave partial files."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)


def numeric_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Convert scalar tensors/numbers to a JSON/CSV-safe metric mapping."""

    result: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def checkpoint_state(path: str | Path) -> Dict[str, torch.Tensor]:
    """Load either a plain GraphNeT state dict or our checkpoint envelope."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")
    return payload
