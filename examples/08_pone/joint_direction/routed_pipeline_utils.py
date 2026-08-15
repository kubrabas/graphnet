"""Safe configuration, path, and checkpoint helpers for routed joint direction.

The existing ``08_pone`` reconstruction pipeline is deliberately left intact.
This module resolves the same categorized train/validation parquet views and
their class-specific RobustScaler, but places the new joint direction result in
an isolated ``zenith_azimuth`` sibling directory.  Source parquet directories
are read-only inputs throughout this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import yaml


DEFAULT_PATHS_PY = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
)
DEFAULT_GRAPHNET_SOURCE = Path("/project/def-nahee/kbas/graphnet/src")
DEFAULT_OUTPUT_ROOT = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Results"
)
DEFAULT_FLAVORS = ("Muon", "Electron", "Tau", "NC")
SUPPORTED_SPLITS = ("train", "val")
TARGET_DIRECTORY_NAME = "zenith_azimuth"
PARQUET_TABLE_BY_MC = {
    "340StringMC": "STRING340MC_PARQUET",
    "Spring2026MC": "SPRING2026MC_PARQUET",
}


@dataclass(frozen=True)
class RoutedSplitPath:
    """One flavor constituent of a routed train or validation dataset."""

    flavor: str
    path: Path

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-safe representation for provenance manifests."""

        return {"flavor": self.flavor, "path": str(self.path)}


def normalize_route_class(route_class: str | int) -> str:
    """Return a numeric route-class identifier without a ``class`` prefix."""

    value = str(route_class).removeprefix("class")
    if not value.isdigit():
        raise ValueError(f"Route class must be a non-negative integer: {route_class!r}")
    return value


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load one non-empty YAML mapping."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {source}")
    return config


def atomic_json_dump(payload: Mapping[str, Any], destination: str | Path) -> None:
    """Atomically write a JSON mapping, avoiding partial files after preemption."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)


def assert_local_graphnet_source(config: Mapping[str, Any]) -> Dict[str, str]:
    """Fail unless Python imported GraphNeT from the configured local checkout."""

    import graphnet

    configured = config.get("environment", {}).get(
        "graphnet_source", DEFAULT_GRAPHNET_SOURCE
    )
    source_root = Path(configured).resolve()
    package_root = (source_root / "graphnet").resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(f"Configured GraphNeT package is absent: {package_root}")

    module_file = getattr(graphnet, "__file__", None)
    if module_file is None:
        raise RuntimeError("graphnet.__file__ is unavailable; source cannot be verified")
    imported = Path(module_file).resolve()
    if imported != package_root / "__init__.py" and package_root not in imported.parents:
        raise RuntimeError(
            "Wrong GraphNeT source imported. "
            f"Expected a module below {package_root}, got {imported}. "
            "The container may provide dependencies, but the local graphnet/src "
            "must appear first on PYTHONPATH."
        )
    return {
        "configured_graphnet_source": str(source_root),
        "imported_graphnet_file": str(imported),
    }


def load_paths_module(path: str | Path = DEFAULT_PATHS_PY):
    """Import the applications path registry without modifying it."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"paths.py does not exist: {source}")
    spec = importlib.util.spec_from_file_location(
        "pone_routed_joint_direction_paths", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import paths module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configured_geometry(config: Mapping[str, Any]) -> str:
    geometry = config.get("geometry", config.get("data", {}).get("geometry"))
    if not geometry:
        raise ValueError("A top-level geometry is required")
    return str(geometry)


def _configured_mc(config: Mapping[str, Any]) -> str:
    mc = config.get("mc")
    if not mc:
        raise ValueError("A top-level mc dataset family is required")
    return str(mc)


def _configured_flavors(config: Mapping[str, Any]) -> List[str]:
    raw = config.get("flavors", DEFAULT_FLAVORS)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("flavors must be a sequence of flavor names")
    flavors = [str(value) for value in raw]
    if not flavors or len(flavors) != len(set(flavors)):
        raise ValueError("flavors must be non-empty and contain no duplicates")
    unknown = sorted(set(flavors) - set(DEFAULT_FLAVORS))
    if unknown:
        raise ValueError(f"Unsupported flavors: {unknown}")
    return flavors


def _safe_component(value: Any, label: str) -> str:
    if value is None:
        raise ValueError(f"Missing {label}")
    result = str(value)
    if (
        not result
        or result in {".", ".."}
        or Path(result).name != result
        or "/" in result
        or "\\" in result
    ):
        raise ValueError(f"Unsafe {label}: {result!r}")
    return result


def target_dir(config: Mapping[str, Any], route_class: str | int) -> Path:
    """Return the exact isolated output directory for one routed joint model.

    The layout intentionally mirrors the completed separate reconstruction
    outputs, making ``zenith_azimuth`` a sibling of ``energy``, ``zenith``, and
    ``azimuth``::

        Results/<mc>/<geometry>/reconstruction/<category>/classN/
            <experiment>/<train_dir>/zenith_azimuth
    """

    route = normalize_route_class(route_class)
    output = config.get("output", {})
    root = Path(output.get("root_dir", DEFAULT_OUTPUT_ROOT))
    mc = _safe_component(_configured_mc(config), "mc")
    geometry = _safe_component(_configured_geometry(config), "geometry")
    category = _safe_component(config.get("routing", {}).get("category"), "routing category")
    experiment = _safe_component(config.get("experiment_name"), "experiment_name")
    train_name = _safe_component(
        output.get("dirs", {}).get("train", "train_and_val"), "train directory"
    )
    configured_target = output.get("target_directory", TARGET_DIRECTORY_NAME)
    if str(configured_target) != TARGET_DIRECTORY_NAME:
        raise ValueError(
            f"output.target_directory must remain {TARGET_DIRECTORY_NAME!r}"
        )
    return (
        root
        / mc
        / geometry
        / "reconstruction"
        / category
        / f"class{route}"
        / experiment
        / train_name
        / TARGET_DIRECTORY_NAME
    )


def _assert_safe_overwrite_target(path: Path) -> None:
    """Reject any recursive deletion that is not the exact task leaf."""

    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite a symlink: {path}")
    resolved = path.resolve()
    if resolved.name != TARGET_DIRECTORY_NAME:
        raise ValueError(
            "Refusing broad deletion: overwrite is permitted only for an exact "
            f"{TARGET_DIRECTORY_NAME!r} directory, got {resolved}"
        )
    parent_names = [parent.name for parent in resolved.parents]
    if "reconstruction" not in parent_names or not any(
        name.startswith("class") for name in parent_names
    ):
        raise ValueError(f"Refusing unsafe routed output deletion: {resolved}")


def prepare_target_dir(
    path: str | Path,
    policy: str,
    *,
    output_prepared: bool = False,
) -> bool:
    """Prepare exactly one ``zenith_azimuth`` output leaf safely.

    Returns ``False`` only when ``policy='skip'`` encounters an existing,
    non-empty target.  ``overwrite`` can remove this exact leaf after strict
    path checks; it can never remove the experiment, class, reconstruction, or
    Results directories.  ``output_prepared`` is used when a checked submitter
    has already performed the policy action before the SLURM worker starts.
    """

    target = Path(path)
    policy = str(policy)
    allowed = {"error", "skip", "overwrite", "resume"}
    if policy not in allowed:
        raise ValueError(f"run.existing_output must be one of {sorted(allowed)}")

    if output_prepared:
        if policy == "error":
            protected_names = {
                "pipeline_config.yml",
                "resolved_config.yml",
                "run_manifest.json",
                "data_audit.json",
                "energy_weight_manifest.json",
                "stage_a_vmf",
                "stage_b_angular_hybrid",
                "inference",
            }
            collisions = sorted(
                child.name
                for child in target.iterdir()
                if child.name in protected_names
            ) if target.exists() else []
            if collisions:
                raise FileExistsError(
                    "OUTPUT_PREPARED cannot bypass existing training artifacts "
                    f"in {target}: {collisions}. Nothing was overwritten."
                )
        target.mkdir(parents=True, exist_ok=True)
        return True

    nonempty = target.exists() and any(target.iterdir())
    if nonempty and policy == "error":
        raise FileExistsError(
            f"Joint-direction target already exists: {target}. Nothing was deleted."
        )
    if nonempty and policy == "skip":
        return False
    if target.exists() and policy == "overwrite":
        _assert_safe_overwrite_target(target)
        shutil.rmtree(target)

    target.mkdir(parents=True, exist_ok=True)
    return True


def resolve_routed_split_paths(
    config: Mapping[str, Any], route_class: str | int
) -> tuple[Dict[str, List[RoutedSplitPath]], Path]:
    """Resolve routed train/val flavor constituents and their class scaler.

    ``does_not_exist`` flavor/class combinations are skipped, matching the
    established reconstruction worker.  Train and validation must contain the
    same flavor constituents.  A test path is never resolved by this training
    helper.
    """

    route = normalize_route_class(route_class)
    mc = _configured_mc(config)
    geometry = _configured_geometry(config)
    category = str(config.get("routing", {}).get("category", ""))
    if not category:
        raise ValueError("routing.category is required")
    flavors = _configured_flavors(config)
    data = config.get("data", {})
    paths_py = data.get("paths_py", DEFAULT_PATHS_PY)
    module = load_paths_module(paths_py)
    table_name = str(
        data.get("parquet_table", PARQUET_TABLE_BY_MC.get(mc, ""))
    )
    if not table_name:
        raise ValueError(f"No parquet table is registered for mc={mc!r}")
    table = getattr(module, table_name, None)
    if not isinstance(table, dict):
        raise KeyError(f"{table_name} is missing from {paths_py}")
    geometry_entry = table.get(geometry)
    if not isinstance(geometry_entry, dict):
        raise KeyError(f"Missing path entry: {table_name}.{geometry}")

    split_paths: Dict[str, List[RoutedSplitPath]] = {
        split: [] for split in SUPPORTED_SPLITS
    }
    for split in SUPPORTED_SPLITS:
        for flavor in flavors:
            flavor_entry = geometry_entry.get(flavor)
            if not isinstance(flavor_entry, dict):
                raise KeyError(f"Missing path entry: {table_name}.{geometry}.{flavor}")
            route_entry = flavor_entry.get(category)
            if not isinstance(route_entry, dict):
                raise KeyError(
                    f"Missing routed path entry: {geometry}.{flavor}.{category}"
                )
            class_entry = route_entry.get(route)
            if class_entry is None:
                class_entry = route_entry.get(f"class{route}")
            if not isinstance(class_entry, dict):
                raise KeyError(
                    f"Missing routed class entry: {geometry}.{flavor}.{category}.class{route}"
                )
            raw = class_entry.get(split)
            if raw == "does_not_exist":
                continue
            if not raw:
                raise ValueError(
                    f"Missing {split} path: {geometry}.{flavor}.{category}.class{route}"
                )
            path = Path(raw)
            if not path.is_dir():
                raise FileNotFoundError(
                    f"{flavor} class{route} {split} directory does not exist: {path}"
                )
            split_paths[split].append(RoutedSplitPath(flavor=flavor, path=path))
        if not split_paths[split]:
            raise ValueError(
                f"No available flavors for {geometry}.{category}.class{route}.{split}"
            )

    train_flavors = [entry.flavor for entry in split_paths["train"]]
    val_flavors = [entry.flavor for entry in split_paths["val"]]
    if train_flavors != val_flavors:
        raise ValueError(
            "Routed train/validation flavor constituents differ: "
            f"train={train_flavors}, val={val_flavors}"
        )

    explicit_scaler = data.get("percentiles_csv")
    if explicit_scaler:
        scaler = Path(explicit_scaler)
    else:
        robust = getattr(module, "ROBUST_SCALER", None)
        if not isinstance(robust, dict):
            raise KeyError(f"ROBUST_SCALER is missing from {paths_py}")
        scaler_raw = (
            robust.get(mc, {})
            .get(geometry, {})
            .get("reconstruction", {})
            .get(category, {})
            .get(route)
        )
        if not scaler_raw:
            raise KeyError(
                "Missing class scaler: "
                f"ROBUST_SCALER.{mc}.{geometry}.reconstruction.{category}.{route}"
            )
        scaler = Path(scaler_raw)
    if not scaler.is_file():
        raise FileNotFoundError(f"Route-specific percentiles CSV is absent: {scaler}")
    return split_paths, scaler


def table_files(split_path: str | Path, table: str) -> List[Path]:
    """Return numerically ordered parquet chunks for one GraphNeT table."""

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
    """Extract numeric chunk suffixes used to pair truth and feature files."""

    result: set[str] = set()
    for path in files:
        match = re.search(r"_(\d+)\.parquet$", path.name)
        if match is None:
            raise ValueError(f"Unexpected parquet filename: {path.name}")
        result.add(match.group(1))
    return result


def read_truth_columns(split_path: str | Path, columns: Sequence[str]):
    """Read selected scalar truth columns only; pulse parquet is never loaded."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read routed truth parquet") from exc

    tables = [
        pq.read_table(path, columns=list(columns))
        for path in table_files(split_path, "truth")
    ]
    if not tables:
        raise ValueError(f"No truth rows found in {split_path}")
    return pa.concat_tables(tables).to_pandas()


def audit_parquet_entry_layout(
    entry: RoutedSplitPath,
    *,
    truth_columns: Sequence[str],
    feature_columns: Sequence[str],
    pulsemaps: str,
    truth_table: str,
    minimum_pulses_per_event: int = 2,
) -> Dict[str, Any]:
    """Audit schema, chunk pairing, and event/pulse coverage for one flavor."""

    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("numpy and pyarrow are required for parquet audit") from exc
    if minimum_pulses_per_event < 1:
        raise ValueError("minimum_pulses_per_event must be positive")

    truth_files = table_files(entry.path, truth_table)
    feature_files = table_files(entry.path, pulsemaps)
    truth_chunks = chunk_ids(truth_files)
    feature_chunks = chunk_ids(feature_files)
    if truth_chunks != feature_chunks:
        raise ValueError(
            f"{entry.flavor}: truth/features chunk mismatch at {entry.path}; "
            f"truth-only={sorted(truth_chunks - feature_chunks)[:10]}, "
            f"features-only={sorted(feature_chunks - truth_chunks)[:10]}"
        )

    for parquet_path in truth_files:
        missing = sorted(
            set(truth_columns) - set(pq.read_schema(parquet_path).names)
        )
        if missing:
            raise ValueError(
                f"{entry.flavor}: missing truth columns in {parquet_path}: {missing}"
            )
    for parquet_path in feature_files:
        missing = sorted(
            set(feature_columns) - set(pq.read_schema(parquet_path).names)
        )
        if missing:
            raise ValueError(
                f"{entry.flavor}: missing feature columns in {parquet_path}: {missing}"
            )

    truth_by_chunk = {
        next(iter(chunk_ids([path]))): path for path in truth_files
    }
    feature_by_chunk = {
        next(iter(chunk_ids([path]))): path for path in feature_files
    }
    minimum_pulses: int | None = None
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
        unique_event_no, pulse_counts = np.unique(pulse_event_no, return_counts=True)
        if len(unique_event_no) != len(truth_event_no) or not np.array_equal(
            np.sort(unique_event_no), np.sort(truth_event_no)
        ):
            missing = np.setdiff1d(truth_event_no, unique_event_no)
            extra = np.setdiff1d(unique_event_no, truth_event_no)
            raise ValueError(
                f"{entry.flavor} chunk {chunk}: truth/feature event coverage "
                f"mismatch; missing={missing[:10].tolist()}, "
                f"extra={extra[:10].tolist()}"
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
                f"{entry.flavor} chunk {chunk}: event with only {chunk_minimum} "
                "pulse(s) would be dropped by GraphNeT's collate function"
            )

    return {
        "flavor": entry.flavor,
        "path": str(entry.path),
        "chunks": len(truth_files),
        "truth_rows": int(
            sum(pq.ParquetFile(path).metadata.num_rows for path in truth_files)
        ),
        "pulse_rows": int(
            sum(pq.ParquetFile(path).metadata.num_rows for path in feature_files)
        ),
        "minimum_pulses_per_event": int(minimum_pulses or 0),
        "maximum_pulses_per_event": int(maximum_pulses),
        "truth_schema": sorted(pq.read_schema(truth_files[0]).names),
        "feature_schema": sorted(pq.read_schema(feature_files[0]).names),
    }


def checkpoint_state(path: str | Path) -> Dict[str, torch.Tensor]:
    """Load either a plain state dict or a Lightning checkpoint envelope."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {source}")
    if not all(isinstance(key, str) for key in payload):
        raise TypeError(f"Checkpoint state has non-string keys: {source}")
    return payload
