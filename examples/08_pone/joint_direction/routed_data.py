"""Read-only mixed-flavor data plumbing for routed joint direction training.

Each route class is represented by the union of its available flavor-specific
categorized parquet views.  Energy weights are fitted once from that combined
training truth.  Validation remains unweighted, and no test loader is exposed.
"""

from __future__ import annotations

from functools import partial
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

import torch

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.models.data_representation import KNNGraph, NodesAsPulses
from graphnet.models.detector.pone import PONE

try:
    from .experiment_config import node_feature_augmentations
except ImportError:  # Direct execution with joint_direction on PYTHONPATH.
    from experiment_config import node_feature_augmentations

try:
    from .pmt_direction_features import (
        PONE_V3_PMT_DIRECTION_CONTRACT,
        PONEV3PMTDirectionNodes,
    )
except ImportError:  # Direct execution with joint_direction on PYTHONPATH.
    from pmt_direction_features import (
        PONE_V3_PMT_DIRECTION_CONTRACT,
        PONEV3PMTDirectionNodes,
    )

try:
    from .routed_pipeline_utils import (
        DEFAULT_FLAVORS,
        RoutedSplitPath,
        SUPPORTED_SPLITS,
        atomic_json_dump,
        audit_parquet_entry_layout,
        normalize_route_class,
        read_truth_columns,
        table_files,
    )
except ImportError:  # Direct execution with joint_direction on PYTHONPATH.
    from routed_pipeline_utils import (
        DEFAULT_FLAVORS,
        RoutedSplitPath,
        SUPPORTED_SPLITS,
        atomic_json_dump,
        audit_parquet_entry_layout,
        normalize_route_class,
        read_truth_columns,
        table_files,
    )


# Reuse the tested, checkpoint-compatible energy-weight manifest implementation
# from the existing joint direction pipeline without copying or modifying it.
EXAMPLES_DIR = Path(__file__).resolve().parents[2]
JOINT_REFERENCE_DIR = EXAMPLES_DIR / "09_pone_muon_direction"
if str(JOINT_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(JOINT_REFERENCE_DIR))
from energy_weighting import (  # noqa: E402
    EnergyWeightManifest,
    fit_energy_weight_manifest,
)


PHYSICAL_EVENT_ID_COLUMNS = ("RunID", "SubrunID", "EventID", "SubEventID")
ROUTING_LABEL_COLUMNS = (
    "category1_isMuonCC",
    "category2_tauCC_others_muonCC",
    "category_3_contains_muon",
)
TRIGGER_BY_GEOMETRY = {
    "102_string_emax1e6": "triggered_nonoise_102_string",
    "160_string_emax1e6": "triggered_nonoise_160_string",
    "full_geometry_emax1e6": "triggered_nonoise_340_string",
}
FLAVOR_PID = {"Muon": 14, "Electron": 12, "Tau": 16, "NC": 12}
SOURCE_FLAVOR_ID = {
    flavor: index for index, flavor in enumerate(DEFAULT_FLAVORS)
}


def loader_feature_names(config: Mapping[str, Any]) -> list[str]:
    """Resolve the parquet columns required to construct every graph node."""

    base = [str(value) for value in config.get("data", {}).get("features", [])]
    if not base or len(base) != len(set(base)):
        raise ValueError("data.features must be non-empty and unique")
    augmentations = node_feature_augmentations(config)
    if not augmentations:
        return base
    contract = PONE_V3_PMT_DIRECTION_CONTRACT
    if tuple(base) != tuple(contract.scaled_features):
        raise ValueError(
            "pmt_direction_v3 requires the established scaled base features "
            f"{list(contract.scaled_features)}, got {base}"
        )
    return list(contract.loader_features)


def _entry_flavor_path(entry: RoutedSplitPath | Mapping[str, Any]) -> tuple[str, Path]:
    """Normalize a routed entry while accepting JSON-like mappings in tests."""

    if isinstance(entry, RoutedSplitPath):
        return entry.flavor, entry.path
    if isinstance(entry, Mapping) and "flavor" in entry and "path" in entry:
        return str(entry["flavor"]), Path(entry["path"])
    raise TypeError(f"Unsupported routed split entry: {entry!r}")


def _geometry(config: Mapping[str, Any]) -> str:
    value = config.get("geometry", config.get("data", {}).get("geometry"))
    if not value:
        raise ValueError("A top-level geometry is required")
    return str(value)


def _trigger_column(config: Mapping[str, Any]) -> str:
    geometry = _geometry(config)
    try:
        expected = TRIGGER_BY_GEOMETRY[geometry]
    except KeyError as exc:
        raise ValueError(
            f"No nonoise trigger column is registered for geometry={geometry!r}"
        ) from exc
    configured = config.get("data", {}).get("trigger_column")
    if configured is not None and str(configured) != expected:
        raise ValueError(
            "data.trigger_column does not match the configured geometry: "
            f"geometry={geometry!r} requires {expected!r}, got {configured!r}"
        )
    return expected


def _id_columns(config: Mapping[str, Any]) -> list[str]:
    configured = config.get("data", {}).get(
        "physical_event_id_columns", PHYSICAL_EVENT_ID_COLUMNS
    )
    columns = [str(value) for value in configured]
    if columns != list(PHYSICAL_EVENT_ID_COLUMNS):
        raise ValueError(
            "data.physical_event_id_columns must be exactly "
            f"{list(PHYSICAL_EVENT_ID_COLUMNS)}"
        )
    return columns


def required_truth_columns(config: Mapping[str, Any]) -> list[str]:
    """Truth fields required for model targets, audits, and provenance."""

    category = str(config.get("routing", {}).get("category", ""))
    if not category:
        raise ValueError("routing.category is required")
    return list(
        dict.fromkeys(
            [
                "event_no",
                *_id_columns(config),
                "zenith",
                "azimuth",
                "totalEnergy",
                "pid",
                "is_CC",
                *ROUTING_LABEL_COLUMNS,
                _trigger_column(config),
            ]
        )
    )


def _numeric_integer_frame(frame, columns: Sequence[str], label: str):
    """Return validated int64 identifiers without silently truncating floats."""

    import numpy as np
    import pandas as pd

    result = frame[list(columns)].copy()
    for column in columns:
        numeric = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{label}: {column} is not a finite integer identifier")
        result[column] = numeric.astype("int64")
    return result


def _validate_flavor_semantics(frame, flavor: str, label: str) -> None:
    """Confirm each registered flavor path contains its advertised interaction."""

    import numpy as np
    import pandas as pd

    if flavor not in FLAVOR_PID:
        raise ValueError(f"Unsupported source flavor: {flavor!r}")
    pid = pd.to_numeric(frame["pid"], errors="raise").to_numpy(dtype=float)
    is_cc = pd.to_numeric(frame["is_CC"], errors="raise").to_numpy(dtype=float)
    if not np.all(np.abs(pid) == FLAVOR_PID[flavor]):
        raise ValueError(f"{label}: pid does not match source flavor {flavor}")
    expected_cc = 0 if flavor == "NC" else 1
    if not np.all(is_cc == expected_cc):
        raise ValueError(
            f"{label}: is_CC does not match source flavor {flavor} "
            f"(expected {expected_cc})"
        )


def deep_data_audit(
    config: Mapping[str, Any],
    split_paths: Mapping[str, Sequence[RoutedSplitPath]],
    route_class: str | int,
) -> Dict[str, Any]:
    """Prove routed selection, geometry trigger, and train/val isolation.

    Physical identifiers are deliberately namespaced by ``source_flavor``.
    This is essential because the four simulation identifier columns have real
    cross-flavor collisions.  The audit reads only train and validation data;
    accepting or constructing a test loader is outside this module's contract.
    """

    import numpy as np
    import pandas as pd

    if set(split_paths) != set(SUPPORTED_SPLITS):
        raise ValueError(
            f"Expected exactly routed splits {SUPPORTED_SPLITS}, got {sorted(split_paths)}"
        )
    route = normalize_route_class(route_class)
    route_value = int(route)
    category = str(config.get("routing", {}).get("category", ""))
    trigger = _trigger_column(config)
    id_columns = _id_columns(config)
    truth_columns = required_truth_columns(config)
    data = config.get("data", {})
    feature_columns = ["event_no", *loader_feature_names(config)]
    histogram_edges = np.asarray(
        config["weighting"]["log10_energy_bin_edges"], dtype=float
    )
    if (
        histogram_edges.ndim != 1
        or histogram_edges.size < 2
        or not np.isfinite(histogram_edges).all()
        or np.any(histogram_edges[1:] <= histogram_edges[:-1])
    ):
        raise ValueError("weighting.log10_energy_bin_edges must increase finitely")

    configured_flavors = [
        str(value) for value in config.get("flavors", DEFAULT_FLAVORS)
    ]
    if len(configured_flavors) != len(set(configured_flavors)):
        raise ValueError("flavors contains duplicates")
    unknown_flavors = sorted(set(configured_flavors) - set(SOURCE_FLAVOR_ID))
    if unknown_flavors:
        raise ValueError(f"Unsupported configured flavors: {unknown_flavors}")
    # Keep this stable even when a config lists a subset or changes list order;
    # validation/reporting use the same canonical 0=Muon,1=Electron,2=Tau,3=NC.
    flavor_id_map = {
        flavor: SOURCE_FLAVOR_ID[flavor] for flavor in configured_flavors
    }

    report: Dict[str, Any] = {"splits": {}}
    split_keys: Dict[str, set[tuple[Any, ...]]] = {}
    for split in SUPPORTED_SPLITS:
        entries = list(split_paths[split])
        if not entries:
            raise ValueError(f"{split}: no routed flavor constituents")
        seen_flavors: set[str] = set()
        keys: set[tuple[Any, ...]] = set()
        flavor_reports: Dict[str, Any] = {}
        split_energies: list[np.ndarray] = []
        total_events = 0
        total_pulses = 0

        for raw_entry in entries:
            flavor, path = _entry_flavor_path(raw_entry)
            if flavor in seen_flavors:
                raise ValueError(f"{split}: duplicate constituent flavor {flavor}")
            if flavor not in flavor_id_map:
                raise ValueError(f"{split}: unconfigured constituent flavor {flavor}")
            seen_flavors.add(flavor)
            entry = RoutedSplitPath(flavor=flavor, path=path)
            layout = audit_parquet_entry_layout(
                entry,
                truth_columns=truth_columns,
                feature_columns=feature_columns,
                pulsemaps=str(data.get("pulsemaps", "features")),
                truth_table=str(data.get("truth_table", "truth")),
                minimum_pulses_per_event=int(data.get("minimum_pulses_per_event", 2)),
            )
            frame = read_truth_columns(path, truth_columns)
            if frame.empty:
                raise ValueError(f"{split}/{flavor}: truth dataset is empty")
            identifiers = _numeric_integer_frame(
                frame, ["event_no", *id_columns], f"{split}/{flavor}"
            )
            if identifiers["event_no"].duplicated().any():
                raise ValueError(f"{split}/{flavor}: duplicate event_no")
            if identifiers[id_columns].duplicated().any():
                raise ValueError(
                    f"{split}/{flavor}: duplicate physical event identifier"
                )

            route_labels = pd.to_numeric(frame[category], errors="raise").to_numpy(
                dtype=float
            )
            triggered = pd.to_numeric(frame[trigger], errors="raise").to_numpy(
                dtype=float
            )
            if not np.all(route_labels == route_value):
                found = sorted(set(route_labels.tolist()))
                raise ValueError(
                    f"{split}/{flavor}: {category} != {route}; found {found}"
                )
            if not np.all(triggered == 1):
                raise ValueError(f"{split}/{flavor}: {trigger} != 1 found")
            _validate_flavor_semantics(frame, flavor, f"{split}/{flavor}")

            energy = pd.to_numeric(frame["totalEnergy"], errors="raise").to_numpy(
                dtype=float
            )
            zenith = pd.to_numeric(frame["zenith"], errors="raise").to_numpy(
                dtype=float
            )
            azimuth = pd.to_numeric(frame["azimuth"], errors="raise").to_numpy(
                dtype=float
            )
            if not np.isfinite(energy).all() or np.any(energy <= 0.0):
                raise ValueError(f"{split}/{flavor}: invalid totalEnergy")
            if not np.isfinite(zenith).all() or np.any(
                (zenith < 0.0) | (zenith > math.pi)
            ):
                raise ValueError(f"{split}/{flavor}: zenith outside [0, pi]")
            if not np.isfinite(azimuth).all() or np.any(
                (azimuth < 0.0) | (azimuth > 2.0 * math.pi)
            ):
                raise ValueError(f"{split}/{flavor}: azimuth outside [0, 2*pi]")
            log_energy = np.log10(energy)
            if np.any(log_energy < histogram_edges[0]) or np.any(
                log_energy > histogram_edges[-1]
            ):
                raise ValueError(
                    f"{split}/{flavor}: energy outside configured weight range"
                )

            flavor_keys = {
                (flavor, *values)
                for values in identifiers[id_columns].itertuples(index=False, name=None)
            }
            if keys & flavor_keys:
                raise ValueError(f"{split}: duplicate flavor-namespaced event key")
            keys.update(flavor_keys)
            split_energies.append(log_energy)
            total_events += int(len(frame))
            total_pulses += int(layout["pulse_rows"])
            layout.update(
                {
                    "source_flavor_id": flavor_id_map[flavor],
                    "events": int(len(frame)),
                    "all_route_labels_correct": True,
                    "all_triggered_nonoise": True,
                    "log10_energy_min": float(log_energy.min()),
                    "log10_energy_median": float(np.median(log_energy)),
                    "log10_energy_max": float(log_energy.max()),
                }
            )
            flavor_reports[flavor] = layout

        combined_log_energy = np.concatenate(split_energies)
        counts, _ = np.histogram(combined_log_energy, bins=histogram_edges)
        if np.any(counts == 0):
            empty = np.flatnonzero(counts == 0).tolist()
            raise ValueError(
                f"{split}: combined routed sample has empty weight bins {empty}"
            )
        if len(keys) != total_events:
            raise AssertionError(
                f"{split}: namespaced key count {len(keys)} != events {total_events}"
            )
        split_keys[split] = keys
        report["splits"][split] = {
            "events": total_events,
            "pulse_rows": total_pulses,
            "constituent_flavors": [
                _entry_flavor_path(entry)[0] for entry in entries
            ],
            "events_by_flavor": {
                flavor: int(flavor_report["events"])
                for flavor, flavor_report in flavor_reports.items()
            },
            "flavors": flavor_reports,
            "all_route_labels_correct": True,
            "all_triggered_nonoise": True,
            "trigger_column": trigger,
            "log10_energy_min": float(combined_log_energy.min()),
            "log10_energy_median": float(np.median(combined_log_energy)),
            "log10_energy_max": float(combined_log_energy.max()),
            "weight_histogram_counts": [int(value) for value in counts],
        }

    overlap = len(split_keys["train"] & split_keys["val"])
    if overlap:
        raise ValueError(
            "Flavor-namespaced physical-event leakage across train/val: "
            f"{overlap}"
        )
    report.update(
        {
            "mc": str(config.get("mc")),
            "geometry": _geometry(config),
            "routing_category": category,
            "routing_class": int(route),
            "selection": f"{category} == {route} and {trigger} == 1",
            "source_flavor_id_map": flavor_id_map,
            "physical_event_id_columns": id_columns,
            "physical_event_key_namespace": ["source_flavor", *id_columns],
            "split_overlap_counts": {"train_val": 0},
            "splits_audited": list(SUPPORTED_SPLITS),
            "test_loader_created": False,
            "parquet_files_modified": False,
        }
    )
    return report


def fit_or_load_energy_manifest(
    config: Mapping[str, Any],
    train_entries: Sequence[RoutedSplitPath],
    manifest_path: str | Path,
) -> EnergyWeightManifest:
    """Fit or verify one train-only manifest over all route constituents."""

    weighting = config["weighting"]
    energy_parts: list[torch.Tensor] = []
    source_files: list[str] = []
    seen_flavors: set[str] = set()
    for raw_entry in train_entries:
        flavor, path = _entry_flavor_path(raw_entry)
        if flavor in seen_flavors:
            raise ValueError(f"Duplicate train flavor in weight fit: {flavor}")
        seen_flavors.add(flavor)
        frame = read_truth_columns(path, ["totalEnergy"])
        energy_parts.append(
            torch.as_tensor(
                frame["totalEnergy"].to_numpy(), dtype=torch.float64
            ).reshape(-1)
        )
        source_files.extend(str(item) for item in table_files(path, "truth"))
    if not energy_parts:
        raise ValueError("Cannot fit weights from an empty routed training sample")
    energies = torch.cat(energy_parts, dim=0)
    expected = fit_energy_weight_manifest(
        energies,
        weighting["log10_energy_bin_edges"],
        alpha=float(weighting["alpha"]),
        clip_min=(
            None if weighting.get("clip_min") is None else float(weighting["clip_min"])
        ),
        clip_max=(
            None if weighting.get("clip_max") is None else float(weighting["clip_max"])
        ),
        out_of_range=str(weighting.get("out_of_range", "error")),
        source_files=source_files,
    )
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        existing = EnergyWeightManifest.load(manifest_path)
        if existing != expected:
            raise ValueError(
                "Existing energy-weight manifest differs from the current "
                f"config/combined train data: {manifest_path}"
            )
        return existing
    expected.save(manifest_path)
    return expected


def build_data_representation(
    config: Mapping[str, Any], percentiles_csv: str | Path | None = None
):
    """Build the established P-ONE KNN graph using the routed class scaler."""

    data = config["data"]
    features = [str(value) for value in data["features"]]
    augmentations = node_feature_augmentations(config)
    configured = percentiles_csv or data.get("percentiles_csv")
    if not configured:
        route_class = config.get("routing", {}).get("class")
        if route_class is not None:
            try:
                from .routed_pipeline_utils import resolve_routed_split_paths
            except ImportError:
                from routed_pipeline_utils import resolve_routed_split_paths
            _, configured = resolve_routed_split_paths(config, route_class)
    if not configured:
        raise ValueError(
            "A route-specific percentiles_csv returned by "
            "resolve_routed_split_paths must be supplied, or routing.class must "
            "be present in the resolved worker config"
        )
    scaler = Path(configured)
    if not scaler.is_file():
        raise FileNotFoundError(f"Percentiles CSV does not exist: {scaler}")
    if not augmentations:
        # Keep the established construction byte-for-byte equivalent for old
        # configs and checkpoints.
        return KNNGraph(
            detector=PONE(percentiles_csv=str(scaler), selected_features=features),
            node_definition=NodesAsPulses(),
            nb_nearest_neighbours=int(config["model"]["nb_neighbours"]),
            distance_as_edge_feature=False,
        )

    contract = PONE_V3_PMT_DIRECTION_CONTRACT
    graph = KNNGraph(
        detector=PONE(
            percentiles_csv=str(scaler),
            selected_features=list(contract.scaled_features),
            replace_with_identity=list(contract.identity_features),
        ),
        node_definition=PONEV3PMTDirectionNodes(),
        input_feature_names=list(contract.loader_features),
        nb_nearest_neighbours=int(config["model"]["nb_neighbours"]),
        distance_as_edge_feature=False,
    )
    if tuple(graph.output_feature_names) != tuple(contract.output_features):
        raise RuntimeError(
            "pmt_direction_v3 output feature contract drifted: "
            f"{graph.output_feature_names}"
        )
    return graph


def _constant_source_flavor_id(_graph, *, value: int) -> torch.Tensor:
    """Picklable top-level custom label callable for spawned DataLoader workers."""

    return torch.tensor([int(value)], dtype=torch.long)


def _source_flavor_id_map(config: Mapping[str, Any]) -> Dict[str, int]:
    flavors = [str(value) for value in config.get("flavors", DEFAULT_FLAVORS)]
    if not flavors or len(flavors) != len(set(flavors)):
        raise ValueError("flavors must be non-empty and unique")
    unknown = sorted(set(flavors) - set(SOURCE_FLAVOR_ID))
    if unknown:
        raise ValueError(f"Unsupported flavors: {unknown}")
    return {flavor: SOURCE_FLAVOR_ID[flavor] for flavor in flavors}


def build_loaders(
    config: Mapping[str, Any],
    split_paths: Mapping[str, Sequence[RoutedSplitPath]],
    data_representation,
    *,
    splits: Sequence[str] = SUPPORTED_SPLITS,
) -> Dict[str, DataLoader]:
    """Construct mixed routed train/val loaders; constructing test is forbidden.

    Every per-flavor ``ParquetDataset`` receives a constant
    ``source_flavor_id`` custom graph label through a picklable ``partial``.
    This preserves source parquet files and keeps event provenance available in
    batched validation predictions.
    """

    requested = tuple(dict.fromkeys(str(value) for value in splits))
    if not requested or any(split not in SUPPORTED_SPLITS for split in requested):
        raise ValueError(
            f"Only train/val loaders are permitted; requested splits={requested}"
        )
    if not set(requested).issubset(split_paths):
        raise ValueError(f"Missing requested routed paths: {requested}")

    data = config["data"]
    loader = config["loader"]
    truth = [
        column for column in required_truth_columns(config) if column != "event_no"
    ]
    flavor_ids = _source_flavor_id_map(config)

    def dataset_for_entry(raw_entry) -> ParquetDataset:
        flavor, path = _entry_flavor_path(raw_entry)
        if flavor not in flavor_ids:
            raise ValueError(f"No source_flavor_id configured for {flavor}")
        dataset = ParquetDataset(
            path=str(path),
            pulsemaps=str(data.get("pulsemaps", "features")),
            truth_table=str(data.get("truth_table", "truth")),
            features=loader_feature_names(config),
            truth=truth,
            data_representation=data_representation,
            cache_size=int(data.get("parquet_cache_size", 1)),
            loss_weight_table=None,
            loss_weight_column=None,
        )
        dataset.add_label(
            partial(_constant_source_flavor_id, value=flavor_ids[flavor]),
            key="source_flavor_id",
        )
        return dataset

    def make(split: str) -> DataLoader:
        entries = list(split_paths[split])
        if not entries:
            raise ValueError(f"{split}: no flavor datasets")
        datasets = [dataset_for_entry(entry) for entry in entries]
        ensemble = EnsembleDataset(datasets)
        is_train = split == "train"
        workers = int(loader["num_workers"])
        if workers < 0:
            raise ValueError("loader.num_workers cannot be negative")
        kwargs: Dict[str, Any] = {
            "batch_size": int(
                loader["batch_size"]
                if is_train
                else loader.get("val_batch_size", loader["batch_size"])
            ),
            "shuffle": is_train,
            "drop_last": is_train,
            "num_workers": workers,
            "persistent_workers": workers > 0,
            "pin_memory": bool(loader.get("pin_memory", True)),
        }
        if workers > 0:
            kwargs["multiprocessing_context"] = str(
                loader.get("multiprocessing_context", "spawn")
            )
        else:
            # GraphNeT forwards this argument to torch DataLoader; None is the
            # valid value when no worker processes are used.
            kwargs["prefetch_factor"] = None
        return DataLoader(ensemble, **kwargs)

    return {split: make(split) for split in requested}


def write_data_audit(report: Mapping[str, Any], destination: str | Path) -> None:
    """Write the routed audit atomically alongside, never inside, source data."""

    atomic_json_dump(report, destination)
