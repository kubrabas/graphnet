"""Read-only Muon-CC parquet audit, weight fitting, and GraphNeT loaders."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.models.data_representation import KNNGraph, NodesAsPulses
from graphnet.models.detector.pone import PONE

from energy_weighting import EnergyWeightManifest, fit_energy_weight_manifest
from pipeline_utils import (
    SPLITS,
    atomic_json_dump,
    audit_parquet_layout,
    read_truth_columns,
    resolve_percentiles_csv,
    table_files,
)


def _trigger_column(config: Mapping[str, Any]) -> str:
    geometry = str(config["data"]["geometry"])
    configured = config["data"].get("trigger_column")
    if configured:
        return str(configured)
    mapping = {
        "102_string_emax1e6": "triggered_nonoise_102_string",
        "160_string_emax1e6": "triggered_nonoise_160_string",
        "full_geometry_emax1e6": "triggered_nonoise_340_string",
    }
    return mapping[geometry]


def required_truth_columns(config: Mapping[str, Any]) -> list[str]:
    """Columns required both for training and leakage/selection checks."""

    identifiers = list(config["data"]["physical_event_id_columns"])
    return list(
        dict.fromkeys(
            [
                "event_no",
                *identifiers,
                "zenith",
                "azimuth",
                "totalEnergy",
                "pid",
                "is_CC",
                "category1_isMuonCC",
                _trigger_column(config),
            ]
        )
    )


def deep_data_audit(
    config: Mapping[str, Any],
    split_paths: Mapping[str, Path],
    *,
    target_value_splits: Sequence[str] = ("train", "val"),
) -> Dict[str, Any]:
    """Prove selection/split invariants without opening blinded test targets.

    Schemas, pulse coverage, physical IDs, Muon-CC flags, and trigger flags are
    checked for every split. Zenith, azimuth, and energy values are read only
    for ``target_value_splits``. Training therefore defaults to train+val and
    leaves test target values unopened until a checkpoint has been frozen.
    """

    import numpy as np
    import pandas as pd

    requested_target_splits = tuple(dict.fromkeys(target_value_splits))
    unknown = sorted(set(requested_target_splits) - set(SPLITS))
    if unknown:
        raise ValueError(f"Unknown target_value_splits: {unknown}")
    truth_columns = required_truth_columns(config)
    feature_columns = ["event_no", *config["data"]["features"]]
    report = audit_parquet_layout(
        split_paths,
        truth_columns=truth_columns,
        feature_columns=feature_columns,
        pulsemaps=str(config["data"]["pulsemaps"]),
        truth_table=str(config["data"]["truth_table"]),
    )
    id_columns = list(config["data"]["physical_event_id_columns"])
    trigger = _trigger_column(config)
    integrity_columns = list(
        dict.fromkeys(
            [
                "event_no",
                *id_columns,
                "pid",
                "is_CC",
                "category1_isMuonCC",
                trigger,
            ]
        )
    )
    keys: Dict[str, set[tuple[int, ...]]] = {}
    histogram_edges = np.asarray(
        config["weighting"]["log10_energy_bin_edges"], dtype=float
    )

    for split in SPLITS:
        inspect_targets = split in requested_target_splits
        columns = truth_columns if inspect_targets else integrity_columns
        frame = read_truth_columns(split_paths[split], columns)
        if frame.empty:
            raise ValueError(f"{split}: truth dataset is empty")
        if frame[id_columns].isna().any().any():
            raise ValueError(f"{split}: null physical event identifier")
        if frame.duplicated(id_columns).any():
            raise ValueError(f"{split}: duplicate physical event identifier")
        if frame["event_no"].duplicated().any():
            raise ValueError(f"{split}: duplicate event_no")

        pid = pd.to_numeric(frame["pid"], errors="raise").to_numpy()
        is_cc = pd.to_numeric(frame["is_CC"], errors="raise").to_numpy()
        muon_cc = pd.to_numeric(
            frame["category1_isMuonCC"], errors="raise"
        ).to_numpy()
        triggered = pd.to_numeric(frame[trigger], errors="raise").to_numpy()
        if not np.all(np.abs(pid) == 14):
            raise ValueError(f"{split}: non-muon pid found in canonical Muon path")
        if not np.all(is_cc == 1):
            raise ValueError(f"{split}: non-CC event found in canonical Muon path")
        if not np.all(muon_cc == 1):
            raise ValueError(f"{split}: category1_isMuonCC != 1 found")
        if not np.all(triggered == 1):
            raise ValueError(f"{split}: {trigger} != 1 found")

        keys[split] = set(
            frame[id_columns].astype("int64").itertuples(index=False, name=None)
        )
        split_report = {
            "events": int(len(frame)),
            "trigger_column": trigger,
            "all_triggered_nonoise": True,
            "all_muon_cc": True,
            "target_values_inspected": inspect_targets,
        }
        if inspect_targets:
            energy = pd.to_numeric(frame["totalEnergy"], errors="raise").to_numpy(
                dtype=float
            )
            zenith = pd.to_numeric(frame["zenith"], errors="raise").to_numpy(
                dtype=float
            )
            azimuth = pd.to_numeric(frame["azimuth"], errors="raise").to_numpy(
                dtype=float
            )
            if not np.isfinite(energy).all() or not np.all(energy > 0.0):
                raise ValueError(f"{split}: invalid totalEnergy")
            if not np.isfinite(zenith).all() or np.any(
                (zenith < 0) | (zenith > math.pi)
            ):
                raise ValueError(f"{split}: zenith outside [0, pi]")
            if not np.isfinite(azimuth).all() or np.any(
                (azimuth < 0) | (azimuth > 2.0 * math.pi)
            ):
                raise ValueError(f"{split}: azimuth outside [0, 2*pi]")

            log_energy = np.log10(energy)
            if np.any(log_energy < histogram_edges[0]) or np.any(
                log_energy > histogram_edges[-1]
            ):
                raise ValueError(f"{split}: energy outside configured weight range")
            counts, _ = np.histogram(log_energy, bins=histogram_edges)
            if np.any(counts == 0):
                raise ValueError(f"{split}: empty configured 0.1-dex energy bin")
            split_report.update(
                {
                    "log10_energy_min": float(log_energy.min()),
                    "log10_energy_median": float(np.median(log_energy)),
                    "log10_energy_max": float(log_energy.max()),
                    "weight_histogram_counts": [int(value) for value in counts],
                }
            )
        report["splits"][split].update(split_report)

    overlaps = {
        "train_val": len(keys["train"] & keys["val"]),
        "train_test": len(keys["train"] & keys["test"]),
        "val_test": len(keys["val"] & keys["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Physical-event leakage across splits: {overlaps}")
    report.update(
        {
            "physical_event_id_columns": id_columns,
            "split_overlap_counts": overlaps,
            "selection": "abs(pid)==14 and is_CC==1 and category1_isMuonCC==1",
            "target_value_splits_inspected": list(requested_target_splits),
            "test_target_values_blinded": "test" not in requested_target_splits,
            "parquet_files_modified": False,
        }
    )
    return report


def fit_or_load_energy_manifest(
    config: Mapping[str, Any],
    train_path: Path,
    manifest_path: Path,
) -> EnergyWeightManifest:
    """Fit weights from train truth once, or validate an existing manifest."""

    weighting = config["weighting"]
    energies_frame = read_truth_columns(train_path, ["totalEnergy"])
    energies = torch.as_tensor(
        energies_frame["totalEnergy"].to_numpy(), dtype=torch.float64
    )
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
        source_files=[str(path) for path in table_files(train_path, "truth")],
    )
    if manifest_path.exists():
        existing = EnergyWeightManifest.load(manifest_path)
        if existing != expected:
            raise ValueError(
                f"Existing weight manifest does not match config/train data: {manifest_path}"
            )
        return existing
    expected.save(manifest_path)
    return expected


def build_data_representation(config: Mapping[str, Any]):
    """Build the same P-ONE graph definition used by the current pipeline."""

    features = list(config["data"]["features"])
    percentiles_csv = resolve_percentiles_csv(config)
    return KNNGraph(
        detector=PONE(
            percentiles_csv=str(percentiles_csv), selected_features=features
        ),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=int(config["model"]["nb_neighbours"]),
        distance_as_edge_feature=False,
    )


def build_loaders(
    config: Mapping[str, Any],
    split_paths: Mapping[str, Path],
    data_representation,
    *,
    splits: Sequence[str] = SPLITS,
) -> Dict[str, DataLoader]:
    """Build direct Muon loaders with no router and no parquet loss weights."""

    data = config["data"]
    loader = config["loader"]
    truth = [value for value in required_truth_columns(config) if value != "event_no"]

    def dataset(split: str) -> ParquetDataset:
        return ParquetDataset(
            path=str(split_paths[split]),
            pulsemaps=str(data["pulsemaps"]),
            truth_table=str(data["truth_table"]),
            features=list(data["features"]),
            truth=truth,
            data_representation=data_representation,
            cache_size=int(data.get("parquet_cache_size", 1)),
            loss_weight_table=None,
            loss_weight_column=None,
        )

    def make(split: str) -> DataLoader:
        is_train = split == "train"
        workers = int(loader["num_workers"])
        kwargs: Dict[str, Any] = {
            "batch_size": int(loader["batch_size"]),
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
        return DataLoader(dataset(split), **kwargs)

    requested_splits = tuple(dict.fromkeys(splits))
    unknown = sorted(set(requested_splits) - set(SPLITS))
    if unknown or not requested_splits:
        raise ValueError(f"Invalid loader splits: {requested_splits}")
    return {split: make(split) for split in requested_splits}


def write_data_audit(report: Mapping[str, Any], destination: str | Path) -> None:
    atomic_json_dump(report, destination)
