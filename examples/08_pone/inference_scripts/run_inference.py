"""
Run the routed P-ONE inference pipeline from a config file.

The pipeline first applies a classification model to a mixed split, then routes
events to class-specific reconstruction models for each configured target.
"""

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
PONE_DIR = THIS_DIR.parent
TRAIN_DIR = PONE_DIR / "train_scripts"
for path in (PONE_DIR, TRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.models.data_representation import KNNGraph, NodesAsPulses
from graphnet.models.detector.pone import PONE

from pipeline_utils import PARQUET_TABLE, load_paths_module
from train_classification import build_model as build_classification_model
from train_reconstruction import build_model as build_reconstruction_model
from utils import _circular_signed_diff, _wrap_to_pi, extract_field, move_batch_to_device


ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]
EVENT_ID_FIELDS = ["event_no", "RunID", "SubrunID", "EventID", "SubEventID"]


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_state_dict(model, checkpoint_path: str, label: str) -> None:
    if not checkpoint_path:
        raise ValueError(f"{label} checkpoint path is empty")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    print(f"[Checkpoint] loaded {label}: {checkpoint_path}")


def resolve_output_dir(cfg: dict) -> Path:
    return (
        Path(cfg["output"]["root_dir"])
        / cfg["mc"]
        / cfg["geometry"]
        / cfg["task"]["type"]
        / cfg["routing"]["category"]
        / cfg["experiment_name"]
        / cfg["output"]["dirs"]["inference"]
    )


def handle_existing_output(out_dir: Path, policy: str) -> None:
    if not out_dir.exists():
        return
    if policy == "error":
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    if policy == "skip":
        print(f"[Run] output exists; skipping: {out_dir}")
        raise SystemExit(0)
    if policy == "overwrite":
        print(f"[Run] removing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)
        return
    raise ValueError(f"Unsupported run.existing_output policy: {policy}")


def resolve_mixed_split_paths(cfg: dict) -> Dict[str, str]:
    mc = cfg["mc"]
    geometry = cfg["geometry"]
    split = cfg["data"].get("split", "test")
    flavors = cfg.get("flavors", ALL_FLAVORS)

    mod = load_paths_module()
    parquet_table = getattr(mod, PARQUET_TABLE[mc])
    geometry_entry = parquet_table.get(geometry)
    if geometry_entry is None:
        raise ValueError(f"{PARQUET_TABLE[mc]}['{geometry}'] is missing in paths.py")

    paths = {}
    print(f"[Paths] mixed {split} inputs:")
    for flavor in flavors:
        split_path = geometry_entry.get(flavor, {}).get(split)
        if not split_path:
            raise ValueError(f"Missing {split} path in paths.py: {geometry}.{flavor}.{split}")
        if not Path(split_path).exists():
            raise FileNotFoundError(f"{flavor} {split} path does not exist: {split_path}")
        paths[flavor] = split_path
        print(f"  {flavor}: {split_path}")
    return paths


def resolve_percentiles(cfg: dict, route_class: str | None = None) -> str:
    explicit = cfg["data"].get("percentiles_csv")
    if explicit:
        return explicit

    mod = load_paths_module()
    robust_scaler = getattr(mod, "ROBUST_SCALER")
    if route_class is None:
        key = "mixed"
    else:
        key = f"{cfg['routing']['category']}_mixed_{route_class}"
    percentiles_csv = robust_scaler.get(cfg["mc"], {}).get(cfg["geometry"], {}).get(key)
    if not percentiles_csv:
        raise ValueError(f"ROBUST_SCALER['{cfg['mc']}']['{cfg['geometry']}']['{key}'] is missing")
    return percentiles_csv


def build_loader(cfg: dict, paths: Dict[str, str], percentiles_csv: str, model_cfg: dict):
    features = cfg["data"]["features"]
    truth_all = unique([*EVENT_ID_FIELDS, *cfg["data"]["truth_all"]])
    loader_cfg = cfg["inference"]

    data_representation = KNNGraph(
        detector=PONE(percentiles_csv=percentiles_csv, selected_features=features),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=model_cfg["model"]["nb_neighbours"],
        distance_as_edge_feature=False,
    )

    datasets = [
        ParquetDataset(
            path=path,
            pulsemaps=cfg["data"]["pulsemaps"],
            truth_table=cfg["data"]["truth_table"],
            features=features,
            truth=truth_all,
            data_representation=data_representation,
        )
        for path in paths.values()
    ]
    dataset = EnsembleDataset(datasets)
    loader = DataLoader(
        dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=loader_cfg["num_workers"],
        multiprocessing_context=loader_cfg.get("multiprocessing_context", "spawn"),
        persistent_workers=loader_cfg["num_workers"] > 0,
        pin_memory=loader_cfg.get("pin_memory", True),
    )
    return data_representation, loader


def field_numpy(batch, field: str, n: int, dtype=float) -> np.ndarray:
    try:
        values = extract_field(batch, field).detach().cpu().view(-1).numpy()
        return values.astype(dtype, copy=False) if dtype is not object else values
    except Exception:
        fill = np.nan if dtype is not object else None
        return np.array([fill] * n)


def event_keys_from_arrays(df: pd.DataFrame) -> pd.Series:
    return df[EVENT_ID_FIELDS].astype("Int64").astype(str).agg(":".join, axis=1)


def event_keys_from_batch(batch, n: int) -> List[str]:
    data = {
        field: field_numpy(batch, field, n)
        for field in EVENT_ID_FIELDS
    }
    return event_keys_from_arrays(pd.DataFrame(data)).tolist()


def truth_frame_from_batch(cfg: dict, batch, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            field: field_numpy(batch, field, n)
            for field in cfg["data"]["truth_all"]
        }
    )


def classification_probabilities(raw: torch.Tensor, cls_cfg: dict) -> torch.Tensor:
    mode = cls_cfg["task"]["mode"]
    if mode == "binary":
        positive = raw.squeeze(-1).detach().float().clamp(0.0, 1.0)
        return torch.stack([1.0 - positive, positive], dim=1)
    if mode == "multiclass":
        return torch.softmax(raw.detach().float(), dim=1)
    raise ValueError(f"Unsupported classification mode: {mode}")


def run_classification(cfg: dict, cls_cfg: dict, paths: Dict[str, str]) -> pd.DataFrame:
    percentiles_csv = resolve_percentiles(cfg, route_class=None)
    data_representation, loader = build_loader(cfg, paths, percentiles_csv, cls_cfg)
    model = build_classification_model(cls_cfg, data_representation, steps_per_epoch_optimizer=1)
    load_state_dict(model, cfg["classification"]["best_model"], "classification")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = list(cls_cfg["task"]["labels"])
    prefix = cls_cfg["task"].get("prediction_prefix", "p_class")
    target = cls_cfg["task"]["target"]
    threshold = cfg["classification"].get("threshold")
    positive_label = cfg["classification"].get("positive_label", labels[-1])

    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False

    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            raw = model(batch)[0]
            probs = classification_probabilities(raw, cls_cfg).cpu().numpy()
            n = probs.shape[0]
            pred_idx = np.argmax(probs, axis=1)
            pred_label = np.asarray([labels[i] for i in pred_idx], dtype=int)
            if cls_cfg["task"]["mode"] == "binary" and threshold is not None:
                pos_idx = labels.index(positive_label)
                pred_label = np.where(probs[:, pos_idx] >= float(threshold), positive_label, labels[0])

            truth_df = truth_frame_from_batch(cfg, batch, n)
            row = {
                "event_key": event_keys_from_batch(batch, n),
                **truth_df.to_dict(orient="list"),
                "true_classification_class": field_numpy(batch, target, n),
                "predicted_route_class": pred_label,
            }
            row[f"true_{target}"] = field_numpy(batch, target, n)
            for idx, label in enumerate(labels):
                row[f"{prefix}_{label}"] = probs[:, idx]
            rows.append(pd.DataFrame(row))

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"[Classification] rows={len(df)}")
    return df


def build_reconstruction_model_for_target(cfg: dict, reco_cfg: dict, target: str, route_class: str, data_representation):
    model = build_reconstruction_model(reco_cfg, data_representation, steps_per_epoch_optimizer=1, target=target)
    checkpoint = cfg["reconstruction"]["models"][str(route_class)][target]
    load_state_dict(model, checkpoint, f"reconstruction class{route_class} {target}")
    return model


def reco_prediction_frame(cfg: dict, reco_cfg: dict, target: str, route_class: str, model, loader) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False

    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            pred = model(batch)[0].detach().float().cpu()
            n = pred.shape[0]
            base = {
                "event_key": event_keys_from_batch(batch, n),
                "route_class": int(route_class),
                "target": target,
            }
            if target == "energy":
                truth_label = reco_cfg["target_settings"]["energy"].get("target_labels", ["totalEnergy"])[0]
                pred_log10 = pred.squeeze(-1).numpy()
                true_energy = field_numpy(batch, truth_label, n)
                base.update(
                    {
                        "true_energy": true_energy,
                        "pred_energy": np.power(10.0, pred_log10),
                        "pred_log10_energy": pred_log10,
                    }
                )
            elif target in ("zenith", "azimuth"):
                pred_angle = pred[:, 0]
                kappa = pred[:, 1]
                true_angle = torch.as_tensor(field_numpy(batch, target, n), dtype=torch.float32)
                residual = _circular_signed_diff(pred_angle, true_angle) if target == "azimuth" else pred_angle - true_angle
                base.update(
                    {
                        f"true_{target}_radian": true_angle.numpy(),
                        f"pred_{target}_radian": pred_angle.numpy(),
                        f"true_{target}_degree": (true_angle * 180.0 / np.pi).numpy(),
                        f"pred_{target}_degree": (pred_angle * 180.0 / np.pi).numpy(),
                        f"residual_{target}_degree": (residual * 180.0 / np.pi).numpy(),
                        f"{target}_kappa": kappa.numpy(),
                    }
                )
                if target == "azimuth":
                    base["pred_azimuth_degree_signed"] = (_wrap_to_pi(pred_angle) * 180.0 / np.pi).numpy()
            else:
                raise ValueError(f"Unsupported reconstruction target: {target}")
            rows.append(pd.DataFrame(base))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_reconstruction(cfg: dict, reco_cfg: dict, paths: Dict[str, str], routed: pd.DataFrame) -> pd.DataFrame:
    targets = [str(target) for target in cfg["reconstruction"]["targets"]]
    route_classes = [str(item) for item in cfg["routing"]["classes"]]
    routed_keys = {
        route_class: set(routed.loc[routed["predicted_route_class"].astype(str) == route_class, "event_key"])
        for route_class in route_classes
    }

    outputs = []
    for route_class in route_classes:
        keep_keys = routed_keys[route_class]
        print(f"[Reconstruction] class{route_class}: routed events={len(keep_keys)}")
        if not keep_keys:
            continue
        percentiles_csv = resolve_percentiles(cfg, route_class=route_class)
        data_representation, loader = build_loader(cfg, paths, percentiles_csv, reco_cfg)
        for target in targets:
            model = build_reconstruction_model_for_target(cfg, reco_cfg, target, route_class, data_representation)
            df = reco_prediction_frame(cfg, reco_cfg, target, route_class, model, loader)
            df = df[df["event_key"].isin(keep_keys)].copy()
            print(f"[Reconstruction] class{route_class} {target}: rows={len(df)}")
            outputs.append(df)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def make_wide_predictions(cls_df: pd.DataFrame, reco_df: pd.DataFrame) -> pd.DataFrame:
    wide = cls_df.copy()
    if reco_df.empty:
        return wide

    for target, target_df in reco_df.groupby("target", sort=False):
        drop_cols = {"route_class", "target"}
        pred_cols = [
            col for col in target_df.columns
            if col not in drop_cols and col != "event_key"
        ]
        pred_df = target_df[["event_key", *pred_cols]].copy()
        rename = {
            col: f"{target}_{col}"
            for col in pred_cols
            if col in wide.columns
        }
        pred_df = pred_df.rename(columns=rename)
        wide = wide.merge(pred_df, on="event_key", how="left")
    return wide


def validate_config(cfg: dict, cls_cfg: dict, reco_cfg: dict) -> None:
    if cfg["task"]["type"] != "inference":
        raise ValueError(f"Expected task.type=inference, got {cfg['task']['type']}")
    if cfg["mc"] != cls_cfg["mc"] or cfg["geometry"] != cls_cfg["geometry"]:
        raise ValueError("Inference and classification configs must use the same mc/geometry")
    if cfg["mc"] != reco_cfg["mc"] or cfg["geometry"] != reco_cfg["geometry"]:
        raise ValueError("Inference and reconstruction configs must use the same mc/geometry")
    if cfg["routing"]["category"] != reco_cfg["routing"]["category"]:
        raise ValueError("Inference routing.category must match reconstruction routing.category")
    labels = {str(label) for label in cls_cfg["task"]["labels"]}
    missing = [route_class for route_class in cfg["routing"]["classes"] if str(route_class) not in labels]
    if missing:
        raise ValueError(f"routing.classes not present in classification labels: {missing}")
    for route_class in cfg["routing"]["classes"]:
        models = cfg["reconstruction"]["models"].get(str(route_class), {})
        for target in cfg["reconstruction"]["targets"]:
            if target not in models:
                raise ValueError(f"Missing reconstruction.models.{route_class}.{target}")


def unique(items: Iterable) -> List:
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to inference YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cls_cfg = load_yaml(cfg["classification"]["config"])
    reco_cfg = load_yaml(cfg["reconstruction"]["config"])
    validate_config(cfg, cls_cfg, reco_cfg)

    out_dir = resolve_output_dir(cfg)
    if os.environ.get("OUTPUT_PREPARED", "0") != "1":
        handle_existing_output(out_dir, cfg.get("run", {}).get("existing_output", "error"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "pipeline_config.yml")

    print("\n========== CONFIG ==========")
    print(yaml.dump(cfg, default_flow_style=False))
    print("============================\n")
    print(f"[Output] inference dir: {out_dir}")

    paths = resolve_mixed_split_paths(cfg)
    cls_df = run_classification(cfg, cls_cfg, paths)
    reco_df = run_reconstruction(cfg, reco_cfg, paths, cls_df)
    wide = make_wide_predictions(cls_df, reco_df)
    wide_path = out_dir / "inference_predictions.csv"
    wide.to_csv(wide_path, index=False)
    print(f"[Output] wrote {wide_path}")


if __name__ == "__main__":
    main()
