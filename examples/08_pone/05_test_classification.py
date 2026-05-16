"""
Evaluate a trained classification model on the test set.

Loads best_model.pth from the training output directory, runs inference
on the test split, and writes a CSV to the global results directory:
  Results/classification/<mc>/<geometry>/<experiment_name>_test_predictions.csv

Metrics printed: accuracy, track/cascade counts.

Usage:
    python3 05_test_classification.py -c configs/classification/exp001.yml
"""

import argparse
import importlib.util
import os

import pandas as pd
import torch
import yaml

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.models.data_representation import KNNGraph, NodesAsPulses
from graphnet.models.detector.pone import PONE
from graphnet.models.gnn import DynEdge
from graphnet.models.standard_model import StandardModel
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss

from utils import (
    extract_field,
    install_logging_filters,
    move_batch_to_device,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATHS_PY     = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
RESULTS_BASE = "/project/def-nahee/kbas/Graphnet-Applications/Results/classification"
ALL_FLAVORS  = ["Muon", "Electron", "Tau", "NC"]
ID_FIELDS    = ["RunID", "SubrunID", "EventID", "SubEventID"]

PARQUET_TABLE = {
    "340StringMC":  "STRING340MC_PARQUET",
    "Spring2026MC": "SPRING2026MC_PARQUET",
}

PARQUET_MIXED_TABLE = {
    "340StringMC":  "STRING340MC_PARQUET_MIXED",
    "Spring2026MC": "SPRING2026MC_PARQUET_MIXED",
}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_paths(cfg: dict):
    mc       = cfg["mc"]
    geometry = cfg["geometry"]
    flavors  = cfg.get("flavors", ALL_FLAVORS)

    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    parquet_table = getattr(mod, PARQUET_TABLE[mc])
    parquet_mixed = getattr(mod, PARQUET_MIXED_TABLE[mc])

    per_flavor = {}
    for flavor in flavors:
        entry = parquet_table.get(geometry, {}).get(flavor, {})
        if not entry.get("test"):
            raise ValueError(f"{PARQUET_TABLE[mc]}['{geometry}']['{flavor}']['test'] is None.")
        per_flavor[flavor] = entry

    percentiles_csv = parquet_mixed.get(geometry, {}).get("percentiles_csv")
    if not percentiles_csv:
        raise ValueError(f"{PARQUET_MIXED_TABLE[mc]}['{geometry}']['percentiles_csv'] is None.")

    return per_flavor, percentiles_csv


# ---------------------------------------------------------------------------
# Data (test only)
# ---------------------------------------------------------------------------

def build_test_loader(cfg: dict, per_flavor: dict, percentiles_csv: str):
    features    = cfg["data"]["features"]
    truth_all   = list(cfg["data"]["truth_all"]) + [f for f in ID_FIELDS if f not in cfg["data"]["truth_all"]]
    pulsemaps   = cfg["data"]["pulsemaps"]
    truth_table = cfg["data"]["truth_table"]
    tcfg        = cfg["training"]

    data_representation = KNNGraph(
        detector=PONE(percentiles_csv=percentiles_csv, selected_features=features),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=cfg["model"]["nb_neighbours"],
        distance_as_edge_feature=False,
    )

    def _make_dataset(path):
        return ParquetDataset(
            path=path,
            pulsemaps=pulsemaps,
            truth_table=truth_table,
            features=features,
            truth=truth_all,
            data_representation=data_representation,
        )

    test_ds = EnsembleDataset([_make_dataset(e["test"]) for e in per_flavor.values()])

    test_loader = DataLoader(
        test_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        multiprocessing_context=tcfg.get("multiprocessing_context", "spawn"),
        persistent_workers=True,
        pin_memory=tcfg.get("pin_memory", True),
    )

    print(f"[Data] test={len(test_loader)} batches")
    return data_representation, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: dict, data_representation) -> StandardModel:
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    backbone = DynEdge(
        nb_inputs=len(cfg["data"]["features"]),
        nb_neighbours=mcfg["nb_neighbours"],
        global_pooling_schemes=mcfg["global_pooling_schemes"],
        add_global_variables_after_pooling=mcfg.get("add_global_variables_after_pooling", True),
        add_norm_layer=mcfg.get("add_norm_layer", False),
        skip_readout=mcfg.get("skip_readout", False),
    )

    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        loss_function=BinaryCrossEntropyLoss(),
        target_labels=["is_track"],
        prediction_labels=["track_score"],
    )

    return StandardModel(
        tasks=task,
        data_representation=data_representation,
        backbone=backbone,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": tcfg["base_lr"]},
    )


def load_model(cfg: dict, data_representation) -> StandardModel:
    model    = build_model(cfg, data_representation)
    best_pth = os.path.join(cfg["output"]["save_dir"], "classification", "best_model.pth")

    if not os.path.exists(best_pth):
        raise FileNotFoundError(f"best_model.pth not found: {best_pth}")

    state = torch.load(best_pth, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    print(f"[Model] Loaded: {best_pth}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_test(cfg: dict, model: StandardModel, test_loader, out_csv: str) -> None:
    install_logging_filters()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            batch       = move_batch_to_device(batch, device)
            track_score = model(batch)[0].detach().float().squeeze(-1).cpu()
            true_label  = extract_field(batch, "is_track").detach().float().view(-1).cpu()

            id_vals = {}
            for f in ID_FIELDS:
                try:
                    id_vals[f] = extract_field(batch, f).detach().cpu().view(-1)
                except Exception:
                    id_vals[f] = None

            for i in range(len(true_label)):
                row = {
                    "RunID":         int(id_vals["RunID"][i].item())      if id_vals["RunID"]      is not None else None,
                    "SubrunID":      int(id_vals["SubrunID"][i].item())   if id_vals["SubrunID"]   is not None else None,
                    "EventID":       int(id_vals["EventID"][i].item())    if id_vals["EventID"]    is not None else None,
                    "SubEventID":    int(id_vals["SubEventID"][i].item()) if id_vals["SubEventID"] is not None else None,
                    "true_is_track": int(true_label[i].item()),
                    "track_score":   float(track_score[i].item()),
                    "pred_is_track": int(track_score[i].item() >= 0.5),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[Test] Wrote {out_csv} | rows={len(df)}")

    acc       = (df["true_is_track"] == df["pred_is_track"]).mean()
    n_track   = int(df["true_is_track"].sum())
    n_cascade = len(df) - n_track
    print(f"[Test] accuracy={acc:.4f} | track={n_track} cascade={n_cascade}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("\n========== CONFIG ==========")
    print(yaml.dump(cfg, default_flow_style=False))
    print("============================\n")

    per_flavor, percentiles_csv = resolve_paths(cfg)
    data_representation, test_loader = build_test_loader(cfg, per_flavor, percentiles_csv)
    model = load_model(cfg, data_representation)

    out_csv = os.path.join(
        RESULTS_BASE,
        cfg["mc"],
        cfg["geometry"],
        f"{cfg['experiment_name']}_test_predictions.csv",
    )

    run_test(cfg, model, test_loader, out_csv)
