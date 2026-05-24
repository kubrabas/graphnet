import argparse
import importlib.util
import math
import os
import shutil

import pytorch_lightning as pl
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
from graphnet.training.callbacks import GraphnetEarlyStopping, PiecewiseLinearLR
from graphnet.training.loss_functions import BinaryCrossEntropyLoss

from utils import (
    EpochCSVLogger,
    EpochTimeLogger,
    _EpochContextCallback,
    extract_field,
    install_logging_filters,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATHS_PY    = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]

PARQUET_TABLE = {
    "340StringMC":  "STRING340MC_PARQUET",
    "Spring2026MC": "SPRING2026MC_PARQUET",
}

PARQUET_MIXED_TABLE = {
    "340StringMC":  "STRING340MC_PARQUET_MIXED",
    "Spring2026MC": "SPRING2026MC_PARQUET_MIXED",
}

EXTRA_KEYS = []


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_paths(cfg: dict):
    """
    Returns:
        per_flavor : dict  flavor -> {train, val, test}
        percentiles_csv : str  path to mixed percentiles CSV
    """
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
        for key in ("train", "val", "test"):
            if not entry.get(key):
                raise ValueError(f"{PARQUET_TABLE[mc]}['{geometry}']['{flavor}']['{key}'] is None — fill in paths.py first.")
        per_flavor[flavor] = entry
        print(f"[Paths] {flavor}: {entry['train']}")

    percentiles_csv = cfg.get("data", {}).get(
        "percentiles_csv",
        parquet_mixed.get(geometry, {}).get("percentiles_csv"),
    )
    if not percentiles_csv:
        raise ValueError(f"{PARQUET_MIXED_TABLE[mc]}['{geometry}']['percentiles_csv'] is None — run compute_mixed_percentiles.py first.")
    print(f"[Paths] percentiles_csv: {percentiles_csv}")

    return per_flavor, percentiles_csv


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_data_classification(cfg: dict, per_flavor: dict, percentiles_csv: str):
    features    = cfg["data"]["features"]
    truth_all   = cfg["data"]["truth_all"]
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

    def _make_loader(ds, shuffle, drop_last=False):
        return DataLoader(
            ds,
            batch_size=tcfg["batch_size"],
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=tcfg["num_workers"],
            multiprocessing_context=tcfg.get("multiprocessing_context", "spawn"),
            persistent_workers=True,
            pin_memory=tcfg.get("pin_memory", True),
        )

    train_ds = EnsembleDataset([_make_dataset(e["train"]) for e in per_flavor.values()])
    val_ds   = EnsembleDataset([_make_dataset(e["val"])   for e in per_flavor.values()])
    test_ds  = EnsembleDataset([_make_dataset(e["test"])  for e in per_flavor.values()])

    train_loader = _make_loader(train_ds, shuffle=True,  drop_last=True)
    val_loader   = _make_loader(val_ds,   shuffle=False)
    test_loader  = _make_loader(test_ds,  shuffle=False)

    print(f"[Data] flavors : {list(per_flavor.keys())}")
    print(f"[Data] train={len(train_loader)} batches | val={len(val_loader)} batches | test={len(test_loader)} batches")

    return data_representation, train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: dict, data_representation, steps_per_epoch_optimizer: int) -> StandardModel:
    features = cfg["data"]["features"]
    mcfg     = cfg["model"]
    tcfg     = cfg["training"]

    backbone = DynEdge(
        nb_inputs=len(features),
        nb_neighbours=mcfg["nb_neighbours"],
        global_pooling_schemes=mcfg["global_pooling_schemes"],
        add_global_variables_after_pooling=mcfg.get("add_global_variables_after_pooling", True),
        add_norm_layer=mcfg.get("add_norm_layer", False),
        skip_readout=mcfg.get("skip_readout", False),
    )

    total_steps     = steps_per_epoch_optimizer * tcfg["max_epochs"]
    warmup_steps    = max(1, int(tcfg.get("warmup_fraction", 0.5) * steps_per_epoch_optimizer))

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
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [0, warmup_steps, total_steps],
            "factors":    [1.0, tcfg["peak_lr"] / tcfg["base_lr"], 1.0],
        },
        scheduler_config={"interval": "step"},
    )


def load_checkpoint_if_available(model: StandardModel, cfg: dict) -> None:
    checkpoint_path = cfg["training"].get("pretrained_weights")
    if not checkpoint_path:
        print("[Classification] no pretrained_weights configured, starting from scratch")
        return

    if not os.path.exists(checkpoint_path):
        print(f"[Classification] checkpoint not found, starting from scratch: {checkpoint_path}")
        return

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    print(f"[Classification] loaded checkpoint: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_classification(cfg: dict, data_representation, train_loader, val_loader) -> None:
    install_logging_filters()

    out_dir = cfg["output"]["save_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tcfg = cfg["training"]
    pl.seed_everything(tcfg["seed"], workers=True)

    steps_per_epoch_optimizer = math.ceil(len(train_loader) / tcfg["accumulate_grad_batches"])
    model = build_model(cfg, data_representation, steps_per_epoch_optimizer)
    load_checkpoint_if_available(model, cfg)

    early_stop = GraphnetEarlyStopping(
        save_dir=out_dir,
        monitor="val_loss",
        mode="min",
        patience=tcfg["early_stopping_patience"],
        check_on_train_epoch_end=False,
        verbose=True,
    )

    metrics_cb   = EpochCSVLogger(out_dir, extra_keys=EXTRA_KEYS)
    time_cb      = EpochTimeLogger(out_dir)
    epoch_ctx_cb = _EpochContextCallback()

    print(f"\n[Classification] cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Classification] GPU: {torch.cuda.get_device_name(0)}")

    trainer = pl.Trainer(
        max_epochs=tcfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        callbacks=[epoch_ctx_cb, early_stop, metrics_cb, time_cb],
        enable_checkpointing=False,
        enable_progress_bar=False,
        accumulate_grad_batches=tcfg["accumulate_grad_batches"],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"[Classification] best_model: {os.path.join(out_dir, 'best_model.pth')}")
    print(f"[Classification] metrics:    {os.path.join(out_dir, 'metrics.csv')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)
    shutil.copy2(args.config, os.path.join(cfg["output"]["save_dir"], "config.yml"))

    print("\n========== CONFIG ==========")
    print(yaml.dump(cfg, default_flow_style=False))
    print("============================\n")

    per_flavor, percentiles_csv = resolve_paths(cfg)

    data_representation, train_loader, val_loader, test_loader = build_data_classification(
        cfg, per_flavor, percentiles_csv
    )

    b0     = next(iter(train_loader))
    labels = extract_field(b0, "is_track").detach().cpu().view(-1)
    labels_float = labels.float()
    print(f"[Sanity] is_track: min={labels.min().item()} max={labels.max().item()} mean={labels_float.mean().item():.3f}")

    run_classification(cfg, data_representation, train_loader, val_loader)
