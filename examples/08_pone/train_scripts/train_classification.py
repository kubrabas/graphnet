"""
Train a P-ONE classification model from a config file.

Usage:
    python3 examples/08_pone/train_scripts/train_classification.py \
        -c examples/08_pone/configs/classification/102_string_emax1e6__category1_isMuonCC.yml

SLURM:
    This script is meant to be called from the external classification SLURM
    script. Python stdout/stderr should be routed by that SLURM script to the
    config-derived train_and_val directory.
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
PONE_DIR = THIS_DIR.parent
if str(PONE_DIR) not in sys.path:
    sys.path.insert(0, str(PONE_DIR))

from graphnet.models.gnn import DynEdge
from graphnet.models.standard_model import StandardModel
from graphnet.models.task.classification import BinaryClassificationTask, MulticlassClassificationTask
from graphnet.training.callbacks import GraphnetEarlyStopping, PiecewiseLinearLR
from graphnet.training.loss_functions import BinaryCrossEntropyLoss, CrossEntropyLoss

from pipeline_utils import (
    build_classification_loaders,
    collect_validation_predictions,
    resolve_classification_paths,
    resolve_stage_dir,
    write_validation_diagnostics,
)
from utils import (
    EpochCSVLogger,
    EpochTimeLogger,
    _EpochContextCallback,
    extract_field,
    install_logging_filters,
)


def build_model(cfg: dict, data_representation, steps_per_epoch_optimizer: int) -> StandardModel:
    features = cfg["data"]["features"]
    task_cfg = cfg["task"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    backbone = DynEdge(
        nb_inputs=len(features),
        nb_neighbours=model_cfg["nb_neighbours"],
        global_pooling_schemes=model_cfg["global_pooling_schemes"],
        add_global_variables_after_pooling=model_cfg.get("add_global_variables_after_pooling", True),
        add_norm_layer=model_cfg.get("add_norm_layer", False),
        skip_readout=model_cfg.get("skip_readout", False),
    )

    loss_weight = None
    weights_cfg = cfg.get("weights", {})
    if weights_cfg.get("enabled", False):
        loss_weight = weights_cfg["loss_weight_column"]

    labels = list(task_cfg["labels"])
    prediction_prefix = task_cfg.get("prediction_prefix", "p_class")
    prediction_labels = [f"{prediction_prefix}_{label}" for label in labels]

    if task_cfg["mode"] == "binary":
        task = BinaryClassificationTask(
            hidden_size=backbone.nb_outputs,
            loss_function=BinaryCrossEntropyLoss(),
            target_labels=[task_cfg["target"]],
            prediction_labels=[prediction_labels[-1]],
            loss_weight=loss_weight,
        )
    elif task_cfg["mode"] == "multiclass":
        task = MulticlassClassificationTask(
            hidden_size=backbone.nb_outputs,
            nb_outputs=len(labels),
            loss_function=CrossEntropyLoss(options={int(label): i for i, label in enumerate(labels)}),
            target_labels=[task_cfg["target"]],
            prediction_labels=prediction_labels,
            loss_weight=loss_weight,
        )
    else:
        raise ValueError(f"Unsupported classification mode: {task_cfg['mode']}")

    total_steps = steps_per_epoch_optimizer * train_cfg["max_epochs"]
    warmup_steps = max(1, int(train_cfg.get("warmup_fraction", 0.5) * steps_per_epoch_optimizer))

    return StandardModel(
        tasks=task,
        data_representation=data_representation,
        backbone=backbone,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": train_cfg["base_lr"]},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [0, warmup_steps, total_steps],
            "factors": [1.0, train_cfg["peak_lr"] / train_cfg["base_lr"], 1.0],
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


def run_training(cfg: dict, data_representation, train_loader, val_loader, out_dir: Path) -> StandardModel:
    install_logging_filters()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["training"]
    pl.seed_everything(train_cfg["seed"], workers=True)

    steps_per_epoch_optimizer = math.ceil(len(train_loader) / train_cfg["accumulate_grad_batches"])
    model = build_model(cfg, data_representation, steps_per_epoch_optimizer)
    load_checkpoint_if_available(model, cfg)

    callbacks = [
        _EpochContextCallback(),
        GraphnetEarlyStopping(
            save_dir=str(out_dir),
            monitor="val_loss",
            mode="min",
            patience=train_cfg["early_stopping_patience"],
            check_on_train_epoch_end=False,
            verbose=True,
        ),
        EpochCSVLogger(out_dir, extra_keys=[], filename="training_history_by_epoch.csv"),
        EpochTimeLogger(out_dir),
    ]

    print(f"\n[Classification] cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Classification] GPU: {torch.cuda.get_device_name(0)}")

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"[Classification] best_model: {out_dir / 'best_model.pth'}")
    print(f"[Classification] history:    {out_dir / 'training_history_by_epoch.csv'}")
    return model


def load_best_weights(model: StandardModel, out_dir: Path) -> None:
    best_path = out_dir / "best_model.pth"
    if not best_path.exists():
        print(f"[Validation] best_model.pth not found, using final epoch weights: {best_path}")
        return
    state = torch.load(best_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    print(f"[Validation] loaded best weights: {best_path}")


def print_sanity(cfg: dict, train_loader) -> None:
    target = cfg["task"]["target"]
    labels = extract_field(next(iter(train_loader)), target).detach().cpu().view(-1)
    labels_float = labels.float()
    print(
        f"[Sanity] {target}: min={labels.min().item()} "
        f"max={labels.max().item()} mean={labels_float.mean().item():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["task"]["type"] != "classification":
        raise ValueError(f"Expected task.type=classification, got {cfg['task']['type']}")
    validate_config(cfg)

    out_dir = resolve_stage_dir(cfg, "train")
    if os.environ.get("OUTPUT_PREPARED", "0") != "1":
        handle_existing_output(out_dir, cfg.get("run", {}).get("existing_output", "error"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "pipeline_config.yml")

    print("\n========== CONFIG ==========")
    print(yaml.dump(cfg, default_flow_style=False))
    print("============================\n")
    print(f"[Output] train_and_val: {out_dir}")

    per_flavor, percentiles_csv = resolve_classification_paths(cfg)
    data_representation, train_loader, val_loader = build_classification_loaders(
        cfg, per_flavor, percentiles_csv
    )
    print_sanity(cfg, train_loader)

    model = run_training(cfg, data_representation, train_loader, val_loader, out_dir)
    load_best_weights(model, out_dir)

    y_true, probabilities, labels = collect_validation_predictions(cfg, model, val_loader)
    write_validation_diagnostics(cfg, y_true, probabilities, labels, out_dir)
    print(f"[Validation] diagnostics written to: {out_dir}")


def validate_config(cfg: dict) -> None:
    task_cfg = cfg["task"]
    labels = task_cfg["labels"]
    class_names = task_cfg.get("class_names", {})

    if task_cfg["target"] not in cfg["data"]["truth_all"]:
        raise ValueError("data.truth_all must include task.target")
    if task_cfg["mode"] not in ("binary", "multiclass"):
        raise ValueError(f"Unsupported classification mode: {task_cfg['mode']}")
    if task_cfg["mode"] == "binary" and len(labels) != 2:
        raise ValueError("binary classification requires exactly two labels")
    if len(set(labels)) != len(labels):
        raise ValueError(f"task.labels contains duplicates: {labels}")
    missing_names = [
        label for label in labels
        if label not in class_names and str(label) not in class_names
    ]
    if missing_names:
        raise ValueError(f"task.class_names is missing labels: {missing_names}")


if __name__ == "__main__":
    main()
