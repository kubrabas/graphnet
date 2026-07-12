"""
Train one routed P-ONE reconstruction model from a config file.

This script is intentionally single-task: one SLURM job trains one routing
class and one reconstruction target. In normal use, call the submit wrapper;
it expands the config into class x target jobs and calls this worker script.

Normal usage:
    python3 /home/kbas/SlurmScripts/GraphNet/submit_reconstruction_pipeline.py \
        -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/reconstruction/102_string_emax1e6__category1_isMuonCC.yml

Worker usage, normally called by the SLURM wrapper:
    python3 train_reconstruction.py \
        --config examples/08_pone/configs/reconstruction/102_string_emax1e6__category1_isMuonCC.yml \
        --route-class 0 \
        --target energy
"""

import argparse
import copy
import importlib.util
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
PONE_DIR = THIS_DIR.parent
if str(PONE_DIR) not in sys.path:
    sys.path.insert(0, str(PONE_DIR))

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.models.data_representation import KNNGraph, NodesAsPulses
from graphnet.models.detector.pone import PONE
from graphnet.models.gnn import DynEdge
from graphnet.models.standard_model import StandardModel
from graphnet.models.task.reconstruction import (
    AzimuthReconstructionWithKappa,
    ZenithReconstructionWithKappa,
)
from graphnet.training.callbacks import GraphnetEarlyStopping, PiecewiseLinearLR
from graphnet.training.loss_functions import LogCoshLoss, VonMisesFisher2DLoss
from graphnet.utilities.maths import eps_like

from utils import (
    DepositedEnergyLog10Task,
    EpochCSVLogger,
    EpochTimeLogger,
    ValidationResidualAndLRMetrics,
    _EpochContextCallback,
    _circular_signed_diff,
    _wrap_to_pi,
    extract_field,
    install_logging_filters,
    move_batch_to_device,
)

PATHS_PY = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]
PARQUET_TABLE = {
    "340StringMC": "STRING340MC_PARQUET",
    "Spring2026MC": "SPRING2026MC_PARQUET",
}
EXTRA_KEYS = {
    "energy": [
        "val_residual_log10_p16", "val_residual_log10_p50", "val_residual_log10_p84",
        "val_W_log10", "val_bias_log10", "val_mae_log10", "val_rmse_log10",
    ],
    "zenith": [
        "val_residual_p16_deg", "val_residual_p50_deg", "val_residual_p84_deg",
        "val_W_deg", "val_kappa_p16", "val_kappa_p50", "val_kappa_p84", "val_kappa_W",
    ],
    "azimuth": [
        "val_residual_p16_deg", "val_residual_p50_deg", "val_residual_p84_deg",
        "val_W_deg", "val_kappa_p16", "val_kappa_p50", "val_kappa_p84", "val_kappa_W",
    ],
}


def load_paths_module():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def effective_cfg_for_target(cfg: dict, target: str) -> dict:
    eff = copy.deepcopy(cfg)
    overrides = cfg.get("target_overrides", {})
    if not overrides.get("enabled", False):
        return eff
    target_overrides = overrides.get(target, {}) or {}
    for section in ("training", "model"):
        if section in target_overrides:
            eff[section] = deep_update(eff.get(section, {}), target_overrides[section])
    if "target_settings" in target_overrides:
        eff["target_settings"][target] = deep_update(
            eff.get("target_settings", {}).get(target, {}),
            target_overrides["target_settings"],
        )
    return eff


def normalize_class_id(route_class) -> str:
    return str(route_class).replace("class", "")


def resolve_experiment_dir(cfg: dict, route_class: str) -> Path:
    return (
        Path(cfg["output"]["root_dir"])
        / cfg["mc"]
        / cfg["geometry"]
        / cfg["task"]["type"]
        / cfg["routing"]["category"]
        / f"class{route_class}"
        / cfg["experiment_name"]
    )


def resolve_target_dir(cfg: dict, route_class: str, target: str) -> Path:
    return resolve_experiment_dir(cfg, route_class) / cfg["output"]["dirs"]["train"] / target


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


def count_parquets(split_path: str) -> Tuple[int, int]:
    p = Path(split_path)
    return (
        len(list((p / "features").glob("*.parquet"))),
        len(list((p / "truth").glob("*.parquet"))),
    )


def resolve_reconstruction_paths(cfg: dict, route_class: str):
    mc = cfg["mc"]
    geometry = cfg["geometry"]
    routing_category = cfg["routing"]["category"]
    flavors = cfg.get("flavors", ALL_FLAVORS)

    mod = load_paths_module()
    parquet_table = getattr(mod, PARQUET_TABLE[mc])
    geometry_entry = parquet_table.get(geometry)
    if geometry_entry is None:
        raise ValueError(f"{PARQUET_TABLE[mc]}['{geometry}'] is missing in paths.py")

    split_paths: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": []}
    split_counts = {"train": {"features": 0, "truth": 0}, "val": {"features": 0, "truth": 0}}

    print(f"[Paths] geometry={geometry} routing={routing_category} class{route_class}")
    for split in ("train", "val"):
        print(f"[Paths] {split} inputs:")
        for flavor in flavors:
            flavor_entry = geometry_entry.get(flavor)
            if flavor_entry is None:
                raise ValueError(f"Missing flavor in paths.py: {geometry}.{flavor}")
            route_entry = flavor_entry.get(routing_category)
            if route_entry is None:
                raise ValueError(f"Missing routing category in paths.py: {geometry}.{flavor}.{routing_category}")
            class_entry = route_entry.get(route_class)
            if class_entry is None:
                raise ValueError(f"Missing route class in paths.py: {geometry}.{flavor}.{routing_category}.{route_class}")
            split_path = class_entry.get(split)
            if split_path is None:
                raise ValueError(f"None path in paths.py: {geometry}.{flavor}.{routing_category}.{route_class}.{split}")
            if split_path == "does_not_exist":
                print(f"  {flavor}: does_not_exist")
                continue
            if not Path(split_path).exists():
                raise FileNotFoundError(f"{flavor} {split} path does not exist: {split_path}")
            features_n, truth_n = count_parquets(split_path)
            split_counts[split]["features"] += features_n
            split_counts[split]["truth"] += truth_n
            split_paths[split].append((flavor, split_path))
            print(f"  {flavor}: {split_path} | features={features_n} parquet | truth={truth_n} parquet")
        print(
            f"[Paths] {split} total: features={split_counts[split]['features']} parquet | "
            f"truth={split_counts[split]['truth']} parquet"
        )

    percentiles_csv = cfg["data"].get("percentiles_csv")
    if not percentiles_csv:
        robust_scaler = getattr(mod, "ROBUST_SCALER")
        percentiles_csv = (
            robust_scaler.get(mc, {})
            .get(geometry, {})
            .get("reconstruction", {})
            .get(routing_category, {})
            .get(str(route_class))
        )
        if not percentiles_csv:
            raise ValueError(
                f"ROBUST_SCALER['{mc}']['{geometry}']['reconstruction']"
                f"['{routing_category}']['{route_class}'] is missing in paths.py"
            )
    print(f"[Paths] percentiles_csv: {percentiles_csv}")
    return split_paths, percentiles_csv


def build_loaders(cfg: dict, split_paths: dict, percentiles_csv: str):
    features = cfg["data"]["features"]
    truth_all = [
        field
        for field in dict.fromkeys(cfg["data"]["truth_all"])
        if field != "event_no"
    ]
    pulsemaps = cfg["data"]["pulsemaps"]
    truth_table = cfg["data"]["truth_table"]
    train_cfg = cfg["training"]
    weights_cfg = cfg.get("weights", {})
    weights_enabled = bool(weights_cfg.get("enabled", False))

    data_representation = KNNGraph(
        detector=PONE(percentiles_csv=percentiles_csv, selected_features=features),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=cfg["model"]["nb_neighbours"],
        distance_as_edge_feature=False,
    )

    def make_dataset(path: str):
        kwargs = {}
        if weights_enabled:
            kwargs = {
                "loss_weight_table": weights_cfg["loss_weight_table"],
                "loss_weight_column": weights_cfg["loss_weight_column"],
                "loss_weight_default_value": weights_cfg.get("loss_weight_default_value", 1.0),
            }
        return ParquetDataset(
            path=path,
            pulsemaps=pulsemaps,
            truth_table=truth_table,
            features=features,
            truth=truth_all,
            data_representation=data_representation,
            **kwargs,
        )

    def make_loader(ds, shuffle: bool, drop_last: bool = False):
        return DataLoader(
            ds,
            batch_size=train_cfg["batch_size"],
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=train_cfg["num_workers"],
            multiprocessing_context=train_cfg.get("multiprocessing_context", "spawn"),
            persistent_workers=True,
            pin_memory=train_cfg.get("pin_memory", True),
        )

    train_ds = EnsembleDataset([make_dataset(path) for _, path in split_paths["train"]])
    val_ds = EnsembleDataset([make_dataset(path) for _, path in split_paths["val"]])
    train_loader = make_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ds, shuffle=False)
    print(f"[Data] train={len(train_loader)} batches | val={len(val_loader)} batches")
    return data_representation, train_loader, val_loader


def loss_from_name(name: str):
    if name == "log_cosh":
        return LogCoshLoss()
    if name == "von_mises_fisher_2d":
        return VonMisesFisher2DLoss()
    raise ValueError(f"Unsupported loss: {name}")


def build_task(cfg: dict, backbone: DynEdge, target: str):
    settings = cfg["target_settings"][target]
    weights_cfg = cfg.get("weights", {})
    loss_weight = weights_cfg["loss_weight_column"] if weights_cfg.get("enabled", False) else None
    loss = loss_from_name(settings["loss"])
    target_labels = settings.get("target_labels", [target])
    prediction_labels = settings.get("prediction_labels")

    if target == "energy":
        if settings.get("transform_target") != "log10" or settings.get("transform_inference") != "pow10":
            raise ValueError("energy currently supports transform_target=log10 and transform_inference=pow10")
        return DepositedEnergyLog10Task(
            hidden_size=backbone.nb_outputs,
            loss_function=loss,
            target_labels=target_labels,
            prediction_labels=prediction_labels or ["log10_energy_pred"],
            transform_target=lambda E: torch.log10(torch.clamp(E, min=eps_like(E))),
            transform_inference=lambda t: torch.pow(10.0, t),
            transform_support=tuple(float(x) for x in cfg["model"].get("transform_support", [1e1, 1e8])),
            loss_weight=loss_weight,
        )
    if target == "zenith":
        return ZenithReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            loss_function=loss,
            target_labels=target_labels,
            prediction_labels=prediction_labels,
            loss_weight=loss_weight,
        )
    if target == "azimuth":
        return AzimuthReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            loss_function=loss,
            target_labels=target_labels,
            prediction_labels=prediction_labels,
            loss_weight=loss_weight,
        )
    raise ValueError(f"Unsupported reconstruction target: {target}")


def build_model(cfg: dict, data_representation, steps_per_epoch_optimizer: int, target: str) -> StandardModel:
    features = cfg["data"]["features"]
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
    task = build_task(cfg, backbone, target)

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


def load_checkpoint_if_available(model: StandardModel, cfg: dict, target: str) -> None:
    checkpoint_path = cfg["training"].get("pretrained_weights")
    if not checkpoint_path:
        print(f"[Reconstruction:{target}] no pretrained_weights configured, starting from scratch")
        return
    if not os.path.exists(checkpoint_path):
        print(f"[Reconstruction:{target}] checkpoint not found, starting from scratch: {checkpoint_path}")
        return
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    print(f"[Reconstruction:{target}] loaded checkpoint: {checkpoint_path}")


def run_training(cfg: dict, target: str, data_representation, train_loader, val_loader, out_dir: Path) -> StandardModel:
    install_logging_filters()
    train_cfg = cfg["training"]
    pl.seed_everything(train_cfg["seed"], workers=True)
    steps_per_epoch_optimizer = math.ceil(len(train_loader) / train_cfg["accumulate_grad_batches"])
    model = build_model(cfg, data_representation, steps_per_epoch_optimizer, target)
    load_checkpoint_if_available(model, cfg, target)

    target_label = cfg.get("target_settings", {}).get(target, {}).get("target_labels", [target])[0]

    callbacks = [
        _EpochContextCallback(),
        ValidationResidualAndLRMetrics(
            target=target,
            val_loader=val_loader,
            max_batches=train_cfg.get("val_metrics_max_batches"),
            target_label=target_label,
        ),
        GraphnetEarlyStopping(
            save_dir=str(out_dir),
            monitor="val_loss",
            mode="min",
            patience=train_cfg["early_stopping_patience"],
            check_on_train_epoch_end=False,
            verbose=True,
        ),
        EpochCSVLogger(out_dir, extra_keys=EXTRA_KEYS[target], filename="training_history_by_epoch.csv"),
        EpochTimeLogger(out_dir),
    ]

    print(f"\n[Reconstruction:{target}] cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Reconstruction:{target}] GPU: {torch.cuda.get_device_name(0)}")

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
    print(f"[Reconstruction:{target}] best_model: {out_dir / 'best_model.pth'}")
    print(f"[Reconstruction:{target}] history:    {out_dir / 'training_history_by_epoch.csv'}")
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


def field_or_nan(batch, field: str, n: int, dtype=float):
    try:
        value = extract_field(batch, field).detach().cpu().view(-1)
        return value.numpy()
    except Exception:
        fill = np.nan if dtype is float else None
        return np.array([fill] * n)


def collect_validation_predictions(cfg: dict, model, val_loader, target: str, route_class: str) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    rows = []
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)
            pred0 = model(batch)[0].detach().float()
            n = pred0.shape[0]
            base = {
                "routing_category": [cfg["routing"]["category"]] * n,
                "routing_class": [int(route_class)] * n,
                "event_no": field_or_nan(batch, "event_no", n),
                "RunID": field_or_nan(batch, "RunID", n),
                "EventID": field_or_nan(batch, "EventID", n),
                "pid": field_or_nan(batch, "pid", n),
                "is_CC": field_or_nan(batch, "is_CC", n),
                "category1_isMuonCC": field_or_nan(batch, "category1_isMuonCC", n),
                "category2_tauCC_others_muonCC": field_or_nan(batch, "category2_tauCC_others_muonCC", n),
                "category_3_contains_muon": field_or_nan(batch, "category_3_contains_muon", n),
            }

            if target == "energy":
                energy_label = cfg["target_settings"]["energy"].get("target_labels", ["totalEnergy"])[0]
                pred_log10 = pred0.squeeze(-1).detach().cpu()
                true_E = extract_field(batch, energy_label).detach().float().view(-1).cpu()
                true_log10 = torch.log10(torch.clamp(true_E, min=eps_like(true_E)))
                pred_E = torch.pow(10.0, pred_log10)
                data = {
                    **base,
                    "true_energy": true_E.numpy(),
                    "pred_energy": pred_E.numpy(),
                    "true_log10_energy": true_log10.numpy(),
                    "pred_log10_energy": pred_log10.numpy(),
                    "residual_log10": (pred_log10 - true_log10).numpy(),
                    "residual": (pred_E - true_E).numpy(),
                }
            elif target in ("zenith", "azimuth"):
                pred_angle = pred0[:, 0].detach().cpu()
                kappa = pred0[:, 1].detach().cpu()
                truth = extract_field(batch, target).detach().float().view(-1).cpu()
                residual_rad = _circular_signed_diff(pred_angle, truth) if target == "azimuth" else pred_angle - truth
                residual_deg = residual_rad * (180.0 / math.pi)
                true_deg = truth * (180.0 / math.pi)
                pred_deg = pred_angle * (180.0 / math.pi)
                data = {
                    **base,
                    f"true_{target}_radian": truth.numpy(),
                    f"pred_{target}_radian": pred_angle.numpy(),
                    f"true_{target}_degree": true_deg.numpy(),
                    f"pred_{target}_degree": pred_deg.numpy(),
                    "kappa": kappa.numpy(),
                    f"residual_{target}_radian": residual_rad.numpy(),
                    f"residual_{target}_degree": residual_deg.numpy(),
                }
                if target == "azimuth":
                    data["true_azimuth_degree_signed"] = (_wrap_to_pi(truth) * (180.0 / math.pi)).numpy()
                    data["pred_azimuth_degree_signed"] = (_wrap_to_pi(pred_angle) * (180.0 / math.pi)).numpy()
            else:
                raise ValueError(f"Unsupported target: {target}")
            rows.append(pd.DataFrame(data))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def quantile_summary(values: np.ndarray) -> Tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    p16, p50, p84 = np.quantile(values, [0.16, 0.50, 0.84])
    return float(p16), float(p50), float(p84), float((p84 - p16) / 2.0)


def write_summary(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    if target == "energy":
        residuals = df["residual_log10"].to_numpy(dtype=float)
        p16, p50, p84, width = quantile_summary(residuals)
        row = {
            "target": target,
            "n_events": len(df),
            "residual_log10_p16": p16,
            "residual_log10_p50": p50,
            "residual_log10_p84": p84,
            "W_log10": width,
            "bias_log10": float(np.nanmean(residuals)) if len(residuals) else float("nan"),
            "mae_log10": float(np.nanmean(np.abs(residuals))) if len(residuals) else float("nan"),
            "rmse_log10": float(np.sqrt(np.nanmean(residuals ** 2))) if len(residuals) else float("nan"),
        }
    else:
        residuals = df[f"residual_{target}_degree"].to_numpy(dtype=float)
        kappas = df["kappa"].to_numpy(dtype=float)
        p16, p50, p84, width = quantile_summary(residuals)
        kp16, kp50, kp84, kwidth = quantile_summary(kappas)
        row = {
            "target": target,
            "n_events": len(df),
            "residual_deg_p16": p16,
            "residual_deg_p50": p50,
            "residual_deg_p84": p84,
            "W_deg": width,
            "kappa_p16": kp16,
            "kappa_p50": kp50,
            "kappa_p84": kp84,
            "kappa_W": kwidth,
        }
    pd.DataFrame([row]).to_csv(out_dir / "validation_metrics_summary.csv", index=False)


def write_plots(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if target == "energy":
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df["residual_log10"].to_numpy(dtype=float), bins=80, histtype="stepfilled", alpha=0.75)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("pred_log10_energy - true_log10_energy")
        ax.set_ylabel("Events")
        ax.set_title("Validation Log10 Energy Residual")
        fig.tight_layout()
        fig.savefig(out_dir / "validation_residual_log10_distribution.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df["true_log10_energy"], df["pred_log10_energy"], s=4, alpha=0.35)
        vals = pd.concat([df["true_log10_energy"], df["pred_log10_energy"]]).to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel("true log10 energy")
        ax.set_ylabel("pred log10 energy")
        ax.set_title("Validation True vs Pred Log10 Energy")
        fig.tight_layout()
        fig.savefig(out_dir / "validation_true_vs_pred_log10_energy.png", dpi=160)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df[f"residual_{target}_degree"].to_numpy(dtype=float), bins=80, histtype="stepfilled", alpha=0.75)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel(f"pred {target} - true {target} [deg]")
        ax.set_ylabel("Events")
        ax.set_title(f"Validation {target.capitalize()} Residual")
        fig.tight_layout()
        fig.savefig(out_dir / "validation_residual_degree_distribution.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(df["kappa"].to_numpy(dtype=float), bins=80, histtype="stepfilled", alpha=0.75)
        ax.set_xlabel("kappa")
        ax.set_ylabel("Events")
        ax.set_title(f"Validation {target.capitalize()} Kappa")
        fig.tight_layout()
        fig.savefig(out_dir / "validation_kappa_distribution.png", dpi=160)
        plt.close(fig)


def write_validation_diagnostics(cfg: dict, model, val_loader, target: str, route_class: str, out_dir: Path) -> None:
    validation_cfg = cfg.get("validation", {})
    df = collect_validation_predictions(cfg, model, val_loader, target, route_class)
    if validation_cfg.get("write_predictions", True):
        df.to_csv(out_dir / "validation_predictions.csv", index=False)
        print(f"[Validation] wrote {out_dir / 'validation_predictions.csv'} | rows={len(df)}")
    if validation_cfg.get("write_summary", True):
        write_summary(df, target, out_dir)
        print(f"[Validation] wrote {out_dir / 'validation_metrics_summary.csv'}")
    if validation_cfg.get("write_plots", True):
        write_plots(df, target, out_dir)
        print(f"[Validation] wrote validation plots in {out_dir}")


def print_sanity(train_loader, target: str) -> None:
    batch = next(iter(train_loader))
    if target == "energy":
        fields = ["totalEnergy"]
    else:
        fields = [target, "totalEnergy"]
    for field in fields:
        values = extract_field(batch, field).detach().cpu().view(-1)
        print(f"[Sanity] {field}: min={values.min():.3e} max={values.max():.3e}")


def validate_config(cfg: dict, target: str) -> None:
    if cfg["task"]["type"] != "reconstruction":
        raise ValueError(f"Expected task.type=reconstruction, got {cfg['task']['type']}")
    if cfg["task"].get("mode") != "separate":
        raise ValueError("Only task.mode=separate is supported for reconstruction")
    if target not in cfg["task"]["targets"]:
        raise ValueError(f"Target {target!r} is not listed in task.targets")
    target_labels = cfg.get("target_settings", {}).get(target, {}).get("target_labels", [target])
    required_truth = ["event_no", "RunID", "EventID", "pid", "is_CC", "category1_isMuonCC", "category2_tauCC_others_muonCC", "category_3_contains_muon", *target_labels]
    missing = [field for field in required_truth if field not in cfg["data"]["truth_all"]]
    if missing:
        raise ValueError(f"data.truth_all is missing required fields: {missing}")
    if target not in cfg.get("target_settings", {}):
        raise ValueError(f"target_settings.{target} is missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    parser.add_argument("--route-class", required=True, help="Routing class id, e.g. 0")
    parser.add_argument("--target", required=True, help="Reconstruction target, e.g. energy")
    args = parser.parse_args()

    route_class = normalize_class_id(args.route_class)
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = effective_cfg_for_target(raw_cfg, args.target)
    validate_config(cfg, args.target)

    out_dir = resolve_target_dir(cfg, route_class, args.target)
    if os.environ.get("OUTPUT_PREPARED", "0") != "1":
        handle_existing_output(out_dir, cfg.get("run", {}).get("existing_output", "error"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "pipeline_config.yml")

    print("\n========== CONFIG ==========")
    print(yaml.dump(cfg, default_flow_style=False))
    print("============================\n")
    print(f"[Output] target dir: {out_dir}")

    split_paths, percentiles_csv = resolve_reconstruction_paths(cfg, route_class)
    data_representation, train_loader, val_loader = build_loaders(cfg, split_paths, percentiles_csv)
    print_sanity(train_loader, args.target)

    model = run_training(cfg, args.target, data_representation, train_loader, val_loader, out_dir)
    load_best_weights(model, out_dir)
    write_validation_diagnostics(cfg, model, val_loader, args.target, route_class, out_dir)
    print(f"[Validation] diagnostics written to: {out_dir}")


if __name__ == "__main__":
    main()
