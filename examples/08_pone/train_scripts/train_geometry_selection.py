"""
Train one reconstruction model with learnable string-selection gates.

This is a single-job worker, not a submit wrapper. It reuses the routed
reconstruction data flow, but adds a global learnable gate per detector string.

Required config section:

geometry_selection:
  string_feature: string_id
  max_string_id: 340
  active_strings: 70
  budget_lambda: 1.0e-3
  gate_mode: soft              # soft | straight_through_topk
  gate_on: charge              # charge | all
  output_dir: /path/to/output  # optional; otherwise derived from output.root_dir
"""

import argparse
import copy
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from torch import Tensor
from torch_geometric.data import Data

THIS_DIR = Path(__file__).resolve().parent
PONE_DIR = THIS_DIR.parent
if str(PONE_DIR) not in sys.path:
    sys.path.insert(0, str(PONE_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.models.data_representation import KNNGraph, NodesAsPulses
from graphnet.models.detector.pone import PONE
from graphnet.models.gnn import DynEdge
from graphnet.models.standard_model import StandardModel
from graphnet.training.callbacks import GraphnetEarlyStopping, PiecewiseLinearLR
from train_reconstruction import (
    EXTRA_KEYS,
    build_task,
    effective_cfg_for_target,
    handle_existing_output,
    load_best_weights,
    normalize_class_id,
    print_sanity,
    resolve_reconstruction_paths,
    write_validation_diagnostics,
)
from utils import (
    EpochCSVLogger,
    EpochTimeLogger,
    ValidationResidualAndLRMetrics,
    _EpochContextCallback,
    install_logging_filters,
)


class PONEWithIdentityAux(PONE):
    """PONE detector that passes auxiliary columns through unchanged."""

    def __init__(self, *args, identity_features: List[str], **kwargs) -> None:
        self._identity_features = list(dict.fromkeys(identity_features))
        super().__init__(*args, **kwargs)

    def feature_map(self) -> Dict[str, object]:
        available = list(self._p50.keys())
        selected = self._selected_features or available
        missing = [
            feature for feature in selected
            if feature not in available and feature not in self._identity_features
        ]
        if missing:
            raise KeyError(f"Features not in percentiles_csv: {missing}")

        fmap = {}
        for feature in selected:
            if feature in self._identity_features:
                fmap[feature] = self._identity
            else:
                fmap[feature] = lambda x, feature=feature: self._robust_scale(x, feature)
        return fmap


class StringGatedDynEdge(DynEdge):
    """DynEdge backbone with learnable per-string gates on node input."""

    def __init__(
        self,
        *,
        string_feature: str,
        aux_feature_indices: List[int],
        charge_feature_index: int,
        max_string_id: int,
        active_strings: int,
        gate_mode: str,
        gate_on: str,
        initial_logit: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if active_strings < 1 or active_strings > max_string_id:
            raise ValueError("active_strings must be in [1, max_string_id]")
        if gate_mode not in {"soft", "straight_through_topk"}:
            raise ValueError("gate_mode must be 'soft' or 'straight_through_topk'")
        if gate_on not in {"charge", "all"}:
            raise ValueError("gate_on must be 'charge' or 'all'")
        self.string_feature = string_feature
        self.aux_feature_indices = sorted(set(int(i) for i in aux_feature_indices))
        self.charge_feature_index = int(charge_feature_index)
        self.max_string_id = int(max_string_id)
        self.active_strings = int(active_strings)
        self.gate_mode = gate_mode
        self.gate_on = gate_on
        self.string_logits = torch.nn.Parameter(
            torch.full((self.max_string_id + 1,), float(initial_logit))
        )

    def gate_scores(self) -> Tensor:
        return torch.sigmoid(self.string_logits[1:])

    def gate_budget(self) -> Tensor:
        return self.gate_scores().sum()

    def _node_gates(self, string_ids: Tensor) -> Tensor:
        string_ids = string_ids.detach().long().clamp(1, self.max_string_id)
        soft = torch.sigmoid(self.string_logits[string_ids])
        if self.gate_mode == "soft":
            return soft

        scores = self.gate_scores()
        selected = torch.topk(scores, k=self.active_strings).indices + 1
        hard_all = torch.zeros_like(self.string_logits)
        hard_all[selected] = 1.0
        hard = hard_all[string_ids]
        return hard.detach() - soft.detach() + soft

    def _drop_aux_features(self, x: Tensor) -> Tensor:
        if not self.aux_feature_indices:
            return x
        keep = [
            idx for idx in range(x.shape[1])
            if idx not in set(self.aux_feature_indices)
        ]
        return x[:, keep]

    def forward(self, data: Data) -> Tensor:
        if self.string_feature not in data:
            raise KeyError(
                f"Graph is missing string feature {self.string_feature!r}. "
                "Add it to data.features and ensure it exists in the parquet feature table."
            )

        string_ids = data[self.string_feature].view(-1).to(data.x.device)
        gates = self._node_gates(string_ids).to(dtype=data.x.dtype, device=data.x.device)
        x = self._drop_aux_features(data.x)

        if self.gate_on == "all":
            x = x * gates[:, None]
        else:
            charge_idx = self.charge_feature_index
            dropped_before_charge = sum(idx < charge_idx for idx in self.aux_feature_indices)
            charge_idx -= dropped_before_charge
            columns = [
                x[:, idx] * gates if idx == charge_idx else x[:, idx]
                for idx in range(x.shape[1])
            ]
            x = torch.stack(columns, dim=1)

        gated = data.clone()
        gated.x = x
        return super().forward(gated)


class GeometrySelectionModel(StandardModel):
    """StandardModel plus a differentiable active-string budget penalty."""

    def __init__(self, *args, budget_lambda: float, budget_target: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.budget_lambda = float(budget_lambda)
        self.budget_target = int(budget_target)

    def budget_penalty(self) -> Tensor:
        budget = self.backbone.gate_budget()
        target = torch.tensor(
            float(self.budget_target),
            dtype=budget.dtype,
            device=budget.device,
        )
        return self.budget_lambda * (budget - target).pow(2)

    def compute_loss(self, preds: Tensor, data: List[Data], verbose: bool = False) -> Tensor:
        reconstruction_loss = super().compute_loss(preds, data, verbose=verbose)
        penalty = self.budget_penalty()
        loss = reconstruction_loss + penalty
        batch_size = len(data)
        self.log(
            "selection_budget",
            self.backbone.gate_budget(),
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            logger=False,
        )
        self.log(
            "selection_budget_penalty",
            penalty,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            logger=False,
        )
        return loss


def validate_selection_config(cfg: dict, target: str) -> dict:
    if cfg["task"]["type"] != "reconstruction":
        raise ValueError(f"Expected task.type=reconstruction, got {cfg['task']['type']}")
    if target not in cfg["task"]["targets"]:
        raise ValueError(f"Target {target!r} is not listed in task.targets")
    sel = cfg.get("geometry_selection")
    if not isinstance(sel, dict):
        raise ValueError("Missing required config section: geometry_selection")
    required = ["string_feature", "max_string_id", "active_strings", "budget_lambda"]
    missing = [key for key in required if key not in sel]
    if missing:
        raise ValueError(f"geometry_selection is missing required keys: {missing}")
    out = copy.deepcopy(sel)
    out["string_feature"] = str(out["string_feature"])
    out["max_string_id"] = int(out["max_string_id"])
    out["active_strings"] = int(out["active_strings"])
    out["budget_lambda"] = float(out["budget_lambda"])
    out["gate_mode"] = str(out.get("gate_mode", "soft"))
    out["gate_on"] = str(out.get("gate_on", "charge"))
    out["initial_logit"] = float(out.get("initial_logit", 0.0))
    return out


def resolve_output_dir(cfg: dict, route_class: str, target: str, output_dir_arg: str = None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    sel = cfg["geometry_selection"]
    if sel.get("output_dir"):
        return Path(sel["output_dir"])
    tag = sel.get("output_tag", "geometry_selection")
    return (
        Path(cfg["output"]["root_dir"])
        / cfg["mc"]
        / cfg["geometry"]
        / tag
        / cfg["routing"]["category"]
        / f"class{route_class}"
        / cfg["experiment_name"]
        / cfg["output"]["dirs"]["train"]
        / target
    )


def build_loaders(cfg: dict, selection_cfg: dict, split_paths: dict, percentiles_csv: str):
    model_features = list(cfg["data"]["features"])
    string_feature = selection_cfg["string_feature"]
    if string_feature in model_features:
        raise ValueError(
            f"Do not include {string_feature!r} in data.features for this script. "
            "It is added as an auxiliary gating column and removed before DynEdge."
        )
    if selection_cfg["gate_on"] == "charge" and "charge" not in model_features:
        raise ValueError("data.features must include 'charge' when geometry_selection.gate_on='charge'")
    loader_features = list(dict.fromkeys([*model_features, string_feature]))
    aux_feature_indices = [
        idx for idx, feature in enumerate(loader_features)
        if feature not in model_features
    ]
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
        detector=PONEWithIdentityAux(
            percentiles_csv=percentiles_csv,
            selected_features=loader_features,
            identity_features=[string_feature],
        ),
        node_definition=NodesAsPulses(),
        input_feature_names=loader_features,
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
            features=loader_features,
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
    print(f"[Data] model_features={model_features}")
    print(f"[Data] loader_features={loader_features}")
    print(f"[Data] aux_feature_indices={aux_feature_indices}")
    print(f"[Data] train={len(train_loader)} batches | val={len(val_loader)} batches")
    charge_feature_index = loader_features.index("charge") if "charge" in loader_features else -1
    return data_representation, train_loader, val_loader, aux_feature_indices, charge_feature_index


def build_model(
    cfg: dict,
    selection_cfg: dict,
    data_representation,
    steps_per_epoch_optimizer: int,
    target: str,
    aux_feature_indices: List[int],
    charge_feature_index: int,
) -> GeometrySelectionModel:
    model_features = cfg["data"]["features"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    backbone = StringGatedDynEdge(
        nb_inputs=len(model_features),
        nb_neighbours=model_cfg["nb_neighbours"],
        global_pooling_schemes=model_cfg["global_pooling_schemes"],
        add_global_variables_after_pooling=model_cfg.get("add_global_variables_after_pooling", True),
        add_norm_layer=model_cfg.get("add_norm_layer", False),
        skip_readout=model_cfg.get("skip_readout", False),
        string_feature=selection_cfg["string_feature"],
        aux_feature_indices=aux_feature_indices,
        charge_feature_index=charge_feature_index,
        max_string_id=selection_cfg["max_string_id"],
        active_strings=selection_cfg["active_strings"],
        gate_mode=selection_cfg["gate_mode"],
        gate_on=selection_cfg["gate_on"],
        initial_logit=selection_cfg["initial_logit"],
    )
    task = build_task(cfg, backbone, target)

    total_steps = steps_per_epoch_optimizer * train_cfg["max_epochs"]
    warmup_steps = max(1, int(train_cfg.get("warmup_fraction", 0.5) * steps_per_epoch_optimizer))

    return GeometrySelectionModel(
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
        budget_lambda=selection_cfg["budget_lambda"],
        budget_target=selection_cfg["active_strings"],
    )


def load_checkpoint_if_available(model: StandardModel, cfg: dict, target: str) -> None:
    checkpoint_path = cfg["training"].get("pretrained_weights")
    if not checkpoint_path:
        print(f"[GeometrySelection:{target}] no pretrained_weights configured, starting from scratch")
        return
    if not os.path.exists(checkpoint_path):
        print(f"[GeometrySelection:{target}] checkpoint not found, starting from scratch: {checkpoint_path}")
        return
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[GeometrySelection:{target}] loaded checkpoint with strict=False: {checkpoint_path}")
    print(f"[GeometrySelection:{target}] missing={list(missing)} unexpected={list(unexpected)}")


def run_training(
    cfg: dict,
    selection_cfg: dict,
    target: str,
    data_representation,
    train_loader,
    val_loader,
    aux_feature_indices: List[int],
    charge_feature_index: int,
    out_dir: Path,
) -> StandardModel:
    install_logging_filters()
    train_cfg = cfg["training"]
    pl.seed_everything(train_cfg["seed"], workers=True)
    steps_per_epoch_optimizer = math.ceil(len(train_loader) / train_cfg["accumulate_grad_batches"])
    model = build_model(
        cfg,
        selection_cfg,
        data_representation,
        steps_per_epoch_optimizer,
        target,
        aux_feature_indices,
        charge_feature_index,
    )
    load_checkpoint_if_available(model, cfg, target)

    target_label = cfg.get("target_settings", {}).get(target, {}).get("target_labels", [target])[0]
    extra_keys = [
        *EXTRA_KEYS[target],
        "selection_budget",
        "selection_budget_penalty",
    ]
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
        EpochCSVLogger(out_dir, extra_keys=extra_keys, filename="training_history_by_epoch.csv"),
        EpochTimeLogger(out_dir),
    ]

    print(f"\n[GeometrySelection:{target}] cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[GeometrySelection:{target}] GPU: {torch.cuda.get_device_name(0)}")

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accumulate_grad_batches=train_cfg["accumulate_grad_batches"],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"[GeometrySelection:{target}] best_model: {out_dir / 'best_model.pth'}")
    return model


def write_string_selection(model: GeometrySelectionModel, out_dir: Path) -> None:
    scores = model.backbone.gate_scores().detach().cpu().numpy()
    active = int(model.backbone.active_strings)
    order = np.argsort(scores)[::-1]
    selected = set((order[:active] + 1).tolist())
    rows = []
    for idx, score in enumerate(scores, start=1):
        rows.append(
            {
                "string_id": idx,
                "gate_score": float(score),
                "selected_topk": idx in selected,
                "rank": int(np.where(order == idx - 1)[0][0] + 1),
            }
        )
    df = pd.DataFrame(rows).sort_values("rank")
    df.to_csv(out_dir / "learned_string_selection.csv", index=False)
    print(f"[GeometrySelection] wrote {out_dir / 'learned_string_selection.csv'}")
    print(f"[GeometrySelection] top{active}: {df.head(active)['string_id'].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    parser.add_argument("--route-class", required=True, help="Routing class id, e.g. 0")
    parser.add_argument("--target", required=True, help="Reconstruction target, e.g. energy")
    parser.add_argument("--output-dir", default=None, help="Override geometry_selection.output_dir")
    args = parser.parse_args()

    route_class = normalize_class_id(args.route_class)
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = effective_cfg_for_target(raw_cfg, args.target)
    selection_cfg = validate_selection_config(cfg, args.target)

    out_dir = resolve_output_dir(cfg, route_class, args.target, args.output_dir)
    if os.environ.get("OUTPUT_PREPARED", "0") != "1":
        handle_existing_output(out_dir, cfg.get("run", {}).get("existing_output", "error"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "pipeline_config.yml")

    print("\n========== CONFIG ==========")
    print(yaml.dump(cfg, default_flow_style=False))
    print("============================\n")
    print(f"[Output] target dir: {out_dir}")
    print(f"[GeometrySelection] settings: {selection_cfg}")

    split_paths, percentiles_csv = resolve_reconstruction_paths(cfg, route_class)
    (
        data_representation,
        train_loader,
        val_loader,
        aux_feature_indices,
        charge_feature_index,
    ) = build_loaders(
        cfg,
        selection_cfg,
        split_paths,
        percentiles_csv,
    )
    print_sanity(train_loader, args.target)

    model = run_training(
        cfg,
        selection_cfg,
        args.target,
        data_representation,
        train_loader,
        val_loader,
        aux_feature_indices,
        charge_feature_index,
        out_dir,
    )
    load_best_weights(model, out_dir)
    write_string_selection(model, out_dir)
    write_validation_diagnostics(cfg, model, val_loader, args.target, route_class, out_dir)
    print(f"[GeometrySelection] diagnostics written to: {out_dir}")


if __name__ == "__main__":
    main()
