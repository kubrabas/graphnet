import argparse
import math
import os

import pytorch_lightning as pl
import torch
import yaml

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
    build_data,
    install_logging_filters,
    move_batch_to_device,
    run_test,
)

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


def build_model(cfg: dict, data_representation, steps_per_epoch_optimizer: int, target: str) -> StandardModel:
    features = cfg["data"]["features"]
    mcfg     = cfg["model"]

    backbone = DynEdge(
        nb_inputs=len(features),
        nb_neighbours=mcfg["nb_neighbours"],
        global_pooling_schemes=mcfg["global_pooling_schemes"],
        add_global_variables_after_pooling=mcfg.get("add_global_variables_after_pooling", True),
        add_norm_layer=mcfg.get("add_norm_layer", False),
        skip_readout=mcfg.get("skip_readout", False),
    )

    tcfg       = cfg["training"]
    base_lr    = tcfg["base_lr"]
    peak_lr    = tcfg["peak_lr"]
    max_epochs = tcfg["max_epochs"]
    accum      = tcfg["accumulate_grad_batches"]

    total_steps      = steps_per_epoch_optimizer * max_epochs
    warmup_fraction  = tcfg.get("warmup_fraction", 0.5)
    warmup_steps     = max(1, int(warmup_fraction * steps_per_epoch_optimizer))

    if target == "zenith":
        task = ZenithReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            loss_function=VonMisesFisher2DLoss(),
            target_labels=["zenith"],
        )
    elif target == "azimuth":
        task = AzimuthReconstructionWithKappa(
            hidden_size=backbone.nb_outputs,
            loss_function=VonMisesFisher2DLoss(),
            target_labels=["azimuth"],
        )
    elif target == "energy":
        task = DepositedEnergyLog10Task(
            hidden_size=backbone.nb_outputs,
            loss_function=LogCoshLoss(),
            target_labels=["energy"],
            prediction_labels=["log10_energy_pred"],
            transform_target=lambda E: torch.log10(torch.clamp(E, min=eps_like(E))),
            transform_inference=lambda t: torch.pow(10.0, t),
            transform_support=tuple(mcfg.get("transform_support", [1e1, 1e8])),
            loss_weight=None,
        )
    else:
        raise ValueError(f"Unknown target: {target}")

    return StandardModel(
        tasks=task,
        data_representation=data_representation,
        backbone=backbone,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": base_lr},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [0, warmup_steps, total_steps],
            "factors":    [1.0, peak_lr / base_lr, 1.0],
        },
        scheduler_config={"interval": "step"},
    )


def run_one(cfg: dict, target: str, data_representation, train_loader, val_loader, test_loader) -> None:
    install_logging_filters()

    save_dir = cfg["output"]["save_dir"]
    out_dir  = os.path.join(save_dir, target)
    os.makedirs(out_dir, exist_ok=True)

    tcfg = cfg["training"]
    pl.seed_everything(tcfg["seed"], workers=True)

    steps_per_epoch_optimizer = math.ceil(len(train_loader) / tcfg["accumulate_grad_batches"])

    model = build_model(cfg, data_representation, steps_per_epoch_optimizer, target)

    early_stop = GraphnetEarlyStopping(
        save_dir=out_dir,
        monitor="val_loss",
        mode="min",
        patience=tcfg["early_stopping_patience"],
        check_on_train_epoch_end=False,
        verbose=True,
    )

    metrics_cb   = EpochCSVLogger(out_dir, extra_keys=EXTRA_KEYS[target])
    time_cb      = EpochTimeLogger(out_dir)
    epoch_ctx_cb = _EpochContextCallback()
    val_metrics_cb = ValidationResidualAndLRMetrics(
        target=target,
        val_loader=val_loader,
        max_batches=tcfg.get("val_metrics_max_batches", None),
    )

    print(f"\n[Run={target}] cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Run={target}] GPU: {torch.cuda.get_device_name(0)}")

    trainer = pl.Trainer(
        max_epochs=tcfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        callbacks=[epoch_ctx_cb, val_metrics_cb, early_stop, metrics_cb, time_cb],
        enable_checkpointing=False,
        enable_progress_bar=False,
        accumulate_grad_batches=tcfg["accumulate_grad_batches"],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"[Run={target}] best_model: {os.path.join(out_dir, 'best_model.pth')}")
    print(f"[Run={target}] metrics:    {os.path.join(out_dir, 'metrics.csv')}")

    run_test(cfg, target=target, model=model, test_loader=test_loader, out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)

    print("\n========== CONFIG ==========")
    print(yaml.dump(cfg, default_flow_style=False))
    print("============================\n")

    data_representation, train_loader, val_loader, test_loader = build_data(cfg)

    from utils import extract_field
    b0 = next(iter(train_loader))
    for field, label in [("azimuth", "azimuth"), ("zenith", "zenith"), ("energy", "energy")]:
        v = extract_field(b0, field).detach().cpu().view(-1)
        print(f"[Sanity] {label}: {v.min():.3e} .. {v.max():.3e}")

    for target in ["energy", "zenith", "azimuth"]:
        run_one(cfg, target, data_representation, train_loader, val_loader, test_loader)
