"""Joint 3D direction model and single-pass validation aggregation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from graphnet.models.gnn import DynEdge
from graphnet.models.standard_model import StandardModel
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappa
from graphnet.training.callbacks import PiecewiseLinearLR

from direction_utils import (
    opening_angle_radians,
    zenith_azimuth_to_unit_vector,
)
from energy_weighting import EnergyWeightManifest
from losses import EnergyWeightedDirectionLoss
from metrics import opening_angle_metrics, weighted_mean
from pipeline_utils import extract_field


PREDICTION_LABELS = [
    "dir_x_pred",
    "dir_y_pred",
    "dir_z_pred",
    "direction_kappa",
]
TARGET_LABELS = ["zenith", "azimuth", "totalEnergy"]


class JointDirectionModel(StandardModel):
    """StandardModel that reuses validation predictions for all diagnostics."""

    def __init__(
        self,
        *args: Any,
        metric_bin_edges: Sequence[float],
        minimum_events_per_bin: int,
        vmf_factor: float,
        objective: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric_bin_edges = tuple(float(value) for value in metric_bin_edges)
        self.minimum_events_per_bin = int(minimum_events_per_bin)
        self.vmf_factor = float(vmf_factor)
        self.direction_objective = str(objective)
        self._validation_cache: list[dict[str, Tensor]] = []
        self.latest_validation_metrics: dict[str, float] = {}
        self.latest_validation_energy_rows: list[dict[str, float]] = []

    @property
    def direction_loss(self) -> EnergyWeightedDirectionLoss:
        loss = self._tasks[0]._loss_function
        if not isinstance(loss, EnergyWeightedDirectionLoss):
            raise TypeError("JointDirectionModel requires EnergyWeightedDirectionLoss")
        return loss

    def on_validation_epoch_start(self) -> None:
        """Drop tensors from the previous epoch before validation starts."""

        self._validation_cache = []

    def validation_step(self, val_batch, batch_idx: int) -> Tensor:
        """Compute validation loss and cache tiny event-level diagnostics once."""

        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        predictions = self(val_batch)
        loss = self.compute_loss(predictions, val_batch)
        batch_size = self._get_batch_size(val_batch)
        self.log(
            "val_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        prediction = predictions[0]
        zenith = extract_field(val_batch, "zenith").reshape(-1).to(prediction)
        azimuth = extract_field(val_batch, "azimuth").reshape(-1).to(prediction)
        energy = extract_field(val_batch, "totalEnergy").reshape(-1).to(prediction)
        target = torch.stack((zenith, azimuth, energy), dim=1)
        target_direction = zenith_azimuth_to_unit_vector(zenith, azimuth)
        angle_rad = opening_angle_radians(prediction[:, :3], target_direction)
        components = self.direction_loss.elementwise_components(prediction, target)
        diagnostic_weights = self.direction_loss.diagnostic_energy_weights(energy)
        hybrid = angle_rad + self.vmf_factor * components["vmf"]
        objective = components["vmf"] if self.direction_objective == "vmf" else hybrid

        if not self.trainer.sanity_checking:
            self._validation_cache.append(
                {
                    "opening_angle_deg": torch.rad2deg(angle_rad).detach().cpu(),
                    "energy": energy.detach().cpu(),
                    "kappa": prediction[:, 3].detach().cpu(),
                    "vmf": components["vmf"].detach().cpu(),
                    "hybrid": hybrid.detach().cpu(),
                    "objective": objective.detach().cpu(),
                    "diagnostic_weights": diagnostic_weights.detach().cpu(),
                }
            )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log global, energy-balanced, and per-energy-bin metrics."""

        if self.trainer.sanity_checking or not self._validation_cache:
            return
        combined = {
            key: torch.cat([batch[key] for batch in self._validation_cache], dim=0)
            for key in self._validation_cache[0]
        }
        metrics, energy_rows = opening_angle_metrics(
            combined["opening_angle_deg"],
            combined["energy"],
            self.metric_bin_edges,
            minimum_events_per_bin=self.minimum_events_per_bin,
            require_all_bins=True,
        )

        weights = combined["diagnostic_weights"]
        for name in ("vmf", "hybrid", "objective"):
            values = combined[name]
            metrics[f"val_{name}_loss_unweighted"] = float(values.mean())
            metrics[f"val_{name}_loss_weighted"] = weighted_mean(values, weights)

        kappa = combined["kappa"].to(torch.float64)
        metrics.update(
            {
                "val_kappa_mean": float(kappa.mean()),
                "val_kappa_median": float(torch.quantile(kappa, 0.50)),
                "val_kappa_q68": float(torch.quantile(kappa, 0.68)),
                "val_kappa_q90": float(torch.quantile(kappa, 0.90)),
                "val_diagnostic_weight_mean": float(weights.mean()),
                "val_diagnostic_weight_min": float(weights.min()),
                "val_diagnostic_weight_max": float(weights.max()),
            }
        )
        self.latest_validation_metrics = metrics
        self.latest_validation_energy_rows = energy_rows

        progress_metrics = {
            "val_global_median_deg",
            "val_macro_median_deg",
        }
        for name, value in metrics.items():
            if not math.isfinite(value):
                raise ValueError(f"Validation metric {name} is not finite: {value}")
            self.log(
                name,
                torch.tensor(value, device=self.device, dtype=torch.float32),
                prog_bar=name in progress_metrics,
                on_epoch=True,
                on_step=False,
                sync_dist=False,
            )
        self._validation_cache = []


def _optional_dynedge_kwargs(model_config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if model_config.get("dynedge_layer_sizes") is not None:
        result["dynedge_layer_sizes"] = [
            tuple(int(value) for value in layer)
            for layer in model_config["dynedge_layer_sizes"]
        ]
    for name in ("post_processing_layer_sizes", "readout_layer_sizes"):
        if model_config.get(name) is not None:
            result[name] = [int(value) for value in model_config[name]]
    if model_config.get("activation_layer") is not None:
        result["activation_layer"] = str(model_config["activation_layer"])
    return result


def build_joint_direction_model(
    config: Mapping[str, Any],
    stage_name: str,
    data_representation,
    energy_manifest: EnergyWeightManifest,
    steps_per_optimizer_epoch: int,
) -> JointDirectionModel:
    """Construct one stage of the joint direction training pipeline."""

    model_config = config["model"]
    stage = config["training"][stage_name]
    features = config["data"]["features"]
    objective = str(stage["objective"])
    vmf_factor = float(config["loss"]["vmf_factor"])

    backbone = DynEdge(
        nb_inputs=len(features),
        nb_neighbours=int(model_config["nb_neighbours"]),
        global_pooling_schemes=list(model_config["global_pooling_schemes"]),
        add_global_variables_after_pooling=bool(
            model_config.get("add_global_variables_after_pooling", True)
        ),
        add_norm_layer=bool(model_config.get("add_norm_layer", False)),
        skip_readout=bool(model_config.get("skip_readout", False)),
        **_optional_dynedge_kwargs(model_config),
    )
    direction_loss = EnergyWeightedDirectionLoss.from_manifest(
        energy_manifest,
        objective=objective,
        angular_surrogate=str(config["loss"]["angular_surrogate"]),
        vmf_factor=vmf_factor,
        weighting_mode="train_only",
    )
    task = DirectionReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        loss_function=direction_loss,
        target_labels=TARGET_LABELS,
        prediction_labels=PREDICTION_LABELS,
        loss_weight=None,
    )

    max_epochs = int(stage["max_epochs"])
    total_steps = max(1, steps_per_optimizer_epoch * max_epochs)
    warmup_steps = max(
        1,
        int(float(stage.get("warmup_fraction", 0.5)) * steps_per_optimizer_epoch),
    )
    base_lr = float(stage["base_lr"])
    peak_lr = float(stage["peak_lr"])
    return JointDirectionModel(
        tasks=task,
        data_representation=data_representation,
        backbone=backbone,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": base_lr},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [0, warmup_steps, total_steps],
            "factors": [1.0, peak_lr / base_lr, 1.0],
        },
        scheduler_config={"interval": "step"},
        metric_bin_edges=config["metrics"]["log10_energy_bin_edges"],
        minimum_events_per_bin=int(config["metrics"]["minimum_events_per_bin"]),
        vmf_factor=vmf_factor,
        objective=objective,
    )

