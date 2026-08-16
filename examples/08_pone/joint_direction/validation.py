"""Validation-only evaluation for routed mixed-flavor joint direction models.

This module deliberately accepts an already-constructed validation loader.  It
does not resolve parquet paths and therefore cannot accidentally inspect the
test split.  Mixed-dataset provenance is carried by ``source_flavor_id`` and
physical identifiers; ``event_no`` is never used as a cross-flavor join key.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd
import torch

# These imports intentionally reuse the approved 09 equations and report
# format rather than maintaining routed copies that could drift. Keep the
# module independently importable as well as usable through the train worker.
EXAMPLES_DIR = Path(__file__).resolve().parents[2]
JOINT_REFERENCE_DIR = EXAMPLES_DIR / "09_pone_muon_direction"
if str(JOINT_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(JOINT_REFERENCE_DIR))

from direction_utils import (
    normalize_unit_vectors,
    opening_angle_radians,
    prediction_to_zenith_azimuth,
    zenith_azimuth_to_unit_vector,
)
from energy_weighting import EnergyWeightManifest
from metrics import opening_angle_metrics, weighted_mean
from model_factory import build_direction_model
from pipeline_utils import checkpoint_state, extract_field, move_batch_to_device
from reporting import write_evaluation_plots

from routed_pipeline_utils import atomic_json_dump


STAGE_NAME = "stage_b"
CHECKPOINT_NAME = "best_macro_median"
INFERENCE_DIRECTORY = "stage_b_best_macro_median"
STAGE_OBJECTIVE = "angular_hybrid"
SOURCE_FLAVOR_NAMES = {
    0: "Muon",
    1: "Electron",
    2: "Tau",
    3: "NC",
}
REQUIRED_ROUTE_LABELS = (
    "category1_isMuonCC",
    "category2_tauCC_others_muonCC",
    "category_3_contains_muon",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_tensor(batch: Any, field: str) -> torch.Tensor:
    """Return one flat event-level tensor, failing clearly on a missing field."""

    try:
        value = extract_field(batch, field)
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError(f"Validation batch is missing required field {field!r}") from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Validation field {field!r} must be a torch.Tensor, got {type(value)!r}"
        )
    return value.reshape(-1)


def _field_numpy(batch: Any, field: str):
    return _field_tensor(batch, field).detach().cpu().numpy()


def _derive_flavor_id(pid: torch.Tensor, is_cc: torch.Tensor) -> torch.Tensor:
    """Derive the source flavor as an independent provenance cross-check."""

    pid = pid.reshape(-1).to(dtype=torch.int64)
    is_cc = is_cc.reshape(-1).to(dtype=torch.int64)
    result = torch.full_like(pid, -1)
    result[is_cc == 0] = 3
    charged_current = is_cc != 0
    absolute_pid = pid.abs()
    result[charged_current & (absolute_pid == 14)] = 0
    result[charged_current & (absolute_pid == 12)] = 1
    result[charged_current & (absolute_pid == 16)] = 2
    if bool((result < 0).any()):
        bad = torch.unique(torch.stack((pid[result < 0], is_cc[result < 0]), dim=1), dim=0)
        raise ValueError(
            "Cannot derive source flavor from pid/is_CC for values "
            f"{bad.detach().cpu().tolist()}"
        )
    return result


def _source_flavors(
    source_flavor_id: torch.Tensor,
    pid: torch.Tensor,
    is_cc: torch.Tensor,
) -> tuple[list[str], torch.Tensor]:
    source_flavor_id = source_flavor_id.reshape(-1).to(dtype=torch.int64)
    unknown = sorted(
        set(source_flavor_id.detach().cpu().tolist()) - SOURCE_FLAVOR_NAMES.keys()
    )
    if unknown:
        raise ValueError(f"Unknown source_flavor_id values: {unknown}")
    derived = _derive_flavor_id(pid, is_cc).to(source_flavor_id.device)
    mismatch = source_flavor_id != derived
    if bool(mismatch.any()):
        examples = torch.stack(
            (
                source_flavor_id[mismatch],
                derived[mismatch],
                pid.reshape(-1).to(source_flavor_id)[mismatch],
                is_cc.reshape(-1).to(source_flavor_id)[mismatch],
            ),
            dim=1,
        )[:10]
        raise ValueError(
            "source_flavor_id disagrees with pid/is_CC-derived flavor; "
            f"examples [source, derived, pid, is_CC]={examples.cpu().tolist()}"
        )
    names = [SOURCE_FLAVOR_NAMES[int(value)] for value in source_flavor_id.cpu().tolist()]
    return names, source_flavor_id


def _validate_output_frame(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    route_class: str,
) -> None:
    id_columns = list(config["data"]["physical_event_id_columns"])
    expected_ids = ["RunID", "SubrunID", "EventID", "SubEventID"]
    if id_columns != expected_ids:
        raise ValueError(
            "data.physical_event_id_columns must be exactly "
            f"{expected_ids}, received {id_columns}"
        )
    safe_key = ["source_flavor", *id_columns]
    if frame[safe_key].isna().any().any():
        raise ValueError("Validation output contains a null mixed physical identifier")
    if frame.duplicated(safe_key).any():
        duplicate = frame.loc[frame.duplicated(safe_key, keep=False), safe_key].head(10)
        raise ValueError(
            "Validation output contains duplicate flavor-namespaced physical IDs: "
            f"{duplicate.to_dict(orient='records')}"
        )

    routing_category = str(config["routing"]["category"])
    expected_class = int(route_class)
    if not (frame["routing_category"] == routing_category).all():
        raise ValueError("Validation output routing_category changed during collection")
    if not (frame["routing_class"] == expected_class).all():
        raise ValueError("Validation output routing_class changed during collection")
    if not (frame[routing_category].astype(int) == expected_class).all():
        raise ValueError("Validation output contains an event from the wrong route class")

    trigger = str(config["data"]["trigger_column"])
    if not (frame[trigger].astype(float) == 1.0).all():
        raise ValueError(f"Validation output contains {trigger} != 1")
    numeric = [
        "totalEnergy",
        "log10_totalEnergy",
        "direction_kappa",
        "opening_angle_rad",
        "opening_angle_deg",
        "vmf_loss",
        "hybrid_loss",
        "train_derived_energy_weight",
    ]
    values = torch.as_tensor(frame[numeric].to_numpy(dtype=float))
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Validation predictions contain NaN or infinite diagnostics")
    if not (frame["direction_kappa"] > 0.0).all():
        raise ValueError("Validation predictions contain non-positive kappa")
    if not frame["opening_angle_deg"].between(0.0, 180.0).all():
        raise ValueError("Validation opening angle lies outside [0, 180] degrees")
    if not (frame["train_derived_energy_weight"] > 0.0).all():
        raise ValueError("Validation diagnostic weights must be positive")


def _collect_predictions(
    model: Any,
    val_loader: Any,
    config: Mapping[str, Any],
    route_class: str,
    device: torch.device,
) -> pd.DataFrame:
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    identifiers = ["event_no", *config["data"]["physical_event_id_columns"]]
    route_labels = list(
        dict.fromkeys(
            [*REQUIRED_ROUTE_LABELS, str(config["routing"]["category"])]
        )
    )
    trigger = str(config["data"]["trigger_column"])
    routing_category = str(config["routing"]["category"])
    frames: list[pd.DataFrame] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)
            prediction = model(batch)[0].detach().float()
            if prediction.ndim != 2 or prediction.shape[1] != 4:
                raise ValueError(
                    "Joint direction prediction must have shape [N, 4], got "
                    f"{tuple(prediction.shape)}"
                )
            zenith = _field_tensor(batch, "zenith").to(prediction)
            azimuth = _field_tensor(batch, "azimuth").to(prediction)
            energy = _field_tensor(batch, "totalEnergy").to(prediction)
            pid = _field_tensor(batch, "pid")
            is_cc = _field_tensor(batch, "is_CC")
            if not (
                prediction.shape[0]
                == zenith.numel()
                == azimuth.numel()
                == energy.numel()
            ):
                raise ValueError("Prediction and validation truth batch sizes differ")

            flavor_names, flavor_ids = _source_flavors(
                _field_tensor(batch, "source_flavor_id"), pid, is_cc
            )
            truth_direction = zenith_azimuth_to_unit_vector(zenith, azimuth)
            predicted_direction = normalize_unit_vectors(prediction[:, :3])
            angle_rad = opening_angle_radians(predicted_direction, truth_direction)
            pred_zenith, pred_azimuth = prediction_to_zenith_azimuth(prediction)
            target = torch.stack((zenith, azimuth, energy), dim=1)
            components = model.direction_loss.elementwise_components(prediction, target)
            energy_weight = model.direction_loss.diagnostic_energy_weights(energy)
            # The approved Stage-B objective is exact opening angle [rad] +
            # 0.05*vMF; use the model value to prevent hard-coded drift.
            hybrid = angle_rad + model.vmf_factor * components["vmf"]
            if not torch.allclose(
                hybrid,
                components["unweighted"],
                rtol=1.0e-5,
                atol=1.0e-6,
            ):
                raise ValueError(
                    "Stage-B model loss is inconsistent with opening angle + "
                    "0.05*vMF"
                )

            row: dict[str, Any] = {
                "split": ["val"] * prediction.shape[0],
                "source_flavor_id": flavor_ids.detach().cpu().numpy(),
                "source_flavor": flavor_names,
                **{field: _field_numpy(batch, field) for field in identifiers},
                "routing_category": [routing_category] * prediction.shape[0],
                "routing_class": [int(route_class)] * prediction.shape[0],
                "pid": pid.detach().cpu().numpy(),
                "is_CC": is_cc.detach().cpu().numpy(),
                **{field: _field_numpy(batch, field) for field in route_labels},
                trigger: _field_numpy(batch, trigger),
                "totalEnergy": energy.cpu().numpy(),
                "log10_totalEnergy": torch.log10(energy).cpu().numpy(),
                "true_zenith_rad": zenith.cpu().numpy(),
                "true_azimuth_rad": azimuth.cpu().numpy(),
                "true_zenith_deg": torch.rad2deg(zenith).cpu().numpy(),
                "true_azimuth_deg": torch.rad2deg(azimuth).cpu().numpy(),
                "true_dir_x": truth_direction[:, 0].cpu().numpy(),
                "true_dir_y": truth_direction[:, 1].cpu().numpy(),
                "true_dir_z": truth_direction[:, 2].cpu().numpy(),
                "pred_zenith_rad": pred_zenith.cpu().numpy(),
                "pred_azimuth_rad": pred_azimuth.cpu().numpy(),
                "pred_zenith_deg": torch.rad2deg(pred_zenith).cpu().numpy(),
                "pred_azimuth_deg": torch.rad2deg(pred_azimuth).cpu().numpy(),
                # DirectionReconstructionWithKappa already returns a unit
                # vector. Persist its exact checkpoint output, matching 09.
                "pred_dir_x": prediction[:, 0].cpu().numpy(),
                "pred_dir_y": prediction[:, 1].cpu().numpy(),
                "pred_dir_z": prediction[:, 2].cpu().numpy(),
                "direction_kappa": prediction[:, 3].cpu().numpy(),
                "opening_angle_rad": angle_rad.cpu().numpy(),
                "opening_angle_deg": torch.rad2deg(angle_rad).cpu().numpy(),
                "vmf_loss": components["vmf"].cpu().numpy(),
                "hybrid_loss": hybrid.cpu().numpy(),
                "train_derived_energy_weight": energy_weight.cpu().numpy(),
            }
            frames.append(pd.DataFrame(row))

    if not frames:
        raise ValueError("Validation inference produced no events")
    frame = pd.concat(frames, ignore_index=True)
    _validate_output_frame(frame, config, route_class)
    return frame


def _write_metrics(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, float]:
    angles = torch.as_tensor(frame["opening_angle_deg"].to_numpy(dtype=float))
    energy = torch.as_tensor(frame["totalEnergy"].to_numpy(dtype=float))
    metrics, energy_rows = opening_angle_metrics(
        angles,
        energy,
        config["metrics"]["log10_energy_bin_edges"],
        minimum_events_per_bin=int(config["metrics"]["minimum_events_per_bin"]),
        require_all_bins=True,
        prefix="val",
    )
    weights = torch.as_tensor(
        frame["train_derived_energy_weight"].to_numpy(dtype=float)
    )
    vmf = torch.as_tensor(frame["vmf_loss"].to_numpy(dtype=float))
    hybrid = torch.as_tensor(frame["hybrid_loss"].to_numpy(dtype=float))
    kappa = torch.as_tensor(frame["direction_kappa"].to_numpy(dtype=float))
    metrics.update(
        {
            # Preserve the established 09 validation-inference column names.
            "vmf_loss_unweighted": float(vmf.mean()),
            "vmf_loss_weighted": weighted_mean(vmf, weights),
            "hybrid_loss_unweighted": float(hybrid.mean()),
            "hybrid_loss_weighted": weighted_mean(hybrid, weights),
            "objective_loss_unweighted": float(hybrid.mean()),
            "objective_loss_weighted": weighted_mean(hybrid, weights),
            "kappa_mean": float(kappa.mean()),
            "kappa_median": float(torch.quantile(kappa, 0.50)),
            "kappa_q68": float(torch.quantile(kappa, 0.68)),
            "kappa_q90": float(torch.quantile(kappa, 0.90)),
            "diagnostic_weight_mean": float(weights.mean()),
            "diagnostic_weight_min": float(weights.min()),
            "diagnostic_weight_max": float(weights.max()),
        }
    )
    for name, value in metrics.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite validation metric {name}: {value}")
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(
        output_dir / "metrics_by_true_energy.csv", index=False
    )
    write_evaluation_plots(frame, energy_rows, output_dir)
    return {str(name): float(value) for name, value in metrics.items()}


def run_validation_inference(
    config: Mapping[str, Any],
    route_class: str,
    data_representation: Any,
    val_loader: Any,
    energy_manifest: EnergyWeightManifest,
    target_dir: str | Path,
    checkpoint: str | Path,
    *,
    device: torch.device | str | None = None,
    graphnet_source: Mapping[str, str] | None = None,
) -> Path:
    """Evaluate Stage-B ``best_macro_median`` on routed validation only.

    Parameters are injected by the training worker.  In particular, no split
    path is accepted or resolved here, making test access impossible through
    this interface.
    """

    route_class = str(route_class).replace("class", "")
    if route_class not in {"0", "1"}:
        raise ValueError("route_class must be 0 or 1")
    if str(config["training"][STAGE_NAME]["objective"]) != STAGE_OBJECTIVE:
        raise ValueError("Validation requires the Stage-B angular_hybrid objective")
    if str(config["loss"]["angular_surrogate"]) != "opening_angle":
        raise ValueError("Validation requires the exact opening_angle surrogate")
    if float(config["loss"]["vmf_factor"]) != 0.05:
        raise ValueError("Validation requires the approved 0.05 vMF factor")
    target_dir = Path(target_dir)
    checkpoint = Path(checkpoint)
    expected_checkpoint = (
        target_dir
        / "stage_b_angular_hybrid"
        / "checkpoints"
        / f"{CHECKPOINT_NAME}.ckpt"
    )
    if checkpoint.resolve() != expected_checkpoint.resolve():
        raise ValueError(
            "Validation is fixed to the Stage-B best_macro_median checkpoint; "
            f"expected {expected_checkpoint}, received {checkpoint}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Validation checkpoint does not exist: {checkpoint}")

    energy_manifest_path = target_dir / "energy_weight_manifest.json"
    if not energy_manifest_path.is_file():
        raise FileNotFoundError(
            f"Train-derived energy manifest is missing: {energy_manifest_path}"
        )
    persisted_energy_manifest = EnergyWeightManifest.load(energy_manifest_path)
    if persisted_energy_manifest != energy_manifest:
        raise ValueError(
            "Injected energy manifest differs from the train-derived manifest "
            f"saved in {energy_manifest_path}"
        )

    output_dir = target_dir / "inference" / "val" / INFERENCE_DIRECTORY
    if output_dir.exists():
        raise FileExistsError(
            f"Validation output already exists: {output_dir}. Nothing was overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    selected_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if (
        bool(config.get("trainer", {}).get("require_gpu", True))
        and selected_device.type != "cuda"
    ):
        raise RuntimeError("A CUDA GPU is required by this validation config")

    model = build_direction_model(
        config,
        STAGE_NAME,
        data_representation,
        energy_manifest,
        steps_per_optimizer_epoch=1,
    )
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    frame = _collect_predictions(
        model, val_loader, config, route_class, selected_device
    )
    frame.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics = _write_metrics(frame, config, output_dir)

    manifest: dict[str, Any] = {
        "split": "val",
        "stage": STAGE_NAME,
        "objective": STAGE_OBJECTIVE,
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "energy_weight_manifest": str(energy_manifest_path.resolve()),
        "energy_weight_manifest_sha256": _sha256(energy_manifest_path),
        "events": int(len(frame)),
        "events_by_source_flavor": {
            str(name): int(count)
            for name, count in frame["source_flavor"].value_counts().sort_index().items()
        },
        "mc": str(config["mc"]),
        "geometry": str(config["geometry"]),
        "routing_category": str(config["routing"]["category"]),
        "routing_class": int(route_class),
        "safe_event_key": [
            "source_flavor",
            *list(config["data"]["physical_event_id_columns"]),
        ],
        "event_no_joined_across_flavors": False,
        "router_model_used": False,
        "source_parquet_modified": False,
        "test_path_resolved": False,
        "test_loader_used": False,
        "metric_prefix": "val",
        "metric_count": len(metrics),
    }
    if graphnet_source:
        manifest.update({str(key): str(value) for key, value in graphnet_source.items()})
    atomic_json_dump(manifest, output_dir / "inference_manifest.json")
    return output_dir


__all__ = ["run_validation_inference"]
