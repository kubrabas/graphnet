#!/usr/bin/env python3
"""Run direct joint-direction inference with no classifier or router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = THIS_DIR.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import pandas as pd
import torch

from data import build_data_representation, build_loaders, deep_data_audit
from direction_utils import (
    opening_angle_radians,
    prediction_to_zenith_azimuth,
    zenith_azimuth_to_unit_vector,
)
from energy_weighting import EnergyWeightManifest
from metrics import opening_angle_metrics, weighted_mean
from model import build_joint_direction_model
from pipeline_utils import (
    assert_local_graphnet_source,
    atomic_json_dump,
    checkpoint_state,
    experiment_dir,
    extract_field,
    load_yaml,
    move_batch_to_device,
    resolve_muon_split_paths,
)
from reporting import write_evaluation_plots


STAGE_DIRS = {
    "stage_a": "stage_a_vmf",
    "stage_b": "stage_b_angular_hybrid",
}


def field_numpy(batch, field: str):
    return extract_field(batch, field).detach().cpu().reshape(-1).numpy()


def resolve_checkpoint(
    root: Path, stage: str, checkpoint_name: str, explicit: Path | None
) -> Path:
    if explicit is not None:
        path = explicit
    else:
        path = root / STAGE_DIRS[stage] / "checkpoints" / f"{checkpoint_name}.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enforce_frozen_test_selection(
    root: Path, split: str, stage: str, checkpoint_name: str, checkpoint: Path
) -> None:
    """Prevent test-driven checkpoint selection."""

    if split != "test":
        return
    path = root / "checkpoint_selection.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Test inference is locked. Run validation inference, then freeze one "
            f"checkpoint with freeze_checkpoint.py. Missing: {path}"
        )
    selection = json.loads(path.read_text(encoding="utf-8"))
    expected = (selection.get("stage"), selection.get("checkpoint_name"))
    requested = (stage, checkpoint_name)
    if requested != expected:
        raise ValueError(
            f"Test checkpoint is frozen as {expected}; requested {requested}. "
            "Selection was not changed."
        )
    if sha256(checkpoint) != selection.get("checkpoint_sha256"):
        raise ValueError("Frozen checkpoint file changed after validation selection")


def collect_predictions(model, loader, split: str, config, device) -> pd.DataFrame:
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    identifiers = ["event_no", *config["data"]["physical_event_id_columns"]]
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            prediction = model(batch)[0].detach().float()
            zenith = extract_field(batch, "zenith").reshape(-1).to(prediction)
            azimuth = extract_field(batch, "azimuth").reshape(-1).to(prediction)
            energy = extract_field(batch, "totalEnergy").reshape(-1).to(prediction)
            truth_direction = zenith_azimuth_to_unit_vector(zenith, azimuth)
            angle_rad = opening_angle_radians(prediction[:, :3], truth_direction)
            pred_zenith, pred_azimuth = prediction_to_zenith_azimuth(prediction)
            target = torch.stack((zenith, azimuth, energy), dim=1)
            components = model.direction_loss.elementwise_components(prediction, target)
            energy_weight = model.direction_loss.diagnostic_energy_weights(energy)
            hybrid = angle_rad + model.vmf_factor * components["vmf"]

            row = {
                "split": [split] * prediction.shape[0],
                **{field: field_numpy(batch, field) for field in identifiers},
                "pid": field_numpy(batch, "pid"),
                "is_CC": field_numpy(batch, "is_CC"),
                "category1_isMuonCC": field_numpy(batch, "category1_isMuonCC"),
                str(config["data"]["trigger_column"]): field_numpy(
                    batch, str(config["data"]["trigger_column"])
                ),
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
            rows.append(pd.DataFrame(row))
    if not rows:
        raise ValueError(f"Inference produced no events for split={split}")
    result = pd.concat(rows, ignore_index=True)
    physical_ids = list(config["data"]["physical_event_id_columns"])
    if result.duplicated(physical_ids).any():
        raise ValueError("Inference output contains duplicate physical event IDs")
    return result


def write_metrics(frame: pd.DataFrame, config, stage: str, output_dir: Path) -> None:
    angles = torch.as_tensor(frame["opening_angle_deg"].to_numpy())
    energy = torch.as_tensor(frame["totalEnergy"].to_numpy())
    metrics, energy_rows = opening_angle_metrics(
        angles,
        energy,
        config["metrics"]["log10_energy_bin_edges"],
        minimum_events_per_bin=int(config["metrics"]["minimum_events_per_bin"]),
        require_all_bins=True,
        prefix=str(frame["split"].iloc[0]),
    )
    weights = torch.as_tensor(frame["train_derived_energy_weight"].to_numpy())
    vmf = torch.as_tensor(frame["vmf_loss"].to_numpy())
    hybrid = torch.as_tensor(frame["hybrid_loss"].to_numpy())
    metrics.update(
        {
            "vmf_loss_unweighted": float(vmf.mean()),
            "vmf_loss_weighted": weighted_mean(vmf, weights),
            "hybrid_loss_unweighted": float(hybrid.mean()),
            "hybrid_loss_weighted": weighted_mean(hybrid, weights),
            "objective_loss_unweighted": float(
                (vmf if stage == "stage_a" else hybrid).mean()
            ),
            "objective_loss_weighted": weighted_mean(
                vmf if stage == "stage_a" else hybrid, weights
            ),
        }
    )
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(
        output_dir / "metrics_by_true_energy.csv", index=False
    )
    write_evaluation_plots(frame, energy_rows, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--stage", choices=("stage_a", "stage_b"), default="stage_b")
    parser.add_argument(
        "--checkpoint-name",
        default="best_macro_median",
        help="Named checkpoint stem, e.g. best_macro_median or best_global_median",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    graphnet_source = assert_local_graphnet_source(config)
    print(f"[GraphNeT] local source verified: {graphnet_source['imported_graphnet_file']}")
    root = experiment_dir(config)
    checkpoint = resolve_checkpoint(
        root, args.stage, args.checkpoint_name, args.checkpoint
    )
    enforce_frozen_test_selection(
        root, args.split, args.stage, args.checkpoint_name, checkpoint
    )
    output_dir = (
        root
        / "inference"
        / args.split
        / f"{args.stage}_{args.checkpoint_name}"
    )
    output_prepared = os.environ.get("OUTPUT_PREPARED", "0") == "1"
    if output_dir.exists() and any(output_dir.iterdir()) and not output_prepared:
        raise FileExistsError(
            f"Inference output exists: {output_dir}. Nothing was overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = resolve_muon_split_paths(config)
    deep_data_audit(config, split_paths, target_value_splits=(args.split,))
    manifest = EnergyWeightManifest.load(root / "energy_weight_manifest.json")
    data_representation = build_data_representation(config)
    loaders = build_loaders(
        config, split_paths, data_representation, splits=(args.split,)
    )
    model = build_joint_direction_model(
        config,
        args.stage,
        data_representation,
        manifest,
        steps_per_optimizer_epoch=1,
    )
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(config["trainer"].get("require_gpu", True)) and device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required by this config")
    frame = collect_predictions(model, loaders[args.split], args.split, config, device)
    frame.to_parquet(output_dir / "predictions.parquet", index=False)
    write_metrics(frame, config, args.stage, output_dir)
    atomic_json_dump(
        {
            "split": args.split,
            "stage": args.stage,
            "checkpoint_name": args.checkpoint_name,
            "checkpoint": str(checkpoint),
            "events": int(len(frame)),
            "router_used": False,
            "source_parquet_modified": False,
            **graphnet_source,
        },
        output_dir / "inference_manifest.json",
    )
    print(f"[Inference] wrote {len(frame):,} events to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
