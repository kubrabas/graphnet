#!/usr/bin/env python3
"""Train one routed mixed-flavor joint zenith/azimuth reconstruction model.

One invocation owns exactly one geometry, routing category, and route class.
It trains Stage A (energy-weighted 3D vMF) followed by Stage B
(energy-weighted opening angle + 0.05 * vMF), then evaluates the Stage-B
``best_macro_median`` checkpoint on validation data only.

The existing separate energy/zenith/azimuth pipeline and its outputs are not
modified. Source parquet files are opened read-only and never gain a weight
column.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


THIS_DIR = Path(__file__).resolve().parent
JOINT_DIR = THIS_DIR.parent
PONE_DIR = JOINT_DIR.parent
REFERENCE_DIR = PONE_DIR.parent / "09_pone_muon_direction"

# The routed modules have unique names. The scientifically approved model,
# loss, metrics, callbacks, and reporting modules are imported directly from
# the working 09 pipeline so their equations cannot silently drift.
for path in (PONE_DIR, REFERENCE_DIR, JOINT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping

from callbacks import EnergyBinMetricsCSV, EpochMetricsCSV, NamedCheckpoint, ResourceCSV
from model import build_joint_direction_model
from reporting import plot_training_history
from routed_data import (
    build_data_representation,
    build_loaders,
    deep_data_audit,
    fit_or_load_energy_manifest,
    write_data_audit,
)
from routed_pipeline_utils import (
    assert_local_graphnet_source,
    atomic_json_dump,
    checkpoint_state,
    load_yaml,
    prepare_target_dir,
    resolve_routed_split_paths,
    target_dir,
)
from validation import run_validation_inference


STAGE_DIRS = {
    "stage_a": "stage_a_vmf",
    "stage_b": "stage_b_angular_hybrid",
}
EXPECTED_TARGET = "zenith_azimuth"
EXPECTED_FLAVORS = ("Muon", "Electron", "Tau", "NC")


def _validate_stage(stage: Mapping[str, Any], expected_objective: str, name: str) -> None:
    if str(stage.get("objective")) != expected_objective:
        raise ValueError(f"training.{name}.objective must be {expected_objective!r}")
    if int(stage.get("max_epochs", 0)) <= 0:
        raise ValueError(f"training.{name}.max_epochs must be positive")
    base_lr = float(stage.get("base_lr", 0.0))
    peak_lr = float(stage.get("peak_lr", 0.0))
    if base_lr <= 0.0 or peak_lr <= 0.0 or peak_lr < base_lr:
        raise ValueError(
            f"training.{name} requires 0 < base_lr <= peak_lr, got "
            f"{base_lr} and {peak_lr}"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail fast on a scientific or output contract mismatch."""

    task = config.get("task", {})
    if task.get("type") != "reconstruction":
        raise ValueError("task.type must be reconstruction")
    if task.get("mode") != "joint_direction":
        raise ValueError("task.mode must be joint_direction")
    if list(task.get("targets", [])) != [EXPECTED_TARGET]:
        raise ValueError("task.targets must be exactly [zenith_azimuth]")
    if config.get("mc") != "340StringMC":
        raise ValueError("This routed joint pipeline currently supports mc=340StringMC")
    if not config.get("geometry"):
        raise ValueError("Top-level geometry is required")
    nested_geometry = config["data"].get("geometry")
    if nested_geometry is not None and config["geometry"] != nested_geometry:
        raise ValueError("data.geometry, when set, must match top-level geometry")
    if tuple(config.get("flavors", [])) != EXPECTED_FLAVORS:
        raise ValueError(
            "flavors must preserve the mixed-pipeline order "
            f"{list(EXPECTED_FLAVORS)}"
        )
    routing = config.get("routing", {})
    if routing.get("category") not in {
        "category1_isMuonCC",
        "category_3_contains_muon",
    }:
        raise ValueError("Only the two user-approved routing categories are allowed")
    configured_classes = routing.get("classes")
    if configured_classes != "all":
        if not isinstance(configured_classes, list) or not configured_classes:
            raise ValueError(
                "routing.classes must be 'all' or a non-empty subset of [0, 1]"
            )
        normalized_classes = [str(value) for value in configured_classes]
        if (
            len(normalized_classes) != len(set(normalized_classes))
            or not set(normalized_classes).issubset({"0", "1"})
        ):
            raise ValueError(
                "routing.classes must be 'all' or a non-empty subset of [0, 1]"
            )

    if config.get("weights") or config["data"].get("loss_weight_column"):
        raise ValueError(
            "Native/final_weight is forbidden; use only the train-derived weighting section"
        )
    weighting = config.get("weighting", {})
    if float(weighting.get("alpha", -1.0)) != 0.5:
        raise ValueError("weighting.alpha must remain 0.5 for this baseline")
    if float(weighting.get("clip_min", -1.0)) != 0.2:
        raise ValueError("weighting.clip_min must remain 0.2 for this baseline")
    if float(weighting.get("clip_max", -1.0)) != 5.0:
        raise ValueError("weighting.clip_max must remain 5.0 for this baseline")
    if str(weighting.get("out_of_range")) != "error":
        raise ValueError("weighting.out_of_range must be error")

    if config.get("loss", {}).get("angular_surrogate") != "opening_angle":
        raise ValueError("loss.angular_surrogate must be opening_angle")
    if float(config.get("loss", {}).get("vmf_factor", -1.0)) != 0.05:
        raise ValueError("loss.vmf_factor must remain 0.05")
    _validate_stage(config["training"]["stage_a"], "vmf", "stage_a")
    _validate_stage(config["training"]["stage_b"], "angular_hybrid", "stage_b")

    if config.get("checkpointing", {}).get("primary_monitor") != "val_macro_median_deg":
        raise ValueError("Primary checkpoint/early-stop monitor must be val_macro_median_deg")
    required_monitors = {
        "best_macro_median": "val_macro_median_deg",
        "best_global_median": "val_global_median_deg",
        "best_global_q68": "val_global_q68_deg",
        "best_weighted_objective": "val_objective_loss_weighted",
    }
    if config.get("checkpointing", {}).get("monitors") != required_monitors:
        raise ValueError("checkpointing.monitors differs from the approved joint baseline")
    if int(config.get("metrics", {}).get("minimum_events_per_bin", -1)) != 100:
        raise ValueError("metrics.minimum_events_per_bin must remain 100")

    output = config.get("output", {})
    if output.get("dirs", {}).get("train") != "train_and_val":
        raise ValueError("output.dirs.train must be train_and_val")
    configured_leaf = output.get("target_directory", EXPECTED_TARGET)
    if configured_leaf != EXPECTED_TARGET:
        raise ValueError("output.target_directory, when set, must be zenith_azimuth")
    validation = config.get("validation", {})
    if not bool(validation.get("enabled", False)):
        raise ValueError("validation.enabled must be true for train_and_val jobs")
    if validation.get("checkpoint_name") != "best_macro_median":
        raise ValueError("validation.checkpoint_name must be best_macro_median")
    for key in ("write_predictions", "write_summary", "write_plots"):
        if not bool(validation.get(key, False)):
            raise ValueError(f"validation.{key} must be true")
    if config.get("run", {}).get("existing_output") not in {"error", "skip", "resume"}:
        raise ValueError("run.existing_output must be error, skip, or resume")


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_submission_reservation(config_path: Path, output_dir: Path) -> str:
    """Bind a queued worker to the exact config bytes and output leaf reserved."""

    expected_hash = os.environ.get("CONFIG_SHA256")
    reserved_output = os.environ.get("RESERVED_OUTPUT_DIR")
    if not expected_hash or not reserved_output:
        raise RuntimeError(
            "OUTPUT_PREPARED requires CONFIG_SHA256 and RESERVED_OUTPUT_DIR"
        )
    actual_hash = _sha256(config_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Frozen config SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    resolved_output = output_dir.resolve()
    resolved_reserved = Path(reserved_output).resolve()
    if resolved_output != resolved_reserved:
        raise RuntimeError(
            "Config resolves a different output than the submitter reserved: "
            f"config={resolved_output}, reserved={resolved_reserved}"
        )
    if config_path.resolve().parent != resolved_reserved:
        raise RuntimeError(
            f"Frozen config must live in its reserved output leaf: {config_path}"
        )
    return actual_hash


def write_run_manifest(
    config: Mapping[str, Any],
    output_dir: Path,
    route_class: str,
    graphnet_source: Mapping[str, str],
    submitted_config_sha256: str | None,
) -> None:
    atomic_json_dump(
        {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "pytorch_lightning": pl.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "graphnet_git_revision": _git_revision(),
            "submitted_config_sha256": submitted_config_sha256,
            "mc": config["mc"],
            "geometry": config["geometry"],
            "routing_category": config["routing"]["category"],
            "routing_class": int(route_class),
            "flavors": list(config["flavors"]),
            "target": EXPECTED_TARGET,
            "router_model_used_for_training": False,
            "truth_categorized_training": True,
            "parquet_modified": False,
            "final_weight_used": False,
            "joint_reference_source": str(REFERENCE_DIR),
            **graphnet_source,
        },
        output_dir / "run_manifest.json",
    )


def snapshot_config(
    config: Mapping[str, Any], source: Path, output_dir: Path, route_class: str
) -> None:
    pipeline_config = output_dir / "pipeline_config.yml"
    if pipeline_config.exists():
        existing = load_yaml(pipeline_config)
        if existing != dict(config):
            raise ValueError(
                f"Existing output has a different pipeline config: {pipeline_config}"
            )
    else:
        shutil.copy2(source, pipeline_config)

    resolved = dict(config)
    resolved["resolved_route_class"] = int(route_class)
    temporary = output_dir / "resolved_config.yml.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    os.replace(temporary, output_dir / "resolved_config.yml")


def _load_weights(model, checkpoint: Path, label: str) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {checkpoint}")
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    print(f"[{label}] loaded network state: {checkpoint}")


def train_stage(
    config: Mapping[str, Any],
    stage_name: str,
    output_dir: Path,
    data_representation,
    loaders,
    energy_manifest,
    *,
    initialization_checkpoint: Path | None,
    resume: bool,
) -> Path:
    """Train one stage and return its resumable ``last.ckpt``."""

    stage_config = config["training"][stage_name]
    stage_dir = output_dir / STAGE_DIRS[stage_name]
    stage_dir.mkdir(parents=True, exist_ok=True)
    accumulation = int(config["loader"]["accumulate_grad_batches"])
    optimizer_steps = math.ceil(len(loaders["train"]) / accumulation)
    model = build_joint_direction_model(
        config,
        stage_name,
        data_representation,
        energy_manifest,
        optimizer_steps,
    )

    resume_checkpoint = stage_dir / "checkpoints" / "last.ckpt"
    checkpoint_path: str | None = None
    if resume:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"--resume requested but checkpoint is missing: {resume_checkpoint}"
            )
        checkpoint_path = str(resume_checkpoint)
    elif initialization_checkpoint is not None:
        _load_weights(model, initialization_checkpoint, f"{stage_name} initialization")

    callbacks: list[Any] = [
        EpochMetricsCSV(stage_dir, stage_name),
        EnergyBinMetricsCSV(stage_dir, stage_name),
        ResourceCSV(stage_dir, stage_name),
    ]
    if bool(stage_config.get("early_stopping", True)):
        callbacks.append(
            EarlyStopping(
                monitor=str(config["checkpointing"]["primary_monitor"]),
                mode="min",
                patience=int(stage_config["early_stopping_patience"]),
                min_delta=float(stage_config.get("early_stopping_min_delta", 0.0)),
                check_on_train_epoch_end=False,
                strict=True,
                verbose=True,
            )
        )
    callbacks.append(
        NamedCheckpoint(
            stage_dir,
            stage_name,
            monitors=config["checkpointing"]["monitors"],
            save_every_epoch=bool(
                config["checkpointing"].get("save_every_epoch", False)
            ),
        )
    )

    print(f"\n========== {stage_name}: {stage_config['objective']} ==========")
    print(f"output: {stage_dir}")
    print(f"train batches: {len(loaders['train'])}")
    print(f"validation batches: {len(loaders['val'])}")
    print(f"optimizer steps/epoch: {optimizer_steps}")
    trainer = pl.Trainer(
        max_epochs=int(stage_config["max_epochs"]),
        accelerator=str(config["trainer"].get("accelerator", "gpu")),
        devices=int(config["trainer"].get("devices", 1)),
        precision=config["trainer"].get("precision", "32-true"),
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=bool(config["trainer"].get("progress_bar", False)),
        enable_model_summary=bool(config["trainer"].get("model_summary", True)),
        accumulate_grad_batches=accumulation,
        gradient_clip_val=float(config["trainer"].get("gradient_clip_val", 0.0)),
        num_sanity_val_steps=int(config["trainer"].get("num_sanity_val_steps", 2)),
        deterministic=bool(config["trainer"].get("deterministic", False)),
        log_every_n_steps=int(config["trainer"].get("log_every_n_steps", 50)),
    )
    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
        ckpt_path=checkpoint_path,
    )
    plot_training_history(stage_dir)
    last = stage_dir / "checkpoints" / "last.ckpt"
    if not last.is_file():
        raise FileNotFoundError(f"Training ended without last checkpoint: {last}")
    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return last


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--route-class", required=True)
    parser.add_argument(
        "--stage", choices=("all", "stage_a", "stage_b"), default="all"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume one selected stage from its last Lightning checkpoint",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_class = str(args.route_class).replace("class", "")
    if route_class not in {"0", "1"}:
        raise ValueError("--route-class must be 0 or 1")
    config = load_yaml(args.config)
    validate_config(config)
    configured_classes = config["routing"]["classes"]
    if configured_classes != "all" and route_class not in {
        str(value) for value in configured_classes
    }:
        raise ValueError(
            f"--route-class {route_class} is not enabled by "
            f"routing.classes={configured_classes}"
        )
    if args.resume and args.stage == "all":
        raise ValueError("--resume requires --stage stage_a or --stage stage_b")
    if bool(config["trainer"].get("require_gpu", True)) and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required by this config")

    graphnet_source = assert_local_graphnet_source(config)
    print(f"[GraphNeT] local source verified: {graphnet_source['imported_graphnet_file']}")
    torch.set_float32_matmul_precision(
        str(config["trainer"].get("float32_matmul_precision", "high"))
    )
    pl.seed_everything(int(config["training"]["seed"]), workers=True)

    output_dir = target_dir(config, route_class)
    output_prepared = os.environ.get("OUTPUT_PREPARED", "0") == "1"
    submitted_config_sha256 = (
        verify_submission_reservation(args.config.resolve(), output_dir)
        if output_prepared
        else None
    )
    policy = "resume" if args.resume else str(config["run"]["existing_output"])
    should_run = prepare_target_dir(
        output_dir,
        policy,
        output_prepared=output_prepared,
    )
    if not should_run:
        print(f"[Skip] output already exists: {output_dir}")
        return 0
    snapshot_config(config, args.config.resolve(), output_dir, route_class)
    write_run_manifest(
        config,
        output_dir,
        route_class,
        graphnet_source,
        submitted_config_sha256,
    )

    split_paths, percentiles_csv = resolve_routed_split_paths(config, route_class)
    audit = deep_data_audit(config, split_paths, route_class)
    write_data_audit(audit, output_dir / "data_audit.json")
    print("[Data] routed mixed train/validation schema, trigger, class, and split audit passed")
    for split in ("train", "val"):
        details = audit["splits"][split]
        print(
            f"[Data] {split}: {details['events']:,} events | "
            f"flavors={details['events_by_flavor']}"
        )

    energy_manifest = fit_or_load_energy_manifest(
        config,
        split_paths["train"],
        output_dir / "energy_weight_manifest.json",
    )
    print(
        f"[Weights] alpha={energy_manifest.alpha:g}, "
        f"train mean={energy_manifest.event_weighted_mean:.6f}, "
        f"range=[{min(energy_manifest.bin_weights):.4g}, "
        f"{max(energy_manifest.bin_weights):.4g}]"
    )

    data_representation = build_data_representation(config, percentiles_csv)
    loaders = build_loaders(
        config,
        split_paths,
        data_representation,
        splits=("train", "val"),
    )
    print(
        f"[Loader] train={len(loaders['train'])}, val={len(loaders['val'])}; "
        "no test path or loader was resolved"
    )

    if args.stage in {"all", "stage_a"}:
        stage_a_last = train_stage(
            config,
            "stage_a",
            output_dir,
            data_representation,
            loaders,
            energy_manifest,
            initialization_checkpoint=None,
            resume=args.resume,
        )
    else:
        stage_a_last = (
            output_dir / STAGE_DIRS["stage_a"] / "checkpoints" / "last.ckpt"
        )

    if args.stage in {"all", "stage_b"}:
        train_stage(
            config,
            "stage_b",
            output_dir,
            data_representation,
            loaders,
            energy_manifest,
            initialization_checkpoint=None if args.resume else stage_a_last,
            resume=args.resume,
        )

        if bool(config.get("validation", {}).get("enabled", True)):
            checkpoint_name = str(
                config.get("validation", {}).get(
                    "checkpoint_name", "best_macro_median"
                )
            )
            if checkpoint_name != "best_macro_median":
                raise ValueError(
                    "Automatic validation is fixed to Stage-B best_macro_median"
                )
            checkpoint = (
                output_dir
                / STAGE_DIRS["stage_b"]
                / "checkpoints"
                / f"{checkpoint_name}.ckpt"
            )
            validation_dir = run_validation_inference(
                config,
                route_class,
                data_representation,
                loaders["val"],
                energy_manifest,
                output_dir,
                checkpoint,
                graphnet_source=graphnet_source,
            )
            print(f"[Validation] outputs: {validation_dir}")

    print(f"[Done] Routed joint-direction outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
