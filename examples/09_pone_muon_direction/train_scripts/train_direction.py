#!/usr/bin/env python3
"""Train Stage-A vMF then Stage-B angular-hybrid joint direction models."""

from __future__ import annotations

import argparse
import gc
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
EXAMPLE_DIR = THIS_DIR.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping

from callbacks import (
    EnergyBinMetricsCSV,
    EpochMetricsCSV,
    NamedCheckpoint,
    ResourceCSV,
)
from data import (
    build_data_representation,
    build_loaders,
    deep_data_audit,
    fit_or_load_energy_manifest,
    write_data_audit,
)
from model import build_joint_direction_model
from pipeline_utils import (
    assert_local_graphnet_source,
    atomic_json_dump,
    checkpoint_state,
    experiment_dir,
    load_yaml,
    prepare_experiment_dir,
    resolve_muon_split_paths,
)
from reporting import plot_training_history


STAGE_DIRS = {
    "stage_a": "stage_a_vmf",
    "stage_b": "stage_b_angular_hybrid",
}


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject accidental router/separate-target/native-weight configurations."""

    if "routing" in config:
        raise ValueError("Router configuration is forbidden in the joint pipeline")
    if str(config["data"].get("flavor")) != "Muon":
        raise ValueError("data.flavor must be Muon")
    if str(config["data"].get("parquet_table")) != "STRING340MC_PARQUET":
        raise ValueError("This first pipeline supports STRING340MC_PARQUET only")
    if config.get("weights") or config["data"].get("loss_weight_column"):
        raise ValueError(
            "Native parquet loss_weight is forbidden; use the custom weighting section"
        )
    if config["training"]["stage_a"]["objective"] != "vmf":
        raise ValueError("training.stage_a.objective must be vmf")
    if config["training"]["stage_b"]["objective"] != "angular_hybrid":
        raise ValueError("training.stage_b.objective must be angular_hybrid")
    if config["loss"]["angular_surrogate"] != "opening_angle":
        raise ValueError(
            "The approved Stage-B objective uses opening_angle, not another surrogate"
        )
    if float(config["loss"]["vmf_factor"]) != 0.05:
        raise ValueError("The approved Stage-B vMF coefficient is 0.05")
    if float(config["weighting"]["alpha"]) < 0.0:
        raise ValueError("weighting.alpha must be non-negative")
    if not config.get("experiment_name"):
        raise ValueError("experiment_name is required")


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def write_run_manifest(
    config: Mapping[str, Any], output_dir: Path, graphnet_source: Mapping[str, str]
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
            "geometry": config["data"]["geometry"],
            "experiment_name": config["experiment_name"],
            "router_used": False,
            "parquet_modified": False,
            **graphnet_source,
        },
        output_dir / "run_manifest.json",
    )


def snapshot_config(config: Mapping[str, Any], source: Path, output_dir: Path) -> None:
    destination = output_dir / "pipeline_config.yml"
    resolved = output_dir / "resolved_config.yml"
    if destination.exists():
        existing = load_yaml(destination)
        if existing != dict(config):
            raise ValueError(
                f"Existing experiment has a different config: {destination}"
            )
    else:
        shutil.copy2(source, destination)
    resolved.write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )


def _load_weights(model, checkpoint: Path, label: str) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {checkpoint}")
    state = checkpoint_state(checkpoint)
    model.load_state_dict(state, strict=True)
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
    ckpt_path: str | None = None
    if resume:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"--resume requested but checkpoint is missing: {resume_checkpoint}"
            )
        ckpt_path = str(resume_checkpoint)
    elif initialization_checkpoint is not None:
        _load_weights(model, initialization_checkpoint, f"{stage_name} initialization")

    monitor_config = config["checkpointing"]["monitors"]
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
    # Save after EarlyStopping has consumed the current epoch, so resumable
    # checkpoints also contain the up-to-date patience counter.
    callbacks.append(
        NamedCheckpoint(
            stage_dir,
            stage_name,
            monitors=monitor_config,
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
        ckpt_path=ckpt_path,
    )
    plot_training_history(stage_dir)
    last = stage_dir / "checkpoints" / "last.ckpt"
    if not last.is_file():
        raise FileNotFoundError(f"Training ended without last checkpoint: {last}")
    # Stage A and Stage B can run in one allocation. Explicitly release the
    # first Trainer/model cycle before constructing the second GPU model.
    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return last


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=("all", "stage_a", "stage_b"), default="all"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the selected single stage from its last Lightning checkpoint",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    validate_config(config)
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

    output_dir = experiment_dir(config)
    policy = "resume" if args.resume else str(config["run"]["existing_output"])
    prepare_experiment_dir(
        output_dir,
        policy,
        output_prepared=os.environ.get("OUTPUT_PREPARED", "0") == "1",
    )
    snapshot_config(config, args.config.resolve(), output_dir)
    write_run_manifest(config, output_dir, graphnet_source)

    split_paths = resolve_muon_split_paths(config)
    audit = deep_data_audit(
        config, split_paths, target_value_splits=("train", "val")
    )
    write_data_audit(audit, output_dir / "data_audit.json")
    print("[Data] Muon-CC, triggered-nonoise, schema, and split audit passed")
    for split in ("train", "val", "test"):
        details = audit["splits"][split]
        print(f"[Data] {split}: {details['events']:,} events | {details['path']}")

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

    data_representation = build_data_representation(config)
    loaders = build_loaders(
        config, split_paths, data_representation, splits=("train", "val")
    )
    print(
        f"[Loader] train={len(loaders['train'])}, val={len(loaders['val'])}; "
        "test loader intentionally not constructed before checkpoint freeze"
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
        stage_a_last = output_dir / STAGE_DIRS["stage_a"] / "checkpoints" / "last.ckpt"

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

    print(f"[Done] Joint-direction outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
