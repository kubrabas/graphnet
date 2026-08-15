#!/usr/bin/env python3
"""Fine-tune Stage-B from a pinned source checkpoint using fresh optimizer state."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Mapping


THIS_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = THIS_DIR.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import pytorch_lightning as pl
import torch

from data import (
    build_data_representation,
    build_loaders,
    deep_data_audit,
    fit_or_load_energy_manifest,
    write_data_audit,
)
from energy_weighting import EnergyWeightManifest
from finetune_utils import (
    resolve_finetune_source,
    write_or_validate_source_manifest,
)
from pipeline_utils import (
    assert_local_graphnet_source,
    experiment_dir,
    load_yaml,
    prepare_experiment_dir,
    resolve_muon_split_paths,
)
from train_direction import (
    snapshot_config,
    train_stage,
    validate_config,
    write_run_manifest,
)


def validate_finetune_config(config: Mapping[str, Any]) -> None:
    """Apply the base guards plus fine-tune-specific optimizer constraints."""

    validate_config(config)
    stage = config["training"]["stage_b"]
    max_epochs = int(stage["max_epochs"])
    base_lr = float(stage["base_lr"])
    peak_lr = float(stage["peak_lr"])
    if max_epochs <= 0:
        raise ValueError("training.stage_b.max_epochs must be positive")
    if base_lr <= 0.0 or peak_lr <= 0.0:
        raise ValueError("Fine-tune learning rates must be positive")
    if peak_lr < base_lr:
        raise ValueError("Fine-tune peak_lr must be greater than or equal to base_lr")
    if not bool(stage.get("early_stopping", True)):
        raise ValueError("Fine-tuning requires Stage-B early stopping")
    if int(stage["early_stopping_patience"]) <= 0:
        raise ValueError("Fine-tune early_stopping_patience must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this fine-tune experiment from its own last.ckpt",
    )
    return parser.parse_args()


def _guard_prepared_output(output_dir: Path, *, resume: bool) -> None:
    """Allow only submit-wrapper artifacts before a fresh job starts."""

    if resume or os.environ.get("OUTPUT_PREPARED", "0") != "1":
        return
    allowed = {
        "gpu_telemetry.csv",
        "last_finetune_job_id.txt",
        "train_finetune.out",
    }
    if output_dir.exists():
        unexpected = sorted(
            path.name for path in output_dir.iterdir() if path.name not in allowed
        )
        if unexpected:
            raise FileExistsError(
                "Fresh fine-tune output contains unexpected files; nothing was "
                f"overwritten: {unexpected}"
            )


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    validate_finetune_config(config)

    output_dir = experiment_dir(config)
    source = resolve_finetune_source(config, output_dir)
    _guard_prepared_output(output_dir, resume=args.resume)

    require_gpu = bool(config["trainer"].get("require_gpu", True))
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required by this config")
    graphnet_source = assert_local_graphnet_source(config)
    print(
        "[GraphNeT] local source verified: "
        f"{graphnet_source['imported_graphnet_file']}"
    )
    print(
        "[FineTune] verified source checkpoint: "
        f"{source.checkpoint} (sha256={source.checkpoint_sha256})"
    )
    print(
        "[FineTune] initialization is weights-only; optimizer and LR scheduler "
        "start fresh"
    )

    torch.set_float32_matmul_precision(
        str(config["trainer"].get("float32_matmul_precision", "high"))
    )
    pl.seed_everything(int(config["training"]["seed"]), workers=True)

    policy = "resume" if args.resume else str(config["run"]["existing_output"])
    prepare_experiment_dir(
        output_dir,
        policy,
        output_prepared=os.environ.get("OUTPUT_PREPARED", "0") == "1",
    )
    snapshot_config(config, config_path, output_dir)
    run_manifest = output_dir / "run_manifest.json"
    if args.resume:
        if not run_manifest.is_file():
            raise FileNotFoundError(
                f"Fine-tune resume is missing its original run manifest: {run_manifest}"
            )
        print(
            "[FineTune] preserving original run manifest during resume: "
            f"{run_manifest}"
        )
    else:
        write_run_manifest(config, output_dir, graphnet_source)
    write_or_validate_source_manifest(
        source, output_dir / "finetune_source_manifest.json"
    )

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
    source_energy_manifest = EnergyWeightManifest.load(source.energy_manifest)
    if energy_manifest != source_energy_manifest:
        raise ValueError(
            "Fine-tune energy-weight manifest differs from the source experiment"
        )
    print(
        f"[Weights] exact source manifest verified; alpha={energy_manifest.alpha:g}, "
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

    train_stage(
        config,
        "stage_b",
        output_dir,
        data_representation,
        loaders,
        energy_manifest,
        initialization_checkpoint=None if args.resume else source.checkpoint,
        resume=args.resume,
    )
    print(f"[Done] Joint-direction fine-tune outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
