#!/usr/bin/env python3
"""Submit direct validation/test inference for one named checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
RUN_SCRIPT = HERE / "run_inference.sh"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--stage", choices=("stage_a", "stage_b"), default="stage_b")
    parser.add_argument("--checkpoint-name", default="best_macro_median")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = (
        Path(config["output"]["root_dir"])
        / config["data"]["geometry"]
        / config["experiment_name"]
    )
    output = root / "inference" / args.split / f"{args.stage}_{args.checkpoint_name}"
    if args.split == "test":
        selection_path = root / "checkpoint_selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(
                "Test inference is locked until one validation checkpoint is "
                f"frozen with freeze_checkpoint.py: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (
            selection.get("stage"),
            selection.get("checkpoint_name"),
        ) != (args.stage, args.checkpoint_name):
            raise ValueError(
                "Requested test checkpoint differs from frozen validation selection"
            )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Inference output exists: {output}")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    logfile = output / "inference.out"
    slurm = config["slurm"]
    environment = config["environment"]
    export = (
        f"ALL,CONFIG={config_path},LOGFILE={logfile},SPLIT={args.split},"
        f"STAGE={args.stage},CHECKPOINT_NAME={args.checkpoint_name},OUTPUT_PREPARED=1,"
        f"GRAPHNET_SRC={environment['graphnet_source']},"
        f"CONTAINER_IMAGE={environment['container_image']}"
    )
    command = [
        "sbatch",
        f"--job-name=jointdir_infer_{config['data']['geometry']}"[:128],
        f"--export={export}",
        f"--account={slurm['account']}",
        f"--time={slurm['inference_time']}",
        f"--mem={slurm['inference_mem']}",
        f"--cpus-per-task={slurm['cpus_per_task']}",
        f"--gpus-per-node={slurm['gpus_per_node']}",
    ]
    if slurm.get("exclude"):
        command.append(f"--exclude={slurm['exclude']}")
    command.append(str(RUN_SCRIPT))
    print(" ".join(command))
    if args.dry_run:
        return 0
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    print(completed.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
