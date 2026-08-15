#!/usr/bin/env python3
"""Submit a pinned, weights-only Stage-B fine-tuning job safely."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

import yaml


HERE = Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from finetune_utils import (  # noqa: E402
    resolve_finetune_source,
    validate_submission_destination,
)


RUN_SCRIPT = HERE / "run_finetune.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this fine-tune experiment from its own last.ckpt",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    output = (
        Path(config["output"]["root_dir"])
        / str(config["data"]["geometry"])
        / str(config["experiment_name"])
    ).resolve()
    source = resolve_finetune_source(config, output)
    validate_submission_destination(output, resume=args.resume)

    print(
        f"source_checkpoint={source.checkpoint} "
        f"sha256={source.checkpoint_sha256}"
    )
    print(f"target_output={output}")
    slurm = config["slurm"]
    environment = config["environment"]
    resume_flag = "--resume" if args.resume else ""
    logfile = output / "train_finetune.out"
    export = (
        f"ALL,CONFIG={config_path},LOGFILE={logfile},"
        f"RESUME_FLAG={resume_flag},OUTPUT_PREPARED=1,"
        f"GRAPHNET_SRC={environment['graphnet_source']},"
        f"CONTAINER_IMAGE={environment['container_image']}"
    )
    job_name = (
        f"jointft_{config['data']['geometry']}_{config['experiment_name']}"[:128]
    )
    command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--export={export}",
        f"--account={slurm['account']}",
        f"--time={slurm['time']}",
        f"--mem={slurm['mem']}",
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
    match = re.search(r"Submitted batch job (\d+)", completed.stdout)
    if not match:
        raise RuntimeError(f"Could not parse SLURM job id: {completed.stdout!r}")
    # The job wrapper also creates this directory.  Creating it only after a
    # successful sbatch avoids leaving a misleading empty run after rejection.
    output.mkdir(parents=True, exist_ok=True)
    (output / "last_finetune_job_id.txt").write_text(
        match.group(1) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
