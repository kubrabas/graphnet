#!/usr/bin/env python3
"""Submit one joint-direction training job without deleting existing output."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
RUN_SCRIPT = HERE / "run_train.sh"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("all", "stage_a", "stage_b"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = (
        Path(config["output"]["root_dir"])
        / config["data"]["geometry"]
        / config["experiment_name"]
    )
    if output.exists() and any(output.iterdir()) and not args.resume:
        stage_b_is_new = (
            args.stage == "stage_b"
            and not (output / "stage_b_angular_hybrid").exists()
            and (output / "stage_a_vmf" / "checkpoints" / "last.ckpt").is_file()
        )
        if not stage_b_is_new:
            raise FileExistsError(
                f"Output exists: {output}. Use a new experiment_name or --resume; "
                "nothing was deleted."
            )
    if args.resume and args.stage == "all":
        raise ValueError("--resume requires --stage stage_a or --stage stage_b")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    logfile = output / f"train_{args.stage}.out"
    slurm = config["slurm"]
    environment = config["environment"]
    resume_flag = "--resume" if args.resume else ""
    export = (
        f"ALL,CONFIG={config_path},LOGFILE={logfile},STAGE={args.stage},"
        f"RESUME_FLAG={resume_flag},OUTPUT_PREPARED=1,"
        f"GRAPHNET_SRC={environment['graphnet_source']},"
        f"CONTAINER_IMAGE={environment['container_image']}"
    )
    job_name = f"jointdir_{config['data']['geometry']}_{config['experiment_name']}"[:128]
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
    (output / "last_train_job_id.txt").write_text(match.group(1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
