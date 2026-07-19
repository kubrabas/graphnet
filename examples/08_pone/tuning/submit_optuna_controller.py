#!/usr/bin/env python3
"""Submit one resumable CPU Optuna controller job."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_optuna_controller.sh"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)

    campaign_dir = Path(cfg["campaign_dir"]).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    slurm = cfg.get("controller_slurm", {})
    job_name = f"optuna_{cfg['study_name']}"[:128]
    logfile = campaign_dir / "controller_%j.out"

    command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--export=STUDY_CONFIG={config_path}",
        f"--output={logfile}",
        f"--error={logfile}",
        f"--account={slurm.get('account', 'def-nahee')}",
        f"--time={slurm.get('time', '7-00:00:00')}",
        f"--mem={slurm.get('mem', '4G')}",
        f"--cpus-per-task={slurm.get('cpus_per_task', 1)}",
        str(RUNNER),
    ]

    print(" ".join(command))
    if args.dry_run:
        return 0

    result = subprocess.run(command, text=True, capture_output=True, check=True)
    print(result.stdout.strip())
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse controller job id: {result.stdout!r}")
    job_id = match.group(1)
    (campaign_dir / "last_controller_job_id.txt").write_text(job_id + "\n")
    print(f"controller job: {job_id}")
    print(f"controller log: {str(logfile).replace('%j', job_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
