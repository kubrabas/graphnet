#!/usr/bin/env python3
"""Submit an isolated one-step CUDA memory probe; never reserve train output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyYAML is required on the login environment. Deactivate the venv or "
        "load a scipy-stack module before submitting."
    ) from exc


HERE = Path(__file__).resolve().parent
JOINT_DIR = HERE.parent
if str(JOINT_DIR) not in sys.path:
    sys.path.insert(0, str(JOINT_DIR))

from experiment_config import validate_experiment_extensions  # noqa: E402


RUN_SCRIPT = HERE / "run_gpu_memory_probe.sh"
EXPECTED_GRAPHNET_SOURCE = Path("/project/def-nahee/kbas/graphnet/src")
EXPECTED_GEOMETRY = "102_string_emax1e6"
EXPECTED_CATEGORY = "category1_isMuonCC"


def configured_model_name(config: dict[str, Any]) -> str:
    return str(config.get("model", {}).get("name", "dynedge"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument(
        "--time",
        default="01:00:00",
        help="SLURM walltime for the four-microbatch probe (default: 01:00:00)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config, raw


def _route_one_is_enabled(config: dict[str, Any]) -> bool:
    classes = config.get("routing", {}).get("classes", "all")
    if classes == "all":
        return True
    if not isinstance(classes, list):
        return False
    return "1" in {str(value).removeprefix("class") for value in classes}


def validate_config(config: dict[str, Any]) -> None:
    validate_experiment_extensions(config)
    experiment_name = str(config.get("experiment_name", ""))
    if (
        not experiment_name
        or experiment_name in {".", ".."}
        or Path(experiment_name).is_absolute()
        or len(Path(experiment_name).parts) != 1
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_name)
    ):
        raise ValueError(
            "experiment_name must be one safe, non-empty relative path component"
        )
    if config.get("geometry") != EXPECTED_GEOMETRY:
        raise ValueError(f"Probe requires geometry={EXPECTED_GEOMETRY}")
    if config.get("routing", {}).get("category") != EXPECTED_CATEGORY:
        raise ValueError(f"Probe requires routing.category={EXPECTED_CATEGORY}")
    if not _route_one_is_enabled(config):
        raise ValueError("Probe config must enable route class1")
    loader = config.get("loader", {})
    batch_size = int(loader.get("batch_size", -1))
    accumulation = int(loader.get("accumulate_grad_batches", -1))
    precision = str(config.get("trainer", {}).get("precision"))
    if batch_size <= 0 or accumulation <= 0:
        raise ValueError("Probe requires positive batch size and accumulation")
    model_name = configured_model_name(config)
    if model_name == "dynedge" and (
        batch_size != 256 or accumulation != 4 or precision != "32-true"
    ):
        raise ValueError(
            "DynEdge probe requires batch=256, accumulation=4, precision=32-true"
        )
    if model_name == "fourier_spacetime_transformer":
        if precision != "bf16-mixed":
            raise ValueError("Transformer probe requires bf16-mixed precision")
    elif model_name != "dynedge":
        raise ValueError(f"Unsupported probe model.name={model_name!r}")
    source = Path(config.get("environment", {}).get("graphnet_source", ""))
    if source.resolve() != EXPECTED_GRAPHNET_SOURCE.resolve():
        raise ValueError(
            f"environment.graphnet_source must be {EXPECTED_GRAPHNET_SOURCE}"
        )
    if not (source / "graphnet" / "__init__.py").is_file():
        raise FileNotFoundError(f"Local GraphNeT package is missing: {source}")
    if not config.get("environment", {}).get("container_image"):
        raise ValueError("environment.container_image is required")
    if not RUN_SCRIPT.is_file():
        raise FileNotFoundError(f"SLURM worker is missing: {RUN_SCRIPT}")


def diagnostic_output(config: dict[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    unique = uuid.uuid4().hex[:8]
    return (
        Path(config["output"]["root_dir"])
        / config["mc"]
        / "diagnostics"
        / "joint_direction_gpu_memory_probe"
        / config["geometry"]
        / config["routing"]["category"]
        / str(config["experiment_name"])
        / f"{stamp}_{unique}"
    )


def job_name(config: dict[str, Any]) -> str:
    safe_experiment = re.sub(r"[^A-Za-z0-9_-]+", "_", str(config["experiment_name"]))
    return f"jointmem_{safe_experiment}"[:128]


def sbatch_command(
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    output_dir: Path,
    logfile: Path,
    walltime: str,
) -> list[str]:
    environment = config["environment"]
    slurm = config["slurm"]
    exports = (
        f"ALL,CONFIG={config_path},CONFIG_SHA256={config_sha256},"
        f"OUTPUT_DIR={output_dir},LOGFILE={logfile},"
        f"GRAPHNET_SRC={environment['graphnet_source']},"
        f"CONTAINER_IMAGE={environment['container_image']}"
    )
    command = [
        "sbatch",
        f"--job-name={job_name(config)}",
        f"--export={exports}",
        f"--account={slurm['account']}",
        f"--time={walltime}",
        f"--mem={slurm['mem']}",
        f"--cpus-per-task={slurm['cpus_per_task']}",
        f"--gpus-per-node={slurm['gpus_per_node']}",
    ]
    if slurm.get("exclude"):
        command.append(f"--exclude={slurm['exclude']}")
    command.append(str(RUN_SCRIPT))
    return command


def main() -> int:
    args = parse_args()
    source_config = args.config.resolve()
    config, config_bytes = load_config(source_config)
    validate_config(config)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    output_dir = diagnostic_output(config)
    frozen_config = output_dir / "submitted_config.yml"
    logfile = output_dir / "gpu_memory_probe.out"
    command = sbatch_command(
        config,
        frozen_config,
        config_sha256,
        output_dir,
        logfile,
        args.time,
    )

    print(f"source config    : {source_config}")
    print(f"config sha256   : {config_sha256}")
    print(f"diagnostic only : {output_dir}")
    print("training output : not reserved or modified")
    loader = config["loader"]
    print(
        "probe workload  : class1 train, "
        f"batch {loader['batch_size']} x min({loader['accumulate_grad_batches']},4), "
        f"{config['trainer']['precision']}, one optimizer step"
    )
    print(" ".join(command))
    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=False)
    with frozen_config.open("xb") as handle:
        handle.write(config_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as error:
        details = (
            f"command: {' '.join(command)}\n"
            f"returncode: {error.returncode}\n"
            f"stdout:\n{error.stdout or ''}\n"
            f"stderr:\n{error.stderr or ''}\n"
        )
        (output_dir / "submission_error.txt").write_text(
            details, encoding="utf-8"
        )
        raise RuntimeError(
            "GPU probe submission failed; see "
            f"{output_dir / 'submission_error.txt'}"
        ) from error
    print(completed.stdout.strip())
    match = re.search(r"Submitted batch job (\d+)", completed.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse SLURM job id: {completed.stdout!r}")
    (output_dir / "job_id.txt").write_text(match.group(1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
