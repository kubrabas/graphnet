#!/usr/bin/env python3
"""Submit one routed joint-direction training job per configured route class."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
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


RUN_SCRIPT = HERE / "run_train.sh"
EXPECTED_CLASSES = ("0", "1")
EXPECTED_GRAPHNET_SOURCE = Path("/project/def-nahee/kbas/graphnet/src")
EXPECTED_TARGET = "zenith_azimuth"
ALL_FLAVORS = ("Muon", "Electron", "Tau", "NC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")
    raw = path.read_bytes()
    payload = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload, raw


def load_paths_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"paths.py does not exist: {path}")
    spec = importlib.util.spec_from_file_location("pone_joint_route_paths", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import paths module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_classes(values: Any) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("routing.classes must be 'all' or a non-empty list")
    result: list[str] = []
    for value in values:
        token = str(value)
        if token.startswith("class"):
            token = token[5:]
        if token not in EXPECTED_CLASSES:
            raise ValueError(
                f"This binary routed pipeline supports only classes 0 and 1, got {value!r}"
            )
        if token not in result:
            result.append(token)
    return sorted(result, key=int)


def discover_route_classes(config: dict[str, Any]) -> list[str]:
    requested = config["routing"].get("classes", "all")
    if requested != "all":
        return normalize_classes(requested)

    paths_py = Path(config["data"]["paths_py"]).resolve()
    module = load_paths_module(paths_py)
    table_name = str(config["data"].get("parquet_table", "STRING340MC_PARQUET"))
    table = getattr(module, table_name, None)
    if not isinstance(table, dict):
        raise KeyError(f"{table_name} is missing from {paths_py}")

    geometry = str(config["geometry"])
    category = str(config["routing"]["category"])
    geometry_entry = table.get(geometry)
    if not isinstance(geometry_entry, dict):
        raise KeyError(f"{table_name}[{geometry!r}] is missing")

    expected = set(EXPECTED_CLASSES)
    for flavor in config["flavors"]:
        route_entry = geometry_entry.get(flavor, {}).get(category)
        if not isinstance(route_entry, dict):
            raise KeyError(
                f"Missing routed data: {table_name}.{geometry}.{flavor}.{category}"
            )
        available = {
            str(value)[5:] if str(value).startswith("class") else str(value)
            for value in route_entry
        }
        if available != expected:
            raise ValueError(
                f"{geometry}.{flavor}.{category} classes={sorted(available)}; "
                f"expected {list(EXPECTED_CLASSES)}"
            )
    return list(EXPECTED_CLASSES)


def validate_config(config: dict[str, Any]) -> None:
    task = config.get("task", {})
    if task.get("type") != "reconstruction":
        raise ValueError("task.type must be reconstruction")
    if task.get("mode") != "joint_direction":
        raise ValueError("task.mode must be joint_direction")
    if task.get("targets") != [EXPECTED_TARGET]:
        raise ValueError(f"task.targets must be [{EXPECTED_TARGET}]")
    if config.get("mc") != "340StringMC":
        raise ValueError("mc must be 340StringMC")
    if config.get("flavors") != list(ALL_FLAVORS):
        raise ValueError(f"flavors must be {list(ALL_FLAVORS)}")
    if config.get("routing", {}).get("category") not in {
        "category1_isMuonCC",
        "category_3_contains_muon",
    }:
        raise ValueError("Unsupported binary routing category")
    if config.get("run", {}).get("existing_output") != "error":
        raise ValueError("run.existing_output must be error; overwrites are forbidden")
    graphnet_source = Path(config.get("environment", {}).get("graphnet_source", ""))
    if graphnet_source.resolve() != EXPECTED_GRAPHNET_SOURCE.resolve():
        raise ValueError(
            f"environment.graphnet_source must be {EXPECTED_GRAPHNET_SOURCE}"
        )
    if not (graphnet_source / "graphnet" / "__init__.py").is_file():
        raise FileNotFoundError(f"Local GraphNeT package is missing: {graphnet_source}")
    if not config.get("environment", {}).get("container_image"):
        raise ValueError("environment.container_image is required")
    if not RUN_SCRIPT.is_file():
        raise FileNotFoundError(f"SLURM worker is missing: {RUN_SCRIPT}")
    validate_experiment_extensions(config)


def output_dir(config: dict[str, Any], route_class: str) -> Path:
    return (
        Path(config["output"]["root_dir"])
        / config["mc"]
        / config["geometry"]
        / config["task"]["type"]
        / config["routing"]["category"]
        / f"class{route_class}"
        / config["experiment_name"]
        / config["output"]["dirs"]["train"]
        / EXPECTED_TARGET
    )


def short_geometry(value: str) -> str:
    return (
        value.replace("full_geometry", "fullgeo")
        .replace("102_string", "102str")
        .replace("160_string", "160str")
    )


def job_name(config: dict[str, Any], route_class: str) -> str:
    raw = (
        f"jointdir_{short_geometry(config['geometry'])}_"
        f"{config['routing']['category']}_c{route_class}_{config['experiment_name']}"
    )
    return raw[:128]


def parse_job_id(stdout: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if match is None:
        raise RuntimeError(f"Could not parse SLURM job id: {stdout!r}")
    return match.group(1)


def sbatch_command(
    config: dict[str, Any],
    frozen_config_path: Path,
    config_sha256: str,
    route_class: str,
    logfile: Path,
    reserved_output_dir: Path,
) -> list[str]:
    slurm = config["slurm"]
    environment = config["environment"]
    exports = (
        f"ALL,CONFIG={frozen_config_path},CONFIG_SHA256={config_sha256},"
        f"RESERVED_OUTPUT_DIR={reserved_output_dir},LOGFILE={logfile},"
        f"ROUTE_CLASS={route_class},"
        "STAGE=all,OUTPUT_PREPARED=1,"
        f"GRAPHNET_SRC={environment['graphnet_source']},"
        f"CONTAINER_IMAGE={environment['container_image']}"
    )
    command = [
        "sbatch",
        f"--job-name={job_name(config, route_class)}",
        f"--export={exports}",
        f"--account={slurm['account']}",
        f"--time={slurm['time']}",
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
    config_path = args.config.resolve()
    config, config_bytes = load_config(config_path)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    validate_config(config)
    route_classes = discover_route_classes(config)
    destinations = {
        route_class: output_dir(config, route_class)
        for route_class in route_classes
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Joint-direction output already exists; no jobs were submitted:\n  - "
            + "\n  - ".join(str(path) for path in existing)
        )

    print(f"config          : {config_path}")
    print(f"mc/geometry     : {config['mc']} / {config['geometry']}")
    print(f"routing category: {config['routing']['category']}")
    print(f"route classes   : {route_classes}")
    print(f"target          : {EXPECTED_TARGET}")
    print("stage           : all")
    print("existing_output : error")
    print(f"config sha256   : {config_sha256}")

    submitted: list[tuple[str, str]] = []
    for route_class in route_classes:
        destination = destinations[route_class]
        frozen_config = destination / "submitted_config.yml"
        logfile = destination / f"{job_name(config, route_class)}.out"
        command = sbatch_command(
            config,
            frozen_config,
            config_sha256,
            route_class,
            logfile,
            destination,
        )
        print(f"\n[class{route_class}] output : {destination}")
        print(f"[class{route_class}] config : {frozen_config}")
        print(f"[class{route_class}] logfile: {logfile}")
        print(" ".join(command))
        if args.dry_run:
            submitted.append((route_class, "DRYRUN"))
            continue
        # Reserve this exact task leaf before sbatch. This closes the race
        # between the existence check and the worker opening its logfile; a
        # second submitter will now fail without touching any existing result.
        destination.mkdir(parents=True, exist_ok=False)
        # Persist the exact bytes parsed above. The queued worker never reloads
        # the mutable source YAML, and verifies this digest before any write.
        with frozen_config.open("xb") as handle:
            handle.write(config_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        print(completed.stdout.strip())
        job_id = parse_job_id(completed.stdout)
        (destination / "last_train_job_id.txt").write_text(
            job_id + "\n", encoding="utf-8"
        )
        submitted.append((route_class, job_id))

    print("\nJoint-direction jobs:")
    for route_class, job_id in submitted:
        print(f"  class{route_class}: {job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
