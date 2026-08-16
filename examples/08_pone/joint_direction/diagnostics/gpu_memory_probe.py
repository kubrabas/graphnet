#!/usr/bin/env python3
"""Measure real routed joint-direction training memory on one visible CUDA GPU.

This is deliberately a diagnostic, not a shortened training run.  It resolves
the real 102-string ``category1_isMuonCC/class1`` train parquet views, builds
the same data representation and model as production, profiles up to four
full-size microbatches, and performs one optimizer step so optimizer state
memory is included. The production accumulation factor is still recorded;
profiling more than four microbatches does not increase retained activation,
gradient, or optimizer-state memory.

All writes are confined to the explicitly supplied diagnostic directory.  No
training output directory is prepared or reserved, and source parquet files
remain read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import socket
import sys
import traceback
from typing import Any, Mapping


THIS_DIR = Path(__file__).resolve().parent
JOINT_DIR = THIS_DIR.parent
PONE_DIR = JOINT_DIR.parent
REFERENCE_DIR = PONE_DIR.parent / "09_pone_muon_direction"
for path in (PONE_DIR, REFERENCE_DIR, JOINT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data

from model_factory import (
    build_direction_model,
    build_model_data_representation,
    configured_model_name,
)
from routed_data import (
    build_loaders,
    fit_or_load_energy_manifest,
)
from routed_pipeline_utils import (
    assert_local_graphnet_source,
    atomic_json_dump,
    load_yaml,
    resolve_routed_split_paths,
    target_dir,
)


ROUTE_CLASS = "1"
EXPECTED_GEOMETRY = "102_string_emax1e6"
EXPECTED_CATEGORY = "category1_isMuonCC"
LEGACY_BATCH_SIZE = 256
LEGACY_ACCUMULATION = 4
LEGACY_PRECISION = "32-true"
MAX_PROFILED_MICROBATCHES = 4
DIAGNOSTIC_PATH_COMPONENTS = ("diagnostics", "joint_direction_gpu_memory_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_one_is_enabled(config: Mapping[str, Any]) -> bool:
    classes = config.get("routing", {}).get("classes", "all")
    if classes == "all":
        return True
    if not isinstance(classes, list):
        return False
    normalized = {str(value).removeprefix("class") for value in classes}
    return ROUTE_CLASS in normalized


def validate_probe_contract(config: Mapping[str, Any], output_dir: Path) -> None:
    """Reject a probe that would no longer represent the approved comparison."""

    if config.get("geometry") != EXPECTED_GEOMETRY:
        raise ValueError(f"Probe requires geometry={EXPECTED_GEOMETRY}")
    if config.get("routing", {}).get("category") != EXPECTED_CATEGORY:
        raise ValueError(f"Probe requires routing.category={EXPECTED_CATEGORY}")
    if not _route_one_is_enabled(config):
        raise ValueError("Probe config must enable routing class1")
    loader = config.get("loader", {})
    model_name = configured_model_name(config)
    batch_size = int(loader.get("batch_size", -1))
    accumulation = int(loader.get("accumulate_grad_batches", -1))
    precision = str(config.get("trainer", {}).get("precision"))
    if batch_size <= 0 or accumulation <= 0:
        raise ValueError("Probe requires positive batch size and accumulation")
    if model_name == "dynedge" and (
        batch_size != LEGACY_BATCH_SIZE
        or accumulation != LEGACY_ACCUMULATION
        or precision != LEGACY_PRECISION
    ):
        raise ValueError(
            "DynEdge probe requires the historical batch=256, accumulation=4, "
            "precision=32-true contract"
        )
    if model_name != "dynedge" and precision != "bf16-mixed":
        raise ValueError("Transformer probe requires bf16-mixed precision")
    if int(config.get("trainer", {}).get("devices", 0)) != 1:
        raise ValueError("Probe requires exactly one visible training device")

    resolved_output = output_dir.resolve()
    parts = resolved_output.parts
    required = DIAGNOSTIC_PATH_COMPONENTS
    if not any(
        tuple(parts[index : index + len(required)]) == required
        for index in range(len(parts) - len(required) + 1)
    ):
        raise ValueError(
            "--output-dir must be inside diagnostics/"
            "joint_direction_gpu_memory_probe"
        )
    production_target = target_dir(config, ROUTE_CLASS).resolve()
    if resolved_output == production_target or production_target in resolved_output.parents:
        raise ValueError("Diagnostic output cannot be inside a training result leaf")


def _cuda_batch(raw_batch: Any, device: torch.device) -> list[Data]:
    if isinstance(raw_batch, Data):
        items = [raw_batch]
    elif isinstance(raw_batch, (list, tuple)) and raw_batch:
        items = list(raw_batch)
    else:
        raise TypeError(f"Unsupported train batch type: {type(raw_batch)!r}")
    if not all(isinstance(item, Data) for item in items):
        raise TypeError("Every train batch item must be torch_geometric.data.Data")
    return [item.to(device, non_blocking=True) for item in items]


def _batch_counts(batch: list[Data]) -> tuple[int, int]:
    events = 0
    nodes = 0
    for graph in batch:
        nodes += int(graph.x.shape[0])
        if hasattr(graph, "num_graphs"):
            events += int(graph.num_graphs)
        else:
            events += int(graph["totalEnergy"].reshape(-1).shape[0])
    return events, nodes


def _memory_snapshot(device: torch.device, total_bytes: int) -> dict[str, Any]:
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    free_bytes, visible_total = torch.cuda.mem_get_info(device)
    if int(visible_total) != total_bytes:
        raise RuntimeError(
            "CUDA visible-total disagreement: device properties report "
            f"{total_bytes}, mem_get_info reports {visible_total}"
        )
    return {
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "free_bytes": int(free_bytes),
        "allocated_fraction_of_visible_total": allocated / total_bytes,
        "reserved_fraction_of_visible_total": reserved / total_bytes,
        "peak_allocated_fraction_of_visible_total": peak_allocated / total_bytes,
        "peak_reserved_fraction_of_visible_total": peak_reserved / total_bytes,
    }


def _parameter_statistics(model: torch.nn.Module) -> dict[str, int]:
    all_parameters = list(model.parameters())
    trainable = [parameter for parameter in all_parameters if parameter.requires_grad]
    return {
        "total_parameters": sum(parameter.numel() for parameter in all_parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in all_parameters
        ),
        "trainable_parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in trainable
        ),
    }


def run_probe(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The GPU memory probe requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA device, got {torch.cuda.device_count()}"
        )

    assert_local_graphnet_source(config)
    torch.set_float32_matmul_precision(
        str(config.get("trainer", {}).get("float32_matmul_precision", "high"))
    )
    pl.seed_everything(int(config["training"]["seed"]), workers=True)

    split_paths, percentiles_csv = resolve_routed_split_paths(config, ROUTE_CLASS)
    manifest = fit_or_load_energy_manifest(
        config,
        split_paths["train"],
        output_dir / "energy_weight_manifest.json",
    )
    data_representation = build_model_data_representation(config, percentiles_csv)
    loaders = build_loaders(
        config,
        split_paths,
        data_representation,
        splits=("train",),
    )
    train_loader = loaders["train"]
    accumulation = int(config["loader"]["accumulate_grad_batches"])
    optimizer_steps = math.ceil(len(train_loader) / accumulation)
    model = build_direction_model(
        config,
        "stage_b",
        data_representation,
        manifest,
        steps_per_optimizer_epoch=optimizer_steps,
    )

    device = torch.device("cuda", 0)
    model = model.float().to(device)
    model.train()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Probe model contains non-float32 parameters")
    model_name = configured_model_name(config)
    if model_name == "dynedge":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["training"]["stage_b"]["base_lr"])
        )
    else:
        optimizer_config = config["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["stage_b"]["base_lr"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            eps=float(optimizer_config["eps"]),
        )

    properties = torch.cuda.get_device_properties(device)
    total_bytes = int(properties.total_memory)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    before = _memory_snapshot(device, total_bytes)

    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    microbatches: list[dict[str, Any]] = []
    profiled_microbatches = min(accumulation, MAX_PROFILED_MICROBATCHES)
    use_bf16 = str(config["trainer"]["precision"]) == "bf16-mixed"
    for microbatch_index in range(profiled_microbatches):
        batch = _cuda_batch(next(iterator), device)
        events, nodes = _batch_counts(batch)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
        ):
            predictions = model(batch)
            loss = model.compute_loss(predictions, batch)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Microbatch loss is invalid: {loss}")
        (loss / accumulation).backward()
        torch.cuda.synchronize(device)
        snapshot = _memory_snapshot(device, total_bytes)
        microbatches.append(
            {
                "index": microbatch_index + 1,
                "events": events,
                "nodes": nodes,
                "unscaled_loss": float(loss.detach().cpu()),
                **snapshot,
            }
        )
        del predictions, loss, batch

    optimizer.step()
    torch.cuda.synchronize(device)
    after_step = _memory_snapshot(device, total_bytes)
    parameters = _parameter_statistics(model)
    peak_reserved_fraction = after_step["peak_reserved_fraction_of_visible_total"]

    return {
        "status": "success",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "probe_contract": {
            "geometry": EXPECTED_GEOMETRY,
            "routing_category": EXPECTED_CATEGORY,
            "routing_class": int(ROUTE_CLASS),
            "split": "train",
            "stage": "stage_b",
            "precision": str(config["trainer"]["precision"]),
            "microbatch_size": int(config["loader"]["batch_size"]),
            "gradient_accumulation": accumulation,
            "profiled_microbatches_before_step": profiled_microbatches,
            "effective_batch_size": (
                int(config["loader"]["batch_size"]) * accumulation
            ),
            "optimizer": optimizer.__class__.__name__,
            "model_name": model_name,
            "optimizer_steps_completed": 1,
            "test_loader_created": False,
            "training_result_reserved": False,
            "source_parquet_modified": False,
        },
        "cuda": {
            "visible_device_count": torch.cuda.device_count(),
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "visible_total_bytes": total_bytes,
            "visible_total_gib": total_bytes / (1024**3),
            "before": before,
            "after_optimizer_step": after_step,
            "peak_reserved_below_85_percent": peak_reserved_fraction < 0.85,
        },
        "model": parameters,
        "loader": {
            "train_batches": len(train_loader),
            "microbatches_profiled": len(microbatches),
            "microbatches": microbatches,
        },
    }


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    config = load_yaml(config_path)
    validate_probe_contract(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gpu_memory_report.json"
    if report_path.exists():
        raise FileExistsError(f"Probe report already exists: {report_path}")
    submitted_hash = os.environ.get("CONFIG_SHA256")
    actual_hash = _sha256(config_path)
    if submitted_hash and submitted_hash != actual_hash:
        raise RuntimeError(
            f"Config SHA256 mismatch: expected {submitted_hash}, got {actual_hash}"
        )

    base_report: dict[str, Any] = {
        "config": str(config_path),
        "config_sha256": actual_hash,
        "experiment_name": str(config.get("experiment_name")),
        "diagnostic_output": str(output_dir),
    }
    try:
        report = {**base_report, **run_probe(config, output_dir)}
    except Exception as error:
        failure: dict[str, Any] = {
            **base_report,
            "status": (
                "cuda_out_of_memory"
                if isinstance(error, torch.cuda.OutOfMemoryError)
                else "error"
            ),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            device = torch.device("cuda", 0)
            total = int(torch.cuda.get_device_properties(device).total_memory)
            failure["cuda_memory_at_failure"] = _memory_snapshot(device, total)
        atomic_json_dump(failure, report_path)
        print(f"[Probe] FAILED; report: {report_path}", file=sys.stderr)
        raise

    atomic_json_dump(report, report_path)
    cuda = report["cuda"]
    peak = cuda["after_optimizer_step"]
    print(f"[Probe] report: {report_path}")
    print(f"[Probe] device: {cuda['device_name']}")
    print(
        "[Probe] visible total: "
        f"{cuda['visible_total_gib']:.2f} GiB | "
        "peak allocated: "
        f"{peak['peak_allocated_bytes'] / (1024**3):.2f} GiB "
        f"({peak['peak_allocated_fraction_of_visible_total']:.1%}) | "
        "peak reserved: "
        f"{peak['peak_reserved_bytes'] / (1024**3):.2f} GiB "
        f"({peak['peak_reserved_fraction_of_visible_total']:.1%})"
    )
    print(
        "[Probe] trainable parameters: "
        f"{report['model']['trainable_parameters']:,} | "
        "one Adam step after four accumulated microbatches: complete"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
