"""Safety and provenance helpers for weights-only direction fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


COMPATIBILITY_SECTIONS = (
    "data",
    "weighting",
    "loss",
    "metrics",
    "loader",
    "model",
    "checkpointing",
    "trainer",
)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA256 digest without modifying the input file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_pinned_sha256(
    settings: Mapping[str, Any], key: str, path: Path, label: str
) -> str:
    expected = str(settings.get(key, "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(
            f"fine_tuning.{key} must contain 64 hexadecimal characters"
        )
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch; expected={expected}, actual={actual}"
        )
    return actual


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _nested_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


@dataclass(frozen=True)
class FineTuneSource:
    """Verified source model and the metadata needed for reproducibility."""

    experiment_dir: Path
    checkpoint: Path
    checkpoint_name: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    checkpoint_metric: str
    checkpoint_value: float
    resolved_config: Path
    resolved_config_sha256: str
    energy_manifest: Path
    energy_manifest_sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "initialization_mode": "weights_only",
            "network_state_dict_strict": True,
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "source_experiment_dir": str(self.experiment_dir),
            "source_checkpoint": str(self.checkpoint),
            "source_checkpoint_name": self.checkpoint_name,
            "source_checkpoint_sha256": self.checkpoint_sha256,
            "source_checkpoint_epoch": self.checkpoint_epoch,
            "source_checkpoint_metric": self.checkpoint_metric,
            "source_checkpoint_value": self.checkpoint_value,
            "source_resolved_config": str(self.resolved_config),
            "source_resolved_config_sha256": self.resolved_config_sha256,
            "source_energy_manifest": str(self.energy_manifest),
            "source_energy_manifest_sha256": self.energy_manifest_sha256,
        }


def resolve_finetune_source(
    config: Mapping[str, Any], target_output_dir: str | Path
) -> FineTuneSource:
    """Resolve and verify the immutable source checkpoint for a new experiment."""

    settings = config.get("fine_tuning")
    if not isinstance(settings, Mapping):
        raise ValueError("fine_tuning must be a mapping")
    if settings.get("initialization_mode") != "weights_only":
        raise ValueError("fine_tuning.initialization_mode must be weights_only")

    source_raw = settings.get("source_experiment_dir")
    if not source_raw:
        raise ValueError("fine_tuning.source_experiment_dir is required")
    source_dir = Path(str(source_raw)).resolve()
    target_dir = Path(target_output_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source experiment does not exist: {source_dir}")
    if _nested_within(target_dir, source_dir) or _nested_within(source_dir, target_dir):
        raise ValueError(
            "Fine-tune output and source experiment must be separate, non-nested "
            f"directories: source={source_dir}, output={target_dir}"
        )

    stage_dir_name = str(settings.get("source_stage_dir", "stage_b_angular_hybrid"))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", stage_dir_name):
        raise ValueError(f"Unsafe fine_tuning.source_stage_dir: {stage_dir_name!r}")
    checkpoint_name = str(settings.get("source_checkpoint_name", ""))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", checkpoint_name):
        raise ValueError(
            f"Unsafe fine_tuning.source_checkpoint_name: {checkpoint_name!r}"
        )

    checkpoint_dir = source_dir / stage_dir_name / "checkpoints"
    checkpoint = (checkpoint_dir / f"{checkpoint_name}.ckpt").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {checkpoint}")
    if not _nested_within(checkpoint, checkpoint_dir.resolve()):
        raise ValueError(
            f"Source checkpoint escaped its checkpoint directory: {checkpoint}"
        )

    actual_sha256 = _verify_pinned_sha256(
        settings,
        "source_checkpoint_sha256",
        checkpoint,
        "Source checkpoint",
    )

    checkpoint_index_path = checkpoint_dir / "checkpoint_index.json"
    checkpoint_index = _load_json(checkpoint_index_path)
    if checkpoint_index.get("stage") != "stage_b":
        raise ValueError(
            f"Expected a stage_b source checkpoint index: {checkpoint_index_path}"
        )
    best = checkpoint_index.get("best")
    if not isinstance(best, Mapping) or checkpoint_name not in best:
        raise KeyError(
            f"Checkpoint {checkpoint_name!r} is absent from {checkpoint_index_path}"
        )
    record = best[checkpoint_name]
    if not isinstance(record, Mapping):
        raise ValueError(f"Invalid checkpoint record for {checkpoint_name!r}")
    indexed_checkpoint = Path(str(record.get("checkpoint", ""))).resolve()
    if indexed_checkpoint != checkpoint:
        raise ValueError(
            "Checkpoint index points to a different file: "
            f"index={indexed_checkpoint}, selected={checkpoint}"
        )
    metric = str(record.get("metric", ""))
    primary_monitor = str(config["checkpointing"]["primary_monitor"])
    if metric != primary_monitor:
        raise ValueError(
            "The selected source checkpoint is not the configured primary monitor: "
            f"checkpoint metric={metric}, primary={primary_monitor}"
        )

    source_config_path = source_dir / "resolved_config.yml"
    if not source_config_path.is_file():
        raise FileNotFoundError(
            f"Source resolved config is missing: {source_config_path}"
        )
    source_config_sha256 = _verify_pinned_sha256(
        settings,
        "source_resolved_config_sha256",
        source_config_path,
        "Source resolved config",
    )
    source_config = _load_yaml_mapping(source_config_path)
    for section in COMPATIBILITY_SECTIONS:
        if config.get(section) != source_config.get(section):
            raise ValueError(
                f"Fine-tune config section {section!r} differs from the source run. "
                "Make architecture/data/loss changes in a separate ablation, not in "
                "this optimization-only continuation."
            )
    if (
        config["training"]["stage_b"].get("objective")
        != source_config["training"]["stage_b"].get("objective")
    ):
        raise ValueError("Stage-B objective differs from the source run")

    energy_manifest = source_dir / "energy_weight_manifest.json"
    if not energy_manifest.is_file():
        raise FileNotFoundError(f"Source energy manifest is missing: {energy_manifest}")
    energy_manifest_sha256 = _verify_pinned_sha256(
        settings,
        "source_energy_manifest_sha256",
        energy_manifest,
        "Source energy manifest",
    )

    return FineTuneSource(
        experiment_dir=source_dir,
        checkpoint=checkpoint,
        checkpoint_name=checkpoint_name,
        checkpoint_sha256=actual_sha256,
        checkpoint_epoch=int(record["epoch"]),
        checkpoint_metric=metric,
        checkpoint_value=float(record["value"]),
        resolved_config=source_config_path.resolve(),
        resolved_config_sha256=source_config_sha256,
        energy_manifest=energy_manifest.resolve(),
        energy_manifest_sha256=energy_manifest_sha256,
    )


def validate_submission_destination(output_dir: str | Path, *, resume: bool) -> Path:
    """Refuse overwrite, and allow resume only from the new experiment's checkpoint."""

    output_dir = Path(output_dir).resolve()
    own_last = (
        output_dir / "stage_b_angular_hybrid" / "checkpoints" / "last.ckpt"
    )
    if resume:
        if not own_last.is_file():
            raise FileNotFoundError(
                f"--resume requested but fine-tune checkpoint is missing: {own_last}"
            )
    elif output_dir.exists():
        raise FileExistsError(
            f"Fine-tune output path already exists: {output_dir}. Nothing was "
            "deleted or overwritten. Use --resume only for a genuinely interrupted "
            "run."
        )
    return own_last


def write_or_validate_source_manifest(
    source: FineTuneSource, destination: str | Path
) -> None:
    """Write source provenance once, or require an exact match during resume."""

    destination = Path(destination)
    expected = source.manifest()
    if destination.exists():
        existing = _load_json(destination)
        if existing != expected:
            raise ValueError(
                f"Existing fine-tune source manifest differs: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, destination)
