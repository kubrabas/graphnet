"""CSV logging, resource telemetry, and named multi-metric checkpoints."""

from __future__ import annotations

import csv
import math
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from pytorch_lightning.callbacks import Callback

from pipeline_utils import atomic_json_dump, numeric_metrics


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rewrite_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for preferred in ("stage", "epoch"):
        if any(preferred in row for row in rows):
            fields.append(preferred)
    fields.extend(
        sorted({str(key) for row in rows for key in row if str(key) not in fields})
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class EpochMetricsCSV(Callback):
    """Persist every scalar epoch metric in a single human-readable CSV."""

    def __init__(self, stage_dir: str | Path, stage_name: str) -> None:
        self.stage_dir = Path(stage_dir)
        self.stage_name = stage_name
        self.path = self.stage_dir / "training_history_by_epoch.csv"
        self.started_at = 0.0
        self.epoch_started_at = 0.0

    def on_fit_start(self, trainer, pl_module) -> None:
        self.started_at = time.monotonic()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_started_at = time.monotonic()

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        values = numeric_metrics(trainer.callback_metrics)
        values.update(
            {
                "stage": self.stage_name,
                "epoch": int(trainer.current_epoch),
                "elapsed_min": (time.monotonic() - self.started_at) / 60.0,
                "epoch_duration_min": (
                    time.monotonic() - self.epoch_started_at
                )
                / 60.0,
                "learning_rate": float(
                    trainer.optimizers[0].param_groups[0]["lr"]
                ),
            }
        )
        rows = _read_csv(self.path)
        rows = [
            row
            for row in rows
            if not (
                row.get("stage") == self.stage_name
                and int(float(row.get("epoch", -1))) == trainer.current_epoch
            )
        ]
        rows.append(values)
        _rewrite_csv(self.path, rows)


class EnergyBinMetricsCSV(Callback):
    """Save per-energy-bin resolution metrics for every validation epoch."""

    def __init__(self, stage_dir: str | Path, stage_name: str) -> None:
        self.path = Path(stage_dir) / "validation_metrics_by_energy_epoch.csv"
        self.stage_name = stage_name

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        current = getattr(pl_module, "latest_validation_energy_rows", [])
        if not current:
            return
        rows = _read_csv(self.path)
        rows = [
            row
            for row in rows
            if not (
                row.get("stage") == self.stage_name
                and int(float(row.get("epoch", -1))) == trainer.current_epoch
            )
        ]
        for row in current:
            rows.append(
                {
                    "stage": self.stage_name,
                    "epoch": int(trainer.current_epoch),
                    **row,
                }
            )
        _rewrite_csv(self.path, rows)


def _gpu_snapshot() -> Dict[str, float]:
    result = {
        "gpu_util_pct": float("nan"),
        "gpu_memory_used_gb": float("nan"),
        "gpu_memory_total_gb": float("nan"),
    }
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        if visible and visible.isdigit():
            command[1:1] = ["-i", visible]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10
        )
        first = completed.stdout.strip().splitlines()[0]
        utilization, used, total = [float(value.strip()) for value in first.split(",")]
        result.update(
            {
                "gpu_util_pct": utilization,
                "gpu_memory_used_gb": used / 1024.0,
                "gpu_memory_total_gb": total / 1024.0,
            }
        )
    except Exception:
        pass
    return result


class ResourceCSV(Callback):
    """Record process and GPU snapshots; shell telemetry covers training peaks."""

    def __init__(self, stage_dir: str | Path, stage_name: str) -> None:
        self.path = Path(stage_dir) / "resources_and_time.csv"
        self.stage_name = stage_name

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        rss_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux ru_maxrss is KiB.
        rss_gb /= 1024.0 * 1024.0
        row: Dict[str, Any] = {
            "stage": self.stage_name,
            "epoch": int(trainer.current_epoch),
            "max_parent_rss_gb": rss_gb,
            "cpu_load_1min": float(os.getloadavg()[0]),
            **_gpu_snapshot(),
        }
        rows = _read_csv(self.path)
        rows.append(row)
        _rewrite_csv(self.path, rows)


class NamedCheckpoint(Callback):
    """Save last plus one resumable/raw checkpoint for every chosen monitor."""

    def __init__(
        self,
        stage_dir: str | Path,
        stage_name: str,
        monitors: Mapping[str, str],
        *,
        save_every_epoch: bool = False,
    ) -> None:
        self.stage_dir = Path(stage_dir)
        self.checkpoint_dir = self.stage_dir / "checkpoints"
        self.stage_name = stage_name
        self.monitors = dict(monitors)
        self.save_every_epoch = bool(save_every_epoch)
        self.best: Dict[str, Dict[str, Any]] = {}

    def setup(self, trainer, pl_module, stage: str | None = None) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def state_dict(self) -> Dict[str, Any]:
        return {"best": self.best}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.best = dict(state_dict.get("best", {}))

    def _save(self, trainer, pl_module, stem: str) -> Dict[str, str]:
        checkpoint = self.checkpoint_dir / f"{stem}.ckpt"
        raw = self.checkpoint_dir / f"{stem}.pth"
        temporary_checkpoint = checkpoint.with_suffix(".ckpt.tmp")
        temporary_raw = raw.with_suffix(".pth.tmp")
        trainer.save_checkpoint(temporary_checkpoint, weights_only=False)
        torch.save(pl_module.state_dict(), temporary_raw)
        os.replace(temporary_checkpoint, checkpoint)
        os.replace(temporary_raw, raw)
        return {"checkpoint": str(checkpoint), "raw_state_dict": str(raw)}

    def _write_index(self, trainer) -> None:
        atomic_json_dump(
            {
                "stage": self.stage_name,
                "last_epoch": int(trainer.current_epoch),
                "monitors": self.monitors,
                "best": self.best,
                "last": {
                    "checkpoint": str(self.checkpoint_dir / "last.ckpt"),
                    "raw_state_dict": str(self.checkpoint_dir / "last.pth"),
                },
            },
            self.checkpoint_dir / "checkpoint_index.json",
        )

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        values = numeric_metrics(trainer.callback_metrics)
        improved: list[str] = []
        for stem, metric_name in self.monitors.items():
            if metric_name not in values:
                raise KeyError(
                    f"Checkpoint monitor {metric_name!r} was not logged; "
                    f"available={sorted(values)}"
                )
            value = float(values[metric_name])
            if not math.isfinite(value):
                raise ValueError(f"Checkpoint monitor {metric_name} is not finite")
            previous = self.best.get(stem)
            if previous is None or value < float(previous["value"]):
                paths = {
                    "checkpoint": str(self.checkpoint_dir / f"{stem}.ckpt"),
                    "raw_state_dict": str(self.checkpoint_dir / f"{stem}.pth"),
                }
                self.best[stem] = {
                    "metric": metric_name,
                    "value": value,
                    "epoch": int(trainer.current_epoch),
                    **paths,
                }
                improved.append(stem)

        # Update the in-memory best table before saving.  Consequently the
        # callback state embedded in both best and last Lightning checkpoints
        # is current and a --resume run cannot forget earlier best values.
        for stem in improved:
            self._save(trainer, pl_module, stem)
            record = self.best[stem]
            metric_name = record["metric"]
            value = record["value"]
            print(
                f"[Checkpoint:{self.stage_name}] {stem}: "
                f"{metric_name}={value:.6g} at epoch {trainer.current_epoch}"
            )
        if self.save_every_epoch:
            epoch_dir = self.checkpoint_dir / "epochs"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            stem = f"epochs/epoch_{trainer.current_epoch:03d}"
            self._save(trainer, pl_module, stem)
        self._save(trainer, pl_module, "last")
        self._write_index(trainer)
