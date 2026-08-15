#!/usr/bin/env python3
"""Freeze one validation-selected checkpoint before opening the test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pipeline_utils import atomic_json_dump, experiment_dir, load_yaml


STAGE_DIRS = {
    "stage_a": "stage_a_vmf",
    "stage_b": "stage_b_angular_hybrid",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("stage_a", "stage_b"), default="stage_b")
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument(
        "--reason",
        required=True,
        help="Short validation-only reason for this irreversible test choice",
    )
    args = parser.parse_args()
    if not args.reason.strip():
        raise ValueError("--reason cannot be empty")

    config = load_yaml(args.config)
    root = experiment_dir(config)
    selection_path = root / "checkpoint_selection.json"
    if selection_path.exists():
        raise FileExistsError(
            f"Checkpoint selection is already frozen: {selection_path}. "
            "It was not changed."
        )
    test_root = root / "inference" / "test"
    prior_test_files = (
        [path for path in test_root.rglob("*") if path.is_file()]
        if test_root.exists()
        else []
    )
    if prior_test_files:
        raise FileExistsError(
            "Cannot claim a validation-only freeze because test artifacts already "
            f"exist below {test_root}. First file: {prior_test_files[0]}"
        )
    checkpoint = (
        root
        / STAGE_DIRS[args.stage]
        / "checkpoints"
        / f"{args.checkpoint_name}.ckpt"
    )
    validation_metrics = (
        root
        / "inference"
        / "val"
        / f"{args.stage}_{args.checkpoint_name}"
        / "metrics_summary.csv"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not validation_metrics.is_file():
        raise FileNotFoundError(
            "Run validation inference for the selected checkpoint before freezing: "
            f"{validation_metrics}"
        )
    with validation_metrics.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one validation summary row: {validation_metrics}")
    metrics = {}
    for key, value in rows[0].items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue

    atomic_json_dump(
        {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": args.stage,
            "checkpoint_name": args.checkpoint_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "selection_basis": "validation_only",
            "reason": args.reason.strip(),
            "validation_metrics_file": str(validation_metrics),
            "validation_metrics": metrics,
            "test_metrics_seen_before_selection": False,
            "preexisting_test_artifacts_verified_absent": True,
        },
        selection_path,
    )
    print(f"Frozen checkpoint selection: {selection_path}")
    print(f"Selected: {args.stage}/{args.checkpoint_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
