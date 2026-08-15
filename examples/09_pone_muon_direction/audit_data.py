#!/usr/bin/env python3
"""Read-only preflight audit for Muon-CC parquet splits and schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data import deep_data_audit, write_data_audit
from pipeline_utils import (
    assert_local_graphnet_source,
    load_yaml,
    resolve_muon_split_paths,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional JSON destination; source parquet files are never modified",
    )
    parser.add_argument(
        "--include-test-targets",
        action="store_true",
        help="Also inspect test zenith/azimuth/energy values (off by default)",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    assert_local_graphnet_source(config)
    paths = resolve_muon_split_paths(config)
    target_splits = (
        ("train", "val", "test")
        if args.include_test_targets
        else ("train", "val")
    )
    report = deep_data_audit(config, paths, target_value_splits=target_splits)
    if args.output is not None:
        write_data_audit(report, args.output)
        print(f"Audit report: {args.output}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
