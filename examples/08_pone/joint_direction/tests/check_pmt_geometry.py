#!/usr/bin/env python3
"""Compare the v3 PMT lookup directions with one real parquet pulse shard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
JOINT_DIR = THIS_DIR.parent
PONE_DIR = JOINT_DIR.parent
for path in (PONE_DIR, JOINT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pyarrow.parquet as pq

from pmt_direction_features import PONE_V3_PMT_DIRECTIONS
from routed_pipeline_utils import (
    load_yaml,
    resolve_routed_split_paths,
    table_files,
)


COLUMNS = (
    "pmt_number",
    "pmt_x",
    "pmt_y",
    "pmt_z",
    "dom_x",
    "dom_y",
    "dom_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--route-class", required=True, choices=("0", "1"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    split_paths, _ = resolve_routed_split_paths(config, args.route_class)
    entry = split_paths["val"][0]
    shard = table_files(entry.path, str(config["data"]["pulsemaps"]))[0]
    table = pq.read_table(shard, columns=list(COLUMNS))
    frame = table.to_pandas()

    pmt_number = frame["pmt_number"].to_numpy(dtype=np.float64)
    rounded = np.rint(pmt_number)
    if not np.isfinite(pmt_number).all() or not np.array_equal(pmt_number, rounded):
        raise ValueError("Real parquet contains a non-integer pmt_number")
    ids = rounded.astype(np.int64)
    if ids.min() < 1 or ids.max() > 16:
        raise ValueError(f"Real parquet PMT ids outside [1, 16]: {ids.min()}..{ids.max()}")

    pmt_xyz = frame[["pmt_x", "pmt_y", "pmt_z"]].to_numpy(dtype=np.float64)
    dom_xyz = frame[["dom_x", "dom_y", "dom_z"]].to_numpy(dtype=np.float64)
    displacement = pmt_xyz - dom_xyz
    radius = np.linalg.norm(displacement, axis=1)
    if not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise ValueError("Real parquet contains invalid PMT displacement vectors")
    geometric_direction = displacement / radius[:, None]
    lookup = np.asarray(PONE_V3_PMT_DIRECTIONS, dtype=np.float64)[ids - 1]
    maximum_component_error = float(np.max(np.abs(geometric_direction - lookup)))
    if maximum_component_error > 1.0e-10:
        raise ValueError(
            "PMT lookup differs from real geometry: "
            f"max_component_error={maximum_component_error:.12g}"
        )

    print(f"config={args.config.resolve()}")
    print(f"route_class={args.route_class}")
    print(f"feature_shard={shard}")
    print(f"pulses_checked={len(frame):,}")
    print(f"pmt_ids={ids.min()}..{ids.max()}")
    print(f"radius_min_m={radius.min():.12g}")
    print(f"radius_max_m={radius.max():.12g}")
    print(f"max_direction_component_error={maximum_component_error:.12g}")
    print("real_parquet_pmt_geometry_check=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
