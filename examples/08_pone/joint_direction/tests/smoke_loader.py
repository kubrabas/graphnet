#!/usr/bin/env python3
"""Read one routed validation batch without creating training outputs.

This is an integration smoke test for the real GraphNeT container. It checks
path/scaler resolution, EnsembleDataset construction, and the in-memory
``source_flavor_id`` label. It never resolves the test split and never writes
to source parquet or Results. The default single-process loader also makes the
short test exit deterministic; production configs continue to use 8 spawned
workers.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
JOINT_DIR = THIS_DIR.parent
PONE_DIR = JOINT_DIR.parent
REFERENCE_DIR = PONE_DIR.parent / "09_pone_muon_direction"
for path in (PONE_DIR, REFERENCE_DIR, JOINT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from energy_weighting import fit_energy_weight_manifest
from model import build_joint_direction_model
from pipeline_utils import extract_field
from routed_data import build_data_representation, build_loaders
from routed_pipeline_utils import checkpoint_state, load_yaml, resolve_routed_split_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument("--route-class", required=True, choices=("0", "1"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--consume-all",
        action="store_true",
        help="Exhaust the validation loader before shutting spawned workers down",
    )
    parser.add_argument(
        "--forward",
        action="store_true",
        help="Also run one CPU Stage-B forward/loss pass on the four graphs",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Strict-load this existing checkpoint before the forward pass",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = copy.deepcopy(load_yaml(args.config))
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    config["loader"]["batch_size"] = args.batch_size
    config["loader"]["num_workers"] = args.workers

    split_paths, scaler = resolve_routed_split_paths(config, args.route_class)
    graph = build_data_representation(config, scaler)
    loaders = build_loaders(
        config, split_paths, graph, splits=("val",)
    )
    iterator = iter(loaders["val"])
    batches_consumed = 0
    try:
        batch = next(iterator)
        batches_consumed = 1
        if args.consume_all:
            for _ in iterator:
                batches_consumed += 1
    finally:
        # The production trainer owns the loader lifecycle. This short test
        # exits after one batch, so shut spawned workers down explicitly to
        # avoid a Python-interpreter teardown race.
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()

    fields = (
        "event_no",
        "RunID",
        "SubrunID",
        "EventID",
        "SubEventID",
        "zenith",
        "azimuth",
        "totalEnergy",
        "source_flavor_id",
        str(config["routing"]["category"]),
        str(config["data"]["trigger_column"]),
    )
    tensors = {name: extract_field(batch, name).reshape(-1) for name in fields}
    sizes = {name: int(value.numel()) for name, value in tensors.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Event-level field sizes differ: {sizes}")
    if not bool((tensors[str(config["routing"]["category"])] == int(args.route_class)).all()):
        raise ValueError("Smoke batch contains the wrong routed class")
    if not bool((tensors[str(config["data"]["trigger_column"])] == 1).all()):
        raise ValueError("Smoke batch contains a non-triggered event")
    flavor_ids = sorted(set(int(value) for value in tensors["source_flavor_id"].tolist()))
    if not set(flavor_ids).issubset({0, 1, 2, 3}):
        raise ValueError(f"Unknown source_flavor_id values: {flavor_ids}")
    if not bool(torch.isfinite(tensors["totalEnergy"]).all()):
        raise ValueError("Smoke batch contains non-finite energy")

    if args.forward or args.checkpoint is not None:
        edges = torch.as_tensor(
            config["weighting"]["log10_energy_bin_edges"], dtype=torch.float64
        )
        synthetic_energy = torch.pow(10.0, 0.5 * (edges[:-1] + edges[1:]))
        manifest = fit_energy_weight_manifest(
            synthetic_energy,
            edges.tolist(),
            alpha=float(config["weighting"]["alpha"]),
            clip_min=float(config["weighting"]["clip_min"]),
            clip_max=float(config["weighting"]["clip_max"]),
        )
        model = build_joint_direction_model(
            config,
            "stage_b",
            graph,
            manifest,
            steps_per_optimizer_epoch=1,
        ).eval()
        if args.checkpoint is not None:
            if not args.checkpoint.is_file():
                raise FileNotFoundError(args.checkpoint)
            model.load_state_dict(checkpoint_state(args.checkpoint), strict=True)
        with torch.no_grad():
            predictions = model(batch)
            prediction = predictions[0]
            loss = model.compute_loss(predictions, [batch])
        if prediction.shape != (next(iter(sizes.values())), 4):
            raise ValueError(f"Unexpected joint prediction shape: {prediction.shape}")
        norms = torch.linalg.vector_norm(prediction[:, :3], dim=1)
        if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5)):
            raise ValueError(f"Predicted directions are not unit vectors: {norms}")
        if not bool((prediction[:, 3] > 0.0).all()):
            raise ValueError("Forward smoke test produced non-positive kappa")
        if not bool(torch.isfinite(loss)):
            raise ValueError("Forward smoke test produced non-finite loss")

    print(f"config={args.config.resolve()}")
    print(f"route_class={args.route_class}")
    print(f"constituent_flavors={[entry.flavor for entry in split_paths['val']]}")
    print(f"scaler={scaler}")
    print(f"batch_events={next(iter(sizes.values()))}")
    print(f"batches_consumed={batches_consumed}")
    print(f"source_flavor_ids={flavor_ids}")
    print(f"graph_input_features={graph.nb_inputs}")
    print(f"graph_output_features={graph.nb_outputs}")
    print(f"graph_output_feature_names={graph.output_feature_names}")
    if args.forward or args.checkpoint is not None:
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        print(f"trainable_parameters={trainable_parameters}")
        print(f"prediction_shape={tuple(prediction.shape)}")
        print(f"stage_b_loss={float(loss):.8g}")
        if args.checkpoint is not None:
            print(f"strict_checkpoint_load=passed:{args.checkpoint.resolve()}")
        print("model_forward_test=passed")
    print("loader_smoke_test=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
