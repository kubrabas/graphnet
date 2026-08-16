#!/usr/bin/env python3
"""Read-only real-parquet smoke test for the two transformer configs."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
JOINT_DIR = THIS_DIR.parent
PONE_DIR = JOINT_DIR.parent
REFERENCE_DIR = PONE_DIR.parent / "09_pone_muon_direction"
for path in (PONE_DIR, REFERENCE_DIR, JOINT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from energy_weighting import EnergyWeightManifest
from model_factory import build_direction_model, build_model_data_representation
from routed_data import build_loaders
from routed_pipeline_utils import load_yaml, resolve_routed_split_paths
from train_scripts.train_routed_joint_direction import validate_config


CONFIG_DIR = PONE_DIR / "configs" / "joint_direction"
CONFIGS = (
    CONFIG_DIR / "102_string_emax1e6__category1_isMuonCC__fourier_t_v1.yml",
    CONFIG_DIR
    / "102_string_emax1e6__category1_isMuonCC__fourier_t_pmt_v1.yml",
)
ENERGY_MANIFEST = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Results/340StringMC/"
    "102_string_emax1e6/reconstruction/category1_isMuonCC/class1/"
    "pmt_direction_v1/train_and_val/zenith_azimuth/energy_weight_manifest.json"
)


def main() -> int:
    if not ENERGY_MANIFEST.is_file():
        raise FileNotFoundError(ENERGY_MANIFEST)
    manifest = EnergyWeightManifest.load(ENERGY_MANIFEST)
    for path in CONFIGS:
        config = load_yaml(path)
        validate_config(config)
        split_paths, scaler = resolve_routed_split_paths(config, "1")
        representation = build_model_data_representation(config, scaler)
        smoke_config = copy.deepcopy(config)
        smoke_config["loader"].update(
            {
                "batch_size": 2,
                "val_batch_size": 2,
                "num_workers": 0,
                "accumulate_grad_batches": 1,
            }
        )
        # Keep the complete production network and only limit sequence length
        # so a CPU backward is a fast interface smoke, not a performance run.
        smoke_config["model"]["transformer"].update(
            {"train_max_pulses": 32, "eval_max_pulses": 32}
        )
        loader = build_loaders(
            smoke_config,
            split_paths,
            representation,
            splits=("train",),
        )["train"]
        batch = next(iter(loader))
        if tuple(representation.output_feature_names[:5]) != (
            "pmt_x",
            "pmt_y",
            "pmt_z",
            "dom_time",
            "charge",
        ):
            raise RuntimeError("Canonical feature order drifted")
        # Identity preprocessing means real metre/ns/charge units must survive.
        if float(batch.x[:, :3].abs().max()) < 100.0:
            raise RuntimeError("Transformer positions appear to have been scaled")
        if float(batch.x[:, 4].min()) <= 0.0:
            raise RuntimeError("Transformer charge must remain raw and positive")
        model = build_direction_model(
            smoke_config,
            "stage_b",
            representation,
            manifest,
            steps_per_optimizer_epoch=1,
        )
        model.train()
        prediction = model(batch)
        loss = model.compute_loss(prediction, [batch])
        if tuple(prediction[0].shape) != (2, 4) or not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"Invalid transformer output/loss: {prediction[0].shape}, {loss}"
            )
        loss.backward()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"PASS {path.name}: parquet={list(representation._input_feature_names)}, "
            f"model={list(representation.output_feature_names)}, "
            f"events=2 nodes={batch.num_nodes} params={trainable:,} "
            f"loss={float(loss.detach()):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
