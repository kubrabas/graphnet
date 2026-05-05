"""
Train a single multi-task reconstruction network that jointly predicts
energy, zenith, and azimuth for a given event topology (track or cascade).

One DynEdge backbone, three task heads trained simultaneously.

Usage:
    python3 02_train_reconstruction_combined.py -c configs/track/combined/exp001.yml
    python3 02_train_reconstruction_combined.py -c configs/cascade/combined/exp001.yml
"""

import argparse
import os

import yaml


def build_model(cfg: dict, data_representation, steps_per_epoch_optimizer: int):
    pass


def run_combined(cfg: dict, data_representation, train_loader, val_loader) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
