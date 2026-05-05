"""
Evaluate trained reconstruction models (separate or combined) on the test set.

Loads best_model.pth for each target, runs inference, and writes prediction
CSVs with true/predicted values and residuals for energy, zenith, azimuth.

Works for both track and cascade reconstruction models.

Usage:
    python3 03_test_reconstruction.py -c configs/track/separate/exp001.yml
    python3 03_test_reconstruction.py -c configs/cascade/separate/exp001.yml
    python3 03_test_reconstruction.py -c configs/track/combined/exp001.yml
    python3 03_test_reconstruction.py -c configs/cascade/combined/exp001.yml
"""

import argparse
import os

import yaml


def load_model(cfg: dict, target: str):
    pass


def run_test(cfg: dict, model, target: str, test_loader, out_dir: str) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
