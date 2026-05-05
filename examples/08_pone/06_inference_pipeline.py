"""
Full inference pipeline: classify events, then route to track or cascade
reconstruction models to produce energy / zenith / azimuth predictions.

Steps:
  1. Load trained classification model.
  2. Load trained track reconstruction models (energy, zenith, azimuth).
  3. Load trained cascade reconstruction models (energy, zenith, azimuth).
  4. For each event in the test set:
       - Run classification → track_score
       - If track_score >= threshold → track reconstruction models
       - Else                        → cascade reconstruction models
  5. Write a single CSV:
       event_id | track_score | pred_topology |
       energy | zenith | azimuth

Usage:
    python3 04_inference_pipeline.py -c configs/inference/exp001.yml
"""

import argparse
import os

import yaml


def load_classification_model(cfg: dict):
    pass


def load_reconstruction_models(cfg: dict, topology: str) -> dict:
    """Load energy, zenith, azimuth models for 'track' or 'cascade'."""
    pass


def run_pipeline(cfg: dict, classification_model, track_models: dict, cascade_models: dict, test_loader, out_dir: str) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
