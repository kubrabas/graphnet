"""Plots and compact tables for training and direction evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


def plot_training_history(stage_dir: str | Path) -> None:
    """Create loss and opening-angle curves from the epoch CSV."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    stage_dir = Path(stage_dir)
    path = stage_dir / "training_history_by_epoch.csv"
    if not path.is_file():
        return
    frame = pd.read_csv(path).sort_values("epoch")
    if frame.empty:
        return

    loss_columns = [
        column
        for column in (
            "train_loss",
            "val_loss",
            "val_objective_loss_unweighted",
            "val_objective_loss_weighted",
        )
        if column in frame
    ]
    if loss_columns:
        figure, axis = plt.subplots(figsize=(8, 5))
        for column in loss_columns:
            axis.plot(frame["epoch"], frame[column], marker="o", label=column)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("Training and validation losses")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(stage_dir / "training_validation_loss.png", dpi=170)
        plt.close(figure)

    angle_columns = [
        column
        for column in (
            "val_global_median_deg",
            "val_macro_median_deg",
            "val_global_q68_deg",
            "val_macro_q68_deg",
            "val_global_mean_deg",
        )
        if column in frame
    ]
    if angle_columns:
        figure, axis = plt.subplots(figsize=(8, 5))
        for column in angle_columns:
            axis.plot(frame["epoch"], frame[column], marker="o", label=column)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="1 degree")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Opening angle [deg]")
        axis.set_title("Validation opening-angle metrics")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(stage_dir / "validation_opening_angle_by_epoch.png", dpi=170)
        plt.close(figure)


def write_evaluation_plots(
    predictions,
    energy_rows,
    output_dir: str | Path,
) -> None:
    """Plot final opening-angle distribution, CDF, and energy dependence."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    angles = predictions["opening_angle_deg"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(8, 5))
    upper = max(10.0, float(np.quantile(angles, 0.99)))
    axis.hist(angles, bins=np.linspace(0.0, upper, 100), histtype="step", linewidth=1.8)
    axis.axvline(float(np.median(angles)), color="tab:orange", label="median")
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1, label="1 degree")
    axis.set_xlabel("Opening angle [deg]")
    axis.set_ylabel("Events")
    axis.set_title("Opening-angle distribution")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "opening_angle_distribution.png", dpi=170)
    plt.close(figure)

    ordered = np.sort(angles)
    cdf = np.arange(1, ordered.size + 1) / ordered.size
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(ordered, cdf)
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0.0, max(10.0, float(np.quantile(angles, 0.99))))
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Opening angle [deg]")
    axis.set_ylabel("Cumulative fraction")
    axis.set_title("Opening-angle CDF")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "opening_angle_cdf.png", dpi=170)
    plt.close(figure)

    if energy_rows:
        centers = [
            0.5 * (row["log10_energy_low"] + row["log10_energy_high"])
            for row in energy_rows
        ]
        figure, axis = plt.subplots(figsize=(8, 5))
        for key, label in (
            ("median_deg", "median"),
            ("q68_deg", "q68"),
            ("q90_deg", "q90"),
        ):
            axis.plot(centers, [row[key] for row in energy_rows], marker="o", label=label)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="1 degree")
        axis.set_xlabel("True log10(E / GeV)")
        axis.set_ylabel("Opening angle [deg]")
        axis.set_title("Angular resolution versus true energy")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / "opening_angle_by_true_energy.png", dpi=170)
        plt.close(figure)

