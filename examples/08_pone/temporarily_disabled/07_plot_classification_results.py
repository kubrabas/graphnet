"""
Create diagnostic plots from a classification test prediction CSV.

Usage:
    python3 07_plot_classification_results.py \
        /path/to/exp001_test_predictions.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PID_LABELS = {
    12: "NuE",
    14: "NuMu",
    16: "NuTau",
    -12: "NuEBar",
    -14: "NuMuBar",
    -16: "NuTauBar",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig, out_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def _label_true_class(value: int) -> str:
    return "Track" if int(value) == 1 else "Cascade"


def _pid_label(pid) -> str:
    if pd.isna(pid):
        return "Unknown"
    pid_int = int(pid)
    return PID_LABELS.get(pid_int, str(pid_int))


def _interaction_label(value) -> str:
    if pd.isna(value):
        return "Unknown"
    value = int(value)
    if value == 1:
        return "CC"
    if value == 2:
        return "NC"
    return str(value)


def _metrics_for(group: pd.DataFrame) -> dict:
    y = group["true_is_track"].astype(int)
    p = group["pred_is_track"].astype(int)
    _, _, auc = _roc_curve_binary(y, group["track_score"])

    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "n": len(group),
        "accuracy": float((y == p).mean()) if len(group) else np.nan,
        "precision_track": precision,
        "recall_track": recall,
        "specificity_cascade": specificity,
        "f1_track": f1,
        "roc_auc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _plot_confusion_matrix(df: pd.DataFrame, out_dir: Path) -> None:
    y = df["true_is_track"].astype(int)
    p = df["pred_is_track"].astype(int)
    matrix = np.array(
        [
            [int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum())],
            [int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())],
        ]
    )

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ["Predicted Cascade", "Predicted Track"])
    ax.set_yticks([0, 1], ["True Cascade", "True Track"])
    ax.set_title("Classification Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color="black")

    _save(fig, out_dir, "confusion_matrix.png")


def _plot_score_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for value, color in [(0, "tab:blue"), (1, "tab:orange")]:
        scores = df.loc[df["true_is_track"].astype(int) == value, "track_score"]
        ax.hist(scores, bins=60, histtype="step", linewidth=1.8, density=True, color=color, label=_label_true_class(value))

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Decision threshold")
    ax.set_xlabel("Track Score")
    ax.set_ylabel("Density")
    ax.set_title("Track Score Distribution by True Class")
    ax.legend()
    _save(fig, out_dir, "track_score_distribution.png")


def _roc_curve_binary(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return None, None, np.nan

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    distinct = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct, y_sorted.size - 1]

    tps = np.cumsum(y_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_sorted == 0)[threshold_idxs]

    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def _plot_roc_curve(df: pd.DataFrame, out_dir: Path) -> None:
    fpr, tpr, auc = _roc_curve_binary(df["true_is_track"], df["track_score"])
    if fpr is None or tpr is None:
        return

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(fpr, tpr, color="tab:blue", linewidth=2.0, label=f"Track vs Cascade (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    _save(fig, out_dir, "roc_curve.png")


def _plot_accuracy_by_energy(df: pd.DataFrame, out_dir: Path) -> None:
    if "true_energy" not in df.columns:
        return

    work = df[df["true_energy"].notna() & (df["true_energy"] > 0)].copy()
    if work.empty:
        return

    work["log10_energy"] = np.log10(work["true_energy"].astype(float))
    bins = np.linspace(work["log10_energy"].min(), work["log10_energy"].max(), 11)
    work["energy_bin"] = pd.cut(work["log10_energy"], bins=bins, include_lowest=True)
    grouped = work.groupby("energy_bin", observed=True)

    centers = np.array([interval.mid for interval in grouped.size().index])
    accuracy = grouped.apply(lambda g: (g["true_is_track"].astype(int) == g["pred_is_track"].astype(int)).mean())
    counts = grouped.size()

    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    ax1.plot(centers, accuracy.values, marker="o", color="tab:green")
    ax1.set_xlabel("log10(True Energy)")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title("Classification Accuracy by True Energy")

    ax2 = ax1.twinx()
    ax2.bar(centers, counts.values, width=(bins[1] - bins[0]) * 0.8, alpha=0.18, color="gray")
    ax2.set_ylabel("Events")
    _save(fig, out_dir, "accuracy_by_true_energy.png")


def _plot_accuracy_by_pid_interaction(df: pd.DataFrame, out_dir: Path) -> None:
    required = {"true_pid", "true_interaction_type"}
    if not required.issubset(df.columns):
        return

    work = df.copy()
    work["particle"] = work["true_pid"].map(_pid_label)
    work["interaction"] = work["true_interaction_type"].map(_interaction_label)
    work["group"] = work["particle"] + " " + work["interaction"]

    rows = []
    for name, group in work.groupby("group"):
        if len(group) < 1:
            continue
        metrics = _metrics_for(group)
        rows.append(
            {
                "group": name,
                "accuracy": metrics["accuracy"],
                "mis_id": 1.0 - metrics["accuracy"],
                "n": metrics["n"],
            }
        )

    if not rows:
        return

    rows.sort(key=lambda item: item["group"])
    labels = [row["group"] for row in rows]
    accuracy = np.array([row["accuracy"] for row in rows])
    mis_id = np.array([row["mis_id"] for row in rows])
    counts = [row["n"] for row in rows]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.8), 5.2))
    ax.bar(x, accuracy, label="Correct class", color="tab:blue")
    ax.bar(x, mis_id, bottom=accuracy, label="Misclassified", color="tab:red", alpha=0.78)

    for i, row in enumerate(rows):
        ax.text(
            i,
            max(row["accuracy"] / 2, 0.04),
            f"{row['accuracy']:.1%}",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
        if row["mis_id"] >= 0.035:
            ax.text(
                i,
                row["accuracy"] + row["mis_id"] / 2,
                f"{row['mis_id']:.1%}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of events")
    ax.set_title("Classification Outcome by Particle and Interaction Type")
    ax.set_xticks(x, [f"{label}\n(n={count:,})" for label, count in zip(labels, counts)], rotation=35, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    _save(fig, out_dir, "metrics_by_particle_interaction.png")


def _plot_accuracy_by_zenith(df: pd.DataFrame, out_dir: Path) -> None:
    if "true_zenith" not in df.columns:
        return

    work = df[df["true_zenith"].notna()].copy()
    if work.empty:
        return

    bins = np.linspace(work["true_zenith"].min(), work["true_zenith"].max(), 11)
    work["zenith_bin"] = pd.cut(work["true_zenith"], bins=bins, include_lowest=True)
    grouped = work.groupby("zenith_bin", observed=True)

    centers = np.array([interval.mid for interval in grouped.size().index])
    accuracy = grouped.apply(lambda g: (g["true_is_track"].astype(int) == g["pred_is_track"].astype(int)).mean())
    counts = grouped.size()

    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    ax1.plot(centers, accuracy.values, marker="o", color="tab:purple")
    ax1.set_xlabel("True Zenith [rad]")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_title("Classification Accuracy by True Zenith")

    ax2 = ax1.twinx()
    ax2.bar(centers, counts.values, width=(bins[1] - bins[0]) * 0.8, alpha=0.18, color="gray")
    ax2.set_ylabel("Events")
    _save(fig, out_dir, "accuracy_by_true_zenith.png")


def _write_metric_tables(df: pd.DataFrame, out_dir: Path) -> None:
    rows = [dict(group="All", **_metrics_for(df))]

    if {"true_pid", "true_interaction_type"}.issubset(df.columns):
        work = df.copy()
        work["particle"] = work["true_pid"].map(_pid_label)
        work["interaction"] = work["true_interaction_type"].map(_interaction_label)
        for name, group in work.groupby(["particle", "interaction"]):
            rows.append(dict(group=f"{name[0]} {name[1]}", **_metrics_for(group)))

    pd.DataFrame(rows).to_csv(out_dir / "classification_metrics_summary.csv", index=False)


def make_plots(predictions_csv: str, output_dir: str | None = None) -> Path:
    csv_path = Path(predictions_csv)
    if output_dir is None:
        out_dir = csv_path.parent / "plots"
    else:
        out_dir = Path(output_dir)

    _ensure_dir(out_dir)
    df = pd.read_csv(csv_path)

    required = {"true_is_track", "pred_is_track", "track_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    _write_metric_tables(df, out_dir)
    _plot_confusion_matrix(df, out_dir)
    _plot_score_distribution(df, out_dir)
    _plot_roc_curve(df, out_dir)
    _plot_accuracy_by_energy(df, out_dir)
    _plot_accuracy_by_pid_interaction(df, out_dir)
    _plot_accuracy_by_zenith(df, out_dir)

    print(f"[Plots] Wrote classification plots to {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv", help="Path to *_test_predictions.csv")
    parser.add_argument("-o", "--output-dir", default=None, help="Directory for plots")
    args = parser.parse_args()

    make_plots(args.predictions_csv, args.output_dir)
