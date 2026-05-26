import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import EnsembleDataset
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.models.data_representation import KNNGraph, NodesAsPulses
from graphnet.models.detector.pone import PONE

from utils import extract_field, move_batch_to_device


PATHS_PY = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]

PARQUET_TABLE = {
    "340StringMC": "STRING340MC_PARQUET",
    "Spring2026MC": "SPRING2026MC_PARQUET",
}


def load_paths_module():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_experiment_dir(cfg: dict) -> Path:
    return (
        Path(cfg["output"]["root_dir"])
        / cfg["mc"]
        / cfg["geometry"]
        / cfg["task"]["type"]
        / cfg["task"]["target"]
        / cfg["experiment_name"]
    )


def resolve_stage_dir(cfg: dict, stage: str) -> Path:
    experiment_dir = resolve_experiment_dir(cfg)
    return experiment_dir / cfg["output"]["dirs"][stage]


def resolve_classification_paths(cfg: dict) -> Tuple[Dict[str, dict], str]:
    mc = cfg["mc"]
    geometry = cfg["geometry"]
    flavors = cfg.get("flavors", ALL_FLAVORS)

    mod = load_paths_module()
    parquet_table = getattr(mod, PARQUET_TABLE[mc])

    per_flavor = {}
    for flavor in flavors:
        entry = parquet_table.get(geometry, {}).get(flavor, {})
        for split in ("train", "val"):
            if not entry.get(split):
                raise ValueError(
                    f"{PARQUET_TABLE[mc]}['{geometry}']['{flavor}']['{split}'] "
                    "is missing in paths.py."
                )
        per_flavor[flavor] = entry
        print(f"[Paths] {flavor}: train={entry['train']} | val={entry['val']}")

    percentiles_csv = cfg["data"].get("percentiles_csv")
    if not percentiles_csv:
        robust_scaler = getattr(mod, "ROBUST_SCALER")
        percentiles_csv = robust_scaler.get(mc, {}).get(geometry, {}).get("mixed")
    if not percentiles_csv:
        raise ValueError(f"ROBUST_SCALER['{mc}']['{geometry}']['mixed'] is missing in paths.py.")
    print(f"[Paths] percentiles_csv: {percentiles_csv}")

    return per_flavor, percentiles_csv


def build_classification_loaders(cfg: dict, per_flavor: dict, percentiles_csv: str):
    features = cfg["data"]["features"]
    truth_all = _unique([*cfg["data"]["truth_all"], cfg["task"]["target"]])
    pulsemaps = cfg["data"]["pulsemaps"]
    truth_table = cfg["data"]["truth_table"]
    tcfg = cfg["training"]
    weights_cfg = cfg.get("weights", {})
    weights_enabled = bool(weights_cfg.get("enabled", False))

    data_representation = KNNGraph(
        detector=PONE(percentiles_csv=percentiles_csv, selected_features=features),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=cfg["model"]["nb_neighbours"],
        distance_as_edge_feature=False,
    )

    def _make_dataset(path):
        kwargs = {}
        if weights_enabled:
            kwargs = {
                "loss_weight_table": weights_cfg["loss_weight_table"],
                "loss_weight_column": weights_cfg["loss_weight_column"],
                "loss_weight_default_value": weights_cfg.get("loss_weight_default_value", 1.0),
            }
        return ParquetDataset(
            path=path,
            pulsemaps=pulsemaps,
            truth_table=truth_table,
            features=features,
            truth=truth_all,
            data_representation=data_representation,
            **kwargs,
        )

    def _make_loader(ds, shuffle, drop_last=False):
        return DataLoader(
            ds,
            batch_size=tcfg["batch_size"],
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=tcfg["num_workers"],
            multiprocessing_context=tcfg.get("multiprocessing_context", "spawn"),
            persistent_workers=True,
            pin_memory=tcfg.get("pin_memory", True),
        )

    train_ds = EnsembleDataset([_make_dataset(entry["train"]) for entry in per_flavor.values()])
    val_ds = EnsembleDataset([_make_dataset(entry["val"]) for entry in per_flavor.values()])

    train_loader = _make_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = _make_loader(val_ds, shuffle=False)

    print(f"[Data] flavors={list(per_flavor.keys())}")
    print(f"[Data] train={len(train_loader)} batches | val={len(val_loader)} batches")
    return data_representation, train_loader, val_loader


def collect_validation_predictions(cfg: dict, model, val_loader):
    labels = list(cfg["task"]["labels"])
    mode = cfg["task"]["mode"]
    target = cfg["task"]["target"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    y_true_parts, prob_parts = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)
            raw = model(batch)[0].detach().float()
            if mode == "binary":
                positive = raw.squeeze(-1).clamp(0.0, 1.0)
                probs = torch.stack([1.0 - positive, positive], dim=1)
            elif mode == "multiclass":
                probs = torch.softmax(raw, dim=1)
            else:
                raise ValueError(f"Unsupported classification mode: {mode}")
            y = extract_field(batch, target).detach().long().view(-1)
            y_true_parts.append(y.cpu())
            prob_parts.append(probs.cpu())

    y_true = torch.cat(y_true_parts).numpy()
    probabilities = torch.cat(prob_parts).numpy()
    return y_true, probabilities, labels


def one_vs_rest_roc(y_true: np.ndarray, scores: np.ndarray, positive_label: int) -> dict:
    y_binary = (y_true == positive_label).astype(int)
    positives = int(y_binary.sum())
    negatives = int(len(y_binary) - positives)
    if positives == 0 or negatives == 0:
        return {
            "fpr": np.array([]),
            "tpr": np.array([]),
            "thresholds": np.array([]),
            "auc": np.nan,
            "youden_threshold": np.nan,
            "closest_threshold": np.nan,
            "support": positives,
        }

    finite_scores = scores[np.isfinite(scores)]
    thresholds = np.unique(finite_scores)[::-1]

    fpr = [0.0]
    tpr = [0.0]
    used_thresholds = [np.inf]
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(((pred == 1) & (y_binary == 1)).sum())
        fp = int(((pred == 1) & (y_binary == 0)).sum())
        fpr.append(fp / negatives)
        tpr.append(tp / positives)
        used_thresholds.append(float(threshold))
    fpr.append(1.0)
    tpr.append(1.0)
    used_thresholds.append(-np.inf)

    fpr_arr = np.asarray(fpr, dtype=float)
    tpr_arr = np.asarray(tpr, dtype=float)
    thr_arr = np.asarray(used_thresholds, dtype=float)
    auc = float(np.trapz(tpr_arr, fpr_arr))

    finite = np.isfinite(thr_arr)
    finite_idx = np.where(finite)[0]
    youden_idx = finite_idx[np.argmax(tpr_arr[finite] - fpr_arr[finite])]
    closest_idx = finite_idx[np.argmin(np.sqrt(fpr_arr[finite] ** 2 + (1.0 - tpr_arr[finite]) ** 2))]

    return {
        "fpr": fpr_arr,
        "tpr": tpr_arr,
        "thresholds": thr_arr,
        "auc": auc,
        "youden_threshold": float(thr_arr[youden_idx]),
        "closest_threshold": float(thr_arr[closest_idx]),
        "support": positives,
    }


def compute_roc_by_class(y_true: np.ndarray, probabilities: np.ndarray, labels: List[int]) -> Dict[int, dict]:
    return {
        label: one_vs_rest_roc(y_true, probabilities[:, idx], label)
        for idx, label in enumerate(labels)
    }


def class_name(cfg: dict, label: int) -> str:
    names = cfg["task"].get("class_names") or {}
    return str(names.get(label, names.get(str(label), f"class_{label}")))


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def write_binary_metrics(cfg: dict, y_true: np.ndarray, probabilities: np.ndarray, roc_by_class: Dict[int, dict], out_dir: Path):
    labels = list(cfg["task"]["labels"])
    positive_label = labels[-1]
    positive_idx = labels.index(positive_label)
    positive_name = safe_name(class_name(cfg, positive_label))
    negative_name = safe_name(class_name(cfg, labels[0]))
    y_binary = (y_true == positive_label).astype(int)
    rows = []

    for method, threshold_key in [
        ("youden_j", "youden_threshold"),
        ("closest_top_left", "closest_threshold"),
    ]:
        threshold = roc_by_class[positive_label][threshold_key]
        pred = (probabilities[:, positive_idx] >= threshold).astype(int)
        tp = int(((pred == 1) & (y_binary == 1)).sum())
        fp = int(((pred == 1) & (y_binary == 0)).sum())
        tn = int(((pred == 0) & (y_binary == 0)).sum())
        fn = int(((pred == 0) & (y_binary == 1)).sum())
        pos_precision = _safe_div(tp, tp + fp)
        pos_recall = _safe_div(tp, tp + fn)
        neg_precision = _safe_div(tn, tn + fn)
        neg_recall = _safe_div(tn, tn + fp)
        rows.append(
            {
                "method": method,
                "threshold": threshold,
                "accuracy": _safe_div(tp + tn, tp + fp + tn + fn),
                f"{negative_name}_precision": neg_precision,
                f"{negative_name}_recall": neg_recall,
                f"{negative_name}_f1": _f1(neg_precision, neg_recall),
                f"{positive_name}_precision": pos_precision,
                f"{positive_name}_recall": pos_recall,
                f"{positive_name}_f1": _f1(pos_precision, pos_recall),
                f"tp_{positive_name}": tp,
                f"fp_{positive_name}": fp,
                f"tn_{positive_name}": tn,
                f"fn_{positive_name}": fn,
                "roc_auc": roc_by_class[positive_label]["auc"],
            }
        )
        write_confusion_matrix(
            np.array([[tn, fp], [fn, tp]], dtype=int),
            [class_name(cfg, labels[0]), class_name(cfg, positive_label)],
            out_dir / f"validation_confusion_matrix_{method}.png",
            title=f"Validation Confusion Matrix ({method})",
        )

    pd.DataFrame(rows).to_csv(out_dir / "validation_metrics_summary.csv", index=False)


def write_multiclass_metrics(cfg: dict, y_true: np.ndarray, probabilities: np.ndarray, roc_by_class: Dict[int, dict], out_dir: Path):
    labels = list(cfg["task"]["labels"])
    pred = np.asarray([labels[i] for i in np.argmax(probabilities, axis=1)])
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, pred):
        if true_label in labels and pred_label in labels:
            matrix[labels.index(int(true_label)), labels.index(int(pred_label))] += 1

    by_class = []
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        by_class.append(
            {
                "class": class_name(cfg, label),
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "support": int(matrix[i, :].sum()),
            }
        )

    by_class_df = pd.DataFrame(by_class)
    by_class_df.to_csv(out_dir / "validation_metrics_by_class.csv", index=False)

    supports = by_class_df["support"].to_numpy(dtype=float)
    weights = supports / supports.sum() if supports.sum() > 0 else np.zeros_like(supports)
    aucs = np.array([roc_by_class[label]["auc"] for label in labels], dtype=float)
    summary = {
        "method": "argmax",
        "accuracy": float(np.trace(matrix) / max(matrix.sum(), 1)),
        "macro_precision": float(by_class_df["precision"].mean()),
        "macro_recall": float(by_class_df["recall"].mean()),
        "macro_f1": float(by_class_df["f1"].mean()),
        "weighted_precision": float(np.sum(by_class_df["precision"].to_numpy(dtype=float) * weights)),
        "weighted_recall": float(np.sum(by_class_df["recall"].to_numpy(dtype=float) * weights)),
        "weighted_f1": float(np.sum(by_class_df["f1"].to_numpy(dtype=float) * weights)),
        "macro_roc_auc": float(np.nanmean(aucs)),
        "weighted_roc_auc": float(np.nansum(aucs * weights)),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "validation_metrics_summary.csv", index=False)
    write_confusion_matrix(
        matrix,
        [class_name(cfg, label) for label in labels],
        out_dir / "validation_confusion_matrix_argmax.png",
        title="Validation Confusion Matrix (argmax)",
    )


def write_validation_diagnostics(cfg: dict, y_true: np.ndarray, probabilities: np.ndarray, labels: List[int], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    roc_by_class = compute_roc_by_class(y_true, probabilities, labels)
    write_roc_plot(cfg, roc_by_class, labels, out_dir / "validation_roc_curve.png")
    write_score_distribution_plot(cfg, y_true, probabilities, labels, roc_by_class, out_dir / "validation_score_distribution_by_true_class.png")
    if cfg["task"]["mode"] == "binary":
        write_binary_metrics(cfg, y_true, probabilities, roc_by_class, out_dir)
    else:
        write_multiclass_metrics(cfg, y_true, probabilities, roc_by_class, out_dir)


def write_roc_plot(cfg: dict, roc_by_class: Dict[int, dict], labels: List[int], out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, text_ax) = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.8),
        gridspec_kw={"width_ratios": [3.2, 1.25]},
    )
    colors = plt.get_cmap("tab10")
    aucs, supports = [], []
    table_lines = ["class | Youden J | Closest"]
    for idx, label in enumerate(labels):
        roc = roc_by_class[label]
        color = colors(idx % 10)
        if roc["fpr"].size:
            ax.plot(
                roc["fpr"],
                roc["tpr"],
                linewidth=2.0,
                color=color,
                label=f"{class_name(cfg, label)} AUC={roc['auc']:.4f}",
            )
        aucs.append(roc["auc"])
        supports.append(roc["support"])
        table_lines.append(
            f"{class_name(cfg, label)} | {roc['youden_threshold']:.4f} | {roc['closest_threshold']:.4f}"
        )

    weights = np.asarray(supports, dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.zeros_like(weights)
    auc_arr = np.asarray(aucs, dtype=float)
    macro_auc = float(np.nanmean(auc_arr))
    weighted_auc = float(np.nansum(auc_arr * weights))

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0, label="Random")
    ax.set_title(f"Validation ROC | macro AUC={macro_auc:.4f} | weighted AUC={weighted_auc:.4f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    text_ax.axis("off")
    text_ax.text(
        0.0,
        1.0,
        "\n".join(table_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_score_distribution_plot(cfg: dict, y_true: np.ndarray, probabilities: np.ndarray, labels: List[int], roc_by_class: Dict[int, dict], out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_classes = len(labels)
    ncols = min(2, n_classes)
    nrows = int(np.ceil(n_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.0 * nrows), squeeze=False)
    colors = plt.get_cmap("tab10")

    for idx, label in enumerate(labels):
        ax = axes[idx // ncols][idx % ncols]
        color = colors(idx % 10)
        mask = y_true == label
        scores = probabilities[mask, idx]
        ax.hist(scores, bins=60, histtype="step", density=True, linewidth=1.8, color=color, label=class_name(cfg, label))
        _draw_threshold_tick(ax, roc_by_class[label]["youden_threshold"], color, 0.03, "Youden J")
        _draw_threshold_tick(ax, roc_by_class[label]["closest_threshold"], color, 0.08, "Closest top-left", linestyle="--")
        ax.set_title(f"True {class_name(cfg, label)} -> p_class_{label}")
        ax.set_xlabel(f"p_class_{label}")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

    for empty_idx in range(n_classes, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_confusion_matrix(matrix: np.ndarray, labels: List[str], out_path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5.2, len(labels) * 1.2), max(4.8, len(labels) * 1.0)))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _draw_threshold_tick(ax, x: float, color, y_frac: float, label: str, linestyle: str = "-"):
    if not np.isfinite(x):
        return
    trans = ax.get_xaxis_transform()
    ax.plot([x, x], [0.0, y_frac], transform=trans, color=color, linestyle=linestyle, linewidth=2.2, label=f"{label}={x:.3f}")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def _unique(items: Iterable) -> List:
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out
