import csv
import logging
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data

from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.models.data_representation import KNNGraph, NodesAsPulses
from graphnet.models.detector.pone import PONE
from graphnet.models.task import StandardLearnedTask
from graphnet.utilities.maths import eps_like


# =======================
# Log suppression
# =======================

class _SuppressGraphnetOptionalDepsAtImport(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith("graphnet"):
            return True
        msg = record.getMessage()
        if (
            "has_jammy_flows_package" in msg
            or "jammy_flows" in msg
            or "has_icecube_package" in msg
            or "`icecube` not available" in msg
            or "has_km3net_package" in msg
            or "`km3net` not available" in msg
        ):
            return False
        return True

if not getattr(logging, "_GRAPHNET_OPTIONAL_DEPS_FILTER_INSTALLED", False):
    _f = _SuppressGraphnetOptionalDepsAtImport()
    logging.getLogger("graphnet").addFilter(_f)
    logging.getLogger("graphnet.utilities.imports").addFilter(_f)
    logging._GRAPHNET_OPTIONAL_DEPS_FILTER_INSTALLED = True


_EPOCH_CTX = {"epoch": None}

_LOG_FILTERS_INSTALLED = False

def install_logging_filters() -> None:
    global _LOG_FILTERS_INSTALLED
    if _LOG_FILTERS_INSTALLED:
        return
    _LOG_FILTERS_INSTALLED = True
    es_filter = _InjectEpochIntoEarlyStoppingLog()
    logging.getLogger("pytorch_lightning.callbacks.early_stopping").addFilter(es_filter)
    logging.getLogger("lightning.pytorch.callbacks.early_stopping").addFilter(es_filter)
    logging.getLogger("graphnet.training.callbacks").addFilter(es_filter)


# =======================
# Callbacks (epoch context + early stopping log inject)
# =======================

from pytorch_lightning.callbacks import Callback

class _EpochContextCallback(Callback):
    def _set_epoch(self, trainer):
        if trainer.sanity_checking:
            return
        _EPOCH_CTX["epoch"] = trainer.current_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        self._set_epoch(trainer)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._set_epoch(trainer)


class _InjectEpochIntoEarlyStoppingLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        ep = _EPOCH_CTX.get("epoch", None)
        if ep is not None and msg.startswith("Metric ") and "New best score" in msg:
            record.msg = msg + f" | epoch: {ep}"
            record.args = ()
        return True


# =======================
# Resource helpers
# =======================
# TODO: gpu_util_pct measured at on_validation_epoch_end is unreliable — GPU is idle at that point.
# Better approach: add periodic nvidia-smi polling to the sbatch script, e.g.:
#   while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 60; done &
# Run this in the background before srun, output goes to the .out file.

def _read_rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 / 1024.0
    except Exception:
        return float("nan")



def _cpu_load_pct() -> float:
    try:
        return 100.0 * os.getloadavg()[0] / (os.cpu_count() or 1)
    except Exception:
        return float("nan")


def _gpu_snapshot() -> Dict[str, float]:
    out = {k: float("nan") for k in ["gpu_util_pct", "gpu_mem_used_gb", "gpu_mem_total_gb", "gpu_mem_util_pct"]}
    try:
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_id = dev.split(",")[0].strip() if dev else None
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
        if gpu_id:
            cmd = ["nvidia-smi", "-i", gpu_id] + cmd[1:]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0 or not res.stdout.strip():
            return out
        util_s, used_s, total_s = [x.strip() for x in res.stdout.strip().splitlines()[0].split(",")]
        used_mb, total_mb = float(used_s), float(total_s)
        out["gpu_util_pct"]     = float(util_s)
        out["gpu_mem_used_gb"]  = used_mb / 1024.0
        out["gpu_mem_total_gb"] = total_mb / 1024.0
        out["gpu_mem_util_pct"] = 100.0 * used_mb / max(total_mb, 1.0)
    except Exception:
        pass
    return out


# =======================
# Callbacks
# =======================

class EpochTimeLogger(Callback):
    def __init__(self, out_dir, filename: str = "resources_and_time.csv"):
        self.out_dir = Path(out_dir)
        self.file = self.out_dir / filename
        self.t0 = None
        self.prev_elapsed_min = 0.0
        self.fieldnames = [
            "epoch", "elapsed_min", "epoch_duration_min",
            "rss_gb", "cpu_load_pct",
            "gpu_util_pct", "gpu_mem_used_gb", "gpu_mem_total_gb", "gpu_mem_util_pct",
        ]

    def _snapshot_row(self, epoch, elapsed_min, epoch_dur_min):
        gpu = _gpu_snapshot()
        return {
            "epoch":              epoch,
            "elapsed_min":        f"{elapsed_min:.3f}",
            "epoch_duration_min": f"{epoch_dur_min:.3f}",
            "rss_gb":             f"{_read_rss_gb():.3f}",
            "cpu_load_pct":       f"{_cpu_load_pct():.2f}",
            "gpu_util_pct":       f"{gpu['gpu_util_pct']:.1f}",
            "gpu_mem_used_gb":    f"{gpu['gpu_mem_used_gb']:.3f}",
            "gpu_mem_total_gb":   f"{gpu['gpu_mem_total_gb']:.3f}",
            "gpu_mem_util_pct":   f"{gpu['gpu_mem_util_pct']:.1f}",
        }

    def on_fit_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.t0 = time.time()
        self.prev_elapsed_min = 0.0
        self.last_written_epoch = None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with self.file.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.t0 is None:
            return
        ep = trainer.current_epoch
        elapsed_min = (time.time() - self.t0) / 60.0
        epoch_dur_min = elapsed_min - self.prev_elapsed_min
        with self.file.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                self._snapshot_row(ep, elapsed_min, epoch_dur_min)
            )
        self.prev_elapsed_min = elapsed_min
        self.last_written_epoch = ep

    def on_fit_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.t0 is None:
            return
        total_min = (time.time() - self.t0) / 60.0
        print(f"[Resources] Wrote {self.file} | last_epoch={self.last_written_epoch} | total={total_min:.2f} min")


class EpochCSVLogger(Callback):
    def __init__(self, out_dir, extra_keys: List[str], filename: str = "metrics.csv"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.out_dir / filename
        self.extra_keys_in_order = extra_keys
        self.best_model_path = self.out_dir / "best_model.pth"
        self._best_sig_prev: Optional[Tuple[int, int]] = None
        self._last_epoch_written: Optional[int] = None

    def _best_sig(self):
        try:
            st = self.best_model_path.stat()
            mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
            return (int(mtime_ns), int(st.st_size))
        except FileNotFoundError:
            return None

    def on_fit_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._best_sig_prev = self._best_sig()
        self._last_epoch_written = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        cur_sig = self._best_sig()
        updated_now = cur_sig is not None and (self._best_sig_prev is None or cur_sig != self._best_sig_prev)

        row: Dict = {
            "epoch":                trainer.current_epoch,
            "train_loss":           metrics.get("train_loss_epoch", metrics.get("train_loss")),
            "val_loss":             metrics.get("val_loss"),
            "lr":                   metrics.get("lr", float("nan")),
            "best_model_is_updated": bool(updated_now),
        }
        for k in self.extra_keys_in_order:
            row[k] = metrics.get(k, float("nan"))
        for k, v in list(row.items()):
            if torch.is_tensor(v):
                row[k] = v.item()

        write_header = not self.file.exists()
        with self.file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._last_epoch_written = trainer.current_epoch
        if cur_sig is not None:
            self._best_sig_prev = cur_sig

    def _patch_last_row_best_flag(self, epoch: int) -> None:
        if not self.file.exists():
            return
        with self.file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        if not rows or "best_model_is_updated" not in fieldnames:
            return
        for r in reversed(rows):
            if str(r.get("epoch", "")) == str(epoch):
                r["best_model_is_updated"] = "True"
                break
        tmp = self.file.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(self.file)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        cur_sig = self._best_sig()
        updated_late = cur_sig is not None and (self._best_sig_prev is None or cur_sig != self._best_sig_prev)
        if updated_late and self._last_epoch_written == trainer.current_epoch:
            self._patch_last_row_best_flag(trainer.current_epoch)
        if cur_sig is not None:
            self._best_sig_prev = cur_sig


class ValidationResidualAndLRMetrics(Callback):
    def __init__(
        self,
        target: str,
        val_loader,
        max_batches: Optional[int] = None,
        target_label: Optional[str] = None,
    ):
        self.target = target
        self.val_loader = val_loader
        self.max_batches = max_batches
        self.target_label = target_label or target

    @staticmethod
    def _quantiles_and_W(x: torch.Tensor):
        if x.numel() == 0:
            nan = float("nan")
            return nan, nan, nan, nan
        qs = torch.tensor([0.16, 0.50, 0.84], dtype=torch.float32)
        p16, p50, p84 = torch.quantile(x.to(torch.float32), qs)
        return float(p16), float(p50), float(p84), float((p84 - p16) / 2.0)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        lr = float("nan")
        try:
            if trainer.optimizers:
                lr = float(trainer.optimizers[0].param_groups[0].get("lr", float("nan")))
        except Exception:
            pass
        pl_module.log("lr", lr, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        residual_deg_all, kappa_all, residual_log10_all = [], [], []

        with torch.no_grad():
            for ib, batch in enumerate(self.val_loader):
                if self.max_batches is not None and ib >= self.max_batches:
                    break
                batch = move_batch_to_device(batch, device)
                pred0 = pl_module(batch)[0].detach().float()

                if self.target in ["zenith", "azimuth"]:
                    pred_angle = pred0[:, 0]
                    pred_kappa = pred0[:, 1]
                    truth = extract_field(batch, self.target).detach().float().view(-1).to(device)
                    residual_rad = (
                        _circular_signed_diff(pred_angle, truth)
                        if self.target == "azimuth"
                        else pred_angle - truth
                    )
                    residual_deg_all.append((residual_rad * (180.0 / math.pi)).cpu())
                    kappa_all.append(pred_kappa.cpu())

                elif self.target == "energy":
                    pred_log10 = pred0.squeeze(-1)
                    true_E = extract_field(batch, self.target_label).detach().float().view(-1).to(device)
                    true_log10 = torch.log10(torch.clamp(true_E, min=eps_like(true_E)))
                    residual_log10_all.append((pred_log10 - true_log10).cpu())

        if was_training:
            pl_module.train()

        def _log(key, val):
            pl_module.log(key, val, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        if self.target in ["zenith", "azimuth"]:
            residuals = torch.cat(residual_deg_all) if residual_deg_all else torch.empty(0)
            kappas    = torch.cat(kappa_all)        if kappa_all        else torch.empty(0)
            p16, p50, p84, W = self._quantiles_and_W(residuals)
            _log("val_residual_p16_deg", p16); _log("val_residual_p50_deg", p50)
            _log("val_residual_p84_deg", p84); _log("val_W_deg", W)
            kp16, kp50, kp84, kW = self._quantiles_and_W(kappas)
            _log("val_kappa_p16", kp16); _log("val_kappa_p50", kp50)
            _log("val_kappa_p84", kp84); _log("val_kappa_W", kW)

        elif self.target == "energy":
            residuals = torch.cat(residual_log10_all) if residual_log10_all else torch.empty(0)
            p16, p50, p84, W = self._quantiles_and_W(residuals)
            _log("val_residual_log10_p16", p16); _log("val_residual_log10_p50", p50)
            _log("val_residual_log10_p84", p84); _log("val_W_log10", W)
            if residuals.numel() > 0:
                _log("val_bias_log10", float(residuals.mean()))
                _log("val_mae_log10",  float(residuals.abs().mean()))
                _log("val_rmse_log10", float(torch.sqrt((residuals ** 2).mean())))
            else:
                for k in ["val_bias_log10", "val_mae_log10", "val_rmse_log10"]:
                    _log(k, float("nan"))


# =======================
# Energy task
# =======================

class DepositedEnergyLog10Task(StandardLearnedTask):
    default_target_labels = ["energy"]
    default_prediction_labels = ["log10_energy_pred"]
    nb_inputs = 1

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def logarithm(E: torch.Tensor) -> torch.Tensor:
    return torch.log10(torch.clamp(E, min=eps_like(E)))

def exponential(t: torch.Tensor) -> torch.Tensor:
    return torch.pow(10.0, t)


# =======================
# Angle / batch helpers
# =======================

def _wrap_to_pi(d: torch.Tensor) -> torch.Tensor:
    period = 2 * math.pi
    return torch.remainder(d + period / 2, period) - period / 2

def _circular_signed_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _wrap_to_pi(a - b)

def _circular_abs_diff(a: torch.Tensor, b: torch.Tensor, period: float = 2 * math.pi) -> torch.Tensor:
    return (torch.remainder(a - b + period / 2, period) - period / 2).abs()

def extract_field(batch, field: str) -> torch.Tensor:
    if isinstance(batch, Data):
        return batch[field]
    if isinstance(batch, (list, tuple)):
        return torch.cat([b[field] for b in batch], dim=0)
    raise TypeError(f"Unsupported batch type: {type(batch)}")

def move_batch_to_device(batch, device):
    if isinstance(batch, Data):
        return batch.to(device)
    return [b.to(device) for b in batch]

def maybe_extract_event_id(batch) -> Optional[torch.Tensor]:
    for key in ["event_id", "event_no", "event", "idx"]:
        try:
            return extract_field(batch, key)
        except Exception:
            pass
    return None


# =======================
# Data loading
# =======================

def build_data(cfg: dict):
    features     = cfg["data"]["features"]
    truth_all    = cfg["data"]["truth_all"]
    train_path   = cfg["data"]["train_path"]
    val_path     = cfg["data"]["val_path"]
    test_path    = cfg["data"].get("test_path", None)
    pulsemaps    = cfg["data"]["pulsemaps"]
    truth_table  = cfg["data"]["truth_table"]
    batch_size   = cfg["training"]["batch_size"]
    num_workers  = cfg["training"]["num_workers"]
    mp_context   = cfg["training"].get("multiprocessing_context", "spawn")
    pin_memory   = cfg["training"].get("pin_memory", True)
    nb_neighbours = cfg["model"]["nb_neighbours"]

    percentiles_csv = cfg["data"]["percentiles_csv"]

    print("[Data] Building KNNGraph with PONE detector")
    data_representation = KNNGraph(
        detector=PONE(percentiles_csv=percentiles_csv, selected_features=features),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=nb_neighbours,
        distance_as_edge_feature=False,
    )

    def _make_dataset(path):
        return ParquetDataset(
            path=path,
            pulsemaps=pulsemaps,
            truth_table=truth_table,
            features=features,
            truth=truth_all,
            data_representation=data_representation,
        )

    def _make_loader(ds, shuffle, drop_last=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=True,
            pin_memory=pin_memory,
        )

    train_loader = _make_loader(_make_dataset(train_path), shuffle=True, drop_last=True)
    val_loader   = _make_loader(_make_dataset(val_path),   shuffle=False)
    test_loader  = _make_loader(_make_dataset(test_path),  shuffle=False) if test_path else None

    print(f"[Data] train={len(train_loader)} batches | val={len(val_loader)} batches")
    if test_loader:
        print(f"[Data] test={len(test_loader)} batches")

    return data_representation, train_loader, val_loader, test_loader


# =======================
# Test writer (reconstruction)
# =======================

def run_test(cfg: dict, target: str, model: pl.LightningModule, test_loader, out_dir: str) -> None:
    if test_loader is None:
        print(f"[Test={target}] No test_loader, skipping.")
        return

    best_path = os.path.join(out_dir, "best_model.pth")
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"[Test={target}] Loaded best weights: {best_path}")
    else:
        print(f"[Test={target}] WARN: best_model.pth not found, using last weights.")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    device = next(model.parameters()).device
    out_csv = os.path.join(out_dir, cfg.get("test_csv_name", "test_predictions.csv"))

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)
            pred0 = model(batch)[0].detach().float()

            event_id = maybe_extract_event_id(batch)
            if event_id is not None:
                event_id = event_id.detach().cpu().view(-1)

            def _eid(i):
                return int(event_id[i].item()) if event_id is not None and i < len(event_id) else None

            if target in ["zenith", "azimuth"]:
                pred_angle = pred0[:, 0].cpu()
                kappa      = pred0[:, 1].cpu()
                truth      = extract_field(batch, target).detach().float().view(-1).cpu()
                residual_rad = (
                    _circular_signed_diff(pred_angle, truth)
                    if target == "azimuth"
                    else pred_angle - truth
                )
                residual_deg = residual_rad * (180.0 / math.pi)
                true_deg = truth * (180.0 / math.pi)
                pred_deg = pred_angle * (180.0 / math.pi)

                for i in range(len(truth)):
                    if target == "zenith":
                        rows.append({
                            "true_zenith_radian":   float(truth[i]),
                            "pred_zenith_radian":   float(pred_angle[i]),
                            "true_zenith_degree":   float(true_deg[i]),
                            "pred_zenith_degree":   float(pred_deg[i]),
                            "kappa":                float(kappa[i]),
                            "residual_zenith_radian": float(residual_rad[i]),
                            "residual_zenith_degree": float(residual_deg[i]),
                            "event_id":             _eid(i),
                        })
                    else:
                        true_signed_deg = _wrap_to_pi(truth) * (180.0 / math.pi)
                        pred_signed_deg = _wrap_to_pi(pred_angle) * (180.0 / math.pi)
                        rows.append({
                            "true_azimuth_radian":        float(truth[i]),
                            "pred_azimuth_radian":        float(pred_angle[i]),
                            "true_azimuth_degree":        float(true_deg[i]),
                            "pred_azimuth_degree":        float(pred_deg[i]),
                            "true_azimuth_degree_signed": float(true_signed_deg[i]),
                            "pred_azimuth_degree_signed": float(pred_signed_deg[i]),
                            "pred_azimuth_degree_adj":    float(true_deg[i] + float(residual_deg[i])),
                            "kappa":                      float(kappa[i]),
                            "residual_azimuth_radian":    float(residual_rad[i]),
                            "residual_azimuth_degree":    float(residual_deg[i]),
                            "event_id":                   _eid(i),
                        })

            else:
                pred_log10 = pred0.squeeze(-1).cpu()
                true_E     = extract_field(batch, "energy").detach().float().view(-1).cpu()
                true_log10 = logarithm(true_E)
                pred_E     = exponential(pred_log10)
                for i in range(len(pred_log10)):
                    rows.append({
                        "true_energy":       float(true_E[i]),
                        "pred_energy":       float(pred_E[i]),
                        "pred_log10_energy": float(pred_log10[i]),
                        "true_log10_energy": float(true_log10[i]),
                        "residual_log10":    float(pred_log10[i] - true_log10[i]),
                        "residual":          float(pred_E[i] - true_E[i]),
                        "event_id":          _eid(i),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[Test={target}] Wrote {out_csv} | rows={len(df)}")

    if target in ["zenith", "azimuth"]:
        col = f"residual_{target}_degree"
        rdeg = torch.tensor(df[col].to_numpy(), dtype=torch.float32)
        p16, p50, p84, W = (
            torch.quantile(rdeg, 0.16).item(),
            torch.quantile(rdeg, 0.50).item(),
            torch.quantile(rdeg, 0.84).item(),
            (torch.quantile(rdeg, 0.84) - torch.quantile(rdeg, 0.16)).item() / 2.0,
        )
        print(f"[Test={target}] residual_deg: p16={p16:.3f} p50={p50:.3f} p84={p84:.3f} W={W:.3f} | kappa_mean={df['kappa'].mean():.3f}")
    else:
        r = torch.tensor(df["residual_log10"].to_numpy(), dtype=torch.float32)
        p16 = torch.quantile(r, 0.16).item()
        p50 = torch.quantile(r, 0.50).item()
        p84 = torch.quantile(r, 0.84).item()
        W   = (torch.quantile(r, 0.84) - torch.quantile(r, 0.16)).item() / 2.0
        print(f"[Test=energy] residual_log10: p16={p16:.4f} p50={p50:.4f} p84={p84:.4f} W={W:.4f} | mae={r.abs().mean():.4f} rmse={torch.sqrt((r**2).mean()):.4f}")
