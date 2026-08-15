"""Run validation inference for router comparison.

For every validation event, this script records reconstruction predictions from
the route selected by the classifier and from the truth-defined route.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


THIS_DIR = Path(__file__).resolve().parent
PONE_DIR = THIS_DIR.parent
INFERENCE_DIR = PONE_DIR / "inference_scripts"
for path in (PONE_DIR, INFERENCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_inference as inference


ID_COLUMNS = ["pid", "is_CC", "RunID", "EventID"]
TRUTH_COLUMNS = ["totalEnergy", "zenith", "azimuth"]


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validation_config(cfg: dict) -> tuple[dict, dict, dict, dict | None]:
    base_path = Path(cfg["base_inference_config"]).expanduser().resolve()
    base_cfg = load_yaml(base_path)
    base_cfg["data"]["split"] = "val"
    base_cfg["notebook"] = {"enabled": False}

    cls_cfg = load_yaml(base_cfg["classification"]["config"])
    reco_cfg = load_yaml(base_cfg["reconstruction"]["config"])
    joint_path = (base_cfg.get("joint_direction", {}) or {}).get("config")
    joint_cfg = load_yaml(joint_path) if joint_path else None
    inference.validate_config(base_cfg, cls_cfg, reco_cfg, joint_cfg)
    return base_cfg, cls_cfg, reco_cfg, joint_cfg


def output_path(cfg: dict) -> Path:
    return (
        Path(cfg["output"]["root_dir"])
        / cfg["mc"]
        / cfg["geometry"]
        / "inference_validation"
        / cfg["output"]["filename"]
    )


def handle_existing_output(path: Path, policy: str) -> None:
    if not path.exists():
        return
    if policy == "error":
        raise FileExistsError(f"Output file already exists: {path}")
    if policy == "skip":
        print(f"[Run] output exists; skipping: {path}")
        raise SystemExit(0)
    if policy == "overwrite":
        print(f"[Run] removing existing output: {path}")
        path.unlink()
        return
    raise ValueError(f"Unsupported run.existing_output policy: {policy}")


def reconstruction_frame(
    reco_df: pd.DataFrame,
    router_name: str,
    route_kind: str,
) -> pd.DataFrame:
    output = None
    columns_by_target = {
        "energy": {
            "pred_energy": f"{router_name}_{route_kind}_energy",
            "pred_log10_energy": f"{router_name}_{route_kind}_log10_energy",
        },
        "zenith": {
            "pred_zenith_radian": f"{router_name}_{route_kind}_zenith",
        },
        "azimuth": {
            "pred_azimuth_radian": f"{router_name}_{route_kind}_azimuth",
        },
    }

    for target, rename in columns_by_target.items():
        target_df = reco_df.loc[
            reco_df["target"] == target,
            ["event_key", *rename],
        ].copy()
        if target_df.duplicated("event_key").any():
            raise ValueError(f"Duplicate event keys in {route_kind} {target} predictions")
        target_df = target_df.rename(columns=rename)
        output = (
            target_df
            if output is None
            else output.merge(target_df, on="event_key", how="outer", validate="one_to_one")
        )

    if output is None:
        raise ValueError(f"No {route_kind} reconstruction predictions were produced")
    return output


def comparison_frame(
    cfg: dict,
    cls_cfg: dict,
    cls_df: pd.DataFrame,
    routed_reco: pd.DataFrame,
    oracle_reco: pd.DataFrame,
) -> pd.DataFrame:
    router_name = cfg["router_name"]
    probability_prefix = cls_cfg["task"].get("prediction_prefix", "p_class")
    probability_columns = [
        f"{probability_prefix}_{label}" for label in cls_cfg["task"]["labels"]
    ]

    base = cls_df[
        [
            "event_key",
            *ID_COLUMNS,
            *TRUTH_COLUMNS,
            "true_classification_class",
            "predicted_route_class",
            *probability_columns,
        ]
    ].copy()
    base = base.rename(
        columns={
            "true_classification_class": f"{router_name}_correct_class",
            "predicted_route_class": f"{router_name}_predicted_class",
            **{
                column: f"{router_name}_{column}"
                for column in probability_columns
            },
        }
    )
    base["true_log10_energy"] = np.log10(base["totalEnergy"])
    base[f"{router_name}_routed_correctly"] = (
        base[f"{router_name}_correct_class"].astype(int)
        == base[f"{router_name}_predicted_class"].astype(int)
    ).astype(int)

    routed = reconstruction_frame(routed_reco, router_name, "routed")
    oracle = reconstruction_frame(oracle_reco, router_name, "oracle")
    output = base.merge(routed, on="event_key", how="left", validate="one_to_one")
    output = output.merge(oracle, on="event_key", how="left", validate="one_to_one")

    prediction_columns = [
        f"{router_name}_{kind}_{target}"
        for kind in ("routed", "oracle")
        for target in ("energy", "log10_energy", "zenith", "azimuth")
    ]
    missing = output[prediction_columns].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"Missing routed/oracle predictions for {int(missing.sum())} validation events"
        )
    if output.duplicated(ID_COLUMNS).any():
        raise ValueError("pid, is_CC, RunID, and EventID do not uniquely identify validation events")

    ordered_columns = [
        *ID_COLUMNS,
        "totalEnergy",
        "true_log10_energy",
        "zenith",
        "azimuth",
        f"{router_name}_correct_class",
        f"{router_name}_predicted_class",
        f"{router_name}_routed_correctly",
        *[f"{router_name}_{column}" for column in probability_columns],
        *prediction_columns,
    ]
    return output[ordered_columns].sort_values(ID_COLUMNS).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if cfg["task"]["type"] != "inference_validation":
        raise ValueError("Expected task.type=inference_validation")

    base_cfg, cls_cfg, reco_cfg, joint_cfg = validation_config(cfg)
    if cfg["mc"] != base_cfg["mc"] or cfg["geometry"] != base_cfg["geometry"]:
        raise ValueError("Validation and base inference configs must use the same mc/geometry")

    inference.preflight(base_cfg, cls_cfg, reco_cfg, joint_cfg)
    inference.require_cuda(base_cfg)
    destination = output_path(cfg)
    if os.environ.get("OUTPUT_PREPARED", "0") != "1":
        handle_existing_output(
            destination,
            cfg.get("run", {}).get("existing_output", "error"),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        args.config,
        destination.parent / f"{cfg['router_name']}_pipeline_config.yml",
    )

    paths = inference.resolve_mixed_split_paths(base_cfg)
    cls_df = inference.run_classification(base_cfg, cls_cfg, paths)
    routed_reco, oracle_reco = inference.run_reconstruction(
        base_cfg, reco_cfg, joint_cfg, paths, cls_df
    )

    output = comparison_frame(
        cfg,
        cls_cfg,
        cls_df,
        routed_reco,
        oracle_reco,
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(destination)
    print(f"[Output] wrote {len(output):,} validation events: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
