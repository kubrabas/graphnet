"""Scientific-deviation gates for routed joint-direction experiments."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import yaml


THIS_DIR = Path(__file__).resolve().parent
JOINT_DIR = THIS_DIR.parent
PONE_DIR = JOINT_DIR.parent
REFERENCE_DIR = PONE_DIR.parent / "09_pone_muon_direction"
TRAIN_DIR = JOINT_DIR / "train_scripts"
for path in (PONE_DIR, REFERENCE_DIR, JOINT_DIR, TRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_routed_joint_direction import validate_config  # noqa: E402


CONFIG_DIR = PONE_DIR / "configs" / "joint_direction"
BASELINES = tuple(
    CONFIG_DIR / name
    for name in (
        "102_string_emax1e6__category1_isMuonCC.yml",
        "102_string_emax1e6__category_3_contains_muon.yml",
        "160_string_emax1e6__category1_isMuonCC.yml",
        "160_string_emax1e6__category_3_contains_muon.yml",
        "full_geometry_emax1e6__category1_isMuonCC.yml",
        "full_geometry_emax1e6__category_3_contains_muon.yml",
    )
)
BASELINE = BASELINES[0]
EXPERIMENTS = (
    CONFIG_DIR
    / "102_string_emax1e6__category1_isMuonCC__pmt_direction_v1.yml",
    CONFIG_DIR / "102_string_emax1e6__category1_isMuonCC__wide_2p75m_v1.yml",
    CONFIG_DIR / "102_string_emax1e6__category1_isMuonCC__alpha025_v1.yml",
    CONFIG_DIR / "102_string_emax1e6__category1_isMuonCC__alpha075_v1.yml",
    CONFIG_DIR
    / "102_string_emax1e6__category1_isMuonCC__pmt_direction_seed20260203_v1.yml",
    CONFIG_DIR
    / "102_string_emax1e6__category1_isMuonCC__pmt_direction_wide_2p75m_v1.yml",
    CONFIG_DIR
    / "102_string_emax1e6__category1_isMuonCC__pmt_direction_v1_class0.yml",
    CONFIG_DIR
    / "160_string_emax1e6__category1_isMuonCC__pmt_direction_v1.yml",
    CONFIG_DIR
    / "102_string_emax1e6__category1_isMuonCC__fourier_t_v1.yml",
    CONFIG_DIR
    / "102_string_emax1e6__category1_isMuonCC__fourier_t_pmt_v1.yml",
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


class ExperimentConfigContractTest(unittest.TestCase):
    """Ablations must be isolated and old configs must remain valid."""

    def test_baseline_and_all_experiments_are_valid(self) -> None:
        for path in (*BASELINES, *EXPERIMENTS):
            with self.subTest(path=path.name):
                validate_config(_load(path))

    def test_absent_and_explicit_empty_augmentation_are_equivalent(self) -> None:
        config = _load(BASELINE)
        self.assertNotIn("node_feature_augmentations", config["data"])
        validate_config(config)
        config["data"]["node_feature_augmentations"] = []
        validate_config(config)

    def test_undeclared_alpha_change_is_rejected(self) -> None:
        config = _load(BASELINE)
        config["weighting"]["alpha"] = 0.25
        with self.assertRaisesRegex(ValueError, "declaration"):
            validate_config(config)

    def test_accidental_second_change_is_rejected(self) -> None:
        config = _load(EXPERIMENTS[0])
        config["weighting"]["alpha"] = 0.75
        with self.assertRaisesRegex(ValueError, "declaration"):
            validate_config(config)

    def test_declared_but_unchanged_field_is_rejected(self) -> None:
        config = copy.deepcopy(_load(BASELINE))
        config["experiment_contract"] = {"varied_fields": ["weighting.alpha"]}
        with self.assertRaisesRegex(ValueError, "declaration"):
            validate_config(config)

    def test_undeclared_seed_change_is_rejected(self) -> None:
        config = _load(BASELINE)
        config["training"]["seed"] = 20260203
        with self.assertRaisesRegex(ValueError, "declaration"):
            validate_config(config)

    def test_transformer_pmt_contract_cannot_silently_drift(self) -> None:
        config = _load(EXPERIMENTS[-1])
        config["model"]["transformer"]["use_pmt_direction"] = False
        with self.assertRaisesRegex(ValueError, "use_pmt_direction"):
            validate_config(config)

    def test_transformer_requires_paper_optimizer(self) -> None:
        config = _load(EXPERIMENTS[-2])
        config["optimizer"]["name"] = "adam"
        with self.assertRaisesRegex(ValueError, "adamw"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
