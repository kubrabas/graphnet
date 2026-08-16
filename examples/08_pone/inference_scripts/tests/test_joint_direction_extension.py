"""Focused, dependency-free contracts for routed joint inference.

The production module imports CUDA/dataframe dependencies that are available
inside the GraphNeT container, not necessarily on a login node. These tests
therefore execute the two path resolvers directly from its AST and inspect the
loader call graph without importing the full runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Mapping
import unittest

import yaml


THIS_DIR = Path(__file__).resolve().parent
SCRIPT = THIS_DIR.parent / "run_inference.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(SCRIPT))
ROUTED_DATA_SCRIPT = (
    THIS_DIR.parent.parent / "joint_direction" / "routed_data.py"
)
ROUTED_DATA_TREE = ast.parse(
    ROUTED_DATA_SCRIPT.read_text(encoding="utf-8"),
    filename=str(ROUTED_DATA_SCRIPT),
)
PMT_CONFIG = (
    THIS_DIR.parent.parent
    / "configs"
    / "joint_direction"
    / "102_string_emax1e6__category1_isMuonCC__pmt_direction_v1.yml"
)
SHIPPED_JOINT_INFERENCE_CONFIGS = (
    THIS_DIR.parent.parent / "configs" / "inference_joint_direction"
)


def _load_functions(*names: str) -> dict:
    selected = [
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    found = {node.name for node in selected}
    if found != set(names):
        raise AssertionError(f"Missing functions: {sorted(set(names) - found)}")
    namespace = {"Path": Path}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, filename=str(SCRIPT), mode="exec"), namespace)
    return namespace


FUNCTIONS = _load_functions(
    "resolve_joint_direction_experiment_name",
    "resolve_joint_direction_root",
    "resolve_joint_direction_checkpoint",
    "validate_config",
)
resolve_joint_direction_root = FUNCTIONS["resolve_joint_direction_root"]
resolve_joint_direction_checkpoint = FUNCTIONS[
    "resolve_joint_direction_checkpoint"
]
validate_config = FUNCTIONS["validate_config"]


def _config() -> tuple[dict, dict]:
    inference = {
        "output": {"root_dir": "/results"},
        "mc": "340StringMC",
        "geometry": "102_string_emax1e6",
        "routing": {"category": "category1_isMuonCC"},
        "reconstruction": {"experiment_name": "energy_baseline"},
        "joint_direction": {
            "stage": "stage_b",
            "checkpoint_name": "best_macro_median",
        },
    }
    joint = {"output": {"dirs": {"train": "train_and_val"}}}
    return inference, joint


class JointExperimentPathTest(unittest.TestCase):
    def test_absent_override_is_exact_legacy_path(self) -> None:
        config, joint = _config()
        root = resolve_joint_direction_root(config, joint, "1")
        self.assertEqual(
            root,
            Path(
                "/results/340StringMC/102_string_emax1e6/reconstruction/"
                "category1_isMuonCC/class1/energy_baseline/train_and_val/"
                "zenith_azimuth"
            ),
        )

    def test_all_shipped_configs_keep_the_legacy_experiment_fallback(self) -> None:
        configs = sorted(SHIPPED_JOINT_INFERENCE_CONFIGS.glob("*.yml"))
        self.assertTrue(configs)
        for config_path in configs:
            with self.subTest(config=config_path.name):
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                self.assertNotIn(
                    "experiment_name", config.get("joint_direction", {})
                )
                joint_path = Path(config["joint_direction"]["config"])
                joint = yaml.safe_load(joint_path.read_text(encoding="utf-8"))
                for route_class in config["routing"]["classes"]:
                    actual = resolve_joint_direction_root(
                        config, joint, str(route_class)
                    )
                    expected = (
                        Path(config["output"]["root_dir"])
                        / config["mc"]
                        / config["geometry"]
                        / "reconstruction"
                        / config["routing"]["category"]
                        / f"class{route_class}"
                        / config["reconstruction"]["experiment_name"]
                        / joint["output"]["dirs"]["train"]
                        / "zenith_azimuth"
                    )
                    self.assertEqual(actual, expected)

    def test_explicit_joint_experiment_is_independent_of_energy(self) -> None:
        config, joint = _config()
        config["joint_direction"]["experiment_name"] = "pmt_direction_v1"
        root = resolve_joint_direction_root(config, joint, "1")
        self.assertEqual(root.parts[-3], "pmt_direction_v1")
        self.assertNotIn("energy_baseline", root.parts)
        self.assertEqual(
            resolve_joint_direction_checkpoint(config, joint, "1"),
            root
            / "stage_b_angular_hybrid"
            / "checkpoints"
            / "best_macro_median.pth",
        )

    def test_explicit_joint_experiment_cannot_escape_result_tree(self) -> None:
        for invalid in ("", ".", "..", "../baseline", "/tmp/baseline", "a/b"):
            with self.subTest(invalid=invalid):
                config, joint = _config()
                config["joint_direction"]["experiment_name"] = invalid
                with self.assertRaisesRegex(ValueError, "one non-empty"):
                    resolve_joint_direction_root(config, joint, "1")

    def test_legacy_experiment_fallback_cannot_escape_result_tree(self) -> None:
        for invalid in ("", ".", "..", "../baseline", "/tmp/baseline", "a/b"):
            with self.subTest(invalid=invalid):
                config, joint = _config()
                config["reconstruction"]["experiment_name"] = invalid
                with self.assertRaisesRegex(ValueError, "one non-empty"):
                    resolve_joint_direction_root(config, joint, "1")


class JointLoaderRoutingTest(unittest.TestCase):
    def test_only_joint_branch_uses_joint_feature_loader(self) -> None:
        functions = {
            node.name: node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
        }
        run_reconstruction = functions["run_reconstruction"]
        calls = [
            node.func.id
            for node in ast.walk(run_reconstruction)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertEqual(calls.count("build_joint_loader"), 1)
        self.assertEqual(calls.count("build_loader"), 1)

        joint_loader = functions["build_joint_loader"]
        joint_names = {
            node.func.id
            for node in ast.walk(joint_loader)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("joint_loader_feature_names", joint_names)
        self.assertIn("build_joint_data_representation", joint_names)

    def test_inference_rejects_joint_experiment_config_mismatch(self) -> None:
        config = {
            "task": {"type": "inference"},
            "mc": "340StringMC",
            "geometry": "102_string_emax1e6",
            "routing": {"category": "category1_isMuonCC", "classes": [1]},
            "reconstruction": {"targets": ["zenith_azimuth"]},
            "joint_direction": {"experiment_name": "pmt_direction_v1"},
        }
        classification = {
            "mc": config["mc"],
            "geometry": config["geometry"],
            "task": {
                "labels": [0, 1],
                "target": config["routing"]["category"],
            },
        }
        reconstruction = {
            "mc": config["mc"],
            "geometry": config["geometry"],
            "routing": {"category": config["routing"]["category"]},
        }
        joint = {
            "experiment_name": "baseline",
            "mc": config["mc"],
            "geometry": config["geometry"],
            "routing": {"category": config["routing"]["category"]},
        }
        with self.assertRaisesRegex(ValueError, "experiment mismatch"):
            validate_config(config, classification, reconstruction, joint)

    def test_absent_override_rejects_nonbaseline_joint_config(self) -> None:
        config = {
            "task": {"type": "inference"},
            "mc": "340StringMC",
            "geometry": "102_string_emax1e6",
            "routing": {"category": "category1_isMuonCC", "classes": [1]},
            "reconstruction": {
                "experiment_name": "baseline",
                "targets": ["zenith_azimuth"],
            },
            # This intentionally omits experiment_name, exercising the legacy
            # reconstruction fallback that must not silently select baseline
            # for a non-baseline joint training config.
            "joint_direction": {},
        }
        classification = {
            "mc": config["mc"],
            "geometry": config["geometry"],
            "task": {
                "labels": [0, 1],
                "target": config["routing"]["category"],
            },
        }
        reconstruction = {
            "mc": config["mc"],
            "geometry": config["geometry"],
            "routing": {"category": config["routing"]["category"]},
        }
        joint = {
            "experiment_name": "pmt_direction_v1",
            "mc": config["mc"],
            "geometry": config["geometry"],
            "routing": {"category": config["routing"]["category"]},
        }
        with self.assertRaisesRegex(
            ValueError, "Set joint_direction.experiment_name explicitly"
        ):
            validate_config(config, classification, reconstruction, joint)

    def test_absent_override_still_accepts_matching_baseline_configs(self) -> None:
        config = {
            "task": {"type": "inference"},
            "mc": "340StringMC",
            "geometry": "102_string_emax1e6",
            "routing": {"category": "category1_isMuonCC", "classes": [1]},
            "reconstruction": {
                "experiment_name": "baseline",
                "targets": ["zenith_azimuth"],
            },
            "joint_direction": {},
        }
        classification = {
            "mc": config["mc"],
            "geometry": config["geometry"],
            "task": {
                "labels": [0, 1],
                "target": config["routing"]["category"],
            },
        }
        reconstruction = {
            "mc": config["mc"],
            "geometry": config["geometry"],
            "routing": {"category": config["routing"]["category"]},
        }
        joint = {
            "experiment_name": "baseline",
            "mc": config["mc"],
            "geometry": config["geometry"],
            "routing": {"category": config["routing"]["category"]},
        }
        validate_config(config, classification, reconstruction, joint)

    def test_pmt_yaml_reaches_joint_loader_with_auxiliary_column(self) -> None:
        """Exercise the production call contract without runtime dependencies."""

        joint_config = yaml.safe_load(PMT_CONFIG.read_text(encoding="utf-8"))

        class _Contract:
            scaled_features = (
                "pmt_x",
                "pmt_y",
                "pmt_z",
                "dom_time",
                "charge",
            )
            identity_features = ("pmt_number",)

            @property
            def loader_features(self):
                return self.scaled_features + self.identity_features

        resolver_node = next(
            node
            for node in ROUTED_DATA_TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "loader_feature_names"
        )
        resolver_namespace = {
            "Any": Any,
            "Mapping": Mapping,
            "node_feature_augmentations": lambda config: tuple(
                config.get("data", {}).get("node_feature_augmentations", [])
            ),
            "PONE_V3_PMT_DIRECTION_CONTRACT": _Contract(),
        }
        exec(
            compile(
                ast.Module(body=[resolver_node], type_ignores=[]),
                filename=str(ROUTED_DATA_SCRIPT),
                mode="exec",
            ),
            resolver_namespace,
        )
        loader_features = resolver_namespace["loader_feature_names"]
        self.assertEqual(
            loader_features(joint_config),
            [
                "pmt_x",
                "pmt_y",
                "pmt_z",
                "dom_time",
                "charge",
                "pmt_number",
            ],
        )

        dataset_calls = []

        class _Dataset:
            def __init__(self, **kwargs):
                dataset_calls.append(kwargs)

        class _Ensemble:
            def __init__(self, datasets):
                self.datasets = datasets

        class _Loader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset
                self.kwargs = kwargs

        class _Graph:
            output_feature_names = [
                "pmt_x",
                "pmt_y",
                "pmt_z",
                "dom_time",
                "charge",
                "pmt_dir_x",
                "pmt_dir_y",
                "pmt_dir_z",
            ]

        graph = _Graph()
        graph_calls = []

        def _build_graph(config, percentiles_csv):
            graph_calls.append((config, percentiles_csv))
            return graph

        def _unique(items):
            result = []
            for item in items:
                if item not in result:
                    result.append(item)
            return result

        build_node = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "build_joint_loader"
        )
        namespace = {
            "Dict": Dict,
            "EVENT_ID_FIELDS": [
                "event_no",
                "RunID",
                "SubrunID",
                "EventID",
                "SubEventID",
            ],
            "unique": _unique,
            "joint_loader_feature_names": loader_features,
            "build_joint_data_representation": _build_graph,
            "ParquetDataset": _Dataset,
            "EnsembleDataset": _Ensemble,
            "DataLoader": _Loader,
        }
        exec(
            compile(
                ast.Module(body=[build_node], type_ignores=[]),
                filename=str(SCRIPT),
                mode="exec",
            ),
            namespace,
        )
        inference_config = {
            "data": {
                "truth_all": ["event_no", "zenith", "azimuth"],
                "pulsemaps": "features",
                "truth_table": "truth",
            },
            "inference": {
                "batch_size": 256,
                "num_workers": 0,
                "pin_memory": True,
            },
        }
        returned_graph, loader = namespace["build_joint_loader"](
            inference_config,
            {"Muon": "/data/muon", "Electron": "/data/electron"},
            "/metadata/class1_percentiles.csv",
            joint_config,
        )

        self.assertIs(returned_graph, graph)
        self.assertEqual(
            graph_calls,
            [(joint_config, "/metadata/class1_percentiles.csv")],
        )
        self.assertEqual(len(dataset_calls), 2)
        self.assertTrue(
            all(call["features"] == loader_features(joint_config)
                for call in dataset_calls)
        )
        self.assertEqual(len(loader.dataset.datasets), 2)
        self.assertEqual(loader.kwargs["batch_size"], 256)


if __name__ == "__main__":
    unittest.main()
