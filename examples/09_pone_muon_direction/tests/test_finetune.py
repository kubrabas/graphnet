"""Dependency-light tests for safe weights-only fine-tune initialization."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from finetune_utils import (  # noqa: E402
    resolve_finetune_source,
    sha256_file,
    validate_submission_destination,
    write_or_validate_source_manifest,
)


class FineTuneSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source = root / "source_experiment"
        self.output = root / "new_finetune_experiment"
        checkpoint_dir = (
            self.source / "stage_b_angular_hybrid" / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True)
        self.checkpoint = checkpoint_dir / "best_macro_median.ckpt"
        self.checkpoint.write_bytes(b"pinned checkpoint bytes")

        common = {
            "data": {"geometry": "102", "features": ["x", "t"]},
            "weighting": {"alpha": 0.5},
            "loss": {"angular_surrogate": "opening_angle", "vmf_factor": 0.05},
            "metrics": {"bins": [2.0, 2.5, 3.0]},
            "loader": {"batch_size": 256},
            "model": {"name": "dynedge"},
            "checkpointing": {"primary_monitor": "val_macro_median_deg"},
            "training": {"stage_b": {"objective": "angular_hybrid"}},
        }
        with (self.source / "resolved_config.yml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(common, handle, sort_keys=False)
        (self.source / "energy_weight_manifest.json").write_text(
            '{"alpha": 0.5}\n', encoding="utf-8"
        )
        index = {
            "stage": "stage_b",
            "best": {
                "best_macro_median": {
                    "checkpoint": str(self.checkpoint),
                    "epoch": 29,
                    "metric": "val_macro_median_deg",
                    "value": 5.646633148193359,
                }
            },
        }
        with (checkpoint_dir / "checkpoint_index.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(index, handle)

        self.config = copy.deepcopy(common)
        self.config["fine_tuning"] = {
            "initialization_mode": "weights_only",
            "source_experiment_dir": str(self.source),
            "source_stage_dir": "stage_b_angular_hybrid",
            "source_checkpoint_name": "best_macro_median",
            "source_checkpoint_sha256": sha256_file(self.checkpoint),
            "source_resolved_config_sha256": sha256_file(
                self.source / "resolved_config.yml"
            ),
            "source_energy_manifest_sha256": sha256_file(
                self.source / "energy_weight_manifest.json"
            ),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_source_is_pinned_and_manifest_is_idempotent(self) -> None:
        source = resolve_finetune_source(self.config, self.output)
        self.assertEqual(source.checkpoint_epoch, 29)
        self.assertEqual(source.checkpoint_metric, "val_macro_median_deg")
        self.assertAlmostEqual(source.checkpoint_value, 5.646633148193359)
        self.assertEqual(source.checkpoint_sha256, sha256_file(self.checkpoint))

        manifest = self.output / "finetune_source_manifest.json"
        write_or_validate_source_manifest(source, manifest)
        first = manifest.read_bytes()
        write_or_validate_source_manifest(source, manifest)
        self.assertEqual(manifest.read_bytes(), first)
        payload = json.loads(first)
        self.assertTrue(payload["network_state_dict_strict"])
        self.assertFalse(payload["optimizer_state_restored"])
        self.assertFalse(payload["scheduler_state_restored"])
        payload["source_checkpoint_epoch"] = 999
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source manifest differs"):
            write_or_validate_source_manifest(source, manifest)

    def test_sha256_mismatch_is_rejected(self) -> None:
        self.config["fine_tuning"]["source_checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            resolve_finetune_source(self.config, self.output)

    def test_source_metadata_hash_mismatch_is_rejected(self) -> None:
        self.config["fine_tuning"]["source_energy_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "energy manifest SHA256 mismatch"):
            resolve_finetune_source(self.config, self.output)

    def test_source_and_output_must_be_separate(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate, non-nested"):
            resolve_finetune_source(self.config, self.source)
        with self.assertRaisesRegex(ValueError, "separate, non-nested"):
            resolve_finetune_source(self.config, self.source / "nested")

    def test_architecture_or_data_change_is_rejected(self) -> None:
        self.config["model"] = {"name": "different-model"}
        with self.assertRaisesRegex(ValueError, "section 'model' differs"):
            resolve_finetune_source(self.config, self.output)

    def test_non_primary_checkpoint_is_rejected(self) -> None:
        self.config["checkpointing"]["primary_monitor"] = "val_global_median_deg"
        with self.assertRaisesRegex(ValueError, "not the configured primary"):
            resolve_finetune_source(self.config, self.output)

    def test_destination_refuses_overwrite_and_requires_own_resume(self) -> None:
        expected_last = validate_submission_destination(self.output, resume=False)
        self.assertTrue(str(expected_last).startswith(str(self.output)))

        self.output.mkdir()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            validate_submission_destination(self.output, resume=False)
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint is missing"):
            validate_submission_destination(self.output, resume=True)

        expected_last.parent.mkdir(parents=True)
        expected_last.write_bytes(b"new experiment checkpoint")
        self.assertEqual(
            validate_submission_destination(self.output, resume=True),
            expected_last.resolve(),
        )


if __name__ == "__main__":
    unittest.main()
