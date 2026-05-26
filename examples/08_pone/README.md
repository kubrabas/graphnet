# 08_pone - P-ONE GraphNeT Pipeline

Config-driven training scripts for the P-ONE / String340MC GraphNeT studies.
The active workflow is split into two stages:

1. Classification models for event routing.
2. Routed reconstruction models trained separately for every routing class and target.

The scripts are written to run inside the GraphNeT training container through the
SLURM wrappers in `/home/kbas/SlurmScripts/GraphNet`.

## Active Scripts

| Path | Purpose |
| --- | --- |
| `train_scripts/train_classification.py` | Trains one classification model from a config. Called by the SLURM wrapper. |
| `train_scripts/train_reconstruction.py` | Trains one reconstruction model for one routing class and one target. Called by the SLURM wrapper. |
| `pipeline_utils.py` | Classification path resolution, loaders, validation diagnostics, plotting helpers. |
| `utils.py` | Shared GraphNeT callbacks, resource logging, reconstruction task helpers, residual metrics. |
| `/home/kbas/SlurmScripts/GraphNet/submit_classification_pipeline.py` | Submit one classification training job. |
| `/home/kbas/SlurmScripts/GraphNet/submit_reconstruction_pipeline.py` | Expand one reconstruction config into class x target SLURM jobs. |
| `/home/kbas/SlurmScripts/GraphNet/train_classification.sh` | Container/environment wrapper for classification. |
| `/home/kbas/SlurmScripts/GraphNet/train_reconstruction.sh` | Container/environment wrapper for reconstruction. |

Older top-level scripts such as `02_train_reconstruction_combined.py` and
`03_train_reconstruction_separate.py` are kept as historical references. The
clean scripts under `train_scripts/` are the current workflow.

## Classification

Classification configs live in:

```text
configs/classification/
```

Current examples:

| Config | Geometry | Target | Mode |
| --- | --- | --- | --- |
| `exp001.yml` | `102_string_emax1e6` | `category1` | binary |
| `exp002.yml` | `102_string_emax1e6` | `category2` | multiclass |
| `exp003.yml` | `full_geometry_emax1e6` | `category1` | binary |
| `exp004.yml` | `full_geometry_emax1e6` | `category2` | multiclass |

Submit one classification job:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_classification_pipeline.py \
  -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/classification/exp001.yml
```

Exclude a bad node if needed:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_classification_pipeline.py \
  -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/classification/exp003.yml \
  --exclude fc10713
```

Output layout:

```text
/project/def-nahee/kbas/Graphnet-Applications/Results/
  <mc>/<geometry>/classification/<target>/<experiment_name>/train_and_val/
```

Example:

```text
/project/def-nahee/kbas/Graphnet-Applications/Results/
  340StringMC/full_geometry_emax1e6/classification/category1/exp003/train_and_val/
```

Classification training writes the config snapshot, best model, training history,
resource log, and validation diagnostics into `train_and_val/`.

## Reconstruction

Reconstruction configs live in:

```text
configs/reconstruction/
```

The current example is:

```text
configs/reconstruction/exp001.yml
```

Reconstruction is routed by a classification category. The config chooses the
routing category and class selection:

```yaml
routing:
  category: first_category
  classes: all
```

`classes: all` means the submit wrapper discovers distinct class ids from
`Metadata/paths.py`. You can restrict it for a rerun:

```yaml
routing:
  category: first_category
  classes: [0]
```

Targets are also selected in the config:

```yaml
task:
  type: reconstruction
  mode: separate
  targets: [energy, zenith, azimuth]
```

To train only one target, edit the config copy:

```yaml
task:
  targets: [energy]
```

Submit reconstruction jobs:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_reconstruction_pipeline.py \
  -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/reconstruction/exp001.yml
```

Dry-run first:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_reconstruction_pipeline.py \
  -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/reconstruction/exp001.yml \
  --dry-run
```

For the default `exp001.yml`, `full_geometry_emax1e6 + first_category` resolves
classes `0` and `1`; with targets `energy`, `zenith`, and `azimuth`, the submit
wrapper creates 6 independent SLURM jobs:

```text
class0 energy
class0 zenith
class0 azimuth
class1 energy
class1 zenith
class1 azimuth
```

Each job trains exactly one model. The worker script usage shown in
`train_scripts/train_reconstruction.py` is for the SLURM wrapper, not the normal
manual entry point.

Reconstruction output layout:

```text
/project/def-nahee/kbas/Graphnet-Applications/Results/
  <mc>/<geometry>/reconstruction/<routing_category>/class<class_id>/<experiment_name>/train_and_val/<target>/
```

Example:

```text
/project/def-nahee/kbas/Graphnet-Applications/Results/
  340StringMC/full_geometry_emax1e6/reconstruction/first_category/class0/exp001/train_and_val/energy/
```

Each target folder contains:

```text
config.yml
best_model.pth
training_history_by_epoch.csv
resources_and_time.csv
validation_predictions.csv
validation_metrics_summary.csv
validation_residual_log10_distribution.png        # energy
validation_true_vs_pred_log10_energy.png          # energy
validation_residual_degree_distribution.png       # zenith/azimuth
validation_kappa_distribution.png                 # zenith/azimuth
```

All reconstruction validation CSVs and plots are computed after loading
`best_model.pth` and using the validation split.

## Existing Output Policy

Reconstruction configs include:

```yaml
run:
  existing_output: error   # error | skip | overwrite
```

Behavior is based on the target output directory:

- `error`: fail if the target directory already exists.
- `skip`: do not submit jobs whose target directory already exists.
- `overwrite`: remove the existing target directory before submitting the job.

## Weights And Overrides

Per-event loss weights are supported through GraphNeT's `ParquetDataset` and task
`loss_weight` mechanism:

```yaml
weights:
  enabled: false
  loss_weight_table: truth
  loss_weight_column: event_weight
  loss_weight_default_value: 1.0
```

Target-specific training/model/task overrides are present but disabled by default:

```yaml
target_overrides:
  enabled: false
  energy:
    training: {}
    model: {}
    target_settings: {}
```

SLURM settings are also in the reconstruction config. `fc10713` is excluded by
default because it produced CUDA unavailable errors in classification jobs.

```yaml
slurm:
  account: def-nahee
  time: "12:00:00"
  mem: 48G
  cpus_per_task: 8
  gpus_per_node: nvidia_h100_80gb_hbm3_3g.40gb:1
  exclude: fc10713
```

Target-specific SLURM overrides are available but disabled by default.

## Data And Paths

Training data and percentiles are resolved from:

```text
/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py
```

Classification uses flavor-level train/val paths and mixed robust-scaler
percentiles. Reconstruction uses categorized paths:

```python
STRING340MC_PARQUET[geometry][flavor][routing.category][class_id][split]
```

For each routing class, available flavor paths are mixed with `EnsembleDataset`.
`does_not_exit` paths are skipped; `None` paths are treated as configuration
errors. The job log prints the number of feature/truth parquet files found for
each input path.

## Environment

GPU training uses:

```text
docker://rorsoe/graphnet:graphnet-1.8.0-cu126-torch26-ubuntu-22.04
```

The SLURM wrappers load:

```bash
module --force purge
module load StdEnv/2020 gcc/11.3.0 apptainer scipy-stack/2023b
```

and set:

```bash
PYTHONPATH=/project/def-nahee/kbas/graphnet/src:/project/def-nahee/kbas/graphnet/examples/08_pone
```
