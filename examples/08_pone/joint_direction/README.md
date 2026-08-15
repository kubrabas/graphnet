# Routed mixed-flavor joint direction reconstruction

This is a parallel extension of the existing `examples/08_pone` reconstruction
pipeline. It adds one joint `zenith_azimuth` model beside the completed
separate `energy`, `zenith`, and `azimuth` models. Existing source scripts,
parquet files, and result directories are not modified.

## Approved scope

Six configs expand to twelve independent jobs:

- geometries: `102_string_emax1e6`, `160_string_emax1e6`,
  `full_geometry_emax1e6`;
- routing categories: `category1_isMuonCC`,
  `category_3_contains_muon`;
- route classes: `class0` and `class1`.

Each model receives the same truth-categorized train/validation constituents
as the existing reconstruction model for that geometry, category, and class.
The test split and classifier/router model are not used by this training
pipeline.

## Scientific contract

The direction model and losses are the already-tested implementation in
`examples/09_pone_muon_direction`:

- one DynEdge predicts one unit direction `(x, y, z)` and `kappa`;
- Stage A: three epochs of 3D vMF;
- Stage B: at most 30 epochs of
  `opening_angle [radian] + 0.05 * vMF`;
- Stage B starts from the Stage-A `last` checkpoint;
- early stopping and the primary checkpoint use the unweighted validation
  macro-median opening angle;
- validation energy bins are `[2.0, 2.5, ..., 6.0]` in
  `log10(E / GeV)`, with at least 100 events required in every bin;
- the four named checkpoints preserve best macro median, global median,
  global Q68, and weighted objective independently.

Energy weights are fitted separately for every geometry/category/class from
the combined constituent **training** energies only:

- 40 bins from `log10(E / GeV) = 2.0` to `6.0`;
- `weight proportional to N_bin^-0.5`;
- configured clip range `0.2` to `5.0`, followed by normalization to mean one;
- training loss is weighted; validation model-selection metrics are
  unweighted;
- `final_weight`, GraphNeT native loss weights, and parquet weight columns are
  not used.

The six shipped configs preserve these baseline choices. The worker also
hard-rejects changes to the weighting, loss, target, and monitoring contracts
that would make the runs scientifically incomparable.

## Data safety and provenance

Before training, every job checks:

- truth/feature schemas and chunk pairing;
- pulse coverage;
- the correct routing label and `triggered_nonoise_<geometry> == 1`;
- finite physical direction and energy targets;
- all configured weight bins are occupied;
- no train/validation overlap using
  `(source_flavor, RunID, SubrunID, EventID, SubEventID)`.

`event_no` and the four physical IDs alone are not globally unique after
mixing flavors. The loader therefore adds an in-memory `source_flavor_id` to
each graph without changing parquet. Validation predictions persist both the
flavor and all four physical IDs.

`run.existing_output: error` is used in all six configs. The submitter refuses
an already-existing `zenith_azimuth` directory and has no overwrite mode. It
atomically reserves each new task leaf, stores the exact submitted YAML there,
and pins both its SHA256 and resolved output path into the queued worker. A
later edit to the source YAML therefore cannot change a waiting job.

## Output layout

For example, `102_string_emax1e6`, `category1_isMuonCC`, `class0` writes only
below:

```text
Results/340StringMC/102_string_emax1e6/reconstruction/
  category1_isMuonCC/class0/baseline/train_and_val/zenith_azimuth/
```

Important contents are:

```text
pipeline_config.yml
submitted_config.yml
resolved_config.yml
run_manifest.json
last_train_job_id.txt
data_audit.json
energy_weight_manifest.json
stage_a_vmf/
  training_history_by_epoch.csv
  validation_metrics_by_energy_epoch.csv
  resources_and_time.csv
  checkpoints/{best_*,last}.{ckpt,pth}
stage_b_angular_hybrid/
  training_history_by_epoch.csv
  validation_metrics_by_energy_epoch.csv
  resources_and_time.csv
  checkpoints/{best_*,last}.{ckpt,pth}
inference/val/stage_b_best_macro_median/
  predictions.parquet
  metrics_summary.csv
  metrics_by_true_energy.csv
  inference_manifest.json
  opening_angle_*.png
```

Automatic validation is locked to the Stage-B `best_macro_median` checkpoint.
No test prediction is produced.

## Dry-run and submission

Run from the local GraphNeT checkout:

```bash
cd /project/def-nahee/kbas/graphnet

python3 examples/08_pone/joint_direction/slurm/submit_train.py \
  -c examples/08_pone/configs/joint_direction/102_string_emax1e6__category1_isMuonCC.yml \
  --dry-run
```

Remove `--dry-run` to submit the two class jobs for that config. Repeat for:

```text
102_string_emax1e6__category1_isMuonCC.yml
102_string_emax1e6__category_3_contains_muon.yml
160_string_emax1e6__category1_isMuonCC.yml
160_string_emax1e6__category_3_contains_muon.yml
full_geometry_emax1e6__category1_isMuonCC.yml
full_geometry_emax1e6__category_3_contains_muon.yml
```

The 102/160 configs request 24 hours, 64 GB, and one H100 MIG slice. The full
geometry configs preserve the established 3-day, 128-GB, full-H100 request.
All jobs use the local source at `/project/def-nahee/kbas/graphnet/src` and the
same GraphNeT 1.8.0 CUDA container as the working pipelines.

## Read-only smoke test

This reads four validation events and writes no result:

```bash
python3 examples/08_pone/joint_direction/tests/smoke_loader.py \
  -c examples/08_pone/configs/joint_direction/102_string_emax1e6__category1_isMuonCC.yml \
  --route-class 0
```

Run this inside the configured GraphNeT container. The production Slurm runner
sets BLAS thread counts to one per DataLoader process to prevent CPU
oversubscription while retaining eight loader workers.
