# Routed mixed-flavor joint direction reconstruction

This is a parallel extension of the existing `examples/08_pone` reconstruction
pipeline. It adds one joint `zenith_azimuth` model beside the completed
separate `energy`, `zenith`, and `azimuth` models. The separate-training
configs and outputs, source parquet, and percentile CSV files are not
modified; every new ablation is opt-in and writes to a new experiment leaf.

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

The six baseline configs preserve these choices. Controlled ablations may
change a small allow-listed subset only when every changed dotted config path
is declared in `experiment_contract.varied_fields`. Both the login-node
submitter and the training worker verify that the declared and observed
changes match exactly. Undeclared changes, declared-but-unchanged fields, and
unsupported fields are rejected before training.

## Opt-in 102-string class-1 experiments

Four independent ablations focus on
`102_string_emax1e6/category1_isMuonCC/class1`. They use the same routed
train/validation event sets, seed, losses, optimizer schedule, energy bins,
batch size 256, gradient accumulation 4, eight CPU workers, 64 GB of memory,
and validation checkpoint policy as the baseline. They do not use
`final_weight`.

| Experiment | Declared varied fields | Exact change | Trainable parameters | Time / GPU request |
| --- | --- | --- | ---: | --- |
| `pmt_direction_v1` | `data.node_feature_augmentations` | `[pmt_direction_v3]`; five model features become eight | 1,377,731 | 12 h / H100 40-GB MIG |
| `pmt_direction_seed20260203_v1` | `data.node_feature_augmentations`, `training.seed` | exact PMT replication with seed `20260203` instead of `20260202` | 1,377,731 | 12 h / H100 40-GB MIG |
| `pmt_direction_wide_2p75m_v1` | `data.node_feature_augmentations`, all three model-size fields | PMT direction features combined with the exact wide architecture | 2,749,547 | 18 h / H100 40-GB MIG |
| `wide_2p75m_v1` | `model.dynedge_layer_sizes`, `model.post_processing_layer_sizes`, `model.readout_layer_sizes` | DynEdge `[[176,360],[480,360],[480,360],[480,360]]`, post-processing `[480,360]`, readout `[176]` | 2,746,523 | 18 h / H100 40-GB MIG, subject to the memory gate below |
| `alpha025_v1` | `weighting.alpha` | `alpha = 0.25` | 1,375,571 | 12 h / H100 40-GB MIG |
| `alpha075_v1` | `weighting.alpha` | `alpha = 0.75` | 1,375,571 | 12 h / H100 40-GB MIG |

The PMT-only experiment is also defined for the remaining
`category1_isMuonCC` route/geometry combinations:

| Config | Submitted route classes | Time / GPU request |
| --- | --- | --- |
| `102_string_emax1e6__category1_isMuonCC__pmt_direction_v1_class0.yml` | class0 only; complements the completed 102-string class1 model | 12 h / H100 40-GB MIG |
| `160_string_emax1e6__category1_isMuonCC__pmt_direction_v1.yml` | class0 and class1 | 18 h each / H100 40-GB MIG |

All three write below the shared experiment name `pmt_direction_v1`, but in
separate geometry/class leaves. This preserves output isolation while making a
complete two-class PMT routed inference possible for both geometries.

The name `wide_2p75m_v1` means an approximately 2.75-million-parameter
capacity trial; the YAML layer sizes above, not the rounded name, are the
reproducible definition. The two alpha experiments change only the exponent
in `weight proportional to N_bin^-alpha`; clipping remains `0.2` to `5.0` and
normalization remains mean one. The PMT and wide experiments retain the
baseline `alpha = 0.5`.

Every experiment writes to its own leaf, for example:

```text
Results/340StringMC/102_string_emax1e6/reconstruction/
  category1_isMuonCC/class1/pmt_direction_v1/train_and_val/zenith_azimuth/
```

No existing baseline result or config is reused as an output target.

### PMT-direction feature contract

`pmt_direction_v3` is an opt-in node-feature augmentation rather than a new
parquet format:

- the original five columns
  `(pmt_x, pmt_y, pmt_z, dom_time, charge)` are transformed by the existing
  route-class percentile scaler exactly as before;
- the already-present, one-based `pmt_number` column is loaded as an auxiliary
  identity feature, so it is never treated as a continuous scaled model
  input;
- after scaling, `pmt_number` is replaced in memory by the exact P-ONE offline
  v3 unit direction `(pmt_dir_x, pmt_dir_y, pmt_dir_z)` for PMT IDs 1 through
  16; the lookup direction agrees with the normalized
  `pmt_xyz - dom_xyz` displacement stored in the real parquet geometry;
- the GNN therefore sees eight features in canonical order:
  `(pmt_x, pmt_y, pmt_z, dom_time, charge, pmt_dir_x, pmt_dir_y, pmt_dir_z)`;
- the first three columns remain the spatial KNN coordinates, so the graph
  construction geometry is unchanged.

The direction components are fixed unit vectors and are not percentile
scaled. No parquet or percentile CSV is edited or regenerated. Invalid,
non-integer, or out-of-range PMT IDs fail loudly instead of being clipped.

The augmentation is registered through a generic node-feature interface; the
training code does not branch on `experiment_name`. Future fixed categorical
node features can use the same loader/scaler/output contract and must declare
their controlled config changes.

### Backward compatibility

When `data.node_feature_augmentations` is absent, as in every historical
config, the loader retains the established `PONE + NodesAsPulses` construction
with the original five inputs. An explicit empty list is equivalent to an
absent field. Model input width is now read from the graph representation's
actual output width: this is still five for old configs and eight only for the
PMT opt-in config. Consequently, old config semantics and old checkpoint
tensor shapes are preserved.

Custom DynEdge widths also remain opt-in: `null` continues to select the
historical GraphNeT defaults. The experiment-contract tests additionally
guard against accidentally combining PMT direction, a wider network, and a
different alpha in one purported single-variable ablation.

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

`run.existing_output: error` is used in all baseline and opt-in experiment
configs. The submitter refuses an already-existing `zenith_azimuth` directory
and has no overwrite mode. It atomically reserves each new task leaf, stores
the exact submitted YAML there, and pins both its SHA256 and resolved output
path into the queued worker. A later edit to the source YAML therefore cannot
change a waiting job.

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
node_feature_contract.json
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
The `inference/val/...` directory above is validation inference, not routed
test inference.

The four focused configs run training plus this automatic validation only.
They do not produce or submit routed test predictions. The routed inference
loader can select an experiment-specific joint config and feature contract,
but tonight's experiments train class1 only. A full two-class routed test run
therefore requires either a matching class0 experiment or a later per-class
joint-config mapping. Do not silently combine a class0 baseline checkpoint
with a class1 PMT/wide checkpoint under one global experiment name.

## Mandatory wide-model GPU memory gate

The historical `gpu_telemetry.csv` reports physical-GPU aggregate memory and
cannot establish how much of the visible 40-GB MIG slice one process used.
Before submitting `wide_2p75m_v1`, run the isolated probe. It reads real class1
training parquet, uses float32 batch 256, accumulates four microbatches, and
performs one Adam step so optimizer state is included. It creates no training
result and no test loader.

```bash
cd /project/def-nahee/kbas/graphnet

python3 examples/08_pone/joint_direction/slurm/submit_gpu_memory_probe.py \
  -c examples/08_pone/configs/joint_direction/102_string_emax1e6__category1_isMuonCC__wide_2p75m_v1.yml \
  --dry-run

# Remove --dry-run to submit only the diagnostic job.
```

The report is written below
`Results/340StringMC/diagnostics/joint_direction_gpu_memory_probe/` as
`gpu_memory_report.json`. Keep the configured 40-GB MIG request only if the
probe succeeds and `peak_reserved_below_85_percent` is true. If it OOMs or
crosses that threshold, change only the wide config's
`slurm.gpus_per_node` to `h100:1`; keep batch 256 and accumulation 4 unchanged
so capacity remains the sole scientific ablation.

The probe measures four seeded, shuffled real microbatches. It is a practical
headroom gate, not a mathematical upper bound on the pulse count of every
later batch. Treat a result close to 85% conservatively and use the full H100
for wide rather than changing batch size.

## Focused experiment dry-run and submission

Each command below targets class1 only and reserves a distinct experiment
leaf. Always dry-run first. Do not submit the wide command until the memory
gate above has passed.

```bash
cd /project/def-nahee/kbas/graphnet

for name in pmt_direction_v1 alpha025_v1 alpha075_v1; do
  python3 examples/08_pone/joint_direction/slurm/submit_train.py \
    -c "examples/08_pone/configs/joint_direction/102_string_emax1e6__category1_isMuonCC__${name}.yml" \
    --dry-run
done

# Submit one experiment by removing --dry-run from its command.
# The wide config is named:
# 102_string_emax1e6__category1_isMuonCC__wide_2p75m_v1.yml
```

## Baseline dry-run and submission

Run from the local GraphNeT checkout:

```bash
cd /project/def-nahee/kbas/graphnet

python3 examples/08_pone/joint_direction/slurm/submit_train.py \
  -c examples/08_pone/configs/joint_direction/102_string_emax1e6__category1_isMuonCC.yml \
  --dry-run
```

Remove `--dry-run` to submit the two class jobs for that baseline config.
Repeat for:

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

## Paper-inspired Fourier space-time transformer

The paper's first-place TITO ensemble has the best overall leaderboard score,
but its implementation is **not publicly available**; the paper states that it
may be provided on reasonable request. The local GraphNeT checkout contains a
`DynEdgeTITO` reimplementation of the published EdgeConv+Transformer design,
which can be evaluated as a separate paper-inspired ablation, but it must not
be described as the original first-place source or a bit-for-bit reproduction.

The opt-in transformer follows the open second-place IceCube Kaggle solution,
not an invented generic encoder:

- paper: Bukhari et al., EPJC 84 (2024) 646,
  <https://arxiv.org/abs/2310.15674>;
- source: <https://github.com/DrHB/icecube-2nd-place>;
- inspected source revision:
  `484cdcfed01af5255dce148122b095a7427ec1cb`;
- upstream license: MIT; the complete notice is retained in
  `THIRD_PARTY_NOTICES.md`.

The historical repository pins PyTorch 1.11, PyG 2.0.4, fastai 2.7, and timm
0.6.12. Do **not** install that requirements file into the production GraphNeT
environment. The local implementation in `fourier_spacetime_transformer.py`
ports the relevant mathematics to the existing PyTorch 2.6 container and adds
no Python dependency. Nothing needs to be installed on a laptop or login-node
venv.

The implemented T model has the paper's 192-dimensional representation,
32-dimensional heads, four relative space-time blocks, 12 regular transformer
blocks, Fourier encoding, relative interval encoding, and CLS-token event
aggregation. The base P-ONE adaptation has 7,756,579 trainable parameters; the
PMT-direction form has 7,863,651. The paper reports 7.57M for its IceCube T
model; the small difference comes from the P-ONE input projection.

The publication describes four relative-bias blocks. The inspected open-source
repository exposes four relative blocks but defaults `n_rel=1`, which removes
the explicit relative bias after the first one. These first P-ONE configs follow
the publication text and keep the bias in all four blocks; this choice is
machine-recorded as `relative_bias_blocks: 4` and should later be ablated rather
than silently changed.

Two isolated class1 configs are provided:

| Config suffix | Purpose |
| --- | --- |
| `fourier_t_v1` | clean transformer architecture comparison using raw x/y/z/time/charge |
| `fourier_t_pmt_v1` | the same transformer plus the proven P-ONE PMT direction |

Both retain the exact routed 102-string MuonCC train/validation split, training
energy weights (`alpha=0.5`, clip `[0.2,5]`), Stage A vMF, Stage B opening angle
`+ 0.05*vMF`, and best-macro checkpoint selection. They do not read test data
and do not use `final_weight`.

### Explicit P-ONE adaptations

The IceCube competition data had median sequence length 62, mean 163.4, and
only 1.4% longer than 768 pulses. The current 102-string MuonCC train data has
164,226 events, 49,768,672 pulses, median 105, mean 303.05, and maximum 33,407;
32.5% exceed 192 and 6.78% exceed 768. Copying every competition data-loader
choice would therefore be inappropriate.

For these first configs:

- train sequences are randomly sampled to at most 256 pulses;
- validation sequences use a deterministic time-uniform selection up to 512;
- the full pre-selection pulse count is Fourier-encoded as an event feature;
- time is measured relative to the event's first pulse and divided by 30,000
  ns;
- coordinates are divided by 500 m and charge uses `log10(charge)/3`;
- no HLC/auxiliary flag is fabricated because the selected P-ONE input is
  nonoise PMT response;
- the graph representation is edgeless, so no unused KNN is constructed;
- source parquet and percentile CSV files remain unchanged.

These choices are persisted in the submitted YAML and
`node_feature_contract.json`; the run should be described as a P-ONE
adaptation of the paper model, not a bit-for-bit reproduction of the IceCube
training run.

### Transformer GPU gate

First probe only the larger PMT form. The probe uses the real class1 train
loader, batch 16, bf16 mixed precision, full 256-pulse training limit, backward
passes, and an AdamW step. It writes only below the diagnostics root.

```bash
cd /project/def-nahee/kbas/graphnet

python3 examples/08_pone/joint_direction/slurm/submit_gpu_memory_probe.py \
  -c examples/08_pone/configs/joint_direction/102_string_emax1e6__category1_isMuonCC__fourier_t_pmt_v1.yml
```

Do not submit training until the diagnostic finishes with exit code 0 and
`peak_reserved_below_85_percent: true`. If it exceeds that gate, change only
the two transformer configs' `slurm.gpus_per_node` to `h100:1`; do not change
the scientific batch/sequence config in response to an OOM.

After the gate passes, dry-run both isolated experiments:

```bash
cd /project/def-nahee/kbas/graphnet

for name in fourier_t_v1 fourier_t_pmt_v1; do
  python3 examples/08_pone/joint_direction/slurm/submit_train.py \
    -c "examples/08_pone/configs/joint_direction/102_string_emax1e6__category1_isMuonCC__${name}.yml" \
    --dry-run
done
```

Remove `--dry-run` only after checking both printed output leaves. Each config
submits exactly one class1 train+validation job. Automatic validation uses the
Stage-B `best_macro_median` checkpoint and 512-pulse evaluation. Routed test
inference is intentionally not claimed for these new backbones yet.
