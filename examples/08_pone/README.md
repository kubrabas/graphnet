# 08_pone — P-ONE Event Reconstruction Pipeline

A config-driven training pipeline for P-ONE neutrino event reconstruction.

## Pipeline Overview

```
Test Event
    |
    v
01_train_classification.py        ->  cascade or track?
    |
    +-- track   ->  02_train_reconstruction_separate.py   (energy / zenith / azimuth, 3 networks)
    |          OR   03_train_reconstruction_combined.py   (energy / zenith+azimuth,   2 networks)
    |
    +-- cascade ->  02_train_reconstruction_separate.py
               OR   03_train_reconstruction_combined.py
```

After training, `04_inference_pipeline.py` runs the full chain end-to-end.
`05_plot_results.py` generates performance plots from the saved CSVs.

Shared code (PONE detector, data loading, callbacks) lives in `utils.py`.

## Scripts

| Script | Task |
|--------|------|
| `utils.py` | Shared: PONE detector, data loading, callbacks |
| `01_train_classification.py` | Cascade vs track classification |
| `02_train_reconstruction_separate.py` | Energy, zenith, azimuth — 3 independent networks |
| `03_train_reconstruction_combined.py` | Energy separate + zenith+azimuth shared backbone |
| `04_inference_pipeline.py` | classify -> reconstruct, end-to-end |
| `05_plot_results.py` | Performance plots from results CSVs |

## Usage

Each script takes a config file path:

```bash
python 01_train_classification.py          -c configs/classification/exp001.yml

python 02_train_reconstruction_separate.py -c configs/track/separate/exp001.yml
python 02_train_reconstruction_separate.py -c configs/cascade/separate/exp001.yml

python 03_train_reconstruction_combined.py -c configs/track/combined/exp001.yml
python 03_train_reconstruction_combined.py -c configs/cascade/combined/exp001.yml
```

To run the full pipeline with a single SLURM submission:

```bash
bash submit_pipeline.sh configs/classification/exp001.yml \
                        configs/track/separate/exp001.yml \
                        configs/cascade/separate/exp001.yml
```

SLURM chains jobs with dependencies: classification, track reco, and cascade reco run in
parallel; inference and plots start only after all three finish.

## Config Structure

```
configs/
├── classification/
│   └── exp001.yml
├── track/
│   ├── separate/
│   │   └── exp001.yml
│   └── combined/
│       └── exp001.yml
└── cascade/
    ├── separate/
    │   └── exp001.yml
    └── combined/
        └── exp001.yml
```

Key config fields:

```yaml
experiment_name: exp001
event_type: track          # track or cascade
output_dir: /project/def-nahee/kbas/Graphnet-Applications/Results/training/track_reconstruction/separate/exp001

training:
  seed: 20260202
  max_epochs: 30
  early_stopping_patience: 5
  batch_size: 256
  accumulate_grad_batches: 4
  base_lr: 1e-5
  peak_lr: 1e-3
  num_workers: 8

model:
  nb_neighbours: 8
  global_pooling_schemes: [min, max, mean, sum]

data:
  train_path: /project/def-nahee/kbas/POM_Response_Parquet/merged/train_reindexed
  val_path:   /project/def-nahee/kbas/POM_Response_Parquet/merged/val_reindexed
  test_path:  /project/def-nahee/kbas/POM_Response_Parquet/merged/test_reindexed
  pulsemaps:  features
  truth_table: truth
  features: [pmt_x, pmt_y, pmt_z, dom_time, charge]
```

## Results

All outputs (models, metrics, predictions) are saved to:

```
/project/def-nahee/kbas/Graphnet-Applications/Results/
├── training/
│   ├── classification/
│   ├── track_reconstruction/
│   │   ├── separate/
│   │   └── combined/
│   └── cascade_reconstruction/
│       ├── separate/
│       └── combined/
└── inference/
```

Experiment log: `/project/def-nahee/kbas/Graphnet-Applications/Results/EXPERIMENTS.md`
