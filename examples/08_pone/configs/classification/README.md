# Classification Config Notes

This folder contains the training config for classification experiments.

The classification training script should train on the train split, select the
best model with the validation split, then run validation diagnostics with that
best model. These diagnostics are written under:

```text
root_dir / mc / geometry / classification / task.target / experiment_name / train_and_val
```

Examples:

```text
/project/def-nahee/kbas/Graphnet-Applications/Results/340StringMC/102_string_emax1e6/classification/category1/exp001/train_and_val
/project/def-nahee/kbas/Graphnet-Applications/Results/340StringMC/102_string_emax1e6/classification/category2/exp002/train_and_val
/project/def-nahee/kbas/Graphnet-Applications/Results/340StringMC/full_geometry_emax1e6/classification/category1/exp003/train_and_val
/project/def-nahee/kbas/Graphnet-Applications/Results/340StringMC/full_geometry_emax1e6/classification/category2/exp004/train_and_val
```


## Current Configs

| Config | Geometry | Target | Mode | Output branch |
| --- | --- | --- | --- | --- |
| `exp001.yml` | `102_string_emax1e6` | `category1` | binary | `classification/category1/exp001` |
| `exp002.yml` | `102_string_emax1e6` | `category2` | multiclass | `classification/category2/exp002` |
| `exp003.yml` | `full_geometry_emax1e6` | `category1` | binary | `classification/category1/exp003` |
| `exp004.yml` | `full_geometry_emax1e6` | `category2` | multiclass | `classification/category2/exp004` |

Submit a config through the SLURM wrapper:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_classification_pipeline.py \
  -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/classification/exp001.yml
```

Dry-run is available:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_classification_pipeline.py \
  -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/classification/exp001.yml \
  --dry-run
```

A bad compute node can be excluded at submit time:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_classification_pipeline.py \
  -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/classification/exp003.yml \
  --exclude fc10713
```

The submit wrapper writes a config snapshot to the target `train_and_val/`
directory at job start. After the job has started and the snapshot exists,
editing the source config does not affect that running job.

PONE truth parquet files use `totalEnergy` for event energy. Classification
configs should therefore include `totalEnergy`, not `energy`, in `data.truth_all`
when energy is needed as diagnostic context.

## Training Outputs

The training directory should always contain:

```text
best_model.pth
training_history_by_epoch.csv
resources_and_time.csv
config.yml
validation_roc_curve.png
validation_score_distribution_by_true_class.png
validation_metrics_summary.csv
```

Binary classification additionally writes:

```text
validation_confusion_matrix_youden_j.png
validation_confusion_matrix_closest_top_left.png
```

Multiclass classification additionally writes:

```text
validation_metrics_by_class.csv
validation_confusion_matrix_argmax.png
```

`training_history_by_epoch.csv` is the epoch-by-epoch training history. It is
not the same as `validation_metrics_summary.csv`.

Expected columns include:

```text
epoch, train_loss, val_loss, lr, best_model_is_updated
```

`validation_metrics_summary.csv` is produced after training, using the best
model on the validation split.

## ROC Curve

`validation_roc_curve.png` should use the same plotting logic for binary and
multiclass classification: one-vs-rest ROC curves for every class.

For binary classification with classes `[0, 1]`:

```text
class 0 vs rest, score = p_class_0
class 1 vs rest, score = p_class_1
```

For multiclass classification with classes `[0, 1, 2]`:

```text
class 0 vs rest, score = p_class_0
class 1 vs rest, score = p_class_1
class 2 vs rest, score = p_class_2
```

The ROC plot should show:

```text
per-class ROC curves
per-class AUC values
macro AUC
weighted AUC
```

Threshold markers should not be drawn on the ROC curves. Instead, the plot
should include a right-side text/table panel with per-class threshold candidates:

```text
Youden J: max(TPR - FPR)
Closest top-left: min distance to (FPR=0, TPR=1)
```


## Score Distribution

`validation_score_distribution_by_true_class.png` should also use the same
logic for binary and multiclass classification.

For every true class, plot the distribution of that class probability score:

```text
true 0 -> p_class_0 histogram
true 1 -> p_class_1 histogram
true 2 -> p_class_2 histogram
```

Each class should have its own color. The threshold candidates for that class
should use the same color as the class histogram.

The score distribution should show:

```text
Youden J threshold as a short x-axis tick
Closest top-left threshold as a short x-axis tick
```

These should not be full-height vertical lines. They should be small ticks near
the x-axis so they do not cover the histogram.

The score distribution should not include:

```text
p = 0.5 reference line
threshold table
```

The threshold values are already listed in the ROC plot's right-side panel.

## Validation Metrics

`validation_metrics_summary.csv` should be computed with the best model on the
validation split.

For binary classification, `validation_metrics_summary.csv` is threshold-based.
It should contain one row for each threshold method:

```text
Youden J threshold
Closest top-left threshold
```

Recommended binary columns:

```text
method
threshold
accuracy
others_precision
others_recall
others_f1
muon_CC_precision
muon_CC_recall
muon_CC_f1
tp_muon_CC
fp_muon_CC
tn_muon_CC
fn_muon_CC
roc_auc
```

Binary class-metric columns should use `class_names` from the config. The
example above assumes:

```yaml
class_names:
  0: others
  1: muon_CC
```

For multiclass classification, metrics should use argmax predictions:

```text
pred_class = argmax(p_class_0, p_class_1, ...)
```

Recommended multiclass summary columns:

```text
method
accuracy
macro_precision
macro_recall
macro_f1
weighted_precision
weighted_recall
weighted_f1
macro_roc_auc
weighted_roc_auc
```

Per-class multiclass metrics should be written separately:

```text
validation_metrics_by_class.csv
```

Recommended per-class columns:

```text
class
precision
recall
f1
support
```

## Confusion Matrices

For binary classification, write:

```text
validation_confusion_matrix_youden_j.png
validation_confusion_matrix_closest_top_left.png
```

For multiclass classification, write one argmax confusion matrix:

```text
validation_confusion_matrix_argmax.png
```

Threshold-based multiclass confusion matrices should not be produced by default,
because one-vs-rest thresholds are diagnostics rather than final multiclass
routing behavior.

## Inference Threshold

The final probability threshold `p` is not part of the training config. It
belongs in the inference config because it controls event routing in the final
pipeline.
