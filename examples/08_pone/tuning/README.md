# Optuna tuning controller

This directory contains a tuning pipeline that is isolated from baseline
training and inference outputs.

The CPU controller is the only process that reads and writes the Optuna SQLite
database. It adaptively asks for trials, creates trial-specific copies of the
production YAML config, submits the existing GraphNeT GPU workers, waits for
their exact `rc=0` completion markers, and reports the validation objective to
Optuna. Restarting the controller with the same study config resumes the same
study.

For reconstruction, one trial is a bundle containing every configured route
class for one geometry, category, and target. The objective is the
validation-event-count-weighted mean of the best validation loss across the
route classes.

## Pilot

Run the preflight without submitting anything:

```bash
/project/def-nahee/kbas/.venv_optuna/bin/python \
  graphnet/examples/08_pone/tuning/optuna_controller.py \
  --config graphnet/examples/08_pone/tuning/study_configs/archive/160_string_emax1e6__category1_isMuonCC__energy_pilot.yml \
  --preflight
```

Preview the controller submission:

```bash
/project/def-nahee/kbas/.venv_optuna/bin/python \
  graphnet/examples/08_pone/tuning/submit_optuna_controller.py \
  --config graphnet/examples/08_pone/tuning/study_configs/archive/160_string_emax1e6__category1_isMuonCC__energy_pilot.yml \
  --dry-run
```

Submit or resume the controller:

```bash
/project/def-nahee/kbas/.venv_optuna/bin/python \
  graphnet/examples/08_pone/tuning/submit_optuna_controller.py \
  --config graphnet/examples/08_pone/tuning/study_configs/archive/160_string_emax1e6__category1_isMuonCC__energy_pilot.yml
```

The archived pilot used three trial slots. Each reconstruction trial trains class 0 and class 1, so at most four GPU jobs were active. Production studies use more trials than TPE startup trials so that later suggestions use completed results.

## 102-string contains-muon studies

The first production campaign tunes the complete `category_3_contains_muon`
pipeline for `102_string_emax1e6`. It contains four independent studies:
one classification study and one reconstruction study for each of energy,
zenith, and azimuth. Reconstruction trials train both route classes with the
same sampled parameters and minimize their validation-event-count-weighted
mean best validation loss.

Every study targets 12 successful trials. Up to four failed attempts may be replaced, so a study can use at most 16 trial numbers. The first four suggestions are TPE startup trials. Training
uses a fixed 60-epoch ceiling and patience of 8; these are not search
dimensions. Baseline outputs and test data are not modified or used for trial
selection.

Submit all four resumable controllers with:

```bash
cd /project/def-nahee/kbas
for config in graphnet/examples/08_pone/tuning/study_configs/102_string_emax1e6__category_3_contains_muon__*.yml; do
  /project/def-nahee/kbas/.venv_optuna/bin/python \
    graphnet/examples/08_pone/tuning/submit_optuna_controller.py \
    --config "$config"
done
```

Each controller can be resubmitted with the same config to resume its SQLite
study if the seven-day Fir walltime expires.

## Outputs

All state lives below the configured `campaign_dir`:

* `optuna.db`: persistent Optuna study
* `trials/trial_XXXX/config.yml`: generated GraphNeT config
* `trials/trial_XXXX/manifest.json`: job ids and controller state
* `trials_summary.csv`: compact trial table
* `best_trial.json`: current best completed trial
* `model_outputs/`: checkpoints, histories, validation products, and GPU logs
* `controller_<jobid>.out`: CPU controller log

Only the final selected hyperparameters are later copied into fixed production
configs. The pilot does not modify baseline configs or results.
