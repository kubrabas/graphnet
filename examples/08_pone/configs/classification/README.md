# Classification configs

There is one config for each detector geometry and classification target.

Geometries:

- `102_string_emax1e6`
- `160_string_emax1e6`
- `full_geometry_emax1e6`

Classification targets:

- `category1_isMuonCC`: binary classes `0=not_muon_cc`, `1=muon_cc`
- `category2_tauCC_others_muonCC`: classes `0=tau_cc`, `1=electron_cc_or_nc`, `2=muon_cc`
- `category_3_contains_muon`: binary classes `0=no_muon`, `1=contains_muon`

Every classification uses all four flavors and the geometry-wide mixed scaler:

```python
ROBUST_SCALER[mc][geometry]["classification"]
```

Dry-run one submission:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_classification_pipeline.py \
  -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/classification/102_string_emax1e6__category1_isMuonCC.yml \
  --dry-run
```

Remove `--dry-run` to submit the job.

Dry-run all nine configs:

```bash
cd /project/def-nahee/kbas

for config in graphnet/examples/08_pone/configs/classification/*.yml; do
  python3 /home/kbas/SlurmScripts/GraphNet/submit_classification_pipeline.py \
    -c "$config" \
    --dry-run
done
```

Submit all nine training jobs:

```bash
cd /project/def-nahee/kbas

for config in graphnet/examples/08_pone/configs/classification/*.yml; do
  python3 /home/kbas/SlurmScripts/GraphNet/submit_classification_pipeline.py \
    -c "$config"
done
```
