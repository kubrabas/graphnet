# Reconstruction configs

There is one config for each detector geometry and routing category. Each config
submits one independent SLURM job for every route-class and reconstruction-target
pair.

Geometries:

- `102_string_emax1e6`
- `160_string_emax1e6`
- `full_geometry_emax1e6`

Routing categories and classes:

- `category1_isMuonCC`: classes `0`, `1`
- `category2_tauCC_others_muonCC`: classes `0`, `1`, `2`
- `category_3_contains_muon`: classes `0`, `1`

Targets: `energy`, `zenith`, `azimuth`.

The nine configs expand to 63 jobs in total. Reconstruction scalers are resolved
per geometry, routing category, and route class:

```python
ROBUST_SCALER[mc][geometry]["reconstruction"][routing_category][route_class]
```

Dry-run one config:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_reconstruction_pipeline.py \
  -c /project/def-nahee/kbas/graphnet/examples/08_pone/configs/reconstruction/102_string_emax1e6__category1_isMuonCC.yml \
  --dry-run
```

Dry-run all nine configs:

```bash
cd /project/def-nahee/kbas

for config in graphnet/examples/08_pone/configs/reconstruction/*.yml; do
  python3 /home/kbas/SlurmScripts/GraphNet/submit_reconstruction_pipeline.py \
    -c "$config" \
    --dry-run
done
```

Submit all 63 training jobs:

```bash
cd /project/def-nahee/kbas

for config in graphnet/examples/08_pone/configs/reconstruction/*.yml; do
  python3 /home/kbas/SlurmScripts/GraphNet/submit_reconstruction_pipeline.py \
    -c "$config"
done
```
