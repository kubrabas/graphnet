# Routed joint-direction inference

These configs preserve the existing classifier and energy reconstruction but
replace the separate zenith and azimuth models with the routed joint
`zenith_azimuth` models. Outputs use the isolated experiment name
`baseline_joint_direction`; existing baseline inference results are untouched.

Each output `inference_predictions.csv` retains the legacy event,
classification, energy, zenith, and azimuth fields and adds:

- `true_route_class` and `router_correct`;
- joint direction-vector, kappa, and opening-angle fields;
- `oracle_*` reconstruction fields selected with the true route class.

The oracle fields answer what reconstruction would have been obtained if the
router had selected the correct class. They are diagnostics and must not be
used for test-driven model or threshold selection.

Submit one config only after both of its route-class joint models and their
validation inference have completed:

```bash
cd /project/def-nahee/kbas
python3 /home/kbas/SlurmScripts/GraphNet/submit_inference_pipeline.py \
  -c graphnet/examples/08_pone/configs/inference_joint_direction/102_string_emax1e6__category1_isMuonCC.yml \
  --dry-run
```

Remove `--dry-run` to submit.
