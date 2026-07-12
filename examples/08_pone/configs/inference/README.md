# Routed Inference Configs

This directory contains the active String340MC inference configs for
`102_string_emax1e6` and `160_string_emax1e6`. Each geometry has one config for
each of the three routing categories. Full-geometry inference is intentionally
excluded until its classification and reconstruction models are trained.

From `/project/def-nahee/kbas`, submit only the completed 102-string
`category1_isMuonCC` pipeline with:

```bash
python3 /home/kbas/SlurmScripts/GraphNet/submit_inference_pipeline.py \
  -c graphnet/examples/08_pone/configs/inference/102_string_emax1e6__category1_isMuonCC.yml
```

Submit all currently active configs with:

```bash
for geometry in 102_string_emax1e6 160_string_emax1e6; do
  for config in graphnet/examples/08_pone/configs/inference/${geometry}__*.yml; do
    python3 /home/kbas/SlurmScripts/GraphNet/submit_inference_pipeline.py \
      -c "$config"
  done
done
```

The wrapper runs a dependency preflight before creating an output directory or
submitting a job. Use `--dry-run` to validate one config without submission.
Archived legacy configs are in `../inference_old/`.
