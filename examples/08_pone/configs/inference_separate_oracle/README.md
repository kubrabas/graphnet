# Separate-direction inference with oracle routing diagnostics

These configs rerun the established separate `energy`, `zenith`, and `azimuth`
inference without overwriting the existing baseline outputs. The isolated
experiment name is `baseline_separate_direction`.

The normal routed columns retain their existing names. Additional
`true_route_class`, `router_correct`, and `oracle_*` columns show the result
that the same reconstruction models would have produced under correct routing.

```bash
cd /project/def-nahee/kbas
python3 /home/kbas/SlurmScripts/GraphNet/submit_inference_pipeline.py \
  -c graphnet/examples/08_pone/configs/inference_separate_oracle/102_string_emax1e6__category1_isMuonCC.yml \
  --dry-run
```

Remove `--dry-run` to submit.
