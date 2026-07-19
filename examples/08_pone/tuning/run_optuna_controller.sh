#!/bin/bash

set -euo pipefail

CONTROLLER="/project/def-nahee/kbas/graphnet/examples/08_pone/tuning/optuna_controller.py"
PYTHON="/project/def-nahee/kbas/.venv_optuna/bin/python"

echo "--- HOST: JOB=${SLURM_JOB_ID:-} HOST=$(hostname)"
echo "--- STUDY_CONFIG: ${STUDY_CONFIG}"
echo "--- CONTROLLER: ${CONTROLLER}"

set +e
"${PYTHON}" -u "${CONTROLLER}" --config "${STUDY_CONFIG}"
rc=$?
set -e

echo "--- optuna_controller finished (rc=${rc})"
exit "${rc}"
