#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Required environment:
#   CONFIG, LOGFILE, ROUTE_CLASS, TARGET
# Optional:
#   OUTPUT_DIR

set -euo pipefail

mkdir -p "$(dirname "${LOGFILE}")"
exec > >(tee -a "${LOGFILE}") 2>&1

module --force purge
module load StdEnv/2020 gcc/11.3.0 apptainer scipy-stack/2023b

GRAPHNET_SRC="/project/def-nahee/kbas/graphnet/src"
EXAMPLE_DIR="/project/def-nahee/kbas/graphnet/examples/08_pone"
TRAIN_SCRIPT="${EXAMPLE_DIR}/train_scripts/train_geometry_selection.py"
IMAGE="docker://rorsoe/graphnet:graphnet-1.8.0-cu126-torch26-ubuntu-22.04"

echo "--- HOST: JOB=${SLURM_JOB_ID:-}  HOST=$(hostname)"
echo "--- CONFIG: ${CONFIG}"
echo "--- ROUTE_CLASS: ${ROUTE_CLASS}"
echo "--- TARGET: ${TARGET}"
echo "--- OUTPUT_DIR: ${OUTPUT_DIR:-}"
echo "--- OUTPUT_PREPARED: ${OUTPUT_PREPARED:-0}"
echo "--- LOGFILE: ${LOGFILE}"
echo "--- TRAIN_SCRIPT: ${TRAIN_SCRIPT}"

cmd=(python3 -u "${TRAIN_SCRIPT}" --config "${CONFIG}" --route-class "${ROUTE_CLASS}" --target "${TARGET}")
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

set +e
apptainer exec --nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="${GRAPHNET_SRC}:${EXAMPLE_DIR}:${EXAMPLE_DIR}/train_scripts" \
  --env OUTPUT_PREPARED="${OUTPUT_PREPARED:-0}" \
  --bind /project \
  "${IMAGE}" \
  "${cmd[@]}"

rc=$?
set -e
echo "--- train_geometry_selection finished (rc=${rc})"
exit $rc
