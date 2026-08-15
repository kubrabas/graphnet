#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${CONFIG:?CONFIG is required}"
: "${LOGFILE:?LOGFILE is required}"
: "${GRAPHNET_SRC:?GRAPHNET_SRC is required}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
SPLIT="${SPLIT:-test}"
STAGE="${STAGE:-stage_b}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_macro_median}"

mkdir -p "$(dirname "${LOGFILE}")"
exec > >(tee -a "${LOGFILE}") 2>&1

module --force purge
module load StdEnv/2020 gcc/11.3.0 apptainer scipy-stack/2023b

EXAMPLE_DIR="/project/def-nahee/kbas/graphnet/examples/09_pone_muon_direction"
SCRIPT="${EXAMPLE_DIR}/inference_scripts/run_direction_inference.py"
IMAGE="${CONTAINER_IMAGE}"

echo "--- JOB=${SLURM_JOB_ID:-} HOST=$(hostname)"
echo "--- CONFIG=${CONFIG} SPLIT=${SPLIT} STAGE=${STAGE} CHECKPOINT=${CHECKPOINT_NAME}"
echo "--- LOCAL_GRAPHNET_SRC=${GRAPHNET_SRC}"
echo "--- CONTAINER_IMAGE=${IMAGE}"

apptainer exec --nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH="${GRAPHNET_SRC}:${EXAMPLE_DIR}" \
  --env OUTPUT_PREPARED="${OUTPUT_PREPARED:-0}" \
  --bind /project \
  "${IMAGE}" \
  python3 -u "${SCRIPT}" --config "${CONFIG}" --split "${SPLIT}" \
    --stage "${STAGE}" --checkpoint-name "${CHECKPOINT_NAME}"
