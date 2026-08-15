#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${CONFIG:?CONFIG is required}"
: "${LOGFILE:?LOGFILE is required}"
: "${GRAPHNET_SRC:?GRAPHNET_SRC is required}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
STAGE="${STAGE:-all}"
RESUME_FLAG="${RESUME_FLAG:-}"

mkdir -p "$(dirname "${LOGFILE}")"
exec > >(tee -a "${LOGFILE}") 2>&1

module --force purge
module load StdEnv/2020 gcc/11.3.0 apptainer scipy-stack/2023b

EXAMPLE_DIR="/project/def-nahee/kbas/graphnet/examples/09_pone_muon_direction"
TRAIN_SCRIPT="${EXAMPLE_DIR}/train_scripts/train_direction.py"
IMAGE="${CONTAINER_IMAGE}"
TELEMETRY="$(dirname "${LOGFILE}")/gpu_telemetry.csv"

echo "--- JOB=${SLURM_JOB_ID:-} HOST=$(hostname)"
echo "--- CONFIG=${CONFIG}"
echo "--- LOCAL_GRAPHNET_SRC=${GRAPHNET_SRC}"
echo "--- CONTAINER_IMAGE=${IMAGE}"
echo "--- STAGE=${STAGE} RESUME_FLAG=${RESUME_FLAG}"
echo "--- OUTPUT=${LOGFILE}"

telemetry_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    if [[ ! -s "${TELEMETRY}" ]]; then
      echo "timestamp,index,utilization_gpu_pct,memory_used_mib,memory_total_mib"
    fi
    while true; do
      timestamp=$(date --iso-8601=seconds)
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits | sed "s/^/${timestamp},/"
      sleep 60
    done
  ) >> "${TELEMETRY}" 2>&1 &
  telemetry_pid=$!
fi

cleanup() {
  if [[ -n "${telemetry_pid}" ]]; then
    kill "${telemetry_pid}" >/dev/null 2>&1 || true
    wait "${telemetry_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

set +e
train_args=(--config "${CONFIG}" --stage "${STAGE}")
if [[ "${RESUME_FLAG}" == "--resume" ]]; then
  train_args+=(--resume)
elif [[ -n "${RESUME_FLAG}" ]]; then
  echo "Unsupported RESUME_FLAG=${RESUME_FLAG}"
  exit 2
fi
apptainer exec --nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONPATH="${GRAPHNET_SRC}:${EXAMPLE_DIR}" \
  --env OUTPUT_PREPARED="${OUTPUT_PREPARED:-0}" \
  --bind /project \
  "${IMAGE}" \
  python3 -u "${TRAIN_SCRIPT}" "${train_args[@]}"
rc=$?
set -e

echo "--- train_direction finished rc=${rc}"
exit "${rc}"
