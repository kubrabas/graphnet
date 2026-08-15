#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${CONFIG:?CONFIG is required}"
: "${CONFIG_SHA256:?CONFIG_SHA256 is required}"
: "${RESERVED_OUTPUT_DIR:?RESERVED_OUTPUT_DIR is required}"
: "${LOGFILE:?LOGFILE is required}"
: "${ROUTE_CLASS:?ROUTE_CLASS is required}"
: "${GRAPHNET_SRC:?GRAPHNET_SRC is required}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
STAGE="${STAGE:-all}"
if [[ "${STAGE}" != "all" ]]; then
  echo "This submit path requires STAGE=all, received ${STAGE}" >&2
  exit 2
fi

EXPECTED_GRAPHNET_SRC="$(readlink -f /project/def-nahee/kbas/graphnet/src)"
RESOLVED_GRAPHNET_SRC="$(readlink -f "${GRAPHNET_SRC}")"
if [[ "${RESOLVED_GRAPHNET_SRC}" != "${EXPECTED_GRAPHNET_SRC}" ]]; then
  echo "Refusing non-local GraphNeT source: ${GRAPHNET_SRC}" >&2
  exit 2
fi
if [[ ! -f "${GRAPHNET_SRC}/graphnet/__init__.py" ]]; then
  echo "Local GraphNeT package is missing below ${GRAPHNET_SRC}" >&2
  exit 2
fi

if [[ ! "${CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Invalid CONFIG_SHA256: ${CONFIG_SHA256}" >&2
  exit 2
fi
actual_config_sha256="$(sha256sum "${CONFIG}")"
actual_config_sha256="${actual_config_sha256%% *}"
if [[ "${actual_config_sha256}" != "${CONFIG_SHA256}" ]]; then
  echo "Frozen config SHA256 mismatch: ${CONFIG}" >&2
  exit 2
fi
resolved_reserved_output="$(readlink -f "${RESERVED_OUTPUT_DIR}")"
resolved_config_parent="$(readlink -f "$(dirname "${CONFIG}")")"
if [[ "${resolved_config_parent}" != "${resolved_reserved_output}" ]]; then
  echo "Frozen config is outside the reserved output leaf: ${CONFIG}" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOGFILE}")"
exec > >(tee -a "${LOGFILE}") 2>&1
# Keep GraphNeT's relative logs and every controllable runtime artifact inside
# this job's already-reserved zenith_azimuth leaf.
cd "${RESERVED_OUTPUT_DIR}"

module --force purge
module load StdEnv/2020 gcc/11.3.0 apptainer scipy-stack/2023b

EXAMPLE_DIR="/project/def-nahee/kbas/graphnet/examples/08_pone"
JOINT_DIR="${EXAMPLE_DIR}/joint_direction"
REFERENCE_DIR="/project/def-nahee/kbas/graphnet/examples/09_pone_muon_direction"
TRAIN_SCRIPT="${JOINT_DIR}/train_scripts/train_routed_joint_direction.py"
TELEMETRY="$(dirname "${LOGFILE}")/gpu_telemetry.csv"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training worker is missing: ${TRAIN_SCRIPT}" >&2
  exit 2
fi

echo "--- JOB=${SLURM_JOB_ID:-} HOST=$(hostname)"
echo "--- CONFIG=${CONFIG}"
echo "--- CONFIG_SHA256=${CONFIG_SHA256}"
echo "--- RESERVED_OUTPUT_DIR=${RESERVED_OUTPUT_DIR}"
echo "--- ROUTE_CLASS=${ROUTE_CLASS}"
echo "--- STAGE=${STAGE}"
echo "--- LOCAL_GRAPHNET_SRC=${GRAPHNET_SRC}"
echo "--- REFERENCE_DIR=${REFERENCE_DIR}"
echo "--- CONTAINER_IMAGE=${CONTAINER_IMAGE}"
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
apptainer exec --nv --cleanenv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 \
  --env PYTHONPATH="${GRAPHNET_SRC}:${JOINT_DIR}:${REFERENCE_DIR}:${EXAMPLE_DIR}" \
  --env GRAPHNET_SRC="${GRAPHNET_SRC}" \
  --env CONFIG_SHA256="${CONFIG_SHA256}" \
  --env RESERVED_OUTPUT_DIR="${RESERVED_OUTPUT_DIR}" \
  --env OUTPUT_PREPARED="${OUTPUT_PREPARED:-0}" \
  --bind /project \
  "${CONTAINER_IMAGE}" \
  python3 -u "${TRAIN_SCRIPT}" -c "${CONFIG}" --route-class "${ROUTE_CLASS}" --stage all
rc=$?
set -e

echo "--- train_routed_joint_direction finished rc=${rc}"
exit "${rc}"
