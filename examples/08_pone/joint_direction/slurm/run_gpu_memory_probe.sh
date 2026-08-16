#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${CONFIG:?CONFIG is required}"
: "${CONFIG_SHA256:?CONFIG_SHA256 is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${LOGFILE:?LOGFILE is required}"
: "${GRAPHNET_SRC:?GRAPHNET_SRC is required}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"

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

RESOLVED_OUTPUT_DIR="$(readlink -f "${OUTPUT_DIR}")"
case "${RESOLVED_OUTPUT_DIR}" in
  */diagnostics/joint_direction_gpu_memory_probe/*) ;;
  *)
    echo "Refusing non-diagnostic output directory: ${OUTPUT_DIR}" >&2
    exit 2
    ;;
esac
RESOLVED_CONFIG_PARENT="$(readlink -f "$(dirname "${CONFIG}")")"
if [[ "${RESOLVED_CONFIG_PARENT}" != "${RESOLVED_OUTPUT_DIR}" ]]; then
  echo "Frozen probe config must live in OUTPUT_DIR: ${CONFIG}" >&2
  exit 2
fi
ACTUAL_CONFIG_SHA256="$(sha256sum "${CONFIG}")"
ACTUAL_CONFIG_SHA256="${ACTUAL_CONFIG_SHA256%% *}"
if [[ "${ACTUAL_CONFIG_SHA256}" != "${CONFIG_SHA256}" ]]; then
  echo "Frozen config SHA256 mismatch: ${CONFIG}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${LOGFILE}") 2>&1
cd "${OUTPUT_DIR}"

module --force purge
module load StdEnv/2020 gcc/11.3.0 apptainer scipy-stack/2023b

EXAMPLE_DIR="/project/def-nahee/kbas/graphnet/examples/08_pone"
JOINT_DIR="${EXAMPLE_DIR}/joint_direction"
REFERENCE_DIR="/project/def-nahee/kbas/graphnet/examples/09_pone_muon_direction"
PROBE_SCRIPT="${JOINT_DIR}/diagnostics/gpu_memory_probe.py"
if [[ ! -f "${PROBE_SCRIPT}" ]]; then
  echo "GPU memory probe is missing: ${PROBE_SCRIPT}" >&2
  exit 2
fi

echo "--- JOB=${SLURM_JOB_ID:-} HOST=$(hostname)"
echo "--- CONFIG=${CONFIG}"
echo "--- CONFIG_SHA256=${CONFIG_SHA256}"
echo "--- OUTPUT_DIR=${OUTPUT_DIR}"
echo "--- LOCAL_GRAPHNET_SRC=${GRAPHNET_SRC}"
echo "--- CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "--- LOGFILE=${LOGFILE}"

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
  --bind /project \
  "${CONTAINER_IMAGE}" \
  python3 -u "${PROBE_SCRIPT}" -c "${CONFIG}" --output-dir "${OUTPUT_DIR}"
RC=$?
set -e

echo "--- gpu_memory_probe finished rc=${RC}"
exit "${RC}"
