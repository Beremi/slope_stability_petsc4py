#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m petsc_ssr.runners.comsol_seepage \
  --backend scipy \
  --elem-type P2 \
  --parse-only

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" mpiexec -n "${RANKS:-2}" "$PYTHON" -m petsc_ssr.runners.comsol_seepage \
  --backend petsc \
  --elem-type "${ELEM_TYPE:-P1}" \
  --pc-variant "${PC_VARIANT:-pmg}" \
  --output-dir "${OUTPUT_DIR:-.local/tmp/comsol_seepage_smoke}"

if [[ "${RUN_P2_FULL:-0}" == "1" ]]; then
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" mpiexec -n "${RANKS:-2}" "$PYTHON" -m petsc_ssr.runners.comsol_seepage \
    --backend petsc \
    --elem-type P2 \
    --pc-variant "${PC_VARIANT:-pmg}" \
    --output-dir "${OUTPUT_DIR_P2:-.local/tmp/comsol_seepage_p2}"
fi
