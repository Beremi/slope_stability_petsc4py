#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RANKS="${RANKS:-32}"
OUT="${OUT:-$ROOT/.local/tmp/l1_r${RANKS}_omega7}"
CASE_TOML="${CASE_TOML:-$ROOT/benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml}"

cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${SKIP_LOAD_CHECK:-0}" != "1" ]]; then
  "$PYTHON" "$ROOT/tools/check_local_benchmark_load.py"
fi

# Use MPIEXEC_FLAGS="--map-by :OVERSUBSCRIBE" for local 64-rank runs on smaller workstations.
# shellcheck disable=SC2086
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" mpiexec ${MPIEXEC_FLAGS:-} -n "$RANKS" "$PYTHON" -m petsc_ssr.cli.main run "$CASE_TOML" \
  --omega-max "${OMEGA_MAX:-7000000}" \
  --continuation-step-max "${CONTINUATION_STEP_MAX:-100}" \
  --linear-rtol "${LINEAR_RTOL:-1e-1}" \
  --ksp-max-it "${KSP_MAX_IT:-200}" \
  --output "$OUT"
