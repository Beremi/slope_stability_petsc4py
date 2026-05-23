#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_OMEGA="${TARGET_OMEGA:-6.754292217973787e6}"
OMEGA_WINDOW="${OMEGA_WINDOW:-5.0e3}"

python_bounds="$(
python - <<'PY' "${TARGET_OMEGA}" "${OMEGA_WINDOW}"
import sys
target = float(sys.argv[1])
window = float(sys.argv[2])
print(f"{target - window:.17e} {target + window:.17e}")
PY
)"
read -r OMEGA_MIN OMEGA_MAX_EXPORT <<< "${python_bounds}"

RANKS="${RANKS:-32}" \
OUT_ROOT="${OUT_ROOT:-/tmp/ssr_step_replay_omega6754292}" \
RUN_OUT="${RUN_OUT:-/tmp/petsc4py_step_replay_omega6754292_out}" \
CONFIG="${CONFIG:-/tmp/petsc4py_step_replay_omega6754292.toml}" \
OMEGA_MAX="${OMEGA_MAX:-6.76e6}" \
STEP_MAX="${STEP_MAX:-100}" \
EXPORT_MAX="${EXPORT_MAX:-1}" \
EXPORT_NEWTON_ITERS="${EXPORT_NEWTON_ITERS:-1}" \
EXPORT_OMEGA_MIN="${EXPORT_OMEGA_MIN:-${OMEGA_MIN}}" \
EXPORT_OMEGA_MAX="${EXPORT_OMEGA_MAX:-${OMEGA_MAX_EXPORT}}" \
EXPORT_MATRIX="${EXPORT_MATRIX:-1}" \
EXPORT_PROBES="${EXPORT_PROBES:-1}" \
EXPORT_PROBE_FORMAT="${EXPORT_PROBE_FORMAT:-raw}" \
EXPORT_STEP_HISTORY="${EXPORT_STEP_HISTORY:-1}" \
  "${SCRIPT_DIR}/export_petsc4py_linear_state.sh"
