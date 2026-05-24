#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-$SCRIPT_DIR/runs/ssr_py_hotpath_omega7_${STAMP}}"
REFINE_LEVELS_LIST="${REFINE_LEVELS_LIST:-0 1}"

export ENGINES="${ENGINES:-py}"
export PROFILES="${PROFILES:-hotpath}"
export MECHANICS_BACKEND="${MECHANICS_BACKEND:-dmplex_c_hotpath}"
export LAYOUTS="${LAYOUTS:-1:128 2:128}"
export TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
export LINEAR_RTOL="${LINEAR_RTOL:-1e-1}"
export KSP_MAX_IT="${KSP_MAX_IT:-200}"
export OMEGA_MAX="${OMEGA_MAX:-7e6}"
export PMG_COARSE_MAX_IT="${PMG_COARSE_MAX_IT:-5}"
export PMG_SHELL_P2_ACTIVE_RANKS="${PMG_SHELL_P2_ACTIVE_RANKS:-64}"
export PMG_SHELL_P1_ACTIVE_RANKS="${PMG_SHELL_P1_ACTIVE_RANKS:-32}"
export PMG_SHELL_SUBCOMM_TYPE="${PMG_SHELL_SUBCOMM_TYPE:-interlaced}"
export PMG_SHELL_FINE_KSP_MAX_IT="${PMG_SHELL_FINE_KSP_MAX_IT:-5}"
export PMG_SHELL_P2_KSP_MAX_IT="${PMG_SHELL_P2_KSP_MAX_IT:-10}"
export PMG_SHELL_P1_PC_TYPE="${PMG_SHELL_P1_PC_TYPE:-redundant}"
export PMG_SHELL_P1_REDUNDANT_NUMBER="${PMG_SHELL_P1_REDUNDANT_NUMBER:-1}"
export PMG_SHELL_P1_REDUNDANT_KSP_TYPE="${PMG_SHELL_P1_REDUNDANT_KSP_TYPE:-fgmres}"
export PMG_SHELL_P1_REDUNDANT_KSP_RTOL="${PMG_SHELL_P1_REDUNDANT_KSP_RTOL:-1e-3}"
export PMG_SHELL_P1_REDUNDANT_PC_TYPE="${PMG_SHELL_P1_REDUNDANT_PC_TYPE:-gamg}"

echo "RUN_ROOT_BASE=$RUN_ROOT_BASE"
echo "REFINE_LEVELS_LIST=$REFINE_LEVELS_LIST"
echo "LAYOUTS=$LAYOUTS ENGINES=$ENGINES PROFILES=$PROFILES MECHANICS_BACKEND=$MECHANICS_BACKEND"

for refine in $REFINE_LEVELS_LIST; do
  echo "Submitting petsc4py C-hotpath refine_levels=$refine"
  RUN_ROOT="$RUN_ROOT_BASE/ref${refine}" REFINE_LEVELS="$refine" "$SCRIPT_DIR/submit_omega7_grid.sh"
done

echo "Collect each completed campaign with:"
for refine in $REFINE_LEVELS_LIST; do
  echo "  \${PYTHON_BIN:-../../.venv/bin/python} $SCRIPT_DIR/collect_omega7_results.py $RUN_ROOT_BASE/ref${refine}"
done
