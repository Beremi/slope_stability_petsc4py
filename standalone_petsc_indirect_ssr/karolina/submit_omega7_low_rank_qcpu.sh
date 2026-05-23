#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PARTITION="${PARTITION:-qcpu}"
export TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
export LAYOUTS="${LAYOUTS:-1:4 1:8}"
export ENGINES="${ENGINES:-c}"
export PROFILES="${PROFILES:-split}"
export REFINE_LEVELS="${REFINE_LEVELS:-1}"
export PMG_COARSE_MAX_IT="${PMG_COARSE_MAX_IT:-5}"
export RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/ssr_omega7_lowrank_qcpu_$(date +%Y%m%d_%H%M%S)}"

exec "$SCRIPT_DIR/submit_omega7_grid.sh"
