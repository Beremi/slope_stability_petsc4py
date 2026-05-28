#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT/.local/tmp/karolina_$(date +%Y%m%d_%H%M%S)}"
NODE_COUNTS="${NODE_COUNTS:-1 2}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
PARTITION="${PARTITION:-qcpu}"
CASE_TOML="${CASE_TOML:-$ROOT/benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml}"

mkdir -p "$RUN_ROOT"
for nodes in $NODE_COUNTS; do
  run_dir="$RUN_ROOT/nodes_${nodes}"
  mkdir -p "$run_dir"
  echo "Submitting nodes=$nodes run_dir=$run_dir"
  sbatch \
    --partition="$PARTITION" \
    --nodes="$nodes" \
    --ntasks-per-node=128 \
    --time="$TIME_LIMIT" \
    --output="$run_dir/slurm_%j.out" \
    --error="$run_dir/slurm_%j.err" \
    --export=ALL,ROOT="$ROOT",RUN_DIR="$run_dir",CASE_TOML="$CASE_TOML",OMEGA_MAX="${OMEGA_MAX:-7000000}",CONTINUATION_STEP_MAX="${CONTINUATION_STEP_MAX:-100}",LINEAR_RTOL="${LINEAR_RTOL:-1e-1}",KSP_MAX_IT="${KSP_MAX_IT:-200}",EXTRA_ARGS="${EXTRA_ARGS:-}" \
    "$SCRIPT_DIR/run_case.sbatch"
done

echo "RUN_ROOT=$RUN_ROOT"
