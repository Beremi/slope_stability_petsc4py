#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/slope_stability_petsc4py}"
SCRIPT="benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/karolina_multinode_full_occupancy.sbatch"
RANKS_PER_NODE="${RANKS_PER_NODE:-128}"
SOCKETS_PER_NODE="${SOCKETS_PER_NODE:-8}"
PARTITION="${PARTITION:-qcpu}"
RANKS_PER_SOCKET=$((RANKS_PER_NODE / SOCKETS_PER_NODE))
NODE_COUNTS=(${NODE_COUNTS:-2 4 8 16})

cd "$ROOT"

for nodes in "${NODE_COUNTS[@]}"; do
  ranks=$((nodes * RANKS_PER_NODE))
  echo "Submitting baseline_${nodes}n with ${nodes} node(s), ${ranks} ranks"
  sbatch \
    --partition="$PARTITION" \
    --nodes="$nodes" \
    --ntasks-per-node="$RANKS_PER_NODE" \
    --export=ALL,CASE_NAME="baseline_${nodes}n",CASE_KIND=baseline,CASE_NODES="$nodes",CASE_RANKS="$ranks",CASE_SOCKETS=0,CASE_RANKS_PER_SOCKET=0 \
    "$SCRIPT"

  sockets=$((nodes * SOCKETS_PER_NODE))
  echo "Submitting gasm_${sockets}x${RANKS_PER_SOCKET} with ${nodes} node(s), ${ranks} ranks, ${sockets} fake sockets"
  sbatch \
    --partition="$PARTITION" \
    --nodes="$nodes" \
    --ntasks-per-node="$RANKS_PER_NODE" \
    --export=ALL,CASE_NAME="gasm_${sockets}x${RANKS_PER_SOCKET}",CASE_KIND=gasm,CASE_NODES="$nodes",CASE_RANKS="$ranks",CASE_SOCKETS="$sockets",CASE_RANKS_PER_SOCKET="$RANKS_PER_SOCKET" \
    "$SCRIPT"
done
