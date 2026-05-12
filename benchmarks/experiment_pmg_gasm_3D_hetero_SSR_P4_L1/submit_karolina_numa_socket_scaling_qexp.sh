#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${SCRIPT:-benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/karolina_numa_socket_scaling_qexp.sbatch}"
OUT_ROOT="${OUT_ROOT:-$HOME/slope_stability_petsc4py/artifacts/experiments/pmg_numa_coalesced_karolina_socket_scaling_p4_l1_omega7}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"

CASE_NAMES=(numa_1x16 numa_2x16 numa_4x16 numa_8x16 numa_16x16)
CASE_NODES=(1 1 1 1 2)
CASE_NTASKS_PER_NODE=(16 32 64 128 128)
CASE_NUMA_DOMAINS_PER_NODE=(1 2 4 8 8)

for idx in "${!CASE_NAMES[@]}"; do
  name="${CASE_NAMES[$idx]}"
  nodes="${CASE_NODES[$idx]}"
  ntasks_per_node="${CASE_NTASKS_PER_NODE[$idx]}"
  numa_per_node="${CASE_NUMA_DOMAINS_PER_NODE[$idx]}"
  echo "Submitting $name: nodes=$nodes ntasks_per_node=$ntasks_per_node numa_per_node=$numa_per_node"
  sbatch \
    --job-name="pmg_${name}" \
    --nodes="$nodes" \
    --ntasks-per-node="$ntasks_per_node" \
    --time="$TIME_LIMIT" \
    --export=ALL,CASE_NAME="$name",CASE_NODES="$nodes",CASE_NTASKS_PER_NODE="$ntasks_per_node",CASE_NUMA_DOMAINS_PER_NODE="$numa_per_node",OUT_ROOT="$OUT_ROOT" \
    "$SCRIPT"
done
