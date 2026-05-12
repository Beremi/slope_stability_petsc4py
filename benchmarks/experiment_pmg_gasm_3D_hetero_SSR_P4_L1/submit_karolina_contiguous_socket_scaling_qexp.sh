#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${SCRIPT:-benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/karolina_contiguous_socket_scaling_qexp.sbatch}"
OUT_ROOT="${OUT_ROOT:-$HOME/slope_stability_petsc4py/artifacts/experiments/pmg_gasm_contiguous_karolina_socket_scaling_p4_l1_omega7}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-59m}"
RANKS_PER_SOCKET="${RANKS_PER_SOCKET:-16}"
OMEGA_MAX="${OMEGA_MAX:-7000000}"
STEP_MAX="${STEP_MAX:-100}"
MPI_TRANSPORT="${MPI_TRANSPORT:-ob1}"

CASE_NAMES=(gasm_1x16 gasm_2x16 gasm_4x16 gasm_8x16 gasm_16x16)
CASE_NODES=(1 1 1 1 2)
CASE_NTASKS_PER_NODE=(16 32 64 128 128)
CASE_SOCKETS=(1 2 4 8 16)

for idx in "${!CASE_NAMES[@]}"; do
  name="${CASE_NAMES[$idx]}"
  nodes="${CASE_NODES[$idx]}"
  ntasks_per_node="${CASE_NTASKS_PER_NODE[$idx]}"
  sockets="${CASE_SOCKETS[$idx]}"
  total_ranks=$((nodes * ntasks_per_node))
  expected_total_ranks=$((sockets * RANKS_PER_SOCKET))
  if [[ "$total_ranks" -ne "$expected_total_ranks" ]]; then
    echo "Refusing to submit $name: total ranks=$total_ranks, expected $expected_total_ranks." >&2
    exit 2
  fi

  echo "Submitting $name: nodes=$nodes ntasks_per_node=$ntasks_per_node sockets=$sockets ranks_per_socket=$RANKS_PER_SOCKET"
  sbatch \
    --job-name="pmg_c${name#gasm_}" \
    --nodes="$nodes" \
    --ntasks-per-node="$ntasks_per_node" \
    --time="$TIME_LIMIT" \
    --export=ALL,CASE_NAME="$name",CASE_NODES="$nodes",CASE_NTASKS_PER_NODE="$ntasks_per_node",CASE_SOCKETS="$sockets",RANKS_PER_SOCKET="$RANKS_PER_SOCKET",OUT_ROOT="$OUT_ROOT",OMEGA_MAX="$OMEGA_MAX",STEP_MAX="$STEP_MAX",COMMAND_TIMEOUT="$COMMAND_TIMEOUT",MPI_TRANSPORT="$MPI_TRANSPORT" \
    "$SCRIPT"
done

echo
echo "After the jobs finish, summarize with:"
echo "./.venv/bin/python benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/summarize_karolina_qexp.py --out-root \"$OUT_ROOT\""
