#!/usr/bin/env bash
#SBATCH --job-name=pmg_cgasm_socket
#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6328
#SBATCH --partition=qcpu_exp
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${CASE_NAME:?CASE_NAME is required, e.g. gasm_8x16}"
: "${CASE_NODES:?CASE_NODES is required}"
: "${CASE_NTASKS_PER_NODE:?CASE_NTASKS_PER_NODE is required}"
: "${CASE_SOCKETS:?CASE_SOCKETS is required}"

ROOT="${ROOT:-$HOME/slope_stability_petsc4py}"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/experiments/pmg_gasm_contiguous_karolina_socket_scaling_p4_l1_omega7}"
OMEGA_MAX="${OMEGA_MAX:-7000000}"
STEP_MAX="${STEP_MAX:-100}"
RANKS_PER_SOCKET="${RANKS_PER_SOCKET:-16}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-59m}"
MPI_TRANSPORT="${MPI_TRANSPORT:-ob1}"

RUN_DIR="$OUT_ROOT/runs/$CASE_NAME/${SLURM_JOB_ID:-manual}"
RUN_OUT="$RUN_DIR/run"
CASE_CONFIG="$RUN_DIR/configs/$CASE_NAME.toml"
TOTAL_RANKS=$((CASE_NODES * CASE_NTASKS_PER_NODE))
EXPECTED_TOTAL_RANKS=$((CASE_SOCKETS * RANKS_PER_SOCKET))

if [[ "$TOTAL_RANKS" -ne "$EXPECTED_TOTAL_RANKS" ]]; then
  echo "Invalid case shape for $CASE_NAME" >&2
  echo "Expected total ranks=$EXPECTED_TOTAL_RANKS from $CASE_SOCKETS sockets * $RANKS_PER_SOCKET ranks/socket." >&2
  echo "Got CASE_NODES=$CASE_NODES CASE_NTASKS_PER_NODE=$CASE_NTASKS_PER_NODE total ranks=$TOTAL_RANKS." >&2
  exit 2
fi

cd "$ROOT"

if command -v ml >/dev/null 2>&1; then
  ml -f purge
  ml GCC/14.3.0 OpenMPI/5.0.8-GCC-14.3.0 Python/3.13.5-GCCcore-14.3.0
fi

if [[ -f build_scripts/activate_local_petsc_env.sh ]]; then
  # shellcheck disable=SC1091
  source build_scripts/activate_local_petsc_env.sh
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PRTE_MCA_prte_base_help_aggregate=0

case "$MPI_TRANSPORT" in
  ob1)
    MPI_ARGS=(--mca pml ob1 --mca btl self,sm,tcp)
    ;;
  ucx)
    MPI_ARGS=(--mca pml ucx)
    ;;
  default)
    MPI_ARGS=()
    ;;
  *)
    echo "Unknown MPI_TRANSPORT=$MPI_TRANSPORT; use ob1, ucx, or default" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_DIR"

./.venv/bin/python benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/prepare_karolina_contiguous_socket_cases.py \
  --out-root "$RUN_DIR" \
  --only "$CASE_NAME" \
  --omega-max "$OMEGA_MAX" \
  --step-max "$STEP_MAX"

cat > "$RUN_DIR/case_metadata.json" <<EOF
{
  "case_name": "$CASE_NAME",
  "kind": "gasm",
  "nodes": $CASE_NODES,
  "ntasks_per_node": $CASE_NTASKS_PER_NODE,
  "ranks": $TOTAL_RANKS,
  "sockets": $CASE_SOCKETS,
  "ranks_per_socket": $RANKS_PER_SOCKET,
  "grouping": "contiguous",
  "overlap": 1,
  "omega_max": $OMEGA_MAX,
  "step_max": $STEP_MAX,
  "command_timeout": "$COMMAND_TIMEOUT",
  "mpi_transport": "$MPI_TRANSPORT"
}
EOF

cp "$RUN_DIR/case_metadata.json" "$RUN_DIR/job_metadata.json"

echo "Root: $ROOT"
echo "Output root: $OUT_ROOT"
echo "Case: $CASE_NAME"
echo "Nodes: $CASE_NODES"
echo "Tasks per node: $CASE_NTASKS_PER_NODE"
echo "Total ranks: $TOTAL_RANKS"
echo "Contiguous GASM sockets: $CASE_SOCKETS"
echo "Ranks per socket: $RANKS_PER_SOCKET"
echo "Config: $CASE_CONFIG"
echo "Run output: $RUN_OUT"
echo "MPI transport: $MPI_TRANSPORT"
echo "Command timeout: $COMMAND_TIMEOUT"
echo "Nodes in allocation:"
srun --ntasks=1 hostname
echo "MPI rank placement:"
mpiexec "${MPI_ARGS[@]}" --bind-to core --map-by "ppr:${CASE_NTASKS_PER_NODE}:node" -n "$TOTAL_RANKS" hostname | sort | uniq -c

start_epoch="$(date +%s)"
set +e
timeout --signal=TERM --kill-after=60s "$COMMAND_TIMEOUT" \
  mpiexec "${MPI_ARGS[@]}" --bind-to core --map-by "ppr:${CASE_NTASKS_PER_NODE}:node" -n "$TOTAL_RANKS" \
    ./.venv/bin/python -u -m slope_stability.cli.run_case_from_config "$CASE_CONFIG" --out_dir "$RUN_OUT" \
    2>&1 | tee "$RUN_DIR/stdout.log"
exit_code=${PIPESTATUS[0]}
set -e
end_epoch="$(date +%s)"
wall_seconds=$((end_epoch - start_epoch))

echo "$exit_code" > "$RUN_DIR/exit_code.txt"
echo "$wall_seconds" > "$RUN_DIR/wall_seconds.txt"

./.venv/bin/python benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/summarize_karolina_qexp.py \
  --out-root "$OUT_ROOT" || true

echo "Exit code: $exit_code"
echo "Wall seconds: $wall_seconds"
echo "Summary: $OUT_ROOT/summary.md"
exit "$exit_code"
