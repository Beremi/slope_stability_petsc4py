#!/usr/bin/env bash
#SBATCH --job-name=pmg_numa_socket
#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6328
#SBATCH --partition=qcpu_exp
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

: "${CASE_NAME:?CASE_NAME is required, e.g. numa_8x16}"
: "${CASE_NODES:?CASE_NODES is required}"
: "${CASE_NTASKS_PER_NODE:?CASE_NTASKS_PER_NODE is required}"
: "${CASE_NUMA_DOMAINS_PER_NODE:?CASE_NUMA_DOMAINS_PER_NODE is required}"

ROOT="${ROOT:-$HOME/slope_stability_petsc4py}"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/experiments/pmg_numa_coalesced_karolina_socket_scaling_p4_l1_omega7}"
OMEGA_MAX="${OMEGA_MAX:-7000000}"
STEP_MAX="${STEP_MAX:-100}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-19m}"

RUN_DIR="$OUT_ROOT/runs/$CASE_NAME/${SLURM_JOB_ID:-manual}"
RUN_OUT="$RUN_DIR/run"
CASE_CONFIG="$RUN_DIR/configs/$CASE_NAME.toml"
TOTAL_RANKS=$((CASE_NODES * CASE_NTASKS_PER_NODE))

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

mkdir -p "$RUN_DIR"

./.venv/bin/python benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/prepare_karolina_numa_socket_cases.py \
  --out-root "$RUN_DIR" \
  --only "$CASE_NAME" \
  --omega-max "$OMEGA_MAX" \
  --step-max "$STEP_MAX"

cat > "$RUN_DIR/job_metadata.json" <<EOF
{
  "case_name": "$CASE_NAME",
  "nodes": $CASE_NODES,
  "ntasks_per_node": $CASE_NTASKS_PER_NODE,
  "total_ranks": $TOTAL_RANKS,
  "numa_domains_per_node": $CASE_NUMA_DOMAINS_PER_NODE,
  "omega_max": $OMEGA_MAX,
  "step_max": $STEP_MAX
}
EOF

echo "Root: $ROOT"
echo "Output root: $OUT_ROOT"
echo "Case: $CASE_NAME"
echo "Nodes: $CASE_NODES"
echo "Tasks per node: $CASE_NTASKS_PER_NODE"
echo "Total ranks: $TOTAL_RANKS"
echo "NUMA domains per node in config: $CASE_NUMA_DOMAINS_PER_NODE"
echo "Config: $CASE_CONFIG"
echo "Run output: $RUN_OUT"
echo "Command timeout: $COMMAND_TIMEOUT"

start_epoch="$(date +%s)"
set +e
timeout --signal=TERM --kill-after=60s "$COMMAND_TIMEOUT" \
  srun \
    --nodes="$CASE_NODES" \
    --ntasks-per-node="$CASE_NTASKS_PER_NODE" \
    --cpus-per-task=1 \
    --distribution=block:block \
    --cpu-bind=cores \
    ./.venv/bin/python -u -m slope_stability.cli.run_case_from_config "$CASE_CONFIG" --out_dir "$RUN_OUT" \
    2>&1 | tee "$RUN_DIR/stdout.log"
exit_code=${PIPESTATUS[0]}
set -e
end_epoch="$(date +%s)"
wall_seconds=$((end_epoch - start_epoch))

echo "$exit_code" > "$RUN_DIR/exit_code.txt"
echo "$wall_seconds" > "$RUN_DIR/wall_seconds.txt"

echo "Exit code: $exit_code"
echo "Wall seconds: $wall_seconds"
exit "$exit_code"
