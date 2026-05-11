#!/usr/bin/env bash
#SBATCH --job-name=pmg_gasm_p4l1
#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6328
#SBATCH --partition=qcpu_exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --array=0-7
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

CASE_NAMES=(
  baseline_16
  baseline_32
  baseline_64
  baseline_128
  gasm_1x16
  gasm_2x16
  gasm_4x16
  gasm_8x16
)
CASE_KINDS=(baseline baseline baseline baseline gasm gasm gasm gasm)
CASE_RANKS=(16 32 64 128 16 32 64 128)
CASE_SOCKETS=(0 0 0 0 1 2 4 8)
CASE_RANKS_PER_SOCKET=(0 0 0 0 16 16 16 16)

ROOT="${ROOT:-$HOME/slope_stability_petsc4py}"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/experiments/pmg_gasm_karolina_qexp_one_node_p4_l1_omega7}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-19m}"
MPI_TRANSPORT="${MPI_TRANSPORT:-ob1}"

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
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
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

CASE_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( CASE_INDEX < 0 || CASE_INDEX >= ${#CASE_NAMES[@]} )); then
  echo "Invalid CASE_INDEX=$CASE_INDEX" >&2
  exit 2
fi

CASE_NAME="${CASE_NAMES[$CASE_INDEX]}"
CASE_KIND="${CASE_KINDS[$CASE_INDEX]}"
RANKS="${CASE_RANKS[$CASE_INDEX]}"
SOCKETS="${CASE_SOCKETS[$CASE_INDEX]}"
RANKS_PER_SOCKET="${CASE_RANKS_PER_SOCKET[$CASE_INDEX]}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-manual}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-$CASE_INDEX}"
RUN_DIR="$OUT_ROOT/runs/$CASE_NAME/${JOB_ID}_${TASK_ID}"
RUN_OUT="$RUN_DIR/run"
CASE_CONFIG="$RUN_DIR/configs/$CASE_NAME.toml"

mkdir -p "$RUN_DIR"

./.venv/bin/python benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/prepare_karolina_qexp_cases.py \
  --out-root "$RUN_DIR" \
  --only "$CASE_NAME" \
  --omega-max 7000000 \
  --step-max 100

./.venv/bin/python - "$RUN_DIR/case_metadata.json" "$CASE_NAME" "$CASE_KIND" "$RANKS" "$SOCKETS" "$RANKS_PER_SOCKET" <<'PY'
import json
import sys

path, case_name, kind, ranks, sockets, ranks_per_socket = sys.argv[1:]
payload = {
    "case_name": case_name,
    "kind": kind,
    "ranks": int(ranks),
    "sockets": None if int(sockets) == 0 else int(sockets),
    "ranks_per_socket": None if int(ranks_per_socket) == 0 else int(ranks_per_socket),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

echo "Root: $ROOT"
echo "Output root: $OUT_ROOT"
echo "Case: $CASE_NAME"
echo "Ranks: $RANKS"
echo "Fake sockets: $SOCKETS"
echo "Config: $CASE_CONFIG"
echo "Run output: $RUN_OUT"
echo "MPI transport: $MPI_TRANSPORT"
echo "Command timeout: $COMMAND_TIMEOUT"
echo "Nodes in allocation:"
srun --ntasks=1 hostname
echo "MPI rank placement:"
mpiexec "${MPI_ARGS[@]}" --bind-to core --map-by "ppr:${RANKS}:node" -n "$RANKS" hostname | sort | uniq -c

start_epoch="$(date +%s)"
set +e
timeout --signal=TERM --kill-after=60s "$COMMAND_TIMEOUT" \
  mpiexec "${MPI_ARGS[@]}" --bind-to core --map-by "ppr:${RANKS}:node" -n "$RANKS" \
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
