#!/usr/bin/env bash
#SBATCH --job-name=kar_l1r2_dbg16
#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6328
#SBATCH --partition=qcpu_exp
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:35:00
#SBATCH --output=artifacts/uniform_refined_3d_hetero_ssr_omega7e6/slurm/%x-%j.out
#SBATCH --error=artifacts/uniform_refined_3d_hetero_ssr_omega7e6/slurm/%x-%j.err

# Defaults target the P2(L1^2) debug case. For P4(L1^1), submit with:
#   --export=ALL,LEVEL=l1_r1,ELEM_TYPE=P4,PMG_FINE_HIERARCHY_MODE=p4_p2_intermediate,PMG_CHAIN_LEVELS=l1,PREPARE_MAX_REFINEMENT=1,RANKS_PER_NODE=16,COMMAND_TIMEOUT=59m

set -euo pipefail

ROOT="$HOME/slope_stability_petsc4py"
STUDY="$ROOT/artifacts/uniform_refined_3d_hetero_ssr_omega7e6"
NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-16}}"
RANKS_PER_NODE="${RANKS_PER_NODE:-16}"
TOTAL_RANKS="${SLURM_NTASKS:-$((NODES * RANKS_PER_NODE))}"
COARSE_REDUNDANT_GROUPS="${COARSE_REDUNDANT_GROUPS:-$NODES}"
LEVEL="${LEVEL:-l1_r2}"
ELEM_TYPE="${ELEM_TYPE:-P2}"
PMG_FINE_HIERARCHY_MODE="${PMG_FINE_HIERARCHY_MODE:-default}"
PMG_CHAIN_LEVELS="${PMG_CHAIN_LEVELS:-l1 l1_r1}"
ELEM_TYPE="${ELEM_TYPE^^}"
PMG_FINE_HIERARCHY_MODE="${PMG_FINE_HIERARCHY_MODE,,}"
PREPARE_MAX_REFINEMENT="${PREPARE_MAX_REFINEMENT:-2}"
RUN_NAME="${RUN_TAG:-karolina_debug_${NODES}node_${LEVEL}_${ELEM_TYPE}_unique_exchange_pmg_redundant_mumps_r${TOTAL_RANKS}_${RANKS_PER_NODE}ppn_${SLURM_JOB_ID:-manual}}"
RUN_DIR="$STUDY/runs/$RUN_NAME"
DEBUG_DIR="$RUN_DIR/debug"
CASE_PATH="$RUN_DIR/generated_case_${LEVEL}_${ELEM_TYPE}_${SLURM_JOB_ID:-manual}.toml"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-25m}"
MPI_TRANSPORT="${MPI_TRANSPORT:-ob1}"
MPI_ARGS=()
case "$MPI_TRANSPORT" in
  ob1)
    MPI_ARGS=(--mca pml ob1 --mca btl self,sm,tcp)
    ;;
  ucx)
    MPI_ARGS=()
    ;;
  custom)
    read -r -a MPI_ARGS <<< "${MPIEXEC_EXTRA_ARGS:-}"
    ;;
  *)
    echo "Unsupported MPI_TRANSPORT=$MPI_TRANSPORT; expected ob1, ucx, or custom" >&2
    exit 2
    ;;
esac
PMG_CHAIN_ARGS=()
if [[ -n "${PMG_CHAIN_LEVELS// }" ]]; then
  read -r -a PMG_CHAIN_ARRAY <<< "$PMG_CHAIN_LEVELS"
  if [[ "${#PMG_CHAIN_ARRAY[@]}" -gt 0 ]]; then
    PMG_CHAIN_ARGS=(--pmg-chain-levels "${PMG_CHAIN_ARRAY[@]}")
  fi
fi

mkdir -p "$RUN_DIR" "$DEBUG_DIR" "$STUDY/slurm"
cd "$ROOT"

finish() {
  rc=$?
  {
    echo "FINISH_RC=$rc"
    date
    echo "--- run files ---"
    find "$RUN_DIR" -maxdepth 5 -type f -printf '%p %s bytes\n' | sort || true
    echo "--- solver stdout tail ---"
    tail -n 120 "$RUN_DIR/solver.stdout" 2>/dev/null || true
    echo "--- solver stderr tail ---"
    tail -n 120 "$RUN_DIR/solver.stderr" 2>/dev/null || true
  } > "$RUN_DIR/final_snapshot.txt" 2>&1 || true
  sacct -j "${SLURM_JOB_ID:-}" --format=JobID,JobName,State,Elapsed,AllocNodes,AllocCPUS,MaxRSS,ExitCode -P > "$RUN_DIR/sacct_${SLURM_JOB_ID:-unknown}.txt" 2>/dev/null || true
  exit "$rc"
}
trap finish EXIT

ml -f purge
ml GCC/14.3.0 OpenMPI/5.0.8-GCC-14.3.0 Python/3.13.5-GCCcore-14.3.0
source build_scripts/activate_local_petsc_env.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export SSR_DEBUG_DIR="$DEBUG_DIR/runner"
export SLOPE_STABILITY_STAGE_DEBUG=1
export SLOPE_STABILITY_STAGE_DEBUG_DIR="$DEBUG_DIR/stages"
export PRTE_MCA_prte_base_help_aggregate=0

./.venv/bin/python "$STUDY/run_uniform_refined_study.py" prepare --max-refinement "$PREPARE_MAX_REFINEMENT" > "$RUN_DIR/prepare_meshes.log" 2>&1

module list > "$RUN_DIR/module_list.txt" 2>&1 || true
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/scontrol_job_${SLURM_JOB_ID}.txt" 2>/dev/null || true
env | sort > "$RUN_DIR/env.txt"

{
  echo "JOB_ID=${SLURM_JOB_ID:-}"
  echo "NODELIST=${SLURM_JOB_NODELIST:-}"
  echo "SLURM_NNODES=${SLURM_NNODES:-}"
  echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-}"
  echo "SLURM_NTASKS=$TOTAL_RANKS"
  echo "SLURM_TASKS_PER_NODE=${SLURM_TASKS_PER_NODE:-}"
  echo "RANKS_PER_NODE=$RANKS_PER_NODE"
  echo "COARSE_REDUNDANT_GROUPS=$COARSE_REDUNDANT_GROUPS"
  echo "LEVEL=$LEVEL"
  echo "ELEM_TYPE=$ELEM_TYPE"
  echo "PMG_FINE_HIERARCHY_MODE=$PMG_FINE_HIERARCHY_MODE"
  echo "PMG_CHAIN_LEVELS=$PMG_CHAIN_LEVELS"
  echo "PREPARE_MAX_REFINEMENT=$PREPARE_MAX_REFINEMENT"
  echo "MPI_TRANSPORT=$MPI_TRANSPORT"
  printf 'MPI_ARGS='
  printf '%q ' "${MPI_ARGS[@]}"
  printf '\n'
  printf 'PMG_CHAIN_ARGS='
  printf '%q ' "${PMG_CHAIN_ARGS[@]}"
  printf '\n'
  echo "RUN_NAME=$RUN_NAME"
  echo "RUN_DIR=$RUN_DIR"
  echo "DEBUG_DIR=$DEBUG_DIR"
  echo "CASE_PATH=$CASE_PATH"
  echo "COMMAND_TIMEOUT=$COMMAND_TIMEOUT"
  echo "ROOT=$ROOT"
  echo "STUDY=$STUDY"
  echo "PETSC_DIR=${PETSC_DIR:-}"
  echo "PETSC_ARCH=${PETSC_ARCH:-}"
  date
  git status --short
  git rev-parse HEAD
  ./.venv/bin/python - <<'PY'
from petsc4py import PETSc
print("PETSc", ".".join(map(str, PETSc.Sys.getVersion())))
PY
} | tee "$RUN_DIR/job_metadata.txt"

srun --ntasks="$TOTAL_RANKS" hostname | sort | uniq -c > "$RUN_DIR/allocated_hosts.txt" 2>&1 || true

CMD=(
  mpiexec
  "${MPI_ARGS[@]}"
  --report-bindings
  --bind-to core
  --map-by "ppr:${RANKS_PER_NODE}:node"
  -n "$TOTAL_RANKS"
  ./.venv/bin/python
  -u
  "$STUDY/run_uniform_refined_study.py"
  run-level
  --level "$LEVEL"
  --elem-type "$ELEM_TYPE"
  --omega-max 7000000
  --step-max 100
  --no-store-step-u
  --constitutive-mode unique_exchange
  --out-dir "$RUN_DIR/run"
  --case-path "$CASE_PATH"
  --debug-dir "$DEBUG_DIR/runner"
  --pmg-fine-hierarchy-mode "$PMG_FINE_HIERARCHY_MODE"
  "${PMG_CHAIN_ARGS[@]}"
  --mg-coarse-pc-type redundant
  --mg-coarse-factor-solver-type mumps
  --petsc-opt "mg_coarse_pc_redundant_number=${COARSE_REDUNDANT_GROUPS}"
  --petsc-opt mg_coarse_psubcomm_type=contiguous
  --petsc-opt mg_coarse_redundant_ksp_type=preonly
  --petsc-opt mg_coarse_redundant_pc_type=lu
  --petsc-opt mg_coarse_redundant_pc_factor_mat_solver_type=mumps
  --petsc-opt mg_levels_ksp_type=chebyshev
  --petsc-opt mg_levels_pc_type=jacobi
  --petsc-opt mg_levels_ksp_max_it=3
)

printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

set +e
/usr/bin/time -v -o "$RUN_DIR/time.txt" \
  timeout --signal=TERM --kill-after=90s "$COMMAND_TIMEOUT" \
  "${CMD[@]}" > "$RUN_DIR/solver.stdout" 2> "$RUN_DIR/solver.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$RUN_DIR/exit_code.txt"
{
  echo "--- STDOUT ---"
  cat "$RUN_DIR/solver.stdout"
  echo "--- STDERR ---"
  cat "$RUN_DIR/solver.stderr"
} > "$RUN_DIR/solver.log" || true

if [[ "$rc" -ne 0 ]]; then
  echo "debug run failed with rc=$rc; see $RUN_DIR" >&2
  exit "$rc"
fi

./.venv/bin/python - <<PY > "$RUN_DIR/quick_summary.txt"
import json
from pathlib import Path
run_dir = Path("$RUN_DIR")
info = run_dir / "run" / "data" / "run_info.json"
print("run_dir:", run_dir)
print("run_info_exists:", info.exists())
if info.exists():
    data = json.loads(info.read_text())
    ri = data["run_info"]
    print("runtime_seconds:", ri.get("runtime_seconds"))
    print("step_count:", ri.get("step_count"))
    print("unknowns:", ri.get("unknowns"))
PY
