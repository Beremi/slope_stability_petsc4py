#!/usr/bin/env bash
# Slurm resources are supplied by submit_material_sweep.sh with sbatch command-line
# options. The job body runs one material-sweep mode on fully occupied CPU nodes.
#SBATCH --cpus-per-task=1

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
fi
ELASTICITY_DIR="$REPO_ROOT/standalone_petsc_p4_plasticity/p4_elasticity"

CASE="${CASE:-l1}"
MATERIAL_SWEEP_MODE="${MATERIAL_SWEEP_MODE:?Set MATERIAL_SWEEP_MODE=fresh|refresh|reuse_pc}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-$SCRIPT_DIR/runs/material_manual}"
TOTAL_RANKS="${SLURM_NTASKS:-${RANKS:?Set RANKS or submit with --ntasks}}"
NODES="${SLURM_JOB_NUM_NODES:-1}"
JOB_ID="${SLURM_JOB_ID:-manual_$$}"
JOB_NAME="${SLURM_JOB_NAME:-p4e_ms_${CASE}_${MATERIAL_SWEEP_MODE}_${TOTAL_RANKS}r}"
RUN_LABEL="${RUN_LABEL:-}"

export PETSC_ARCH="${PETSC_ARCH:-linux-c-opt}"
if [[ -z "${PETSC_DIR:-}" ]]; then
  if [[ -d "$REPO_ROOT/.build/src/petsc-3.24.5" ]]; then
    export PETSC_DIR="$REPO_ROOT/.build/src/petsc-3.24.5"
  else
    echo "ERROR: PETSC_DIR is not set and $REPO_ROOT/.build/src/petsc-3.24.5 does not exist." >&2
    exit 2
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# PETSc was built against EasyBuild dependencies on Karolina. Some transitive
# libraries are not visible on compute nodes unless their module paths are in
# the runtime search path.
KAROLINA_BZIP2_ROOT="${KAROLINA_BZIP2_ROOT:-/apps/all/bzip2/1.0.8-GCCcore-14.3.0}"
for libdir in "$KAROLINA_BZIP2_ROOT/lib64" "$KAROLINA_BZIP2_ROOT/lib"; do
  if [[ -d "$libdir" ]]; then
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
  fi
done

safe_label=""
if [[ -n "$RUN_LABEL" ]]; then
  safe_label="_${RUN_LABEL//[^A-Za-z0-9_.-]/_}"
fi
RESULT_DIR="$CAMPAIGN_DIR/results/${JOB_ID}_${CASE}_pmg_${MATERIAL_SWEEP_MODE}_${TOTAL_RANKS}r${safe_label}"
mkdir -p "$RESULT_DIR"

metadata="$RESULT_DIR/job.env"
{
  echo "JOB_ID=$JOB_ID"
  echo "JOB_NAME=$JOB_NAME"
  echo "RUN_LABEL=$RUN_LABEL"
  echo "CASE=$CASE"
  echo "VARIANT=pmg"
  echo "MATERIAL_SWEEP_MODE=$MATERIAL_SWEEP_MODE"
  echo "MATERIAL_SWEEP_COUNT=${MATERIAL_SWEEP_COUNT:-100}"
  echo "RANKS=$TOTAL_RANKS"
  echo "NODES=$NODES"
  echo "TASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-${TASKS_PER_NODE:-unknown}}"
  echo "NODELIST=${SLURM_JOB_NODELIST:-unknown}"
  echo "PARTITION=${SLURM_JOB_PARTITION:-${PARTITION:-unknown}}"
  echo "ACCOUNT=${SLURM_JOB_ACCOUNT:-${ACCOUNT:-unknown}}"
  echo "QOS=${SLURM_JOB_QOS:-${QOS:-unknown}}"
  echo "PMG_GROUP_SIZE=${PMG_GROUP_SIZE:-16}"
  echo "PMG_COARSE_TELESCOPE_FACTOR=${PMG_COARSE_TELESCOPE_FACTOR:-}"
  echo "PMG_COARSE_TELESCOPE_ACTIVE_RANKS=${PMG_COARSE_TELESCOPE_ACTIVE_RANKS:-}"
  echo "PMG_COARSE_TELESCOPE_SUBCOMM_TYPE=${PMG_COARSE_TELESCOPE_SUBCOMM_TYPE:-}"
  echo "EXTRA_PETSC_OPTIONS=${EXTRA_PETSC_OPTIONS:-}"
  echo "PETSC_DIR=$PETSC_DIR"
  echo "PETSC_ARCH=$PETSC_ARCH"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  echo "REPO_ROOT=$REPO_ROOT"
  echo "COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "BRANCH=$(git -C "$REPO_ROOT" branch --show-current)"
  echo "START_TIME=$(date --iso-8601=seconds)"
} | tee "$metadata"

if command -v module >/dev/null 2>&1; then
  module list >"$RESULT_DIR/module_list.txt" 2>&1 || true
fi

if [[ "${BUILD_BEFORE_RUN:-0}" == "1" ]]; then
  make -C "$ELASTICITY_DIR" >"$RESULT_DIR/build.log" 2>&1
fi

case "$CASE" in
  cube)
    EXE="./cube_elasticity"
    PARTITIONER="${CUBE_PARTITIONER:-parmetis}"
    CASE_ARGS=(-cube_faces "${CUBE_FACES:-16,16,16}")
    ;;
  l1)
    EXE="./l1_elasticity"
    PARTITIONER="${L1_PARTITIONER:-parmetis}"
    CASE_ARGS=(-mesh_bc_mode "${L1_BC_MODE:-rollers}")
    ;;
  *)
    echo "ERROR: unknown CASE=$CASE, expected cube or l1." >&2
    exit 2
    ;;
esac

BASE_ARGS=(
  "$EXE"
  "${CASE_ARGS[@]}"
  -petscpartitioner_type "$PARTITIONER"
  -pc_variant pmg
  -pmg_coarse_redundant_group_size "${PMG_GROUP_SIZE:-16}"
  -pmg_coarse_gamg_aggressive_square_graph "${PMG_SQUARE_GRAPH:-false}"
  -ksp_type "${KSP_TYPE:-fgmres}"
  -ksp_gmres_restart "${KSP_GMRES_RESTART:-100}"
  -ksp_rtol "${KSP_RTOL:-1e-3}"
  -ksp_max_it "${KSP_MAX_IT:-200}"
  -mg_coarse_ksp_type "${MG_COARSE_KSP_TYPE:-fgmres}"
  -mg_coarse_ksp_gmres_restart "${MG_COARSE_KSP_GMRES_RESTART:-100}"
  -mg_coarse_ksp_max_it "${MG_COARSE_KSP_MAX_IT:-100}"
  -material_sweep_count "${MATERIAL_SWEEP_COUNT:-100}"
  -material_sweep_seed "${MATERIAL_SWEEP_SEED:-1729}"
  -material_sweep_mode "$MATERIAL_SWEEP_MODE"
)

if [[ -n "${PMG_COARSE_TELESCOPE_FACTOR:-}" ]]; then
  BASE_ARGS+=(
    -mg_coarse_pc_type telescope
    -mg_coarse_pc_telescope_reduction_factor "$PMG_COARSE_TELESCOPE_FACTOR"
    -mg_coarse_pc_telescope_subcomm_type "${PMG_COARSE_TELESCOPE_SUBCOMM_TYPE:-interlaced}"
    -mg_coarse_telescope_ksp_type "${PMG_COARSE_TELESCOPE_KSP_TYPE:-fgmres}"
    -mg_coarse_telescope_ksp_rtol "${PMG_COARSE_TELESCOPE_KSP_RTOL:-1e-3}"
    -mg_coarse_telescope_ksp_max_it "${PMG_COARSE_TELESCOPE_KSP_MAX_IT:-100}"
    -mg_coarse_telescope_ksp_gmres_restart "${PMG_COARSE_TELESCOPE_KSP_GMRES_RESTART:-100}"
    -mg_coarse_telescope_pc_type "${PMG_COARSE_TELESCOPE_PC_TYPE:-gamg}"
  )
fi

if [[ "${LOG_VIEW:-1}" == "1" ]]; then
  BASE_ARGS+=(-log_view)
fi

if [[ -n "${EXTRA_PETSC_OPTIONS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($EXTRA_PETSC_OPTIONS)
  BASE_ARGS+=("${EXTRA_ARGS[@]}")
fi

SRUN_ARGS=(-n "$TOTAL_RANKS" --cpu-bind="${SRUN_CPU_BIND:-cores}" --distribution="${SRUN_DISTRIBUTION:-block:block}")
if [[ -n "${SRUN_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  MORE_SRUN_ARGS=($SRUN_EXTRA_ARGS)
  SRUN_ARGS+=("${MORE_SRUN_ARGS[@]}")
fi

{
  printf 'cd %q\n' "$ELASTICITY_DIR"
  printf 'srun'
  printf ' %q' "${SRUN_ARGS[@]}"
  printf ' '
  printf '%q ' "${BASE_ARGS[@]}"
  printf '\n'
} | tee "$RESULT_DIR/command.sh"

cd "$ELASTICITY_DIR"
run_log="$RESULT_DIR/run.log"
time_log="$RESULT_DIR/time.txt"

set +e
/usr/bin/time -v -o "$time_log" srun "${SRUN_ARGS[@]}" "${BASE_ARGS[@]}" >"$run_log" 2>&1
exit_code=$?
set -e

grep '^RESULT ' "$run_log" | tail -n 1 >"$RESULT_DIR/result_line.txt" || true
grep '^SWEEP_RESULT ' "$run_log" >"$RESULT_DIR/sweep_results.txt" || true
grep -E '^(RESULT|SWEEP_CONFIG|SWEEP_RESULT|PMG_COARSE_SOLVE|BOUNDARY_COUNT)' "$run_log" >"$RESULT_DIR/diagnostics.txt" || true

if command -v sacct >/dev/null 2>&1 && [[ "$JOB_ID" != manual_* ]]; then
  sacct --units=M -j "$JOB_ID" --format=JobID,JobName%48,State,Elapsed,AllocNodes,AllocCPUS,NTasks,MaxRSS,AveRSS,MaxVMSize,ExitCode -P >"$RESULT_DIR/sacct.txt" 2>/dev/null || true
fi

{
  echo "END_TIME=$(date --iso-8601=seconds)"
  echo "EXIT_CODE=$exit_code"
} | tee -a "$metadata"

exit "$exit_code"
