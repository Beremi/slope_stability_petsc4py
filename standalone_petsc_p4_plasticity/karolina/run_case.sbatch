#!/usr/bin/env bash
# Slurm resources are supplied by submit_pmg_scaling.sh with sbatch command-line
# options so this job body can be reused for each PMG profile point.
#SBATCH --cpus-per-task=1

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
fi
PLASTICITY_DIR="$REPO_ROOT/standalone_petsc_p4_plasticity"

CAMPAIGN_DIR="${CAMPAIGN_DIR:-$SCRIPT_DIR/runs/manual}"
TOTAL_RANKS="${SLURM_NTASKS:-${RANKS:?Set RANKS or submit with --ntasks}}"
NODES="${SLURM_JOB_NUM_NODES:-1}"
JOB_ID="${SLURM_JOB_ID:-manual_$$}"
JOB_NAME="${SLURM_JOB_NAME:-p4p_pmg_${TOTAL_RANKS}r}"
RUN_LABEL="${RUN_LABEL:-manual}"

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

KAROLINA_BZIP2_ROOT="${KAROLINA_BZIP2_ROOT:-/apps/all/bzip2/1.0.8-GCCcore-14.3.0}"
for libdir in "$KAROLINA_BZIP2_ROOT/lib64" "$KAROLINA_BZIP2_ROOT/lib"; do
  if [[ -d "$libdir" ]]; then
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
  fi
done

safe_label="${RUN_LABEL//[^A-Za-z0-9_.-]/_}"
RESULT_DIR="$CAMPAIGN_DIR/results/${JOB_ID}_pmg_${TOTAL_RANKS}r_${safe_label}"
mkdir -p "$RESULT_DIR"

metadata="$RESULT_DIR/job.env"
{
  echo "JOB_ID=$JOB_ID"
  echo "JOB_NAME=$JOB_NAME"
  echo "RUN_LABEL=$RUN_LABEL"
  echo "RANKS=$TOTAL_RANKS"
  echo "NODES=$NODES"
  echo "TASKS_PER_NODE=${TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-unknown}}"
  echo "NODELIST=${SLURM_JOB_NODELIST:-unknown}"
  echo "PARTITION=${SLURM_JOB_PARTITION:-${PARTITION:-unknown}}"
  echo "ACCOUNT=${SLURM_JOB_ACCOUNT:-${ACCOUNT:-unknown}}"
  echo "QOS=${SLURM_JOB_QOS:-${QOS:-unknown}}"
  echo "LAUNCHER=${LAUNCHER:-mpiexec}"
  echo "PETSC_DIR=$PETSC_DIR"
  echo "PETSC_ARCH=$PETSC_ARCH"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  echo "REPO_ROOT=$REPO_ROOT"
  echo "COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "BRANCH=$(git -C "$REPO_ROOT" branch --show-current)"
  echo "OPTIONS_FILE=${OPTIONS_FILE:-options/pmg_shell_vcycle.opts}"
  echo "PARTITIONER=${PARTITIONER:-parmetis}"
  echo "REFINE_LEVELS=${REFINE_LEVELS:-1}"
  echo "LAMBDA=${LAMBDA:-1.5}"
  echo "LINEAR_RTOL=${LINEAR_RTOL:-1e-1}"
  echo "KSP_MAX_IT=${KSP_MAX_IT:-200}"
  echo "NEWTON_MAX_IT=${NEWTON_MAX_IT:-20}"
  echo "DEFLATION=${DEFLATION:-true}"
  echo "PMG_COARSE_TELESCOPE_ACTIVE_RANKS=${PMG_COARSE_TELESCOPE_ACTIVE_RANKS:-}"
  echo "PMG_COARSE_TELESCOPE_SUBCOMM_TYPE=${PMG_COARSE_TELESCOPE_SUBCOMM_TYPE:-}"
  echo "PMG_COARSE_REDUNDANT_GROUP_SIZE=${PMG_COARSE_REDUNDANT_GROUP_SIZE:-}"
  echo "PMG_P2_TELESCOPE_ACTIVE_RANKS=${PMG_P2_TELESCOPE_ACTIVE_RANKS:-0}"
  echo "PMG_APPLY_BACKEND=${PMG_APPLY_BACKEND:-}"
  echo "PMG_SHELL_P2_ACTIVE_RANKS=${PMG_SHELL_P2_ACTIVE_RANKS:-}"
  echo "PMG_SHELL_P1_ACTIVE_RANKS=${PMG_SHELL_P1_ACTIVE_RANKS:-}"
  echo "PMG_SHELL_SUBCOMM_TYPE=${PMG_SHELL_SUBCOMM_TYPE:-}"
  echo "PMG_LAG_PRECONDITIONER=${PMG_LAG_PRECONDITIONER:-1}"
  echo "START_TIME=$(date --iso-8601=seconds)"
} | tee "$metadata"

if command -v module >/dev/null 2>&1; then
  module list >"$RESULT_DIR/module_list.txt" 2>&1 || true
fi

if [[ "${BUILD_BEFORE_RUN:-0}" == "1" ]]; then
  make -C "$PLASTICITY_DIR" >"$RESULT_DIR/build.log" 2>&1
fi

BASE_ARGS=(
  ./p4_plasticity
  -options_file "${OPTIONS_FILE:-options/pmg_shell_vcycle.opts}"
  -mesh "${MESH:-data/adaptive_family_a_l1.msh}"
  -refine_levels "${REFINE_LEVELS:-1}"
  -lambda "${LAMBDA:-1.5}"
  -linear_rtol "${LINEAR_RTOL:-1e-1}"
  -ksp_max_it "${KSP_MAX_IT:-200}"
  -newton_max_it "${NEWTON_MAX_IT:-20}"
  -ksp_type "${KSP_TYPE:-fgmres}"
  -ksp_norm_type "${KSP_NORM_TYPE:-unpreconditioned}"
  -deflation "${DEFLATION:-true}"
  -pmg_lag_preconditioner "${PMG_LAG_PRECONDITIONER:-1}"
  -ksp_converged_reason
)

if [[ -n "${PARTITIONER:-parmetis}" ]]; then
  BASE_ARGS+=(-petscpartitioner_type "${PARTITIONER:-parmetis}")
fi
if [[ -n "${PMG_COARSE_TELESCOPE_ACTIVE_RANKS:-}" ]]; then
  BASE_ARGS+=(
    -pmg_coarse_telescope_active_ranks "$PMG_COARSE_TELESCOPE_ACTIVE_RANKS"
    -pmg_coarse_telescope_subcomm_type "${PMG_COARSE_TELESCOPE_SUBCOMM_TYPE:-interlaced}"
  )
fi
if [[ -n "${PMG_COARSE_REDUNDANT_GROUP_SIZE:-}" ]]; then
  BASE_ARGS+=(-pmg_coarse_redundant_group_size "$PMG_COARSE_REDUNDANT_GROUP_SIZE")
fi
if [[ -n "${PMG_P2_TELESCOPE_ACTIVE_RANKS:-0}" ]]; then
  BASE_ARGS+=(-pmg_p2_telescope_active_ranks "${PMG_P2_TELESCOPE_ACTIVE_RANKS:-0}")
fi
if [[ -n "${PMG_APPLY_BACKEND:-}" ]]; then
  BASE_ARGS+=(-pmg_apply_backend "$PMG_APPLY_BACKEND")
fi
if [[ -n "${PMG_SHELL_P2_ACTIVE_RANKS:-}" ]]; then
  BASE_ARGS+=(-pmg_shell_p2_active_ranks "$PMG_SHELL_P2_ACTIVE_RANKS")
fi
if [[ -n "${PMG_SHELL_P1_ACTIVE_RANKS:-}" ]]; then
  BASE_ARGS+=(-pmg_shell_p1_active_ranks "$PMG_SHELL_P1_ACTIVE_RANKS")
fi
if [[ -n "${PMG_SHELL_SUBCOMM_TYPE:-}" ]]; then
  BASE_ARGS+=(-pmg_shell_subcomm_type "$PMG_SHELL_SUBCOMM_TYPE")
fi
if [[ "${LOG_VIEW:-1}" == "1" ]]; then
  BASE_ARGS+=(-log_view)
fi
if [[ -n "${EXTRA_PETSC_OPTIONS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($EXTRA_PETSC_OPTIONS)
  BASE_ARGS+=("${EXTRA_ARGS[@]}")
fi

LAUNCHER="${LAUNCHER:-mpiexec}"
case "$LAUNCHER" in
  srun)
    LAUNCH_ARGS=(-n "$TOTAL_RANKS" --cpu-bind="${SRUN_CPU_BIND:-cores}" --distribution="${SRUN_DISTRIBUTION:-block:block}")
    if [[ -n "${SRUN_EXTRA_ARGS:-}" ]]; then
      # shellcheck disable=SC2206
      MORE_LAUNCH_ARGS=($SRUN_EXTRA_ARGS)
      LAUNCH_ARGS+=("${MORE_LAUNCH_ARGS[@]}")
    fi
    ;;
  mpiexec)
    LAUNCH_ARGS=(-n "$TOTAL_RANKS")
    if [[ -n "${MPIEXEC_EXTRA_ARGS:-}" ]]; then
      # shellcheck disable=SC2206
      MORE_LAUNCH_ARGS=($MPIEXEC_EXTRA_ARGS)
      LAUNCH_ARGS+=("${MORE_LAUNCH_ARGS[@]}")
    fi
    ;;
  *)
    echo "ERROR: unknown LAUNCHER=$LAUNCHER, expected mpiexec or srun." >&2
    exit 2
    ;;
esac

{
  printf 'cd %q\n' "$PLASTICITY_DIR"
  printf '%q' "$LAUNCHER"
  printf ' %q' "${LAUNCH_ARGS[@]}"
  printf ' '
  printf '%q ' "${BASE_ARGS[@]}"
  printf '\n'
} | tee "$RESULT_DIR/command.sh"

cd "$PLASTICITY_DIR"
run_log="$RESULT_DIR/run.log"
time_log="$RESULT_DIR/time.txt"

set +e
/usr/bin/time -v -o "$time_log" "$LAUNCHER" "${LAUNCH_ARGS[@]}" "${BASE_ARGS[@]}" >"$run_log" 2>&1
exit_code=$?
set -e

grep '^RESULT ' "$run_log" | tail -n 1 >"$RESULT_DIR/result_line.txt" || true
grep -E '^(RESULT|DEFLATION_TIMING|PMG_|LINEAR_SOLVER_REUSE|BOUNDARY_COUNT)' "$run_log" >"$RESULT_DIR/diagnostics.txt" || true

if command -v sacct >/dev/null 2>&1 && [[ "$JOB_ID" != manual_* ]]; then
  sacct --units=M -j "$JOB_ID" --format=JobID,JobName%48,State,Elapsed,AllocNodes,AllocCPUS,NTasks,MaxRSS,AveRSS,MaxVMSize,ExitCode -P >"$RESULT_DIR/sacct.txt" 2>/dev/null || true
fi

{
  echo "END_TIME=$(date --iso-8601=seconds)"
  echo "EXIT_CODE=$exit_code"
} | tee -a "$metadata"

exit "$exit_code"
