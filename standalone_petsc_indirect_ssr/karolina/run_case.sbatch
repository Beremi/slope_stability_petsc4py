#!/usr/bin/env bash
# Slurm resources are supplied by submit_omega7_grid.sh so this job body can be
# reused for every engine/profile/occupancy point.
#SBATCH --cpus-per-task=1

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
fi
SSR_DIR="$REPO_ROOT/standalone_petsc_indirect_ssr"

CAMPAIGN_DIR="${CAMPAIGN_DIR:-$SCRIPT_DIR/runs/manual}"
TOTAL_RANKS="${SLURM_NTASKS:-${RANKS:?Set RANKS or submit with --ntasks}}"
NODES="${SLURM_JOB_NUM_NODES:-${NODES:-1}}"
TASKS_PER_NODE="${TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-unknown}}"
JOB_ID="${SLURM_JOB_ID:-manual_$$}"
JOB_NAME="${SLURM_JOB_NAME:-ssr_${TOTAL_RANKS}r}"
RUN_LABEL="${RUN_LABEL:-manual}"
ENGINE="${ENGINE:-c}"
PROFILE="${PROFILE:-baseline}"

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
RESULT_DIR="$CAMPAIGN_DIR/results/${JOB_ID}_ssr_${ENGINE}_${PROFILE}_${TOTAL_RANKS}r_${safe_label}"
mkdir -p "$RESULT_DIR"

metadata="$RESULT_DIR/job.env"
{
  echo "JOB_ID=$JOB_ID"
  echo "JOB_NAME=$JOB_NAME"
  echo "RUN_LABEL=$RUN_LABEL"
  echo "ENGINE=$ENGINE"
  echo "PROFILE=$PROFILE"
  echo "RANKS=$TOTAL_RANKS"
  echo "NODES=$NODES"
  echo "TASKS_PER_NODE=$TASKS_PER_NODE"
  echo "NODELIST=${SLURM_JOB_NODELIST:-unknown}"
  echo "PARTITION=${SLURM_JOB_PARTITION:-${PARTITION:-unknown}}"
  echo "ACCOUNT=${SLURM_JOB_ACCOUNT:-${ACCOUNT:-unknown}}"
  echo "QOS=${SLURM_JOB_QOS:-${QOS:-unknown}}"
  echo "LAUNCHER=${LAUNCHER:-mpiexec}"
  echo "PETSC_DIR=$PETSC_DIR"
  echo "PETSC_ARCH=$PETSC_ARCH"
  echo "PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
  echo "PY_PETSC_OPTIONS=${PY_PETSC_OPTIONS:--log_view}"
  echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  echo "REPO_ROOT=$REPO_ROOT"
  echo "COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "BRANCH=$(git -C "$REPO_ROOT" branch --show-current)"
  echo "OMEGA_MAX=${OMEGA_MAX:-7e6}"
  echo "CONTINUATION_STEP_MAX=${CONTINUATION_STEP_MAX:-100}"
  echo "LINEAR_RTOL=${LINEAR_RTOL:-1e-1}"
  echo "KSP_MAX_IT=${KSP_MAX_IT:-200}"
  echo "PARTITIONER=${PARTITIONER:-parmetis}"
  echo "OPTIONS_FILE=${OPTIONS_FILE:-}"
  echo "START_TIME=$(date --iso-8601=seconds)"
} | tee "$metadata"

if command -v module >/dev/null 2>&1; then
  module list >"$RESULT_DIR/module_list.txt" 2>&1 || true
fi

if [[ "${BUILD_BEFORE_RUN:-0}" == "1" && "$ENGINE" == "c" ]]; then
  make -C "$SSR_DIR" >"$RESULT_DIR/build.log" 2>&1
fi

profile_options_file() {
  case "$PROFILE" in
    baseline) printf '%s\n' "${OPTIONS_FILE:-options/pmg_shell_vcycle.opts}" ;;
    split) printf '%s\n' "${OPTIONS_FILE:-options/pmg_shell_split_smoother.opts}" ;;
    petsc4py) printf '%s\n' "${OPTIONS_FILE:-options/pmg_shell_petsc4py.opts}" ;;
    *) echo "ERROR: unknown PROFILE=$PROFILE, expected baseline, split, or petsc4py." >&2; return 2 ;;
  esac
}

write_py_config() {
  local cfg="$1"
  local petsc_opt

  case "$PROFILE" in
    baseline)
      petsc_opt='"mg_levels_ksp_type=chebyshev", "mg_levels_ksp_max_it=2", "mg_levels_pc_type=jacobi", "mg_coarse_ksp_type=fgmres", "mg_coarse_rtol=1e-3", "mg_coarse_max_it=100", "mg_coarse_pc_type=gamg", "pc_gamg_aggressive_square_graph=false"'
      ;;
    petsc4py)
      petsc_opt='"pc_hypre_boomeramg_max_iter=4", "pc_hypre_boomeramg_tol=0.0"'
      ;;
    split)
      echo "ERROR: PROFILE=split is only implemented for ENGINE=c." >&2
      return 2
      ;;
    *)
      echo "ERROR: unknown PROFILE=$PROFILE, expected baseline, split, or petsc4py." >&2
      return 2
      ;;
  esac

  cat >"$cfg" <<TOML
[benchmark]
title = "3D heterogeneous indirect SSR omega7 ${PROFILE}"
matlab_script = "slope_stability_3D_hetero_SSR.m"
comparison_kind = "continuation"
mpi_ranks = ${TOTAL_RANKS}
suite = false

[problem]
name = "slope_stability_3D_hetero_SSR_omega7_${PROFILE}_${TOTAL_RANKS}r"
asset = "3d_hetero_slope"
mesh_variant = "adaptive_family_a_l1.msh"
analysis = "ssr"
elem_type = "P4"
davis_type = "B"

[execution]
node_ordering = "block_metis"
mpi_distribute_by_nodes = true
constitutive_mode = "overlap"

[continuation]
method = "indirect"
lambda_init = 1.0
d_lambda_init = 0.1
d_lambda_min = 1e-3
d_lambda_diff_scaled_min = 1e-3
omega_max = ${OMEGA_MAX:-7e6}
init_newton_stopping_criterion = "relative_correction"
init_newton_stopping_tol = 1e-3
step_max = ${CONTINUATION_STEP_MAX:-100}

[newton]
it_max = 200
it_damp_max = 10
tol = 1e-4
r_min = 1e-4
stopping_criterion = "absolute_delta_lambda"
stopping_tol = 1e-4

[linear_solver]
solver_type = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"
tolerance = ${LINEAR_RTOL:-1e-1}
max_iterations = ${KSP_MAX_IT:-200}
deflation_basis_tolerance = 1e-3
threads = 1
print_level = 0
use_as_preconditioner = true
compiled_outer = false
recycle_preconditioner = true
pc_backend = "pmg_shell"
pc_hypre_coarsen_type = "HMIS"
pc_hypre_interp_type = "ext+i"
petsc_opt = [${petsc_opt}]

[export]
write_custom_debug_bundle = false
write_history_json = true
write_solution_vtu = false
TOML
}

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

run_log="$RESULT_DIR/run.log"
time_log="$RESULT_DIR/time.txt"

if [[ "$ENGINE" == "c" ]]; then
  opts="$(profile_options_file)"
  BASE_ARGS=(
    ./p4_indirect_ssr
    -options_file "$opts"
    -mesh "${MESH:-data/adaptive_family_a_l1.msh}"
    -omega_max "${OMEGA_MAX:-7e6}"
    -continuation_step_max "${CONTINUATION_STEP_MAX:-100}"
    -linear_rtol "${LINEAR_RTOL:-1e-1}"
    -ksp_max_it "${KSP_MAX_IT:-200}"
    -curve_csv "$RESULT_DIR/continuation_curve.csv"
    -ksp_converged_reason
  )
  if [[ -n "${PARTITIONER:-parmetis}" ]]; then
    BASE_ARGS+=(-petscpartitioner_type "${PARTITIONER:-parmetis}")
  fi
  if [[ "${LOG_VIEW:-1}" == "1" ]]; then
    BASE_ARGS+=(-log_view)
  fi
  if [[ -n "${EXTRA_PETSC_OPTIONS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=($EXTRA_PETSC_OPTIONS)
    BASE_ARGS+=("${EXTRA_ARGS[@]}")
  fi
  {
    printf 'cd %q\n' "$SSR_DIR"
    printf '%q' "$LAUNCHER"
    printf ' %q' "${LAUNCH_ARGS[@]}"
    printf ' '
    printf '%q ' "${BASE_ARGS[@]}"
    printf '\n'
  } | tee "$RESULT_DIR/command.sh"
  cd "$SSR_DIR"
  set +e
  /usr/bin/time -v -o "$time_log" "$LAUNCHER" "${LAUNCH_ARGS[@]}" "${BASE_ARGS[@]}" >"$run_log" 2>&1
  exit_code=$?
  set -e
elif [[ "$ENGINE" == "py" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
  cfg="$RESULT_DIR/case.toml"
  write_py_config "$cfg"
  export PETSC_OPTIONS="${PY_PETSC_OPTIONS:--log_view}"
  BASE_ARGS=(
    "$PYTHON_BIN" -m slope_stability.cli.run_case_from_config
    "$cfg"
    --out_dir "$RESULT_DIR/out"
  )
  {
    printf 'cd %q\n' "$REPO_ROOT"
    printf 'PETSC_OPTIONS=%q ' "$PETSC_OPTIONS"
    printf '%q' "$LAUNCHER"
    printf ' %q' "${LAUNCH_ARGS[@]}"
    printf ' '
    printf '%q ' "${BASE_ARGS[@]}"
    printf '\n'
  } | tee "$RESULT_DIR/command.sh"
  cd "$REPO_ROOT"
  set +e
  /usr/bin/time -v -o "$time_log" "$LAUNCHER" "${LAUNCH_ARGS[@]}" "${BASE_ARGS[@]}" >"$run_log" 2>&1
  exit_code=$?
  set -e
else
  echo "ERROR: unknown ENGINE=$ENGINE, expected c or py." >&2
  exit 2
fi

grep '^RESULT ' "$run_log" | tail -n 1 >"$RESULT_DIR/result_line.txt" || true
grep '^SSR_RESULT ' "$run_log" | tail -n 1 >"$RESULT_DIR/ssr_result_line.txt" || true
grep '^DEFLATION_TIMING ' "$run_log" | tail -n 1 >"$RESULT_DIR/deflation_timing.txt" || true
grep '^PMG_SHELL_APPLY_SUMMARY ' "$run_log" | tail -n 1 >"$RESULT_DIR/pmg_shell_summary.txt" || true
grep -E '^(RESULT|SSR_|DEFLATION_|PMG_|LINEAR_SOLVER_REUSE|BOUNDARY_COUNT)' "$run_log" >"$RESULT_DIR/diagnostics.txt" || true

if command -v sacct >/dev/null 2>&1 && [[ "$JOB_ID" != manual_* ]]; then
  sacct --units=M -j "$JOB_ID" --format=JobID,JobName%48,State,Elapsed,AllocNodes,AllocCPUS,NTasks,MaxRSS,AveRSS,MaxVMSize,ExitCode -P >"$RESULT_DIR/sacct.txt" 2>/dev/null || true
fi

{
  echo "END_TIME=$(date --iso-8601=seconds)"
  echo "EXIT_CODE=$exit_code"
} | tee -a "$metadata"

exit "$exit_code"
