#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
PARTITION="${PARTITION:-qcpu_exp}"
TIME_LIMIT="${TIME_LIMIT:-00:45:00}"
NODE_CORES="${NODE_CORES:-128}"
LAYOUTS="${LAYOUTS:-1:128 2:128}"
ENGINES="${ENGINES:-c}"
PROFILES="${PROFILES:-split}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/ssr_omega7_grid_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"

manifest="$RUN_ROOT/submitted_ssr_omega7_jobs.csv"
echo "job_id,engine,profile,nodes,tasks_per_node,ranks,refine_levels,omega_max,linear_rtol,ksp_max_it,pmg_coarse_max_it,partition,qos,time_limit,run_label" >"$manifest"

echo "RUN_ROOT=$RUN_ROOT"
echo "ACCOUNT=$ACCOUNT QOS=$QOS PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT NODE_CORES=$NODE_CORES"
echo "LAYOUTS=$LAYOUTS"
echo "ENGINES=$ENGINES"
echo "PROFILES=$PROFILES"
echo "REFINE_LEVELS=${REFINE_LEVELS:-1} OMEGA_MAX=${OMEGA_MAX:-7e6} CONTINUATION_STEP_MAX=${CONTINUATION_STEP_MAX:-100} LINEAR_RTOL=${LINEAR_RTOL:-1e-1} KSP_MAX_IT=${KSP_MAX_IT:-200} PMG_COARSE_MAX_IT=${PMG_COARSE_MAX_IT:-5}"

OPENMPI_BIN="${OPENMPI_BIN:-/apps/all/OpenMPI/5.0.8-GCC-14.3.0/bin}"
if ! command -v mpiexec >/dev/null 2>&1 && [[ -x "$OPENMPI_BIN/mpiexec" ]]; then
  export PATH="$OPENMPI_BIN:$PATH"
fi

submit_one() {
  local engine="$1"
  local profile="$2"
  local nodes="$3"
  local tasks_per_node="$4"
  local ranks run_label job_name export_arg job_id launcher srun_extra

  ranks=$(( nodes * tasks_per_node ))
  if (( nodes < 1 || tasks_per_node < 1 || tasks_per_node > NODE_CORES )); then
    echo "ERROR: invalid layout nodes=$nodes tasks_per_node=$tasks_per_node NODE_CORES=$NODE_CORES" >&2
    exit 2
  fi
  case "$engine" in c|py) ;; *) echo "ERROR: unknown engine=$engine" >&2; exit 2 ;; esac
  case "$profile" in baseline|petsc4py|split) ;; *) echo "ERROR: unknown profile=$profile" >&2; exit 2 ;; esac
  if [[ "$engine" != "c" && "$profile" == "split" ]]; then
    echo "ERROR: profile=split is a C-only shell V-cycle profile; use ENGINES=c." >&2
    exit 2
  fi

  run_label="${engine}_${profile}_${nodes}n${tasks_per_node}ppn_r${ranks}_ref${REFINE_LEVELS:-1}_omega${OMEGA_MAX:-7e6}_rtol${LINEAR_RTOL:-1e-1}_p1max${PMG_COARSE_MAX_IT:-5}"
  job_name="ssr_${engine}_${profile}_${nodes}n_${tasks_per_node}ppn"
  launcher="${LAUNCHER:-}"
  srun_extra="${SRUN_EXTRA_ARGS:-}"
  if [[ -z "$launcher" ]]; then
    if (( nodes > 1 )); then
      launcher="srun"
    else
      launcher="mpiexec"
    fi
  fi
  if [[ "$launcher" == "srun" && -z "$srun_extra" ]]; then
    srun_extra="--mpi=pmix_v4"
  fi

  local exports=(
    ALL
    REPO_ROOT="$REPO_ROOT"
    CAMPAIGN_DIR="$RUN_ROOT"
    ACCOUNT="$ACCOUNT"
    QOS="$QOS"
    PARTITION="$PARTITION"
    RANKS="$ranks"
    NODES="$nodes"
    TASKS_PER_NODE="$tasks_per_node"
    RUN_LABEL="$run_label"
    ENGINE="$engine"
    PROFILE="$profile"
    OMEGA_MAX="${OMEGA_MAX:-7e6}"
    REFINE_LEVELS="${REFINE_LEVELS:-1}"
    CONTINUATION_STEP_MAX="${CONTINUATION_STEP_MAX:-100}"
    LINEAR_RTOL="${LINEAR_RTOL:-1e-1}"
    KSP_MAX_IT="${KSP_MAX_IT:-200}"
    PMG_COARSE_MAX_IT="${PMG_COARSE_MAX_IT:-5}"
    PARTITIONER="${PARTITIONER:-parmetis}"
    PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
    PY_PETSC_OPTIONS="${PY_PETSC_OPTIONS:--log_view}"
    BUILD_BEFORE_RUN="${BUILD_BEFORE_RUN:-0}"
    LOG_VIEW="${LOG_VIEW:-1}"
    LAUNCHER="$launcher"
  )
  if [[ -n "${EXTRA_PETSC_OPTIONS:-}" ]]; then
    exports+=(EXTRA_PETSC_OPTIONS="$EXTRA_PETSC_OPTIONS")
  fi
  if [[ -n "${MPIEXEC_EXTRA_ARGS:-}" ]]; then
    exports+=(MPIEXEC_EXTRA_ARGS="$MPIEXEC_EXTRA_ARGS")
  fi
  if [[ -n "$srun_extra" ]]; then
    exports+=(SRUN_EXTRA_ARGS="$srun_extra")
  fi

  export_arg="$(IFS=,; echo "${exports[*]}")"
  local cmd=(
    sbatch
    --parsable
    --account="$ACCOUNT"
    --qos="$QOS"
    --partition="$PARTITION"
    --nodes="$nodes"
    --ntasks="$ranks"
    --ntasks-per-node="$tasks_per_node"
    --cpus-per-task=1
    --time="$TIME_LIMIT"
    --job-name="$job_name"
    --output="$RUN_ROOT/logs/%x-%j.out"
    --error="$RUN_ROOT/logs/%x-%j.err"
    --export="$export_arg"
  )

  if [[ "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
    cmd+=(--test-only)
  fi
  cmd+=("$SCRIPT_DIR/run_case.sbatch")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "DRY_RUN,$engine,$profile,$nodes,$tasks_per_node,$ranks,${REFINE_LEVELS:-1},${OMEGA_MAX:-7e6},${LINEAR_RTOL:-1e-1},${KSP_MAX_IT:-200},${PMG_COARSE_MAX_IT:-5},$PARTITION,$QOS,$TIME_LIMIT,$run_label" >>"$manifest"
    return 0
  fi

  job_id="$("${cmd[@]}")"
  echo "$job_id,$engine,$profile,$nodes,$tasks_per_node,$ranks,${REFINE_LEVELS:-1},${OMEGA_MAX:-7e6},${LINEAR_RTOL:-1e-1},${KSP_MAX_IT:-200},${PMG_COARSE_MAX_IT:-5},$PARTITION,$QOS,$TIME_LIMIT,$run_label" >>"$manifest"
  echo "submitted job_id=$job_id engine=$engine profile=$profile nodes=$nodes tasks_per_node=$tasks_per_node ranks=$ranks"
}

for layout in $LAYOUTS; do
  IFS=: read -r nodes tasks_per_node extra <<<"$layout"
  if [[ -n "${extra:-}" || -z "${nodes:-}" || -z "${tasks_per_node:-}" ]]; then
    echo "ERROR: LAYOUTS entries must have form nodes:tasks_per_node, got '$layout'" >&2
    exit 2
  fi
  for profile in $PROFILES; do
    for engine in $ENGINES; do
      submit_one "$engine" "$profile" "$nodes" "$tasks_per_node"
    done
  done
done

echo "Manifest: $manifest"
echo "Collect after completion with:"
echo "  $SCRIPT_DIR/collect_omega7_results.py $RUN_ROOT"
