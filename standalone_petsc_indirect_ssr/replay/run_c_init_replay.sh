#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAMPLE_DIR="${1:?usage: run_c_init_replay.sh SAMPLE_DIR [petsc4py|baseline] [all|exported|c|exportedA_cRhs|cA_exportedRhs]}"
PROFILE="${2:-petsc4py}"
MODE="${3:-all}"
RANKS="${RANKS:-32}"
OUT_DIR="${OUT_DIR:-/tmp/c_init_replay_logs}"
mkdir -p "${OUT_DIR}"

case "${PROFILE}" in
  petsc4py) OPTS="${SOLVER_DIR}/options/pmg_shell_petsc4py.opts" ;;
  baseline) OPTS="${SOLVER_DIR}/options/pmg_shell_vcycle.opts" ;;
  *) echo "Unknown profile '${PROFILE}' (use petsc4py or baseline)" >&2; exit 2 ;;
esac

run_one() {
  local matrix_flag="$1"
  local rhs_flag="$2"
  local tag="$3"
  local sample_name
  sample_name="$(basename "${SAMPLE_DIR}")"
  local log="${OUT_DIR}/${sample_name}_${PROFILE}_${tag}.log"
  local -a extra_args=()
  if [[ -n "${EXTRA_PETSC_OPTS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_PETSC_OPTS})
  fi

  (
    cd "${SOLVER_DIR}"
    mpi_pid=""
    cleanup_mpi() {
      local status=$?
      trap - EXIT INT TERM
      if [[ -n "${mpi_pid}" ]] && kill -0 "${mpi_pid}" 2>/dev/null; then
        kill -TERM "${mpi_pid}" 2>/dev/null || true
        sleep 2
        kill -KILL "${mpi_pid}" 2>/dev/null || true
      fi
      if [[ "${status}" -ne 0 ]]; then
        pkill -TERM -f "p4_indirect_ssr .*init_replay_dir ${SAMPLE_DIR}" 2>/dev/null || true
        sleep 1
        pkill -KILL -f "p4_indirect_ssr .*init_replay_dir ${SAMPLE_DIR}" 2>/dev/null || true
      fi
      exit "${status}"
    }
    trap cleanup_mpi EXIT INT TERM
    OMP_NUM_THREADS=1 mpiexec -n "${RANKS}" ./p4_indirect_ssr \
      -options_file "${OPTS}" \
      -deflation_solver matlab_dfgmres \
      -petscpartitioner_type parmetis \
      -init_replay_dir "${SAMPLE_DIR}" \
      -init_replay_use_exported_matrix "${matrix_flag}" \
      -init_replay_use_exported_rhs "${rhs_flag}" \
      -init_replay_use_exported_u true \
      -init_replay_check_damping true \
      -ksp_max_it "${KSP_MAX_IT:-200}" \
      -linear_rtol "${LINEAR_RTOL:-1e-1}" \
      -ksp_converged_reason \
      "${extra_args[@]}" &
    mpi_pid=$!
    set +e
    wait "${mpi_pid}"
    status=$?
    set -e
    mpi_pid=""
    trap - EXIT INT TERM
    exit "${status}"
  ) > "${log}" 2>&1
  echo "Wrote ${log}"
}

case "${MODE}" in
  all)
    run_one true true exportedA_exportedRhs
    run_one true false exportedA_cRhs
    run_one false true cA_exportedRhs
    run_one false false cA_cRhs
    ;;
  exported) run_one true true exportedA_exportedRhs ;;
  c) run_one false false cA_cRhs ;;
  exportedA_cRhs) run_one true false exportedA_cRhs ;;
  cA_exportedRhs) run_one false true cA_exportedRhs ;;
  *) echo "Unknown mode '${MODE}'" >&2; exit 2 ;;
esac
