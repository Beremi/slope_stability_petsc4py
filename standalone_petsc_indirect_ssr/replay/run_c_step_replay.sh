#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAMPLE_DIR="${1:?usage: run_c_step_replay.sh SAMPLE_DIR [petsc4py|baseline]}"
PROFILE="${2:-petsc4py}"
RANKS="${RANKS:-32}"
LOG="${LOG:-/tmp/c_step_replay_${PROFILE}.log}"
SUMMARY_CSV="${SUMMARY_CSV:-${LOG%.log}.csv}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

case "${PROFILE}" in
  petsc4py)
    OPTS="${SOLVER_DIR}/options/pmg_shell_petsc4py.opts"
    ;;
  baseline)
    OPTS="${SOLVER_DIR}/options/pmg_shell_vcycle.opts"
    ;;
  *)
    echo "Unknown profile '${PROFILE}' (use petsc4py or baseline)" >&2
    exit 2
    ;;
esac

cd "${SOLVER_DIR}"
read -r -a EXTRA_ARGS <<< "${EXTRA_OPTS}"
OMP_NUM_THREADS=1 mpiexec -n "${RANKS}" ./p4_indirect_ssr \
  -options_file "${OPTS}" \
  -deflation_solver matlab_dfgmres \
  -deflation_monitor true \
  -deflation_intra_newton_recycle false \
  -deflation_krylov_persistent false \
  -indirect_newton_pair_freeze_matrix false \
  -petscpartitioner_type parmetis \
  -step_replay_dir "${SAMPLE_DIR}" \
  -ksp_max_it "${KSP_MAX_IT:-200}" \
  -linear_rtol "${LINEAR_RTOL:-1e-1}" \
  -ksp_converged_reason \
  "${EXTRA_ARGS[@]}" \
  > "${LOG}" 2>&1

echo "Wrote ${LOG}"
"${SCRIPT_DIR}/collect_step_replay.py" \
  --profile "${PROFILE}" \
  --csv-out "${SUMMARY_CSV}" \
  "${SAMPLE_DIR}" "${LOG}" \
  > "${SUMMARY_CSV%.csv}.txt"
echo "Wrote ${SUMMARY_CSV}"
