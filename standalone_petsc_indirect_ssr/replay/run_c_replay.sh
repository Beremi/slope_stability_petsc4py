#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAMPLE_DIR="${1:?usage: run_c_replay.sh SAMPLE_DIR [petsc4py|baseline]}"
PROFILE="${2:-petsc4py}"
RANKS="${RANKS:-4}"
LOG="${LOG:-/tmp/c_linear_replay_${PROFILE}.log}"

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
OMP_NUM_THREADS=1 mpiexec -n "${RANKS}" ./p4_indirect_ssr \
  -options_file "${OPTS}" \
  -deflation_solver matlab_dfgmres \
  -deflation_monitor true \
  -petscpartitioner_type parmetis \
  -linear_replay_dir "${SAMPLE_DIR}" \
  -linear_replay_use_exported_rhs true \
  -ksp_max_it "${KSP_MAX_IT:-20}" \
  -linear_rtol "${LINEAR_RTOL:-1e-1}" \
  -ksp_converged_reason \
  > "${LOG}" 2>&1

echo "Wrote ${LOG}"
