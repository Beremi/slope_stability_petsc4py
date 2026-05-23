#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/ssr_init_replay_state}"
RUN_OUT="${RUN_OUT:-/tmp/petsc4py_p4_l1_init_replay_out}"
CONFIG="${CONFIG:-/tmp/petsc4py_p4_l1_init_replay.toml}"
RANKS="${RANKS:-32}"
PETSC_OPT_ITEMS='"pc_hypre_boomeramg_max_iter=4", "pc_hypre_boomeramg_tol=0.0"'
if [[ -n "${EXTRA_PETSC_OPTS_CSV:-}" ]]; then
  IFS=',' read -r -a _extra_petsc_opts <<< "${EXTRA_PETSC_OPTS_CSV}"
  for _opt in "${_extra_petsc_opts[@]}"; do
    _opt="${_opt#"${_opt%%[![:space:]]*}"}"
    _opt="${_opt%"${_opt##*[![:space:]]}"}"
    [[ -z "${_opt}" ]] && continue
    _opt="${_opt//\\/\\\\}"
    _opt="${_opt//\"/\\\"}"
    PETSC_OPT_ITEMS+=", \"${_opt}\""
  done
fi

cat > "${CONFIG}" <<TOML
[benchmark]
title = "3D heterogeneous SSR fixed-lambda init replay export"
matlab_script = "slope_stability_3D_hetero_SSR.m"
comparison_kind = "continuation"
mpi_ranks = ${RANKS}
suite = false

[problem]
name = "slope_stability_3D_hetero_SSR_init_replay"
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
step_max = ${STEP_MAX:-2}

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
petsc_opt = [${PETSC_OPT_ITEMS}]

[export]
write_custom_debug_bundle = false
write_history_json = true
write_solution_vtu = false
TOML

rm -rf "${OUT_ROOT}" "${RUN_OUT}"

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
    pkill -TERM -f "run_case_from_config ${CONFIG} --out_dir ${RUN_OUT}" 2>/dev/null || true
    sleep 1
    pkill -KILL -f "run_case_from_config ${CONFIG} --out_dir ${RUN_OUT}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup_mpi EXIT INT TERM

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
SSP_INIT_REPLAY_EXPORT_DIR="${OUT_ROOT}" \
SSP_INIT_REPLAY_EXPORT_LAMBDAS="${EXPORT_LAMBDAS:-1.0,1.1}" \
SSP_INIT_REPLAY_EXPORT_NEWTON_ITERS="${EXPORT_NEWTON_ITERS:-all}" \
SSP_INIT_REPLAY_EXPORT_MAX="${EXPORT_MAX:-0}" \
SSP_INIT_REPLAY_EXPORT_MATRIX="${EXPORT_MATRIX:-1}" \
SSP_INIT_REPLAY_EXPORT_DAMPING="${EXPORT_DAMPING:-1}" \
SSP_INIT_REPLAY_EXPORT_PROBES="${EXPORT_PROBES:-0}" \
SSP_INIT_REPLAY_EXPORT_PROBE_FORMAT="${EXPORT_PROBE_FORMAT:-raw}" \
mpiexec -n "${RANKS}" "${ROOT_DIR}/.venv/bin/python" -m slope_stability.cli.run_case_from_config \
  "${CONFIG}" \
  --out_dir "${RUN_OUT}" &
mpi_pid=$!
set +e
wait "${mpi_pid}"
status=$?
set -e
mpi_pid=""
trap - EXIT INT TERM
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

echo "Exported fixed-lambda init replay samples under ${OUT_ROOT}"
