#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/ssr_linear_replay_state}"
RUN_OUT="${RUN_OUT:-/tmp/petsc4py_p4_l1_replay_out}"
CONFIG="${CONFIG:-/tmp/petsc4py_p4_l1_replay.toml}"
RANKS="${RANKS:-4}"

cat > "${CONFIG}" <<'TOML'
[benchmark]
title = "3D heterogeneous SSR replay export"
matlab_script = "slope_stability_3D_hetero_SSR.m"
comparison_kind = "continuation"
mpi_ranks = 4
suite = false

[problem]
name = "slope_stability_3D_hetero_SSR_replay"
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
omega_max = 6.25e6
init_newton_stopping_criterion = "relative_correction"
init_newton_stopping_tol = 1e-3
step_max = 100

[newton]
it_max = 200
it_damp_max = 10
tol = 1e-4
r_min = 1e-4
stopping_criterion = "absolute_delta_lambda"
stopping_tol = 1e-4

[linear_solver]
solver_type = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"
tolerance = 1e-1
max_iterations = 100
deflation_basis_tolerance = 1e-3
threads = 16
print_level = 0
use_as_preconditioner = true
compiled_outer = false
recycle_preconditioner = true
pc_backend = "pmg_shell"
pc_hypre_coarsen_type = "HMIS"
pc_hypre_interp_type = "ext+i"
petsc_opt = ["pc_hypre_boomeramg_max_iter=4", "pc_hypre_boomeramg_tol=0.0"]

[export]
write_custom_debug_bundle = false
write_history_json = true
write_solution_vtu = false
TOML

rm -rf "${OUT_ROOT}" "${RUN_OUT}"
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
SSP_LINEAR_STATE_EXPORT_DIR="${OUT_ROOT}" \
SSP_LINEAR_STATE_EXPORT_MAX="${EXPORT_MAX:-1}" \
SSP_LINEAR_STATE_EXPORT_NEWTON_ITERS="${EXPORT_NEWTON_ITERS:-1}" \
SSP_LINEAR_STATE_EXPORT_OMEGA_MIN="${EXPORT_OMEGA_MIN:-6.24e6}" \
SSP_LINEAR_STATE_EXPORT_OMEGA_MAX="${EXPORT_OMEGA_MAX:-6.25e6}" \
SSP_LINEAR_STATE_EXPORT_MATRIX="${EXPORT_MATRIX:-1}" \
mpiexec -n "${RANKS}" "${ROOT_DIR}/.venv/bin/python" -m slope_stability.cli.run_case_from_config \
  "${CONFIG}" \
  --out_dir "${RUN_OUT}"

echo "Exported replay samples under ${OUT_ROOT}"
