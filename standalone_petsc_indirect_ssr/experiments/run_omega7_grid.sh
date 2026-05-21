#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${SOLVER_DIR}/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/ssr_omega7_grid_$(date +%Y%m%d_%H%M%S)}"
RANKS_LIST="${RANKS_LIST:-16 32}"
ENGINES="${ENGINES:-c py}"
PROFILES="${PROFILES:-baseline petsc4py}"
OMEGA_MAX="${OMEGA_MAX:-7e6}"
STEP_MAX="${STEP_MAX:-100}"
LINEAR_RTOL="${LINEAR_RTOL:-1e-1}"
KSP_MAX_IT="${KSP_MAX_IT:-200}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
PETSC_DIR="${PETSC_DIR:-${ROOT_DIR}/.build/src/petsc-3.24.5}"
PETSC_ARCH="${PETSC_ARCH:-linux-c-opt}"

export PETSC_DIR PETSC_ARCH
mkdir -p "${OUT_ROOT}"

make -C "${SOLVER_DIR}"

write_py_config() {
  local cfg="$1"
  local profile="$2"
  local ranks="$3"
  local petsc_opt

  case "${profile}" in
    petsc4py)
      petsc_opt='"pc_hypre_boomeramg_max_iter=4", "pc_hypre_boomeramg_tol=0.0"'
      ;;
    baseline)
      petsc_opt='"mg_levels_ksp_type=chebyshev", "mg_levels_ksp_max_it=2", "mg_levels_pc_type=jacobi", "mg_coarse_ksp_type=fgmres", "mg_coarse_rtol=1e-3", "mg_coarse_max_it=100", "mg_coarse_pc_type=gamg", "pc_gamg_aggressive_square_graph=false"'
      ;;
    *)
      echo "Unknown profile '${profile}'" >&2
      return 2
      ;;
  esac

  cat > "${cfg}" <<TOML
[benchmark]
title = "3D heterogeneous SSR omega7 ${profile}"
matlab_script = "slope_stability_3D_hetero_SSR.m"
comparison_kind = "continuation"
mpi_ranks = ${ranks}
suite = false

[problem]
name = "slope_stability_3D_hetero_SSR_omega7_${profile}"
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
omega_max = ${OMEGA_MAX}
init_newton_stopping_criterion = "relative_correction"
init_newton_stopping_tol = 1e-3
step_max = ${STEP_MAX}

[newton]
it_max = 200
it_damp_max = 10
tol = 1e-4
r_min = 1e-4
stopping_criterion = "absolute_delta_lambda"
stopping_tol = 1e-4

[linear_solver]
solver_type = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"
tolerance = ${LINEAR_RTOL}
max_iterations = ${KSP_MAX_IT}
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

run_case() {
  local engine="$1"
  local profile="$2"
  local ranks="$3"
  local case_dir="${OUT_ROOT}/${engine}_${profile}_r${ranks}"
  local log="${case_dir}/run.log"
  mkdir -p "${case_dir}"
  echo "RUN_START engine=${engine} profile=${profile} ranks=${ranks} out=${case_dir}" | tee "${case_dir}/status.txt"

  if [[ "${engine}" == "c" ]]; then
    local opts="${SOLVER_DIR}/options/pmg_shell_vcycle.opts"
    if [[ "${profile}" == "petsc4py" ]]; then
      opts="${SOLVER_DIR}/options/pmg_shell_petsc4py.opts"
    fi
    (
      cd "${SOLVER_DIR}"
      time mpiexec -n "${ranks}" ./p4_indirect_ssr \
        -options_file "${opts}" \
        -petscpartitioner_type parmetis \
        -omega_max "${OMEGA_MAX}" \
        -continuation_step_max "${STEP_MAX}" \
        -linear_rtol "${LINEAR_RTOL}" \
        -ksp_max_it "${KSP_MAX_IT}" \
        -curve_csv "${case_dir}/continuation_curve.csv" \
        -ksp_converged_reason
    ) > "${log}" 2>&1
  elif [[ "${engine}" == "py" ]]; then
    local cfg="${case_dir}/case.toml"
    write_py_config "${cfg}" "${profile}" "${ranks}"
    time mpiexec -n "${ranks}" "${PYTHON_BIN}" -m slope_stability.cli.run_case_from_config \
      "${cfg}" \
      --out_dir "${case_dir}/out" \
      > "${log}" 2>&1
  else
    echo "Unknown engine '${engine}'" >&2
    return 2
  fi

  echo "RUN_DONE engine=${engine} profile=${profile} ranks=${ranks} out=${case_dir}" | tee -a "${case_dir}/status.txt"
}

for ranks in ${RANKS_LIST}; do
  for profile in ${PROFILES}; do
    for engine in ${ENGINES}; do
      run_case "${engine}" "${profile}" "${ranks}"
    done
  done
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/collect_omega7_grid.py" "${OUT_ROOT}" | tee "${OUT_ROOT}/summary.txt"
echo "OUT_ROOT=${OUT_ROOT}"
