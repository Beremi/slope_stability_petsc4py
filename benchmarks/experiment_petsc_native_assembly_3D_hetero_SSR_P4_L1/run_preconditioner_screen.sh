#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/cases/experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1/latest}"
RANKS="${RANKS:-32}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cases=(
  owned_csr_pmg_shell_32
  petsc_aij_pmg_shell_32
  petsc_aij_pmg_shell_redundant_coarse_32
  petsc_aij_hypre_32
  petsc_aij_hypre_lagged_pmis_32
  petsc_aij_gamg_32
  petsc_aij_gamg_lagged_lowcomm_32
  petsc_aij_bddc_32
  petsc_aij_bddc_gamg_32
)

if [[ "$#" -gt 0 ]]; then
  cases=("$@")
fi

mkdir -p "$OUT_ROOT"
for name in "${cases[@]}"; do
  case_path="$CASE_DIR/${name}.toml"
  out_dir="$OUT_ROOT/$name"
  echo "[petsc-native-screen] ranks=$RANKS case=$name out=$out_dir"
  mpiexec -n "$RANKS" "$ROOT/.venv/bin/python" -m slope_stability.cli.run_case_from_config "$case_path" --out_dir "$out_dir"
done
