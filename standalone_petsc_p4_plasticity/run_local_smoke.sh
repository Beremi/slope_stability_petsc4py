#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

: "${PETSC_DIR:=$(cd .. && pwd)/.build/src/petsc-3.24.5}"
: "${PETSC_ARCH:=linux-c-opt}"
export PETSC_DIR PETSC_ARCH

make

run_case() {
  local ranks="$1"
  shift
  echo
  echo "== mpiexec -n ${ranks} ./p4_plasticity $* =="
  mpiexec -n "${ranks}" ./p4_plasticity "$@"
}

run_case 1 -mesh data/tiny_box.msh -pc_variant gamg -ksp_type preonly -pc_type lu -refine_levels 0 -newton_max_it 2
run_case 1 -mesh data/tiny_box.msh -pc_variant bddc -refine_levels 0 -newton_max_it 1
run_case 1 -mesh data/tiny_box.msh -pc_variant fetidp -refine_levels 0 -newton_max_it 1
