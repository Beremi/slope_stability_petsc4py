#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT_DIR/.venv/bin/activate"
export PETSC_DIR="${PETSC_DIR:-$ROOT_DIR/.build/src/petsc-3.24.5}"
export PETSC_ARCH="${PETSC_ARCH:-linux-c-opt}"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"
