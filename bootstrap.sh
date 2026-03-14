#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/.build"

ENV_DIR="${ENV_DIR:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_MODE="${BOOTSTRAP_MODE:-local}"

if [[ "$BOOTSTRAP_MODE" == "wheel" ]]; then
  ENV_DIR="$ENV_DIR" PYTHON_BIN="$PYTHON_BIN" "$ROOT/build_scripts/bootstrap_petsc4py_venv.sh"
else
  ENV_DIR="$ENV_DIR" PYTHON_BIN="$PYTHON_BIN" SKIP_PETSC4PY_INSTALL=1 SKIP_PROJECT_INSTALL=1 \
    "$ROOT/build_scripts/bootstrap_petsc4py_venv.sh"
  VENV_DIR="$ENV_DIR" "$ROOT/build_scripts/build_local_petsc_opt.sh"
fi
