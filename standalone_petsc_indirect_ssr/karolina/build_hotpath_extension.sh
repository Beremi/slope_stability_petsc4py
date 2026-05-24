#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

export PETSC_ARCH="${PETSC_ARCH:-linux-c-opt}"
if [[ -z "${PETSC_DIR:-}" ]]; then
  if [[ -d "$REPO_ROOT/.build/src/petsc-3.24.5" ]]; then
    export PETSC_DIR="$REPO_ROOT/.build/src/petsc-3.24.5"
  else
    echo "ERROR: PETSC_DIR is not set and $REPO_ROOT/.build/src/petsc-3.24.5 does not exist." >&2
    exit 2
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

KAROLINA_BZIP2_ROOT="${KAROLINA_BZIP2_ROOT:-/apps/all/bzip2/1.0.8-GCCcore-14.3.0}"
for libdir in "$KAROLINA_BZIP2_ROOT/lib64" "$KAROLINA_BZIP2_ROOT/lib"; do
  if [[ -d "$libdir" ]]; then
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
  fi
done

cd "$REPO_ROOT"
"$PYTHON_BIN" -m pip install -e .
"$PYTHON_BIN" - <<'PY'
import slope_stability._petsc_ssr as ssr
print("slope_stability._petsc_ssr import ok", ssr.__name__)
PY
