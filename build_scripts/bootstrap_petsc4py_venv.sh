#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
ENV_DIR=${ENV_DIR:-$(pwd)/.venv}
PETSC4PY_INDEX=${PETSC4PY_INDEX:-https://pypi.org/simple}
SKIP_PETSC4PY_INSTALL=${SKIP_PETSC4PY_INSTALL:-0}
SKIP_PROJECT_INSTALL=${SKIP_PROJECT_INSTALL:-0}

if [[ ! -d "${ENV_DIR}" ]]; then
  "$PYTHON_BIN" -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --upgrade packaging cython numpy

if [[ "${SKIP_PETSC4PY_INSTALL}" != "1" ]]; then
  # Prefer pip wheels; on systems with no binary support, pip may build PETSc locally.
  PETSC4PY_FORCE_PURE=0 python -m pip install --no-cache-dir -i "${PETSC4PY_INDEX}" petsc4py
fi

if [[ "${SKIP_PROJECT_INSTALL}" != "1" ]]; then
  pip install --no-cache-dir -e .[test,cython,partition]
fi

cat <<MSG
Python environment prepared at ${ENV_DIR}.
Activate with:
  source .venv/bin/activate
Quick check:
  python - <<'PY'
  from petsc4py import PETSc
  import slope_stability
  print('PETSc', PETSc.COMM_WORLD.Get_size())
  print('Package', slope_stability.__version__)
  PY
MSG
