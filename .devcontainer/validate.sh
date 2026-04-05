#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMPORTS_ONLY=0
SMOKE_CASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --imports-only)
      IMPORTS_ONLY=1
      shift
      ;;
    --smoke-case)
      SMOKE_CASE="${2:?missing benchmark name after --smoke-case}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "${ROOT_DIR}"
source "${ROOT_DIR}/build_scripts/activate_local_petsc_env.sh"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "$(id -u)" == "0" ]]; then
  export OMPI_ALLOW_RUN_AS_ROOT=1
  export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

python - <<'PY'
from petsc4py import PETSc
import pyvista as pv
import nbclient
import notebook_support as nb
import slope_stability

status = nb.viz_support_status()
print("PETSc ranks:", PETSc.COMM_WORLD.getSize())
print("PyVista version:", pv.__version__)
print("nbclient version:", nbclient.__version__)
print("Package version:", slope_stability.__version__)
print("Viz status:", status)
assert status["pyvista"] and status["ipywidgets"] and status["trame"]
PY

if [[ "${IMPORTS_ONLY}" == "1" ]]; then
  exit 0
fi

if [[ -n "${SMOKE_CASE}" ]]; then
  python benchmarks/prepare_committed_notebook_artifacts.py --cases "${SMOKE_CASE}" --execution-profile smoke
  python benchmarks/execute_visualisation_notebooks.py --cases "${SMOKE_CASE}" --jupyter-backend static --surface-subdivision 0 --surface-decimate-reduction 0.0
fi
