#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP_FILE="${ROOT_DIR}/.build/devcontainer-ready"

mkdir -p "${ROOT_DIR}/.build"

if [[ ! -f "${STAMP_FILE}" ]]; then
  echo "[devcontainer] Bootstrapping PETSc + Python environment..."
  (
    cd "${ROOT_DIR}"
    PROJECT_EXTRAS="test,viz,cython,partition" \
    JOBS="${JOBS:-$(nproc)}" \
    ./bootstrap.sh
  )
  touch "${STAMP_FILE}"
else
  echo "[devcontainer] Reusing existing bootstrap at ${STAMP_FILE}"
fi

source "${ROOT_DIR}/build_scripts/activate_local_petsc_env.sh"
python -m ipykernel install --user --name slope-stability --display-name "Slope Stability (.venv)"
bash "${ROOT_DIR}/.devcontainer/validate.sh" --imports-only

cat <<'EOF'
[devcontainer] Ready.
Activate manually with:
  source build_scripts/activate_local_petsc_env.sh
EOF
