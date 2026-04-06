#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_PATH="${HOME}/.local/share/jupyter/kernels/slope-stability/kernel.json"

cat <<EOF
[devcontainer] Workspace ready.
- interpreter: ${ROOT_DIR}/.venv/bin/python
- kernel: ${KERNEL_PATH}
- activate shell env: source ${ROOT_DIR}/build_scripts/activate_local_petsc_env.sh
EOF
