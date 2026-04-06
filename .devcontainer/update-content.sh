#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

devcontainer_ensure_toolchain
devcontainer_ensure_project
devcontainer_install_kernel
bash "${ROOT_DIR}/.devcontainer/validate.sh" --imports-only

echo "[devcontainer] updateContent completed."
