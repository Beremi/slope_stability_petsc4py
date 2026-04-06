#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/.build"
PETSC_VERSION="${PETSC_VERSION:-3.24.5}"
PETSC_ARCH="${PETSC_ARCH:-linux-c-opt}"
PETSC_DIR_DEFAULT="${ROOT_DIR}/.build/src/petsc-${PETSC_VERSION}"
VENV_DIR="${ROOT_DIR}/.venv"
PROJECT_EXTRAS_VALUE="${PROJECT_EXTRAS:-test,viz,cython,partition}"
KERNEL_NAME="${DEVCONTAINER_KERNEL_NAME:-slope-stability}"
KERNEL_DISPLAY_NAME="${DEVCONTAINER_KERNEL_DISPLAY_NAME:-Slope Stability (.venv)}"
TOOLCHAIN_STAMP="${BUILD_DIR}/devcontainer-toolchain.sha256"
PROJECT_STAMP="${BUILD_DIR}/devcontainer-project.sha256"

mkdir -p "${BUILD_DIR}"

_sha256_files() {
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi
  sha256sum "$@" | sha256sum | awk '{print $1}'
}

devcontainer_toolchain_signature() {
  _sha256_files \
    "${ROOT_DIR}/bootstrap.sh" \
    "${ROOT_DIR}/build_scripts/bootstrap_petsc4py_venv.sh" \
    "${ROOT_DIR}/build_scripts/build_local_petsc_opt.sh" \
    "${ROOT_DIR}/build_scripts/activate_local_petsc_env.sh"
}

devcontainer_project_signature() {
  mapfile -t tracked < <(
    {
      printf '%s\n' "${ROOT_DIR}/pyproject.toml"
      find "${ROOT_DIR}/src" -type f \( -name '*.pyx' -o -name '*.pxd' -o -name '*.pyi' \) | sort
    } | awk 'NF'
  )
  _sha256_files "${tracked[@]}"
}

devcontainer_activate_env() {
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/build_scripts/activate_local_petsc_env.sh"
  export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/benchmarks${PYTHONPATH:+:${PYTHONPATH}}"
}

devcontainer_env_ready() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    return 1
  fi
  if [[ ! -d "${PETSC_DIR_DEFAULT}/${PETSC_ARCH}" ]]; then
    return 1
  fi
  if [[ ! -f "${PETSC_DIR_DEFAULT}/${PETSC_ARCH}/lib/libpetsc.so" ]]; then
    return 1
  fi
  (
    devcontainer_activate_env
    python - <<'PY' >/dev/null
from petsc4py import PETSc
import slope_stability
import pyvista
import nbclient
import ipykernel
print(PETSc.COMM_WORLD.getSize(), slope_stability.__version__, pyvista.__version__, nbclient.__version__)
PY
  )
}

devcontainer_full_bootstrap() {
  echo "[devcontainer] Building local PETSc toolchain and project environment..."
  (
    cd "${ROOT_DIR}"
    PROJECT_EXTRAS="${PROJECT_EXTRAS_VALUE}" \
    JOBS="${JOBS:-$(nproc)}" \
    ./bootstrap.sh
  )
}

devcontainer_ensure_toolchain() {
  local wanted current=""
  wanted="$(devcontainer_toolchain_signature)"
  if [[ -f "${TOOLCHAIN_STAMP}" ]]; then
    current="$(<"${TOOLCHAIN_STAMP}")"
  fi

  if devcontainer_env_ready; then
    if [[ "${current}" == "${wanted}" ]]; then
      echo "[devcontainer] Reusing prepared PETSc toolchain."
      return 0
    fi
    if [[ -z "${current}" ]]; then
      echo "[devcontainer] Adopting existing PETSc toolchain into devcontainer stamps."
      printf '%s\n' "${wanted}" > "${TOOLCHAIN_STAMP}"
      return 0
    fi
  fi

  devcontainer_full_bootstrap
  printf '%s\n' "${wanted}" > "${TOOLCHAIN_STAMP}"
}

devcontainer_install_project() {
  local target="."
  if [[ -n "${PROJECT_EXTRAS_VALUE}" ]]; then
    target="${target}[${PROJECT_EXTRAS_VALUE}]"
  fi
  (
    devcontainer_activate_env
    python -m pip install --no-cache-dir -e "${target}"
  )
}

devcontainer_ensure_project() {
  local wanted current=""
  wanted="$(devcontainer_project_signature)"
  if [[ -f "${PROJECT_STAMP}" ]]; then
    current="$(<"${PROJECT_STAMP}")"
  fi

  if [[ "${current}" == "${wanted}" ]] && devcontainer_env_ready; then
    echo "[devcontainer] Reusing editable project install."
    return 0
  fi

  echo "[devcontainer] Syncing editable project install into .venv..."
  devcontainer_install_project
  printf '%s\n' "${wanted}" > "${PROJECT_STAMP}"
}

devcontainer_install_kernel() {
  (
    devcontainer_activate_env
    python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${KERNEL_DISPLAY_NAME}" >/dev/null
    ROOT_DIR="${ROOT_DIR}" PETSC_DIR_DEFAULT="${PETSC_DIR_DEFAULT}" PETSC_ARCH="${PETSC_ARCH}" KERNEL_NAME="${KERNEL_NAME}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
petsc_dir = Path(os.environ["PETSC_DIR_DEFAULT"])
petsc_arch = os.environ["PETSC_ARCH"]
kernel_name = os.environ["KERNEL_NAME"]
kernel_path = Path.home() / ".local" / "share" / "jupyter" / "kernels" / kernel_name / "kernel.json"
data = json.loads(kernel_path.read_text(encoding="utf-8"))
ld_parts = []
for part in [str(petsc_dir / petsc_arch / "lib"), *os.environ.get("LD_LIBRARY_PATH", "").split(":")]:
    if part and part not in ld_parts:
        ld_parts.append(part)
data["env"] = {
    **data.get("env", {}),
    "PETSC_DIR": str(petsc_dir),
    "PETSC_ARCH": petsc_arch,
    "LD_LIBRARY_PATH": ":".join(ld_parts),
    "PYTHONPATH": f"{root / 'src'}:{root / 'benchmarks'}",
    "VIRTUAL_ENV": str(root / ".venv"),
}
kernel_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
PY
  )
}
