#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ELASTICITY_DIR="$REPO_ROOT/standalone_petsc_p4_plasticity/p4_elasticity"

cd "$REPO_ROOT"

if [[ "${GIT_UPDATE:-0}" == "1" ]]; then
  git fetch origin
  git pull --ff-only
fi

if [[ -z "${PETSC_DIR:-}" ]]; then
  if [[ -d "$REPO_ROOT/.build/src/petsc-3.24.5" ]]; then
    export PETSC_DIR="$REPO_ROOT/.build/src/petsc-3.24.5"
  else
    echo "ERROR: set PETSC_DIR to the Karolina PETSc 3.24.x build path." >&2
    exit 2
  fi
fi
export PETSC_ARCH="${PETSC_ARCH:-linux-c-opt}"

echo "REPO_ROOT=$REPO_ROOT"
echo "BRANCH=$(git branch --show-current)"
echo "COMMIT=$(git rev-parse HEAD)"
echo "PETSC_DIR=$PETSC_DIR"
echo "PETSC_ARCH=$PETSC_ARCH"
git status --short --branch

if command -v module >/dev/null 2>&1; then
  module list || true
fi

make -C "$ELASTICITY_DIR"

for exe in cube_elasticity l1_elasticity; do
  path="$ELASTICITY_DIR/$exe"
  test -x "$path"
  echo "built: $path"
  ldd "$path" | sed "s/^/  /" || true
done

cat <<EOF

Ready to submit from:
  $SCRIPT_DIR

Example:
  cd "$SCRIPT_DIR"
  PARTITION=qcpu_exp RANKS="16" CASES="l1" VARIANTS="pmg" ./submit_scaling.sh
EOF
