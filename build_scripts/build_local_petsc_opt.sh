#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR=${VENV_DIR:-"$ROOT_DIR/.venv"}
PETSC_VERSION=${PETSC_VERSION:-3.24.5}
PETSC_GIT_TAG=${PETSC_GIT_TAG:-"v${PETSC_VERSION}"}
PETSC_REPO=${PETSC_REPO:-https://gitlab.com/petsc/petsc.git}
PETSC_SRC_DIR=${PETSC_SRC_DIR:-"$ROOT_DIR/.build/src/petsc-${PETSC_VERSION}"}
PETSC_ARCH=${PETSC_ARCH:-linux-c-opt}
JOBS=${JOBS:-24}
COPTFLAGS=${COPTFLAGS:-"-O3 -march=native -mtune=native"}
CXXOPTFLAGS=${CXXOPTFLAGS:-"-O3 -march=native -mtune=native"}
FOPTFLAGS=${FOPTFLAGS:-"-O3 -march=native -mtune=native"}
INSTALL_PROJECT=${INSTALL_PROJECT:-1}

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing virtualenv: $VENV_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$PETSC_SRC_DIR")"

if [[ ! -d "$PETSC_SRC_DIR/.git" ]]; then
  git clone --depth 1 --branch "$PETSC_GIT_TAG" "$PETSC_REPO" "$PETSC_SRC_DIR"
else
  git -C "$PETSC_SRC_DIR" fetch --tags origin
  git -C "$PETSC_SRC_DIR" checkout "$PETSC_GIT_TAG"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging cython numpy
python -m pip uninstall -y petsc petsc4py || true

cd "$PETSC_SRC_DIR"

# Rebuild the selected PETSC_ARCH from a clean state so package/config changes
# do not inherit stale artifacts from an earlier failed external package build.
rm -rf "$PETSC_ARCH"

CONFIGURE_ARGS=(
  "PETSC_ARCH=${PETSC_ARCH}"
  "--with-debugging=0"
  "--with-shared-libraries=1"
  "--with-mpi=1"
  "--with-x=0"
  "--with-fortran-bindings=0"
  "CC=mpicc"
  "CXX=mpicxx"
  "FC=mpif90"
  "COPTFLAGS=${COPTFLAGS}"
  "CXXOPTFLAGS=${CXXOPTFLAGS}"
  "FOPTFLAGS=${FOPTFLAGS}"
  "--download-fblaslapack"
  "--download-metis"
  "--download-parmetis"
  "--download-ptscotch"
  "--download-scalapack"
  "--download-mumps"
  "--download-superlu"
  "--download-superlu_dist"
  "--download-hypre"
)

./configure "${CONFIGURE_ARGS[@]}"
make PETSC_DIR="$PETSC_SRC_DIR" PETSC_ARCH="$PETSC_ARCH" all -j"$JOBS"

export PETSC_DIR="$PETSC_SRC_DIR"
export PETSC_ARCH
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"

python -m pip install --no-build-isolation --no-deps --force-reinstall "$PETSC_SRC_DIR/src/binding/petsc4py"

if [[ "$INSTALL_PROJECT" == "1" ]]; then
  (
    cd "$ROOT_DIR"
    python -m pip install --no-cache-dir -e .[test,cython,partition]
  )
fi

cat <<EOF
Local PETSc build completed.
PETSC_DIR=$PETSC_DIR
PETSC_ARCH=$PETSC_ARCH
Activate with:
  source "$ROOT_DIR/build_scripts/activate_local_petsc_env.sh"
EOF
