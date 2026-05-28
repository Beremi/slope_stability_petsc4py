#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="${1:-$(pwd)}"
shift || true

CASE_DIR="$(cd "$CASE_DIR" && pwd)"
CASE_TOML="${CASE_TOML:-$CASE_DIR/case.toml}"
OUT_DIR="${OUT_DIR:-$CASE_DIR/artifacts/latest}"
MPI_RANKS="${MPI_RANKS:-${RANKS:-1}}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MPIEXEC="${MPIEXEC:-mpiexec}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$MPIEXEC" -n "$MPI_RANKS" "$PYTHON" \
  -m petsc_ssr.cli.main run \
  "$CASE_TOML" \
  --output "$OUT_DIR" \
  "$@"
