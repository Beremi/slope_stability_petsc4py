#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${1:-$ROOT/artifacts/experiments/pmg_gasm_3d_hetero_ssr_p4_l1/latest}"
MPIEXEC="${MPIEXEC:-mpiexec}"
RANKS="${RANKS:-4}"
FAKE_SOCKETS="${FAKE_SOCKETS:-2}"
RUN_BASELINE="${RUN_BASELINE:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMPI_MCA_rmaps_base_oversubscribe="${OMPI_MCA_rmaps_base_oversubscribe:-1}"
export OMPI_MCA_mpi_yield_when_idle="${OMPI_MCA_mpi_yield_when_idle:-1}"

if (( RANKS <= 0 )); then
  echo "RANKS must be positive, got $RANKS" >&2
  exit 2
fi
if (( FAKE_SOCKETS <= 0 )); then
  echo "FAKE_SOCKETS must be positive, got $FAKE_SOCKETS" >&2
  exit 2
fi
if (( RANKS % FAKE_SOCKETS != 0 )); then
  echo "RANKS must be divisible by FAKE_SOCKETS for contiguous fake socket grouping ($RANKS vs $FAKE_SOCKETS)" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
GASM_CONFIG="$OUT_ROOT/gasm_fake_${FAKE_SOCKETS}x$((RANKS / FAKE_SOCKETS)).toml"

"$ROOT/.venv/bin/python" - <<PY
from pathlib import Path

src = Path("$CASE_DIR/gasm.toml")
dst = Path("$GASM_CONFIG")
text = src.read_text(encoding="utf-8")
text = text.replace(
    "pmg_smoother_gasm_total_subdomains = 2",
    "pmg_smoother_gasm_total_subdomains = $FAKE_SOCKETS",
)
dst.write_text(text, encoding="utf-8")
PY

if [[ "$RUN_BASELINE" != "0" ]]; then
  "$MPIEXEC" -n "$RANKS" "$ROOT/.venv/bin/python" -m slope_stability.cli.run_case_from_config \
    "$CASE_DIR/baseline.toml" \
    --out_dir "$OUT_ROOT/baseline"
fi

"$MPIEXEC" -n "$RANKS" "$ROOT/.venv/bin/python" -m slope_stability.cli.run_case_from_config \
  "$GASM_CONFIG" \
  --out_dir "$OUT_ROOT/gasm_fake_${FAKE_SOCKETS}x$((RANKS / FAKE_SOCKETS))"

echo "Wrote runs under $OUT_ROOT"
