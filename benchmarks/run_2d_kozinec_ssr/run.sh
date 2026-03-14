#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$ROOT/artifacts/cases/run_2d_kozinec_ssr/latest}"
"$ROOT/.venv/bin/python" -m slope_stability.cli.run_case_from_config "$CASE_DIR/case.toml" --out_dir "$OUT_DIR"
