#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$ROOT/artifacts/cases/slope_stability_2D_Kozinec_LL/latest}"
"$ROOT/.venv/bin/python" -m slope_stability.cli.run_case_from_config "$CASE_DIR/case.toml" --out_dir "$OUT_DIR"
