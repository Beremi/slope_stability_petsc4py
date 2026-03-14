#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
"$ROOT/.venv/bin/python" -m slope_stability.cli.run_benchmark_case "$ROOT/benchmarks/run_3D_hetero_seepage_capture/case.toml"
