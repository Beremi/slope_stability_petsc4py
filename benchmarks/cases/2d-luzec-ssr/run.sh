#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$CASE_DIR/../../tools/run_standalone_case.sh" "$CASE_DIR" "$@"
