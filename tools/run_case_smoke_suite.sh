#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
RANKS="${RANKS:-4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
CONTINUATION_STEP_MAX="${CONTINUATION_STEP_MAX:-3}"
LINEAR_RTOL="${LINEAR_RTOL:-1e-1}"
KSP_MAX_IT="${KSP_MAX_IT:-100}"
OUT_ROOT="${OUT_ROOT:-$ROOT/.local/tmp/case_smoke_suite}"
SUMMARY="$OUT_ROOT/summary.tsv"

mkdir -p "$OUT_ROOT"
printf "case\tstatus\tlog\tresult\n" > "$SUMMARY"

mapfile -t CASES < <(find "$ROOT/benchmarks/cases" -maxdepth 2 -name case.toml | sort)

for config in "${CASES[@]}"; do
  case_name="$(basename "$(dirname "$config")")"
  out_dir="$OUT_ROOT/$case_name"
  log="$OUT_ROOT/$case_name.log"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  echo "CASE_SMOKE_START case=$case_name ranks=$RANKS timeout=$TIMEOUT_SECONDS"
  set +e
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    timeout "$TIMEOUT_SECONDS" mpiexec -n "$RANKS" "$PYTHON" -m petsc_ssr.runners.run_case_from_config \
      "$config" \
      --output-dir "$out_dir" \
      --continuation-step-max "$CONTINUATION_STEP_MAX" \
      --linear-rtol "$LINEAR_RTOL" \
      --ksp-max-it "$KSP_MAX_IT" \
      >"$log" 2>&1
  status=$?
  set -e
  result="$(rg 'CASE_RESULT|HYDRO_RESULT|^RESULT ' "$log" | tail -n 1 | tr '\t' ' ' || true)"
  if [[ "$status" -eq 0 ]]; then
    echo "CASE_SMOKE_DONE case=$case_name status=pass"
    printf "%s\tpass\t%s\t%s\n" "$case_name" "$log" "$result" >> "$SUMMARY"
  else
    echo "CASE_SMOKE_DONE case=$case_name status=fail exit=$status"
    printf "%s\tfail:%s\t%s\t%s\n" "$case_name" "$status" "$log" "$result" >> "$SUMMARY"
  fi
done

echo "CASE_SMOKE_SUMMARY path=$SUMMARY"
