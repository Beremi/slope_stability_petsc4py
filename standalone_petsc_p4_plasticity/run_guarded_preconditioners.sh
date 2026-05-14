#!/usr/bin/env bash
set -uo pipefail

MEM_LIMIT_GB="${MEM_LIMIT_GB:-120}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.2}"
TIME_LIMIT_SEC="${TIME_LIMIT_SEC:-0}"
MPIEXEC="${MPIEXEC:-mpiexec}"
RUN_LABEL="${RUN_LABEL:-}"
OUTDIR="${OUTDIR:-/tmp/standalone_petsc_p4_plasticity_$(date +%Y%m%d_%H%M%S)}"
MESH="${MESH:-data/adaptive_family_a_l1.msh}"
RANKS="${RANKS:-16 32}"
VARIANTS="${VARIANTS:-gamg pmg bddc fetidp none}"
PARTITIONER="${PARTITIONER:-}"
LINEAR_RTOL="${LINEAR_RTOL:-1e-3}"
KSP_MAX_IT="${KSP_MAX_IT:-200}"
NEWTON_MAX_IT="${NEWTON_MAX_IT:-20}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

mkdir -p "$OUTDIR"

limit_kb=$(awk -v gb="$MEM_LIMIT_GB" 'BEGIN { printf "%.0f", gb * 1024 * 1024 }')
csv="$OUTDIR/summary.csv"
if [[ ! -f "$csv" ]]; then
  printf 'ranks,variant,partitioner,status,exit_code,peak_memory_gb,elastic_its,newton_its,newton_linear_its,total_linear_its,elastic_assembly_time,elastic_solve_time,newton_assembly_time,newton_solve_time,wall_time,global_dofs,log\n' > "$csv"
fi

sample_rss_kb() {
  local sid="$1"
  ps -o rss= -s "$sid" 2>/dev/null | awk '{s += $1} END {print s + 0}'
}

kill_session() {
  local sid="$1"
  kill -TERM -- "-$sid" 2>/dev/null || true
  pkill -TERM -s "$sid" 2>/dev/null || true
  sleep 3
  kill -KILL -- "-$sid" 2>/dev/null || true
  pkill -KILL -s "$sid" 2>/dev/null || true
}

field_from_result() {
  local result="$1"
  local key="$2"
  awk -v k="$key" '{
    for (i = 1; i <= NF; ++i) {
      split($i, a, "=")
      if (a[1] == k) {
        print a[2]
        exit
      }
    }
  }' <<< "$result"
}

run_guarded() {
  local ranks="$1"
  local variant="$2"
  local log="$OUTDIR/${variant}_${ranks}.log"
  if [[ -n "$RUN_LABEL" ]]; then log="$OUTDIR/${variant}_${ranks}_${RUN_LABEL}.log"; fi
  local peak_kb=0
  local status="failed"
  local guard_status=""
  local exit_code=0
  local start_sec
  start_sec=$(date +%s)
  local partitioner_opts=()
  if [[ -n "$PARTITIONER" ]]; then
    partitioner_opts=(-petscpartitioner_type "$PARTITIONER")
  fi

  printf '\n== ranks=%s variant=%s ==\n' "$ranks" "$variant"
  setsid env OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    $MPIEXEC -n "$ranks" ./p4_plasticity \
    -mesh "$MESH" \
    "${partitioner_opts[@]}" \
    -pc_variant "$variant" \
    -linear_rtol "$LINEAR_RTOL" \
    -ksp_max_it "$KSP_MAX_IT" \
    -newton_max_it "$NEWTON_MAX_IT" \
    -ksp_converged_reason \
    $EXTRA_OPTS >"$log" 2>&1 &
  local pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    local rss
    rss=$(sample_rss_kb "$pid")
    if (( rss > peak_kb )); then peak_kb="$rss"; fi
    if (( rss > limit_kb )); then
      printf 'Memory guard tripped: %.3f GiB > %.3f GiB\n' "$(awk -v kb="$rss" 'BEGIN {print kb/1024/1024}')" "$MEM_LIMIT_GB" | tee -a "$log"
      guard_status="memory_guard"
      kill_session "$pid"
      break
    fi
    if (( TIME_LIMIT_SEC > 0 && $(date +%s) - start_sec > TIME_LIMIT_SEC )); then
      printf 'Time guard tripped: %s sec > %s sec\n' "$(( $(date +%s) - start_sec ))" "$TIME_LIMIT_SEC" | tee -a "$log"
      guard_status="time_guard"
      kill_session "$pid"
      break
    fi
    sleep "$SAMPLE_INTERVAL"
  done
  wait "$pid"
  exit_code=$?

  local result
  result=$(grep '^RESULT ' "$log" | tail -1 || true)
  if [[ "$exit_code" -eq 0 && -n "$result" ]]; then status="pass"; fi
  if [[ -n "$guard_status" ]]; then status="$guard_status"; fi
  if grep -q 'Memory guard tripped' "$log"; then status="memory_guard"; fi
  if grep -q 'Time guard tripped' "$log"; then status="time_guard"; fi

  local peak_gb
  peak_gb=$(awk -v kb="$peak_kb" 'BEGIN {printf "%.3f", kb / 1024 / 1024}')
  local partitioner elastic_its newton_its newton_linear_its total_linear_its elastic_assembly_time elastic_solve_time newton_assembly_time newton_solve_time wall_time global_dofs
  partitioner=$(field_from_result "$result" partitioner)
  if [[ -z "$partitioner" ]]; then partitioner="${PARTITIONER:-auto}"; fi
  elastic_its=$(field_from_result "$result" elastic_its)
  newton_its=$(field_from_result "$result" newton_its)
  newton_linear_its=$(field_from_result "$result" newton_linear_its)
  total_linear_its=$(field_from_result "$result" total_linear_its)
  elastic_assembly_time=$(field_from_result "$result" elastic_assembly_time)
  elastic_solve_time=$(field_from_result "$result" elastic_solve_time)
  newton_assembly_time=$(field_from_result "$result" newton_assembly_time)
  newton_solve_time=$(field_from_result "$result" newton_solve_time)
  wall_time=$(field_from_result "$result" wall_time)
  global_dofs=$(field_from_result "$result" global_dofs)

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$ranks" "$variant" "$partitioner" "$status" "$exit_code" "$peak_gb" \
    "$elastic_its" "$newton_its" "$newton_linear_its" "$total_linear_its" \
    "$elastic_assembly_time" "$elastic_solve_time" "$newton_assembly_time" "$newton_solve_time" "$wall_time" "$global_dofs" "$log" | tee -a "$csv"
}

if [[ "$#" -gt 0 ]]; then
  run_guarded "$@"
else
  for ranks in $RANKS; do
    for variant in $VARIANTS; do
      run_guarded "$ranks" "$variant"
    done
  done
fi

printf '\nsummary: %s\n' "$csv"
