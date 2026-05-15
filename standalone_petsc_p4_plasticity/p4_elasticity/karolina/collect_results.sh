#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 runs/<campaign-directory>" >&2
  exit 2
fi

RUN_ROOT="$1"
RESULTS_DIR="$RUN_ROOT/results"
OUT="$RUN_ROOT/summary.csv"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "ERROR: missing results directory: $RESULTS_DIR" >&2
  exit 2
fi

csv_escape() {
  local s="${1:-}"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

env_value() {
  local file="$1"
  local key="$2"
  grep -E "^${key}=" "$file" 2>/dev/null | tail -n 1 | cut -d= -f2- || true
}

result_value() {
  local line="$1"
  local key="$2"
  tr ' ' '\n' <<<"$line" | awk -F= -v k="$key" '$1 == k {print $2; exit}'
}

sacct_value() {
  local file="$1"
  local column="$2"
  awk -F'|' -v col="$column" '
    NR == 1 {
      for (i = 1; i <= NF; ++i) idx[$i] = i
      next
    }
    $1 ~ /\.[0-9]+$/ && idx[col] && $idx[col] != "" {print $idx[col]; printed = 1; exit}
    $1 ~ /\.batch$/ && idx[col] && $idx[col] != "" {batch = $idx[col]}
    NR == 2 && idx[col] {fallback = $idx[col]}
    END {
      if (printed) {}
      else if (batch != "") print batch
      else if (fallback != "") print fallback
    }
  ' "$file" 2>/dev/null || true
}

rss_to_gib() {
  local value="${1:-}"
  if [[ -z "$value" || "$value" == "0" ]]; then
    printf ''
    return 0
  fi
  awk -v s="$value" '
    BEGIN {
      unit = substr(s, length(s), 1)
      val = s + 0.0
      if (unit == "K") kib = val
      else if (unit == "M") kib = val * 1024.0
      else if (unit == "G") kib = val * 1024.0 * 1024.0
      else if (unit == "T") kib = val * 1024.0 * 1024.0 * 1024.0
      else kib = val
      printf "%.6g", kib / (1024.0 * 1024.0)
    }
  '
}

{
  echo "job_id,case,variant,ranks,nodes,partition,qos,state,exit_code,elapsed,maxrss,averss,maxvmsize,maxrss_gib_per_rank,averss_gib_per_rank,approx_total_averss_gib,global_dofs,ksp_its,ksp_reason,solve_time,max_abs_u,result_line,log"
  find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r dir; do
    env_file="$dir/job.env"
    result_file="$dir/result_line.txt"
    sacct_file="$dir/sacct.txt"
    run_log="$dir/run.log"

    result_line=""
    [[ -f "$result_file" ]] && result_line="$(cat "$result_file")"

    job_id="$(env_value "$env_file" JOB_ID)"
    case_name="$(env_value "$env_file" CASE)"
    variant="$(env_value "$env_file" VARIANT)"
    ranks="$(env_value "$env_file" RANKS)"
    nodes="$(env_value "$env_file" NODES)"
    partition="$(env_value "$env_file" PARTITION)"
    qos="$(env_value "$env_file" QOS)"
    exit_code="$(env_value "$env_file" EXIT_CODE)"

    state=""
    elapsed=""
    maxrss=""
    averss=""
    maxvmsize=""
    if [[ ! -f "$sacct_file" && -n "$job_id" ]] && command -v sacct >/dev/null 2>&1; then
      sacct -j "$job_id" --format=JobID,JobName%40,State,Elapsed,AllocNodes,AllocCPUS,NTasks,MaxRSS,AveRSS,MaxVMSize,ExitCode -P >"$sacct_file" 2>/dev/null || true
    fi
    if [[ -f "$sacct_file" ]]; then
      state="$(sacct_value "$sacct_file" State)"
      elapsed="$(sacct_value "$sacct_file" Elapsed)"
      maxrss="$(sacct_value "$sacct_file" MaxRSS)"
      averss="$(sacct_value "$sacct_file" AveRSS)"
      maxvmsize="$(sacct_value "$sacct_file" MaxVMSize)"
    fi

    global_dofs="$(result_value "$result_line" global_dofs)"
    ksp_its="$(result_value "$result_line" ksp_its)"
    ksp_reason="$(result_value "$result_line" ksp_reason)"
    solve_time="$(result_value "$result_line" solve_time)"
    max_abs_u="$(result_value "$result_line" max_abs_u)"
    maxrss_gib="$(rss_to_gib "$maxrss")"
    averss_gib="$(rss_to_gib "$averss")"
    total_averss_gib=""
    if [[ -n "$averss_gib" && -n "$ranks" ]]; then
      total_averss_gib="$(awk -v a="$averss_gib" -v r="$ranks" 'BEGIN {printf "%.6g", a * r}')"
    fi

    csv_escape "$job_id"; printf ','
    csv_escape "$case_name"; printf ','
    csv_escape "$variant"; printf ','
    csv_escape "$ranks"; printf ','
    csv_escape "$nodes"; printf ','
    csv_escape "$partition"; printf ','
    csv_escape "$qos"; printf ','
    csv_escape "$state"; printf ','
    csv_escape "$exit_code"; printf ','
    csv_escape "$elapsed"; printf ','
    csv_escape "$maxrss"; printf ','
    csv_escape "$averss"; printf ','
    csv_escape "$maxvmsize"; printf ','
    csv_escape "$maxrss_gib"; printf ','
    csv_escape "$averss_gib"; printf ','
    csv_escape "$total_averss_gib"; printf ','
    csv_escape "$global_dofs"; printf ','
    csv_escape "$ksp_its"; printf ','
    csv_escape "$ksp_reason"; printf ','
    csv_escape "$solve_time"; printf ','
    csv_escape "$max_abs_u"; printf ','
    csv_escape "$result_line"; printf ','
    csv_escape "$run_log"; printf '\n'
  done
} >"$OUT"

echo "Wrote $OUT"
