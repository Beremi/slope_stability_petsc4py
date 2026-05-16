#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 runs/<campaign-directory>" >&2
  exit 2
fi

RUN_ROOT="$1"
RESULTS_DIR="$RUN_ROOT/results"
SUMMARY_OUT="$RUN_ROOT/material_sweep_summary.csv"
SAMPLES_OUT="$RUN_ROOT/material_sweep_samples.csv"

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
  # Karolina's default sacct -P can emit bare AveRSS values in bytes.
  awk -v s="$value" '
    BEGIN {
      unit = substr(s, length(s), 1)
      val = s + 0.0
      if (unit == "K") kib = val
      else if (unit == "M") kib = val * 1024.0
      else if (unit == "G") kib = val * 1024.0 * 1024.0
      else if (unit == "T") kib = val * 1024.0 * 1024.0 * 1024.0
      else kib = val / 1024.0
      printf "%.6g", kib / (1024.0 * 1024.0)
    }
  '
}

event_time() {
  local log="$1"
  local stage_name="$2"
  local event_name="$3"
  awk -v stage="$stage_name" -v event="$event_name" '
    /^--- Event Stage / {
      in_stage = index($0, stage) > 0
      next
    }
    in_stage {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      split(line, f, /[[:space:]]+/)
      if (f[1] == event) {
        print f[4]
        exit
      }
    }
  ' "$log" 2>/dev/null || true
}

{
  echo "job_id,case,sweep_mode,ranks,nodes,partition,qos,state,exit_code,elapsed,maxrss,averss,maxvmsize,maxrss_gib_per_rank,averss_gib_per_rank,approx_total_averss_gib,global_dofs,sweep_count,converged,total_ksp_its,first_ksp_its,repeated_ksp_its,max_ksp_its,first_solve_time,repeated_avg_solve_time,min_solve_time,max_solve_time,total_solve_time,first_snes,first_jacobian,first_ksp,first_pcsetup,first_matmatmatsym,first_matmatmatnum,repeated_snes,repeated_jacobian,repeated_ksp,repeated_pcsetup,repeated_pcapply,repeated_matmatmatsym,repeated_matmatmatnum,repeated_dmcreateinterp,repeated_dmcreatemat,repeated_dmprealloc,result_line,log"
  find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r dir; do
    env_file="$dir/job.env"
    result_file="$dir/result_line.txt"
    sacct_file="$dir/sacct.txt"
    run_log="$dir/run.log"

    result_line=""
    [[ -f "$result_file" ]] && result_line="$(cat "$result_file")"

    job_id="$(env_value "$env_file" JOB_ID)"
    case_name="$(env_value "$env_file" CASE)"
    sweep_mode="$(env_value "$env_file" MATERIAL_SWEEP_MODE)"
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
    if [[ -n "$job_id" ]] && command -v sacct >/dev/null 2>&1; then
      sacct --units=M -j "$job_id" --format=JobID,JobName%48,State,Elapsed,AllocNodes,AllocCPUS,NTasks,MaxRSS,AveRSS,MaxVMSize,ExitCode -P >"$sacct_file" 2>/dev/null || true
    fi
    if [[ -f "$sacct_file" ]]; then
      state="$(sacct_value "$sacct_file" State)"
      elapsed="$(sacct_value "$sacct_file" Elapsed)"
      maxrss="$(sacct_value "$sacct_file" MaxRSS)"
      averss="$(sacct_value "$sacct_file" AveRSS)"
      maxvmsize="$(sacct_value "$sacct_file" MaxVMSize)"
    fi

    maxrss_gib="$(rss_to_gib "$maxrss")"
    averss_gib="$(rss_to_gib "$averss")"
    total_averss_gib=""
    if [[ -n "$averss_gib" && -n "$ranks" ]]; then
      total_averss_gib="$(awk -v a="$averss_gib" -v r="$ranks" 'BEGIN {printf "%.6g", a * r}')"
    fi

    first_stage="--- Event Stage 1: material_sweep_first_setup"
    repeated_stage="--- Event Stage 2: material_sweep_repeated_solves"

    csv_escape "$job_id"; printf ','
    csv_escape "$case_name"; printf ','
    csv_escape "$sweep_mode"; printf ','
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
    csv_escape "$(result_value "$result_line" global_dofs)"; printf ','
    csv_escape "$(result_value "$result_line" sweep_count)"; printf ','
    csv_escape "$(result_value "$result_line" converged)"; printf ','
    csv_escape "$(result_value "$result_line" total_ksp_its)"; printf ','
    csv_escape "$(result_value "$result_line" first_ksp_its)"; printf ','
    csv_escape "$(result_value "$result_line" repeated_ksp_its)"; printf ','
    csv_escape "$(result_value "$result_line" max_ksp_its)"; printf ','
    csv_escape "$(result_value "$result_line" first_solve_time)"; printf ','
    csv_escape "$(result_value "$result_line" repeated_avg_solve_time)"; printf ','
    csv_escape "$(result_value "$result_line" min_solve_time)"; printf ','
    csv_escape "$(result_value "$result_line" max_solve_time)"; printf ','
    csv_escape "$(result_value "$result_line" total_solve_time)"; printf ','
    csv_escape "$(event_time "$run_log" "$first_stage" SNESSolve)"; printf ','
    csv_escape "$(event_time "$run_log" "$first_stage" DMPlexJacobianFE)"; printf ','
    csv_escape "$(event_time "$run_log" "$first_stage" KSPSolve)"; printf ','
    csv_escape "$(event_time "$run_log" "$first_stage" PCSetUp)"; printf ','
    csv_escape "$(event_time "$run_log" "$first_stage" MatMatMatMultSym)"; printf ','
    csv_escape "$(event_time "$run_log" "$first_stage" MatMatMatMultNum)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" SNESSolve)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" DMPlexJacobianFE)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" KSPSolve)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" PCSetUp)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" PCApply)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" MatMatMatMultSym)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" MatMatMatMultNum)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" DMCreateInterp)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" DMCreateMat)"; printf ','
    csv_escape "$(event_time "$run_log" "$repeated_stage" DMPlexPrealloc)"; printf ','
    csv_escape "$result_line"; printf ','
    csv_escape "$run_log"; printf '\n'
  done
} >"$SUMMARY_OUT"

{
  echo "job_id,case,sweep_mode,ranks,nodes,sample,E,nu,lambda,mu,ksp_its,ksp_reason,solve_time,max_abs_u,log"
  find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r dir; do
    env_file="$dir/job.env"
    run_log="$dir/run.log"
    [[ -f "$run_log" ]] || continue
    job_id="$(env_value "$env_file" JOB_ID)"
    case_name="$(env_value "$env_file" CASE)"
    sweep_mode="$(env_value "$env_file" MATERIAL_SWEEP_MODE)"
    ranks="$(env_value "$env_file" RANKS)"
    nodes="$(env_value "$env_file" NODES)"
    while read -r line; do
      csv_escape "$job_id"; printf ','
      csv_escape "$case_name"; printf ','
      csv_escape "$sweep_mode"; printf ','
      csv_escape "$ranks"; printf ','
      csv_escape "$nodes"; printf ','
      csv_escape "$(result_value "$line" sample)"; printf ','
      csv_escape "$(result_value "$line" E)"; printf ','
      csv_escape "$(result_value "$line" nu)"; printf ','
      csv_escape "$(result_value "$line" lambda)"; printf ','
      csv_escape "$(result_value "$line" mu)"; printf ','
      csv_escape "$(result_value "$line" ksp_its)"; printf ','
      csv_escape "$(result_value "$line" ksp_reason)"; printf ','
      csv_escape "$(result_value "$line" solve_time)"; printf ','
      csv_escape "$(result_value "$line" max_abs_u)"; printf ','
      csv_escape "$run_log"; printf '\n'
    done < <(grep '^SWEEP_RESULT ' "$run_log" || true)
  done
} >"$SAMPLES_OUT"

echo "Wrote $SUMMARY_OUT"
echo "Wrote $SAMPLES_OUT"
