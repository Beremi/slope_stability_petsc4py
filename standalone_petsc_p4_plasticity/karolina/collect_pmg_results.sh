#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 runs/<campaign-directory>" >&2
  exit 2
fi

RUN_ROOT="$1"
RESULTS_DIR="$RUN_ROOT/results"
OUT="$RUN_ROOT/pmg_results.csv"

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
      else kib = val / 1024.0
      printf "%.6g", kib / (1024.0 * 1024.0)
    }
  '
}

event_time() {
  local log="$1"
  local event_name="$2"
  awk -v event="$event_name" '
    {
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

join_diagnostics() {
  local file="$1"
  if [[ -f "$file" ]]; then
    awk 'BEGIN {first = 1} {gsub(/"/, "\"\""); if (!first) printf "; "; printf "%s", $0; first = 0}' "$file"
  fi
}

{
  echo "job_id,run_label,ranks,nodes,tasks_per_node,partition,qos,state,exit_code,elapsed,maxrss,averss,maxvmsize,maxrss_gib_per_rank,averss_gib_per_rank,approx_total_averss_gib,linear_rtol,ksp_max_it,deflation,pmg_apply_backend,pmg_p1_active_ranks,pmg_p1_subcomm,pmg_redundant_group_size,pmg_p2_active_ranks,pmg_shell_p2_active_ranks,pmg_shell_p1_active_ranks,pmg_shell_subcomm_type,pmg_shell_coarse_layout,pmg_lag_preconditioner,global_dofs,elastic_its,newton_its,newton_linear_its,total_linear_its,elastic_assembly_time,elastic_solve_time,newton_assembly_time,newton_solve_time,wall_time,final_rel,deflation_basis_cols,deflation_orthogonalization_time,deflation_coarse_initial_time,deflation_pc_apply_time,deflation_projector_time,PCApply,KSPSolve,MatMult,VecScatterEnd,VecMDot,KSPGMRESOrthog,MatPtAPNumeric,MatPtAPSymbolic,PCSetUp,pmg_diagnostics,result_line,diagnostics,log"
  find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r dir; do
    env_file="$dir/job.env"
    result_file="$dir/result_line.txt"
    sacct_file="$dir/sacct.txt"
    run_log="$dir/run.log"
    diagnostics_file="$dir/diagnostics.txt"

    result_line=""
    [[ -f "$result_file" ]] && result_line="$(cat "$result_file")"

    job_id="$(env_value "$env_file" JOB_ID)"
    run_label="$(env_value "$env_file" RUN_LABEL)"
    ranks="$(env_value "$env_file" RANKS)"
    nodes="$(env_value "$env_file" NODES)"
    tasks_per_node="$(env_value "$env_file" TASKS_PER_NODE)"
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

    csv_escape "$job_id"; printf ','
    csv_escape "$run_label"; printf ','
    csv_escape "$ranks"; printf ','
    csv_escape "$nodes"; printf ','
    csv_escape "$tasks_per_node"; printf ','
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
    csv_escape "$(env_value "$env_file" LINEAR_RTOL)"; printf ','
    csv_escape "$(env_value "$env_file" KSP_MAX_IT)"; printf ','
    csv_escape "$(env_value "$env_file" DEFLATION)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_APPLY_BACKEND)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_COARSE_TELESCOPE_ACTIVE_RANKS)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_COARSE_TELESCOPE_SUBCOMM_TYPE)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_COARSE_REDUNDANT_GROUP_SIZE)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_P2_TELESCOPE_ACTIVE_RANKS)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_SHELL_P2_ACTIVE_RANKS)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_SHELL_P1_ACTIVE_RANKS)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_SHELL_SUBCOMM_TYPE)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_SHELL_COARSE_LAYOUT)"; printf ','
    csv_escape "$(env_value "$env_file" PMG_LAG_PRECONDITIONER)"; printf ','
    csv_escape "$(result_value "$result_line" global_dofs)"; printf ','
    csv_escape "$(result_value "$result_line" elastic_its)"; printf ','
    csv_escape "$(result_value "$result_line" newton_its)"; printf ','
    csv_escape "$(result_value "$result_line" newton_linear_its)"; printf ','
    csv_escape "$(result_value "$result_line" total_linear_its)"; printf ','
    csv_escape "$(result_value "$result_line" elastic_assembly_time)"; printf ','
    csv_escape "$(result_value "$result_line" elastic_solve_time)"; printf ','
    csv_escape "$(result_value "$result_line" newton_assembly_time)"; printf ','
    csv_escape "$(result_value "$result_line" newton_solve_time)"; printf ','
    csv_escape "$(result_value "$result_line" wall_time)"; printf ','
    csv_escape "$(result_value "$result_line" final_rel)"; printf ','
    csv_escape "$(result_value "$result_line" deflation_basis_cols)"; printf ','
    csv_escape "$(result_value "$result_line" deflation_orthogonalization_time)"; printf ','
    csv_escape "$(result_value "$result_line" deflation_coarse_initial_time)"; printf ','
    csv_escape "$(result_value "$result_line" deflation_pc_apply_time)"; printf ','
    csv_escape "$(result_value "$result_line" deflation_projector_time)"; printf ','
    csv_escape "$(event_time "$run_log" PCApply)"; printf ','
    csv_escape "$(event_time "$run_log" KSPSolve)"; printf ','
    csv_escape "$(event_time "$run_log" MatMult)"; printf ','
    csv_escape "$(event_time "$run_log" VecScatterEnd)"; printf ','
    csv_escape "$(event_time "$run_log" VecMDot)"; printf ','
    csv_escape "$(event_time "$run_log" KSPGMRESOrthog)"; printf ','
    csv_escape "$(event_time "$run_log" MatPtAPNumeric)"; printf ','
    csv_escape "$(event_time "$run_log" MatPtAPSymbolic)"; printf ','
    csv_escape "$(event_time "$run_log" PCSetUp)"; printf ','
    csv_escape "$(join_diagnostics "$diagnostics_file")"; printf ','
    csv_escape "$result_line"; printf ','
    csv_escape "$diagnostics_file"; printf ','
    csv_escape "$run_log"; printf '\n'
  done
} >"$OUT"

echo "Wrote $OUT"
