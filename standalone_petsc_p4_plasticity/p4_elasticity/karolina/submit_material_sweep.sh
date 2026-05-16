#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
PARTITION="${PARTITION:-qcpu_exp}"
NODE_CORES="${NODE_CORES:-128}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
CASE="${CASE:-l1}"
MATERIAL_SWEEP_COUNT="${MATERIAL_SWEEP_COUNT:-100}"
MATERIAL_SWEEP_MODES="${MATERIAL_SWEEP_MODES:-fresh refresh reuse_pc}"
NODE_COUNTS="${NODE_COUNTS:-1 2}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/material_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"

manifest="$RUN_ROOT/submitted_material_sweep_jobs.csv"
echo "job_id,case,sweep_mode,sweep_count,ranks,nodes,tasks_per_node,partition,qos,time_limit,pmg_group_size" >"$manifest"

echo "RUN_ROOT=$RUN_ROOT"
echo "ACCOUNT=$ACCOUNT QOS=$QOS PARTITION=$PARTITION NODE_CORES=$NODE_CORES TIME_LIMIT=$TIME_LIMIT"
echo "CASE=$CASE"
echo "MATERIAL_SWEEP_COUNT=$MATERIAL_SWEEP_COUNT"
echo "MATERIAL_SWEEP_MODES=$MATERIAL_SWEEP_MODES"
echo "NODE_COUNTS=$NODE_COUNTS"
echo "PMG_GROUP_SIZE=${PMG_GROUP_SIZE:-16}"

submit_one() {
  local mode="$1"
  local nodes="$2"
  local ranks tasks_per_node job_name

  ranks=$(( nodes * NODE_CORES ))
  tasks_per_node="$NODE_CORES"
  job_name="p4e_ms_${CASE}_${mode}_${nodes}n"

  local cmd=(
    sbatch
    --parsable
    --account="$ACCOUNT"
    --qos="$QOS"
    --partition="$PARTITION"
    --nodes="$nodes"
    --ntasks="$ranks"
    --ntasks-per-node="$tasks_per_node"
    --cpus-per-task=1
    --time="$TIME_LIMIT"
    --job-name="$job_name"
    --output="$RUN_ROOT/logs/%x-%j.out"
    --error="$RUN_ROOT/logs/%x-%j.err"
    --export=ALL,REPO_ROOT="$REPO_ROOT",CAMPAIGN_DIR="$RUN_ROOT",ACCOUNT="$ACCOUNT",QOS="$QOS",CASE="$CASE",MATERIAL_SWEEP_MODE="$mode",MATERIAL_SWEEP_COUNT="$MATERIAL_SWEEP_COUNT",RANKS="$ranks"
  )

  if [[ "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
    cmd+=(--test-only)
  fi

  cmd+=("$SCRIPT_DIR/run_material_sweep.sbatch")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "DRY_RUN,$CASE,$mode,$MATERIAL_SWEEP_COUNT,$ranks,$nodes,$tasks_per_node,$PARTITION,$QOS,$TIME_LIMIT,${PMG_GROUP_SIZE:-16}" >>"$manifest"
    return 0
  fi

  local job_id
  job_id="$("${cmd[@]}")"
  echo "$job_id,$CASE,$mode,$MATERIAL_SWEEP_COUNT,$ranks,$nodes,$tasks_per_node,$PARTITION,$QOS,$TIME_LIMIT,${PMG_GROUP_SIZE:-16}" >>"$manifest"
  echo "submitted job_id=$job_id case=$CASE mode=$mode ranks=$ranks nodes=$nodes"
}

for nodes in $NODE_COUNTS; do
  if (( nodes < 1 )); then
    echo "ERROR: NODE_COUNTS entries must be positive, got $nodes" >&2
    exit 2
  fi
  for mode in $MATERIAL_SWEEP_MODES; do
    case "$mode" in
      fresh|refresh|reuse_pc) ;;
      *)
        echo "ERROR: unknown material sweep mode $mode" >&2
        exit 2
        ;;
    esac
    submit_one "$mode" "$nodes"
  done
done

echo "Manifest: $manifest"
