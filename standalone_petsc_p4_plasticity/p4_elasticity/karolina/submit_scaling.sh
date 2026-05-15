#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
PARTITION="${PARTITION:-qcpu_exp}"
NODE_CORES="${NODE_CORES:-128}"
TIME_LIMIT="${TIME_LIMIT:-00:10:00}"
RANKS="${RANKS:-16 32 64 128 256}"
CASES="${CASES:-cube l1}"
VARIANTS="${VARIANTS:-gamg bddc fetidp pmg}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/karolina_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"

manifest="$RUN_ROOT/submitted_jobs.csv"
echo "job_id,case,variant,ranks,nodes,tasks_per_node,partition,qos,time_limit" >"$manifest"

echo "RUN_ROOT=$RUN_ROOT"
echo "ACCOUNT=$ACCOUNT QOS=$QOS PARTITION=$PARTITION NODE_CORES=$NODE_CORES TIME_LIMIT=$TIME_LIMIT"
echo "CASES=$CASES"
echo "VARIANTS=$VARIANTS"
echo "RANKS=$RANKS"

submit_one() {
  local case_name="$1"
  local variant="$2"
  local ranks="$3"
  local nodes tasks_per_node job_name

  nodes=$(( (ranks + NODE_CORES - 1) / NODE_CORES ))
  if (( nodes < 1 )); then nodes=1; fi
  tasks_per_node=$(( (ranks + nodes - 1) / nodes ))

  if (( tasks_per_node > NODE_CORES )); then
    echo "ERROR: ranks=$ranks requires tasks_per_node=$tasks_per_node > NODE_CORES=$NODE_CORES" >&2
    exit 2
  fi

  job_name="p4e_${case_name}_${variant}_${ranks}r"
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
    --export=ALL,REPO_ROOT="$REPO_ROOT",CAMPAIGN_DIR="$RUN_ROOT",ACCOUNT="$ACCOUNT",QOS="$QOS",CASE="$case_name",VARIANT="$variant",RANKS="$ranks"
  )

  if [[ "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
    cmd+=(--test-only)
  fi

  cmd+=("$SCRIPT_DIR/run_case.sbatch")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "DRY_RUN,$case_name,$variant,$ranks,$nodes,$tasks_per_node,$PARTITION,$QOS,$TIME_LIMIT" >>"$manifest"
    return 0
  fi

  local job_id
  job_id="$("${cmd[@]}")"
  echo "$job_id,$case_name,$variant,$ranks,$nodes,$tasks_per_node,$PARTITION,$QOS,$TIME_LIMIT" >>"$manifest"
  echo "submitted job_id=$job_id case=$case_name variant=$variant ranks=$ranks nodes=$nodes"
}

for case_name in $CASES; do
  for variant in $VARIANTS; do
    for ranks in $RANKS; do
      submit_one "$case_name" "$variant" "$ranks"
    done
  done
done

echo "Manifest: $manifest"
