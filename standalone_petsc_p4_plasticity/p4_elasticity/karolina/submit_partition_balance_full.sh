#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
CASE="${CASE:-l1}"
NODES_LIST="${NODES_LIST:-2 4}"
TASKS_PER_NODE="${TASKS_PER_NODE:-128}"
ACTIVE_COARSE_RANKS_LIST="${ACTIVE_COARSE_RANKS_LIST:-16 32 64}"
TELESCOPE_SUBCOMM_TYPE="${TELESCOPE_SUBCOMM_TYPE:-interlaced}"
MATERIAL_SWEEP_COUNT="${MATERIAL_SWEEP_COUNT:-10}"
TIME_LIMIT="${TIME_LIMIT:-00:15:00}"
PLEX_PARTITION_BALANCE="${PLEX_PARTITION_BALANCE:-true}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/partition_balance_full_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"

manifest="$RUN_ROOT/submitted_partition_balance_full_jobs.csv"
echo "job_id,case,sweep_mode,sweep_count,ranks,nodes,tasks_per_node,active_coarse_ranks,telescope_factor,subcomm_type,partition_balance,partition,qos,time_limit,run_label" >"$manifest"

echo "RUN_ROOT=$RUN_ROOT"
echo "ACCOUNT=$ACCOUNT QOS=$QOS TIME_LIMIT=$TIME_LIMIT"
echo "CASE=$CASE"
echo "NODES_LIST=$NODES_LIST"
echo "TASKS_PER_NODE=$TASKS_PER_NODE"
echo "ACTIVE_COARSE_RANKS_LIST=$ACTIVE_COARSE_RANKS_LIST"
echo "TELESCOPE_SUBCOMM_TYPE=$TELESCOPE_SUBCOMM_TYPE"
echo "MATERIAL_SWEEP_COUNT=$MATERIAL_SWEEP_COUNT"
echo "PLEX_PARTITION_BALANCE=$PLEX_PARTITION_BALANCE"

partition_for_nodes() {
  local nodes="$1"

  if [[ -n "${PARTITION:-}" ]]; then
    echo "$PARTITION"
  elif (( nodes <= 2 )); then
    echo "${PARTITION_2NODES:-qcpu_exp}"
  else
    echo "${PARTITION_4NODES:-qcpu}"
  fi
}

submit_one() {
  local nodes="$1"
  local active="$2"
  local ranks factor run_label job_name partition

  ranks=$(( nodes * TASKS_PER_NODE ))
  if (( active < 1 )); then
    echo "ERROR: active coarse ranks must be positive, got $active" >&2
    exit 2
  fi
  if (( ranks % active != 0 )); then
    echo "ERROR: ranks=$ranks is not divisible by active coarse ranks=$active" >&2
    exit 2
  fi

  factor=$(( ranks / active ))
  partition="$(partition_for_nodes "$nodes")"
  run_label="pbal_tel_rf${factor}_a${active}_${TELESCOPE_SUBCOMM_TYPE}_${nodes}n${TASKS_PER_NODE}ppn"
  job_name="p4e_pbal_${CASE}_${ranks}r_a${active}"

  local cmd=(
    sbatch
    --parsable
    --account="$ACCOUNT"
    --qos="$QOS"
    --partition="$partition"
    --nodes="$nodes"
    --ntasks="$ranks"
    --ntasks-per-node="$TASKS_PER_NODE"
    --cpus-per-task=1
    --time="$TIME_LIMIT"
    --job-name="$job_name"
    --output="$RUN_ROOT/logs/%x-%j.out"
    --error="$RUN_ROOT/logs/%x-%j.err"
    --export=ALL,REPO_ROOT="$REPO_ROOT",CAMPAIGN_DIR="$RUN_ROOT",ACCOUNT="$ACCOUNT",QOS="$QOS",CASE="$CASE",MATERIAL_SWEEP_MODE=refresh,MATERIAL_SWEEP_COUNT="$MATERIAL_SWEEP_COUNT",RANKS="$ranks",TASKS_PER_NODE="$TASKS_PER_NODE",RUN_LABEL="$run_label",PMG_COARSE_TELESCOPE_FACTOR="$factor",PMG_COARSE_TELESCOPE_ACTIVE_RANKS="$active",PMG_COARSE_TELESCOPE_SUBCOMM_TYPE="$TELESCOPE_SUBCOMM_TYPE",PLEX_PARTITION_BALANCE="$PLEX_PARTITION_BALANCE"
  )

  if [[ "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
    cmd+=(--test-only)
  fi

  cmd+=("$SCRIPT_DIR/run_material_sweep.sbatch")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "DRY_RUN,$CASE,refresh,$MATERIAL_SWEEP_COUNT,$ranks,$nodes,$TASKS_PER_NODE,$active,$factor,$TELESCOPE_SUBCOMM_TYPE,$PLEX_PARTITION_BALANCE,$partition,$QOS,$TIME_LIMIT,$run_label" >>"$manifest"
    return 0
  fi

  local job_id
  job_id="$("${cmd[@]}")"
  echo "$job_id,$CASE,refresh,$MATERIAL_SWEEP_COUNT,$ranks,$nodes,$TASKS_PER_NODE,$active,$factor,$TELESCOPE_SUBCOMM_TYPE,$PLEX_PARTITION_BALANCE,$partition,$QOS,$TIME_LIMIT,$run_label" >>"$manifest"
  echo "submitted job_id=$job_id case=$CASE ranks=$ranks nodes=$nodes tasks_per_node=$TASKS_PER_NODE active_coarse_ranks=$active telescope_factor=$factor partition_balance=$PLEX_PARTITION_BALANCE partition=$partition"
}

for nodes in $NODES_LIST; do
  if (( nodes < 1 )); then
    echo "ERROR: NODES_LIST entries must be positive, got $nodes" >&2
    exit 2
  fi
  for active in $ACTIVE_COARSE_RANKS_LIST; do
    submit_one "$nodes" "$active"
  done
done

echo "Manifest: $manifest"
echo "Collect after completion with:"
echo "  $SCRIPT_DIR/collect_material_sweep.sh $RUN_ROOT"
