#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
PARTITION="${PARTITION:-qcpu_exp}"
TIME_LIMIT="${TIME_LIMIT:-00:15:00}"
CASE="${CASE:-l1}"
NODES="${NODES:-2}"
TASKS_PER_NODE_LIST="${TASKS_PER_NODE_LIST:-64 128}"
ACTIVE_COARSE_RANKS_LIST="${ACTIVE_COARSE_RANKS_LIST:-8 16 32}"
TELESCOPE_SUBCOMM_TYPE="${TELESCOPE_SUBCOMM_TYPE:-interlaced}"
MATERIAL_SWEEP_COUNT="${MATERIAL_SWEEP_COUNT:-10}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/telescope_refresh_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"

manifest="$RUN_ROOT/submitted_telescope_refresh_jobs.csv"
echo "job_id,case,sweep_mode,sweep_count,ranks,nodes,tasks_per_node,active_coarse_ranks,telescope_factor,subcomm_type,partition,qos,time_limit,run_label" >"$manifest"

echo "RUN_ROOT=$RUN_ROOT"
echo "ACCOUNT=$ACCOUNT QOS=$QOS PARTITION=$PARTITION TIME_LIMIT=$TIME_LIMIT"
echo "CASE=$CASE"
echo "NODES=$NODES"
echo "TASKS_PER_NODE_LIST=$TASKS_PER_NODE_LIST"
echo "ACTIVE_COARSE_RANKS_LIST=$ACTIVE_COARSE_RANKS_LIST"
echo "TELESCOPE_SUBCOMM_TYPE=$TELESCOPE_SUBCOMM_TYPE"
echo "MATERIAL_SWEEP_COUNT=$MATERIAL_SWEEP_COUNT"

submit_one() {
  local tasks_per_node="$1"
  local active="$2"
  local ranks factor run_label job_name

  ranks=$(( NODES * tasks_per_node ))
  if (( active < 1 )); then
    echo "ERROR: active coarse ranks must be positive, got $active" >&2
    exit 2
  fi
  if (( ranks % active != 0 )); then
    echo "ERROR: ranks=$ranks is not divisible by active coarse ranks=$active" >&2
    exit 2
  fi

  factor=$(( ranks / active ))
  run_label="tel_rf${factor}_a${active}_${TELESCOPE_SUBCOMM_TYPE}_${NODES}n${tasks_per_node}ppn"
  job_name="p4e_tel_${CASE}_${ranks}r_a${active}"

  local cmd=(
    sbatch
    --parsable
    --account="$ACCOUNT"
    --qos="$QOS"
    --partition="$PARTITION"
    --nodes="$NODES"
    --ntasks="$ranks"
    --ntasks-per-node="$tasks_per_node"
    --cpus-per-task=1
    --time="$TIME_LIMIT"
    --job-name="$job_name"
    --output="$RUN_ROOT/logs/%x-%j.out"
    --error="$RUN_ROOT/logs/%x-%j.err"
    --export=ALL,REPO_ROOT="$REPO_ROOT",CAMPAIGN_DIR="$RUN_ROOT",ACCOUNT="$ACCOUNT",QOS="$QOS",CASE="$CASE",MATERIAL_SWEEP_MODE=refresh,MATERIAL_SWEEP_COUNT="$MATERIAL_SWEEP_COUNT",RANKS="$ranks",TASKS_PER_NODE="$tasks_per_node",RUN_LABEL="$run_label",PMG_COARSE_TELESCOPE_FACTOR="$factor",PMG_COARSE_TELESCOPE_ACTIVE_RANKS="$active",PMG_COARSE_TELESCOPE_SUBCOMM_TYPE="$TELESCOPE_SUBCOMM_TYPE"
  )

  if [[ "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
    cmd+=(--test-only)
  fi

  cmd+=("$SCRIPT_DIR/run_material_sweep.sbatch")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "DRY_RUN,$CASE,refresh,$MATERIAL_SWEEP_COUNT,$ranks,$NODES,$tasks_per_node,$active,$factor,$TELESCOPE_SUBCOMM_TYPE,$PARTITION,$QOS,$TIME_LIMIT,$run_label" >>"$manifest"
    return 0
  fi

  local job_id
  job_id="$("${cmd[@]}")"
  echo "$job_id,$CASE,refresh,$MATERIAL_SWEEP_COUNT,$ranks,$NODES,$tasks_per_node,$active,$factor,$TELESCOPE_SUBCOMM_TYPE,$PARTITION,$QOS,$TIME_LIMIT,$run_label" >>"$manifest"
  echo "submitted job_id=$job_id case=$CASE ranks=$ranks nodes=$NODES tasks_per_node=$tasks_per_node active_coarse_ranks=$active telescope_factor=$factor"
}

if (( NODES != 2 )); then
  echo "WARNING: this campaign was designed for two-node runs; NODES=$NODES" >&2
fi

for tasks_per_node in $TASKS_PER_NODE_LIST; do
  if (( tasks_per_node < 1 )); then
    echo "ERROR: TASKS_PER_NODE_LIST entries must be positive, got $tasks_per_node" >&2
    exit 2
  fi
  for active in $ACTIVE_COARSE_RANKS_LIST; do
    submit_one "$tasks_per_node" "$active"
  done
done

echo "Manifest: $manifest"
echo "Collect after completion with:"
echo "  $SCRIPT_DIR/collect_material_sweep.sh $RUN_ROOT"
