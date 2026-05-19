#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

ACCOUNT="${ACCOUNT:-fta-26-40}"
QOS="${QOS:-3571_6328}"
PARTITION="${PARTITION:-qcpu_exp}"
NODE_CORES="${NODE_CORES:-128}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
NODE_COUNTS="${NODE_COUNTS:-1 2}"
ACTIVE_RANKS_LIST="${ACTIVE_RANKS_LIST:-16 32 64}"
SUBCOMM_TYPES="${SUBCOMM_TYPES:-contiguous interlaced}"
DEFLATION_LIST="${DEFLATION_LIST:-false true}"
REDUNDANT_GROUP_SIZES="${REDUNDANT_GROUP_SIZES:-16 32 64}"
SHELL_P2_ACTIVE_RANKS_LIST="${SHELL_P2_ACTIVE_RANKS_LIST:-64 128}"
SHELL_P1_ACTIVE_RANKS_LIST="${SHELL_P1_ACTIVE_RANKS_LIST:-32 64}"
SHELL_SUBCOMM_TYPES="${SHELL_SUBCOMM_TYPES:-interlaced contiguous}"
INCLUDE_TELESCOPE="${INCLUDE_TELESCOPE:-1}"
INCLUDE_REDUNDANT="${INCLUDE_REDUNDANT:-1}"
INCLUDE_SHELL="${INCLUDE_SHELL:-0}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs/pmg_plasticity_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"

manifest="$RUN_ROOT/submitted_pmg_jobs.csv"
echo "job_id,profile,ranks,nodes,tasks_per_node,active_ranks,subcomm_type,redundant_group_size,shell_p2_active_ranks,shell_p1_active_ranks,shell_subcomm_type,deflation,lag,linear_rtol,ksp_max_it,partition,qos,time_limit,run_label" >"$manifest"

echo "RUN_ROOT=$RUN_ROOT"
echo "ACCOUNT=$ACCOUNT QOS=$QOS PARTITION=$PARTITION NODE_CORES=$NODE_CORES TIME_LIMIT=$TIME_LIMIT"
echo "NODE_COUNTS=$NODE_COUNTS"
echo "ACTIVE_RANKS_LIST=$ACTIVE_RANKS_LIST"
echo "SUBCOMM_TYPES=$SUBCOMM_TYPES"
echo "DEFLATION_LIST=$DEFLATION_LIST"
echo "REDUNDANT_GROUP_SIZES=$REDUNDANT_GROUP_SIZES"
echo "SHELL_P2_ACTIVE_RANKS_LIST=$SHELL_P2_ACTIVE_RANKS_LIST"
echo "SHELL_P1_ACTIVE_RANKS_LIST=$SHELL_P1_ACTIVE_RANKS_LIST"
echo "SHELL_SUBCOMM_TYPES=$SHELL_SUBCOMM_TYPES"
echo "LINEAR_RTOL=${LINEAR_RTOL:-1e-1} KSP_MAX_IT=${KSP_MAX_IT:-200} PMG_LAG_PRECONDITIONER=${PMG_LAG_PRECONDITIONER:-1}"

submit_one() {
  local profile="$1"
  local nodes="$2"
  local active="$3"
  local subcomm="$4"
  local group_size="$5"
  local deflation="$6"
  local shell_p2="${7:-}"
  local shell_p1="${8:-}"
  local shell_subcomm="${9:-}"
  local ranks tasks_per_node job_name run_label profile_options_file profile_backend

  ranks=$(( nodes * NODE_CORES ))
  tasks_per_node="$NODE_CORES"
  run_label="${profile}_${nodes}n${tasks_per_node}ppn"
  profile_options_file="${OPTIONS_FILE:-options/pmg_p1_telescope.opts}"
  profile_backend="pcmg"
  if [[ "$profile" == "telescope" ]]; then
    if (( active < 1 )); then
      echo "ERROR: telescope active ranks must be positive, got $active" >&2
      exit 2
    fi
    if (( ranks % active != 0 )); then
      echo "ERROR: ranks=$ranks is not divisible by active ranks=$active" >&2
      exit 2
    fi
    run_label+="_a${active}_${subcomm}"
  elif [[ "$profile" == "redundant" ]]; then
    run_label+="_redundant_g${group_size}"
  elif [[ "$profile" == "shell" ]]; then
    if (( shell_p2 < 1 || shell_p1 < 1 )); then
      echo "ERROR: shell active ranks must be positive, got p2=$shell_p2 p1=$shell_p1" >&2
      exit 2
    fi
    if (( ranks % shell_p2 != 0 || ranks % shell_p1 != 0 )); then
      echo "ERROR: ranks=$ranks must be divisible by shell p2=$shell_p2 and p1=$shell_p1" >&2
      exit 2
    fi
    run_label+="_p2a${shell_p2}_p1a${shell_p1}_${shell_subcomm}"
    profile_options_file="${SHELL_OPTIONS_FILE:-options/pmg_shell_vcycle.opts}"
    profile_backend="shell_vcycle"
  else
    echo "ERROR: unknown profile=$profile" >&2
    exit 2
  fi
  run_label+="_defl${deflation}_lag${PMG_LAG_PRECONDITIONER:-1}_rtol${LINEAR_RTOL:-1e-1}"
  job_name="p4p_${profile}_${nodes}n_${deflation}"

  local exports=(
    ALL
    REPO_ROOT="$REPO_ROOT"
    CAMPAIGN_DIR="$RUN_ROOT"
    ACCOUNT="$ACCOUNT"
    QOS="$QOS"
    RANKS="$ranks"
    TASKS_PER_NODE="$tasks_per_node"
    RUN_LABEL="$run_label"
    DEFLATION="$deflation"
    LINEAR_RTOL="${LINEAR_RTOL:-1e-1}"
    KSP_MAX_IT="${KSP_MAX_IT:-200}"
    NEWTON_MAX_IT="${NEWTON_MAX_IT:-20}"
    REFINE_LEVELS="${REFINE_LEVELS:-1}"
    LAMBDA="${LAMBDA:-1.5}"
    PARTITIONER="${PARTITIONER:-parmetis}"
    PMG_LAG_PRECONDITIONER="${PMG_LAG_PRECONDITIONER:-1}"
    PMG_P2_TELESCOPE_ACTIVE_RANKS=0
    PMG_APPLY_BACKEND="$profile_backend"
    OPTIONS_FILE="$profile_options_file"
  )

  if [[ "$profile" == "telescope" ]]; then
    exports+=(
      PMG_COARSE_TELESCOPE_ACTIVE_RANKS="$active"
      PMG_COARSE_TELESCOPE_SUBCOMM_TYPE="$subcomm"
      PMG_COARSE_REDUNDANT_GROUP_SIZE="${PMG_COARSE_REDUNDANT_GROUP_SIZE:-16}"
    )
  elif [[ "$profile" == "redundant" ]]; then
    exports+=(
      PMG_COARSE_TELESCOPE_ACTIVE_RANKS=0
      PMG_COARSE_TELESCOPE_SUBCOMM_TYPE=interlaced
      PMG_COARSE_REDUNDANT_GROUP_SIZE="$group_size"
    )
  elif [[ "$profile" == "shell" ]]; then
    exports+=(
      PMG_COARSE_TELESCOPE_ACTIVE_RANKS=0
      PMG_COARSE_TELESCOPE_SUBCOMM_TYPE=interlaced
      PMG_COARSE_REDUNDANT_GROUP_SIZE=0
      PMG_SHELL_P2_ACTIVE_RANKS="$shell_p2"
      PMG_SHELL_P1_ACTIVE_RANKS="$shell_p1"
      PMG_SHELL_SUBCOMM_TYPE="$shell_subcomm"
    )
  fi

  local export_arg
  export_arg="$(IFS=,; echo "${exports[*]}")"
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
    --export="$export_arg"
  )

  if [[ "${SBATCH_TEST_ONLY:-0}" == "1" ]]; then
    cmd+=(--test-only)
  fi

  cmd+=("$SCRIPT_DIR/run_case.sbatch")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "DRY_RUN,$profile,$ranks,$nodes,$tasks_per_node,$active,$subcomm,$group_size,$shell_p2,$shell_p1,$shell_subcomm,$deflation,${PMG_LAG_PRECONDITIONER:-1},${LINEAR_RTOL:-1e-1},${KSP_MAX_IT:-200},$PARTITION,$QOS,$TIME_LIMIT,$run_label" >>"$manifest"
    return 0
  fi

  local job_id
  job_id="$("${cmd[@]}")"
  echo "$job_id,$profile,$ranks,$nodes,$tasks_per_node,$active,$subcomm,$group_size,$shell_p2,$shell_p1,$shell_subcomm,$deflation,${PMG_LAG_PRECONDITIONER:-1},${LINEAR_RTOL:-1e-1},${KSP_MAX_IT:-200},$PARTITION,$QOS,$TIME_LIMIT,$run_label" >>"$manifest"
  echo "submitted job_id=$job_id profile=$profile ranks=$ranks active=$active subcomm=$subcomm group_size=$group_size deflation=$deflation"
}

for nodes in $NODE_COUNTS; do
  if (( nodes < 1 )); then
    echo "ERROR: NODE_COUNTS entries must be positive, got $nodes" >&2
    exit 2
  fi
  for deflation in $DEFLATION_LIST; do
    case "$deflation" in
      true|false) ;;
      *) echo "ERROR: DEFLATION_LIST entries must be true or false, got $deflation" >&2; exit 2 ;;
    esac
    if [[ "$INCLUDE_TELESCOPE" == "1" ]]; then
      for active in $ACTIVE_RANKS_LIST; do
        for subcomm in $SUBCOMM_TYPES; do
          case "$subcomm" in
            contiguous|interlaced) ;;
            *) echo "ERROR: SUBCOMM_TYPES entries must be contiguous or interlaced, got $subcomm" >&2; exit 2 ;;
          esac
          submit_one telescope "$nodes" "$active" "$subcomm" "" "$deflation"
        done
      done
    fi
    if [[ "$INCLUDE_REDUNDANT" == "1" ]]; then
      for group_size in $REDUNDANT_GROUP_SIZES; do
        if (( group_size < 1 )); then
          echo "ERROR: REDUNDANT_GROUP_SIZES entries must be positive, got $group_size" >&2
          exit 2
        fi
        submit_one redundant "$nodes" 0 none "$group_size" "$deflation"
      done
    fi
    if [[ "$INCLUDE_SHELL" == "1" ]]; then
      for shell_p2 in $SHELL_P2_ACTIVE_RANKS_LIST; do
        for shell_p1 in $SHELL_P1_ACTIVE_RANKS_LIST; do
          if (( shell_p1 > shell_p2 )); then
            echo "skipping shell p2=$shell_p2 p1=$shell_p1 because p1 active ranks exceed p2 active ranks"
            continue
          fi
          for shell_subcomm in $SHELL_SUBCOMM_TYPES; do
            case "$shell_subcomm" in
              contiguous|interlaced) ;;
              *) echo "ERROR: SHELL_SUBCOMM_TYPES entries must be contiguous or interlaced, got $shell_subcomm" >&2; exit 2 ;;
            esac
            submit_one shell "$nodes" 0 none "" "$deflation" "$shell_p2" "$shell_p1" "$shell_subcomm"
          done
        done
      done
    fi
  done
done

echo "Manifest: $manifest"
echo "Collect after completion with:"
echo "  $SCRIPT_DIR/collect_pmg_results.sh $RUN_ROOT"
