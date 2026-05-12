#!/usr/bin/env bash
#SBATCH --job-name=pmg_numa_p4l1
#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6328
#SBATCH --partition=qcpu_exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

ROOT="${ROOT:-$HOME/slope_stability_petsc4py}"
CONFIG="${CONFIG:-$ROOT/benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/gasm_numa_coalesced.toml}"
OUT_ROOT="${OUT_ROOT:-$ROOT/artifacts/experiments/pmg_numa_coalesced_karolina_p4_l1_omega7}"
RUN_NAME="${RUN_NAME:-nodes${SLURM_JOB_NUM_NODES:-1}_128ppn}"
RUN_OUT="$OUT_ROOT/runs/$RUN_NAME/${SLURM_JOB_ID:-manual}/run"

cd "$ROOT"

if command -v ml >/dev/null 2>&1; then
  ml -f purge
  ml GCC/14.3.0 OpenMPI/5.0.8-GCC-14.3.0 Python/3.13.5-GCCcore-14.3.0
fi

if [[ -f build_scripts/activate_local_petsc_env.sh ]]; then
  # shellcheck disable=SC1091
  source build_scripts/activate_local_petsc_env.sh
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_OUT"

echo "Root: $ROOT"
echo "Config: $CONFIG"
echo "Output: $RUN_OUT"
echo "Nodes: ${SLURM_JOB_NUM_NODES:-1}"
echo "Tasks per node: 128"

srun \
  --nodes="${SLURM_JOB_NUM_NODES:-1}" \
  --ntasks-per-node=128 \
  --cpus-per-task=1 \
  --distribution=block:block \
  --cpu-bind=cores \
  ./.venv/bin/python -u -m slope_stability.cli.run_case_from_config "$CONFIG" --out_dir "$RUN_OUT"
