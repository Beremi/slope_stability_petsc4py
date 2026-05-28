#!/usr/bin/env bash
#SBATCH --job-name=ssr_engine
#SBATCH --partition=qcpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

ROOT="${ROOT:?ROOT must point to the PETSc SSR repository root}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
RANKS="${SLURM_NTASKS:?SLURM_NTASKS missing}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CASE_TOML="${CASE_TOML:-$ROOT/benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml}"

mkdir -p "$RUN_DIR"
cd "$ROOT"

export OMP_NUM_THREADS=1
export PETSC_DIR="${PETSC_DIR:-$ROOT/.build/src/petsc-3.24.5}"
export PETSC_ARCH="${PETSC_ARCH:-linux-c-opt}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

/usr/bin/time -v srun --cpu-bind=cores "$PYTHON" -m petsc_ssr.cli.main run "$CASE_TOML" \
  --omega-max "${OMEGA_MAX:-7000000}" \
  --continuation-step-max "${CONTINUATION_STEP_MAX:-100}" \
  --linear-rtol "${LINEAR_RTOL:-1e-1}" \
  --ksp-max-it "${KSP_MAX_IT:-200}" \
  --output "$RUN_DIR/output" \
  ${EXTRA_ARGS:-} 2>&1 | tee "$RUN_DIR/run.log"

sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%24,State,Elapsed,MaxRSS,AveRSS,AllocTRES%48 > "$RUN_DIR/sacct.txt" || true
