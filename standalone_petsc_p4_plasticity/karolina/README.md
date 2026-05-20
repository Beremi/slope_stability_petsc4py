# Karolina PMG Plasticity Runs

This harness submits and collects full-node PMG runs for the standalone pure
PETSc plasticity executable:

```text
standalone_petsc_p4_plasticity/p4_plasticity
```

Defaults target the refined L1 plasticity case and the P1-only telescope PMG
profile:

```text
OPTIONS_FILE=options/pmg_p1_telescope.opts
REFINE_LEVELS=1
LAMBDA=1.5
LINEAR_RTOL=1e-1
KSP_MAX_IT=200
PARTITIONER=parmetis
NODE_CORES=128
NODE_COUNTS="1 2"
ACTIVE_RANKS_LIST="16 32 64"
SUBCOMM_TYPES="contiguous interlaced"
DEFLATION_LIST="false true"
```

## Prepare

On Karolina, pull the branch and build the standalone solver:

```bash
cd /path/to/slope_stability_petsc4py
export PETSC_DIR=/path/to/petsc-3.24.5
export PETSC_ARCH=linux-c-opt
make -C standalone_petsc_p4_plasticity
```

## Preview

```bash
cd standalone_petsc_p4_plasticity/karolina
DRY_RUN=1 ./submit_pmg_scaling.sh
```

To preview only the two-node P1 telescope matrix:

```bash
DRY_RUN=1 \
NODE_COUNTS="2" \
INCLUDE_REDUNDANT=0 \
./submit_pmg_scaling.sh
```

To preview the opt-in true coarse-level shell V-cycle experiment only:

```bash
DRY_RUN=1 \
INCLUDE_TELESCOPE=0 \
INCLUDE_REDUNDANT=0 \
INCLUDE_SHELL=1 \
SHELL_P2_ACTIVE_RANKS_LIST="64 128" \
SHELL_P1_ACTIVE_RANKS_LIST="32 64" \
SHELL_SUBCOMM_TYPES="interlaced contiguous" \
SHELL_COARSE_LAYOUTS="active_layout repartitioned_dm" \
./submit_pmg_scaling.sh
```

## Submit

The default campaign runs:

- `1x128` and `2x128`
- P1 telescope active ranks `16,32,64`
- subcommunicators `contiguous,interlaced`
- `deflation=false,true`
- redundant P1 coarse comparisons with group sizes `16,32,64`

```bash
PARTITION=qcpu_exp TIME_LIMIT=00:30:00 ./submit_pmg_scaling.sh
```

The shell V-cycle backend is opt-in and uses
`options/pmg_shell_vcycle.opts`. For the first true coarse-level scaling
batch, run only the shell variants against the already-known best baseline.
`active_layout` is the current shell baseline; `repartitioned_dm` switches to
the real repartitioned coarse-DM/operator experiment:

```bash
PARTITION=qcpu_exp \
TIME_LIMIT=00:45:00 \
INCLUDE_TELESCOPE=0 \
INCLUDE_REDUNDANT=0 \
INCLUDE_SHELL=1 \
NODE_COUNTS="1 2" \
DEFLATION_LIST=true \
LINEAR_RTOL=1e-1 \
KSP_MAX_IT=200 \
SHELL_P2_ACTIVE_RANKS_LIST="64" \
SHELL_P1_ACTIVE_RANKS_LIST="32" \
SHELL_SUBCOMM_TYPES="interlaced" \
SHELL_COARSE_LAYOUTS="active_layout repartitioned_dm" \
./submit_pmg_scaling.sh
```

If that first four-job comparison is stable, repeat the same command with
`SHELL_P2_ACTIVE_RANKS_LIST="128"` to test the wider P2 active set.

For the best `1e-1` profile follow-up at tighter tolerance, narrow the lists
explicitly:

```bash
LINEAR_RTOL=1e-3 \
KSP_MAX_IT=500 \
DEFLATION_LIST=true \
NODE_COUNTS="2" \
ACTIVE_RANKS_LIST="32" \
SUBCOMM_TYPES="contiguous" \
INCLUDE_REDUNDANT=0 \
./submit_pmg_scaling.sh
```

## Collect

After jobs finish:

```bash
./collect_pmg_results.sh runs/<campaign-directory>
```

The collector writes `pmg_results.csv` with parsed `RESULT` fields, deflation
timings, PMG diagnostics, Slurm `MaxRSS`/`AveRSS`, aggregate memory estimates,
and PETSc log timings for:

```text
PCApply KSPSolve MatMult VecScatterEnd VecMDot KSPGMRESOrthog
MatPtAPNumeric MatPtAPSymbolic PCSetUp
```

Run artifacts stay under `standalone_petsc_p4_plasticity/karolina/runs/`, which
is ignored by git.
