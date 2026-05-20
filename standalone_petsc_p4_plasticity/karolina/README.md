# Karolina PMG Plasticity Runs

This harness submits and collects full-node runs for the standalone pure
PETSc plasticity executable:

```text
standalone_petsc_p4_plasticity/p4_plasticity
```

The default campaign is the maintained refined-L1 PMG shell V-cycle baseline:

```text
OPTIONS_FILE=options/pmg_shell_vcycle.opts
PMG_APPLY_BACKEND=shell_vcycle
PMG_SHELL_P2_ACTIVE_RANKS=64
PMG_SHELL_P1_ACTIVE_RANKS=32
PMG_SHELL_SUBCOMM_TYPE=interlaced
DEFLATION=true
REFINE_LEVELS=1
LAMBDA=1.5
LINEAR_RTOL=1e-1
KSP_MAX_IT=200
PARTITIONER=parmetis
NODE_CORES=128
NODE_COUNTS="1 2"
```

The older PCMG telescope and redundant comparisons remain available through
explicit `INCLUDE_TELESCOPE=1` and `INCLUDE_REDUNDANT=1` overrides, but they are
not submitted by default.

## Prepare

On Karolina, pull the branch and build the standalone solver:

```bash
cd /path/to/slope_stability_petsc4py
export PETSC_DIR=/path/to/petsc-3.24.5
export PETSC_ARCH=linux-c-opt
make -C standalone_petsc_p4_plasticity
```

## Preview

The default dry run should show only the two baseline shell jobs: `1x128` and
`2x128`, both with deflation enabled.

```bash
cd standalone_petsc_p4_plasticity/karolina
DRY_RUN=1 ./submit_pmg_scaling.sh
```

To preview legacy PCMG P1 telescope comparisons:

```bash
DRY_RUN=1 \
INCLUDE_SHELL=0 \
INCLUDE_TELESCOPE=1 \
INCLUDE_REDUNDANT=0 \
ACTIVE_RANKS_LIST="16 32 64" \
SUBCOMM_TYPES="contiguous interlaced" \
DEFLATION_LIST="false true" \
./submit_pmg_scaling.sh
```

## Submit

Submit the maintained baseline:

```bash
PARTITION=qcpu_exp TIME_LIMIT=00:45:00 ./submit_pmg_scaling.sh
```

For an explicit rerun of only the canonical two-node reference:

```bash
PARTITION=qcpu_exp \
TIME_LIMIT=00:45:00 \
NODE_COUNTS="2" \
DEFLATION_LIST=true \
LINEAR_RTOL=1e-1 \
KSP_MAX_IT=200 \
INCLUDE_TELESCOPE=0 \
INCLUDE_REDUNDANT=0 \
INCLUDE_SHELL=1 \
SHELL_P2_ACTIVE_RANKS_LIST=64 \
SHELL_P1_ACTIVE_RANKS_LIST=32 \
SHELL_SUBCOMM_TYPES=interlaced \
./submit_pmg_scaling.sh
```

For the best `1e-1` profile follow-up at tighter tolerance:

```bash
PARTITION=qcpu_exp \
TIME_LIMIT=01:00:00 \
NODE_COUNTS="2" \
DEFLATION_LIST=true \
LINEAR_RTOL=1e-3 \
KSP_MAX_IT=500 \
INCLUDE_TELESCOPE=0 \
INCLUDE_REDUNDANT=0 \
INCLUDE_SHELL=1 \
SHELL_P2_ACTIVE_RANKS_LIST=64 \
SHELL_P1_ACTIVE_RANKS_LIST=32 \
SHELL_SUBCOMM_TYPES=interlaced \
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
