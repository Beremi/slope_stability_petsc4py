# Karolina Scaling Runs

These scripts submit one Slurm job per elasticity case, preconditioner, and MPI
rank count on IT4I Karolina CPU nodes. They assume the repository has already
been pushed, then pulled on Karolina login storage.

Defaults match the current IT4I Karolina CPU allocation:

```text
ACCOUNT=fta-26-40
QOS=3571_6328
PARTITION=qcpu_exp
NODE_CORES=128
```

Override any of them from the environment. Use `qcpu_exp` for short validation
and `qcpu` for the full campaign if the queue policy requires it.

## Login Preparation

On Karolina, after pulling the pushed branch:

```bash
cd /path/to/slope_stability_petsc4py
export PETSC_DIR=/path/to/petsc-3.24.5
export PETSC_ARCH=linux-c-opt
standalone_petsc_p4_plasticity/p4_elasticity/karolina/prepare_login.sh
```

`prepare_login.sh` only checks the tree and builds the two elasticity binaries.
Set `GIT_UPDATE=1` if you want it to run `git fetch` and `git pull --ff-only`
before building.

## Dry Run

Preview the Slurm commands:

```bash
cd standalone_petsc_p4_plasticity/p4_elasticity/karolina
DRY_RUN=1 \
CASES="cube l1" \
VARIANTS="gamg bddc fetidp pmg" \
RANKS="16 32 64 128 256" \
PARTITION=qcpu_exp \
./submit_scaling.sh
```

You can also ask Slurm to validate without submitting:

```bash
SBATCH_TEST_ONLY=1 RANKS="16" CASES="l1" VARIANTS="pmg" ./submit_scaling.sh
```

## Submit

Short validation:

```bash
PARTITION=qcpu_exp \
TIME_LIMIT=00:10:00 \
RANKS="16" \
CASES="l1" \
VARIANTS="pmg" \
./submit_scaling.sh
```

One-node and two-node scaling campaign:

```bash
PARTITION=qcpu \
TIME_LIMIT=00:10:00 \
RANKS="16 32 64 128 256" \
CASES="cube l1" \
VARIANTS="gamg bddc fetidp pmg" \
./submit_scaling.sh
```

Each submitted combination is one Slurm job. Ranks up to 128 use one Karolina
CPU node; 256 ranks use two nodes. The submitter writes a manifest to
`runs/<timestamp>/submitted_jobs.csv`.

Useful overrides:

```text
CUBE_FACES=16,16,16
L1_BC_MODE=rollers
CUBE_PARTITIONER=parmetis
L1_PARTITIONER=parmetis
KSP_RTOL=1e-3
KSP_MAX_IT=300
PMG_GROUP_SIZE=64
EXTRA_PETSC_OPTIONS="-log_view"
```

## Collect

After jobs finish:

```bash
./collect_results.sh runs/<campaign-directory>
```

The collector writes `summary.csv` with the parsed `RESULT` line and Slurm
accounting fields. `MaxRSS` and `AveRSS` come directly from `sacct`; the
collector also adds GiB conversions and `approx_total_averss_gib = AveRSS *
ranks` as a quick aggregate-memory estimate. Raw logs, PETSc output, command
lines, environment snapshots, and `sacct` output are kept under
`runs/<campaign>/results/<job...>/`.
