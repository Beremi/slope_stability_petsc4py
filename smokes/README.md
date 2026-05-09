# Smoke Checks

Tiny checks for validating a freshly built local or cluster environment before
running expensive benchmark jobs.

## Environment Smoke

Run once in serial:

```bash
./.venv/bin/python smokes/environment_smoke.py
```

Run with a couple of MPI ranks:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
mpiexec -n 2 ./.venv/bin/python smokes/environment_smoke.py --min-mpi-size 2
```

The smoke imports the main runtime dependencies, creates tiny PETSc systems,
checks the asset registry, and reports optional PETSc package probes for HYPRE
and MUMPS.

Use these flags when a cluster environment is expected to provide them:

```bash
./.venv/bin/python smokes/environment_smoke.py --require-hypre --require-mumps
```

## Tiny Case Smoke

Run a one-step serial mechanics case using the smallest 2D homogeneous asset and
PETSc LU:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
./.venv/bin/python smokes/tiny_case_smoke.py
```

Outputs are written under `artifacts/smokes/tiny_2d_homo_ssr/` by default.

On Slurm, keep the case smoke as a one-rank job; use the environment smoke for
multi-rank MPI validation.
