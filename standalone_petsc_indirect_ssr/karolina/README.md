# Karolina Omega=7e6 SSR Scaling Grid

This harness runs the full indirect SSR continuation to `omega_max=7e6`.
The default batch is the current refined C split-smoother scaling check:
`refine_levels=1`, full-node `1x128` and `2x128`, and P1/GAMG coarse
`max_it=5`.

Default grid:

```text
1 node, 128 ranks      full occupancy
2 nodes, 256 ranks     full occupancy
```

Default cases:

```text
engine=c
profile=split
```

The C-only `split` profile runs the current local best-fit scalable variant,
`options/pmg_shell_split_smoother.opts`: shell V-cycle, P2/P1 active ranks
`64/32`, interlaced layout, GAMG coarse solve with max-it `5`, fine smoother
`5`, and P2 smoother `10`.

The submitter uses OpenMPI `mpiexec` for one-node jobs and Slurm
`srun --mpi=pmix_v4` for multi-node jobs. This avoids OpenMPI singleton
launches on Karolina while still using the MPI runtime linked into the PETSc
build.

Submit a dry run first:

```bash
cd standalone_petsc_indirect_ssr/karolina
DRY_RUN=1 ./submit_omega7_grid.sh
```

Submit the default grid:

```bash
./submit_omega7_grid.sh
```

Useful overrides:

```bash
PARTITION=qcpu_exp TIME_LIMIT=00:45:00 ./submit_omega7_grid.sh
LAYOUTS="1:64 1:128 2:128" ENGINES="c py" PROFILES="baseline petsc4py" REFINE_LEVELS=0 ./submit_omega7_grid.sh
BUILD_BEFORE_RUN=1 ENGINES=c PROFILES=baseline ./submit_omega7_grid.sh
```

Submit the petsc4py C-hotpath comparison against the already collected C
baseline artifacts. This runs full-node `1x128` and `2x128` jobs for both the
base L1 mesh and the uniformly refined L1 mesh:

```bash
./build_hotpath_extension.sh
./submit_petsc4py_hotpath_scaling.sh
```

The hotpath profile writes `mechanics_backend = "dmplex_c_hotpath"` into the
petsc4py config, then uses the same maintained C split-smoother path through
the Cython bridge: P2/P1 active ranks `64/32`, interlaced subcommunicators,
fine smoother max-it `5`, P2 smoother max-it `10`, and redundant P1/GAMG
coarse solve max-it `5`. Override `REFINE_LEVELS_LIST`, `LAYOUTS`, or the
`PMG_SHELL_*` variables only when intentionally testing a different profile.

Submit only the split-smoother C scalability check requested after local
validation:

```bash
LAYOUTS="1:128 2:128" ENGINES=c PROFILES=split REFINE_LEVELS=1 PMG_COARSE_MAX_IT=5 TIME_LIMIT=00:45:00 ./submit_omega7_grid.sh
```

Submit the low-rank `qcpu` campaign:

```bash
./submit_omega7_low_rank_qcpu.sh
```

This submits only `1:4` and `1:8` layouts on `qcpu`, keeping the refined split
profile, P1/GAMG coarse `max_it=5`, and the same profiling outputs. Override
`TIME_LIMIT` if queue policy or runtime requires it.

Collect after the jobs finish:

```bash
../../.venv/bin/python ./collect_omega7_results.py runs/ssr_omega7_grid_YYYYMMDD_HHMMSS
```

The collector writes:

```text
ssr_omega7_results.csv        one row per job, including memory and phase timings
ssr_omega7_steps.csv          init/accepted-step timing and iteration rows
ssr_omega7_petsc_events.csv   PETSc -log_view event rows when present
```

The C runs use `-log_view`, `/usr/bin/time -v`, `sacct`, `RESULT`,
`DEFLATION_TIMING`, and `PMG_SHELL_APPLY_SUMMARY`. The Python runs additionally
collect the structured `run_info.json` and `progress.jsonl` timing breakdowns,
including constitutive assembly, linear solve, orthogonalization, preconditioner
setup/apply, and manual PMG phase timings.
