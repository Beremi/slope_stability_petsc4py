# Karolina Omega=7e6 SSR Scaling Grid

This harness runs the full indirect SSR continuation to `omega_max=7e6` for
the standalone C solver and the petsc4py solver, using both the maintained C
PMG baseline and the petsc4py-like PMG profile.

Default grid:

```text
1 node, 64 ranks       half occupancy
1 node, 128 ranks      full occupancy
2 nodes, 256 ranks     full occupancy
```

Default cases:

```text
engine=c|py
profile=baseline|petsc4py
```

The C-only `split` profile runs the current local best-fit scalable variant,
`options/pmg_shell_split_smoother.opts`: shell V-cycle, P2/P1 active ranks
`64/32`, interlaced layout, GAMG coarse solve, fine smoother `5`, and P2
smoother `10`.

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
PARTITION=qcpu_exp TIME_LIMIT=02:00:00 ./submit_omega7_grid.sh
LAYOUTS="1:64 1:128 2:128" ENGINES="c py" PROFILES="baseline petsc4py" ./submit_omega7_grid.sh
BUILD_BEFORE_RUN=1 ENGINES=c PROFILES=baseline ./submit_omega7_grid.sh
```

Submit only the split-smoother C scalability check requested after local
validation:

```bash
LAYOUTS="1:128 2:128" ENGINES=c PROFILES=split TIME_LIMIT=02:00:00 ./submit_omega7_grid.sh
```

Collect after the jobs finish:

```bash
./collect_omega7_results.py runs/ssr_omega7_grid_YYYYMMDD_HHMMSS
```

The collector writes:

```text
ssr_omega7_results.csv        one row per job
ssr_omega7_steps.csv          init/accepted-step timing rows
ssr_omega7_petsc_events.csv   PETSc -log_view event rows when present
```

The C runs use `-log_view`, `/usr/bin/time -v`, `sacct`, `RESULT`,
`DEFLATION_TIMING`, and `PMG_SHELL_APPLY_SUMMARY`. The Python runs additionally
collect the structured `run_info.json` and `progress.jsonl` timing breakdowns,
including constitutive assembly, linear solve, orthogonalization, preconditioner
setup/apply, and manual PMG phase timings.
