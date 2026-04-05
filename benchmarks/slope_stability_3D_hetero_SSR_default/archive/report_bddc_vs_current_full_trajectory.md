# BDDC vs Current Full-Trajectory Status

## Baseline Reuse

The current production baseline was reused and not rerun:

- Summary: `artifacts/p2_p4_compare_rank8_final_memfix/summary.json`
- Narrative report: `benchmarks/slope_stability_3D_hetero_SSR_default/archive/report_p2_vs_p4_rank8_final_memfix.md`

Those remain the reference full trajectories.

## What Changed In This Recovery Cycle

The BDDC branch moved from “not starting reliably” to “working on `P2` with a contained elastic `P = K_elast` path”.

The important fix was not another PETSc option tweak. It was solver reuse:

- the elastic `P` matrix was constant,
- but the solver rebuilt it every Newton call,
- so the first `P2 step_max=10` BDDC run wasted `325.291 s` in setup and rebuilt `75` times.

After treating `preconditioner_matrix_source=elastic` as static after the first build:

- the same `P2 step_max=10` BDDC candidate dropped to `77.764 s`,
- `preconditioner_setup_time_total` dropped to `8.803 s`,
- rebuild count dropped to `2`,
- reuse count rose to `73`.

## Current BDDC Status By Stage

### P2 elastic probe

- contained elastic `MATIS + PCBDDC` probes now complete successfully
- see `benchmarks/slope_stability_3D_hetero_SSR_default/archive/report_bddc_elastic_probe.md`

### P2 short continuation

- Hypre control: completed
- BDDC elastic reuse candidate: completed
- accepted-state count matched Hypre
- runtime improved versus Hypre on this short case
- peak RSS was still much higher than Hypre
- see `benchmarks/slope_stability_3D_hetero_SSR_default/archive/report_bddc_short_runs.md`

### P4 elastic probe

- the branch does not yet clear even the contained elastic `P4` probe
- rank-2 `P4` probe aborted with PETSc `SIGSEGV` before first progress
- rank-1 `P4` backtrace shows the process in SciPy `csr_sort_indices`, not PETSc BDDC setup

That means the next `P4` blocker is the local high-order elastic assembly / CSR path upstream of PETSc BDDC.

## Reused Current Full-Trajectory Baseline

| Case | Runtime [s] | Accepted states | Final lambda | Final omega | Final Umax | Linear preconditioner [s] | Linear solve [s] | Peak RSS [GiB] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current `P2` baseline | 108.002 | 14 | 1.666089658 | 12000000.000000000 | 132.839467255 | 6.420 | 70.694 | not tracked here |
| Current `P4` baseline | 5542.992 | 12 | 1.570672856 | 9922242.674347960 | 83.781198088 | 2038.534 | 2460.268 | 42.676 |

## Full-Trajectory Comparison Status

No new BDDC full trajectory was run.

Reason:

- `P2` short continuation is now working, but still misses the strict memory gate
- `P4` does not yet pass the contained elastic probe
- without a clean `P4` elastic probe, there is no justified path to a new `P4` short run or full trajectory

So there is still no new row to place against the reused full-trajectory baseline.

## Conclusion

The BDDC branch is materially better than it was:

- `P2` elastic probe works
- `P2` short continuation works
- the main runtime pathology from repeated elastic-`P` rebuilds is fixed

But it is still not ready to replace the current production Hypre path for `P4`.

The blocking issue has moved:

- it is no longer the original BDDC startup/rebuild logic on `P2`
- it is now the `P4` local elastic matrix construction / CSR sorting path before PETSc BDDC setup

Current recommendation:

- keep current Hypre as the production preconditioner path
- keep the elastic-first BDDC branch as a working `P2` prototype
- do not run a new `P4` full trajectory until the `P4` elastic probe is made stable and memory-bounded
