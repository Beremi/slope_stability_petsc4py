# P4 Preconditioner Screen: Partial Snapshot

- Date: `2026-03-16`
- Mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Element order: `P4`
- Tangent kernel: `rows`
- Constitutive mode: `overlap`
- Node ordering: `block_metis`
- `recycle_preconditioner = false`
- Screen target: rank `1` and rank `8`, `step_max = 2`
- Status: partial, because the GAMG screen was still in progress when this snapshot was written

## Implemented Solver Changes

- Explicit operator/preconditioner split in the PETSc MATLAB-style DFGMRES path.
- Lagged preconditioner policies with rebuild/reuse diagnostics.
- HYPRE/GAMG/BDDC backend plumbing exposed through config and CLI.
- MATIS + BDDC prototype path implemented and covered by serial/MPI smoke tests.
- Dedicated preconditioner benchmark harness added in `benchmarks/slope_stability_3D_hetero_SSR_default/archive/compare_preconditioners.py`.
- Baseline-relative runtime cutoffs added to the screen stage so obviously non-competitive candidates are pruned early.

## Validation Completed Before Screening

- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_solver_preconditioner_policies.py tests/test_petsc_matis_bddc_helpers.py tests/test_preconditioner_mpi.py tests/test_newton_cleanup.py tests/test_distributed_tangent_rows.py tests/test_owned_constitutive_exchange.py tests/test_owned_constitutive_exchange_mpi.py`
- Result: `29 passed`

## Completed Screen Results

| Variant | Rank | Runtime [s] | Attempt prec [s] | Attempt solve [s] | Rebuilds | Reuses | Age max | Peak RSS [GiB] | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hypre_current` | 1 | 813.593 | 139.390 | 109.863 | 22 | 0 | 0 | 23.541 | baseline complete |
| `hypre_current` | 8 | 300.276 | 44.124 | 45.940 | 23 | 0 | 0 | 28.742 | baseline complete |
| `hypre_lagged_current` | 1 | 629.173 | 19.944 | 259.777 | 2 | 20 | 15 | 24.585 | runtime win |
| `hypre_lagged_current` | 8 | 269.808 | 8.585 | 111.738 | 2 | 20 | 15 | 37.797 | memory disqualified |
| `hypre_lagged_complexity` | 1 | 455.548 | 3.344 | 213.070 | 2 | 20 | 14 | 29.094 | strong runtime win |

## Runtime-Cutoff Results

| Variant | Rank | Last event | Elapsed at stop [s] | Peak RSS [GiB] | Interpretation |
| --- | ---: | --- | ---: | ---: | --- |
| `hypre_lagged_pmis` | 1 | `runtime_cutoff` | 896.457 | 24.523 | slower than the allowed baseline-relative screen limit |
| `hypre_lagged_pmis` | 8 | `runtime_cutoff` | 330.670 | 28.566 | slower than the allowed baseline-relative screen limit |
| `hypre_lagged_complexity` | 8 | `runtime_cutoff` | 330.696 | 33.939 | runtime-cut and already above the rank-8 memory limit |

## Current Interpretation

- `hypre_current` provides the reference behavior for the new solver plumbing, but it is slower than the older no-recycle baseline previously recorded in `report_p2_vs_p4_rank8_final_memfix.md`.
- `hypre_lagged_current` proves that lagging the preconditioner can cut setup cost dramatically:
  - rank-1 attempt preconditioner time dropped from `139.390 s` to `19.944 s`
  - rank-8 attempt preconditioner time dropped from `44.124 s` to `8.585 s`
  - but rank-8 peak RSS rose from `28.742 GiB` to `37.797 GiB`, which breaks the `+5%` memory rule
- `hypre_lagged_pmis` did not survive the runtime screen at either rank.
- `hypre_lagged_complexity` is the most interesting variant so far:
  - rank-1 runtime improved from `813.593 s` to `455.548 s`
  - rank-1 attempt preconditioner time fell to `3.344 s`
  - rank-8 still failed the screen because it hit the runtime cutoff at `330.696 s` and had already reached `33.939 GiB` peak RSS

## Incomplete Work At Snapshot Time

- `gamg_lagged_lowmem` screen was still in progress and is not included in this snapshot.
- No AIJ variant had yet satisfied both the runtime and memory promotion rules.
- Because the screen was not finished, the following were not run from this snapshot:
  - promoted-candidate full step-2 scaling on `1/2/4/8`
  - BDDC rank-8 step-2 gate
  - new full-trajectory `8`-rank `P4` production run
  - final replacement report against the reused baseline

## Next Decision Point

- Finish the `gamg_lagged_lowmem` screen.
- If GAMG also fails the promotion rules, stop the sweep and conclude that the current solver-plumbing branch improved reuse mechanics but did not yet produce a drop-in replacement for the existing P4 production baseline under the strict memory rule.
- If GAMG passes, continue to the planned `1/2/4/8` step-2 scaling and then one full-trajectory run for the winner.
