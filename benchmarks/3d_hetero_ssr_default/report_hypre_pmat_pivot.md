# Hypre Explicit-Pmat Pivot

## Scope

This report records the post-BDDC pivot for the `3D / heterogeneous / P4 / rows / overlap / block_metis` path.

Goals:

1. keep one BDDC topology sanity check as a stop/go diagnostic
2. compare Hypre with explicit `Pmat` choices on frozen nonlinear states
3. run a minimal `pc_hypre_boomeramg_max_iter` sweep on the best reusable source
4. validate the winner on the nonlinear `step_max=1` continuation gate

All Hypre runs below use:

- `pc_backend=hypre`
- `pc_hypre_coarsen_type=HMIS`
- `pc_hypre_interp_type=ext+i`
- `native_ksp_type=fgmres` for the frozen-state probe
- `native_ksp_norm_type=unpreconditioned`
- `linear_tolerance=1e-1`

## BDDC Sanity Result

Artifact:

- `artifacts/bddc_topology_sanity_rank2_xyz_p2_v2`

Result:

- PETSc still reports `candidate faces = 0` on a deliberately geometric rank-2 `xyz` split with forced user-graph analysis
- PETSc coarse size stays `9`
- this keeps BDDC off the production track for the current distributed presentation

## Frozen-State Hypre Table

State sources:

- `S_easy`: first accepted saved state, label `step_1`
- `S_hard`: latest saved accepted state in the short source artifact, label `step_3`

Source artifact:

- `artifacts/p4_step4_memory_fix_smoke/data/petsc_run.npz`
- `artifacts/p4_step4_memory_fix_smoke/data/run_info.json`

Artifacts used:

- easy-state partial batch:
  - `artifacts/hypre_pmat_pivot/screen20_v2/data/summary.json`
- hard-state singles:
  - `artifacts/hypre_pmat_pivot/hard_tangent_single_v1/data/run_info.json`
  - `artifacts/hypre_pmat_pivot/hard_regularized_single_v1/data/run_info.json`
  - `artifacts/hypre_pmat_pivot/hard_elastic_single_v1/data/run_info.json`

Comparison metric:

- `setup_elapsed_s + solve_elapsed_s`
- outer iteration count
- final relative residual

| State | Pmat | Iterations | Setup [s] | Solve [s] | Setup+Solve [s] | Final relative residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| easy | tangent | 1 | 20.823 | 1.193 | 22.016 | 7.603e-2 |
| easy | regularized | 1 | 20.752 | 1.190 | 21.942 | 7.577e-2 |
| easy | elastic | 2 | 23.591 | 2.611 | 26.201 | 9.709e-2 |
| hard | tangent | 1 | 20.776 | 1.199 | 21.975 | 7.554e-2 |
| hard | regularized | 1 | 20.576 | 1.167 | 21.743 | 7.623e-2 |
| hard | elastic | 3 | 23.782 | 3.965 | 27.748 | 8.055e-2 |

Interpretation:

- `elastic` is clearly worse than both `tangent` and `regularized` on both frozen states
- `regularized` is effectively tied with `tangent`
- `regularized` is the only reusable source worth carrying to the next gate, but it is not an obvious improvement over `tangent`

## BoomerAMG Cycle Sweep

Artifact:

- `artifacts/hypre_pmat_pivot/hard_cycle_sweep_v1/data/summary.json`

Configuration:

- `S_hard`
- `Pmat in {tangent, regularized}`
- `pc_hypre_boomeramg_max_iter in {1, 2}`

| Pmat | BoomerAMG max_iter | Iterations | Setup [s] | Solve [s] | Setup+Solve [s] | Final relative residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tangent | 1 | 1 | 20.755 | 1.166 | 21.920 | 7.554e-2 |
| tangent | 2 | 1 | 20.664 | 2.348 | 23.013 | 4.829e-2 |
| regularized | 1 | 1 | 20.519 | 1.168 | 21.688 | 7.623e-2 |
| regularized | 2 | 1 | 20.702 | 2.391 | 23.093 | 4.922e-2 |

Interpretation:

- a second BoomerAMG cycle lowers the one-step residual, but it does not reduce outer iteration count
- `max_iter=2` increases solve time by about `1.2 s`
- the best practical setting remains `pc_hypre_boomeramg_max_iter=1`

## Nonlinear Step-Max-1 Gate

Reused tangent baseline:

- `artifacts/bddc_short_runs/p4_step1_hypre_current_v1/data/run_info.json`

New regularized run:

- `artifacts/hypre_pmat_pivot/p4_step1_regularized_v1/data/run_info.json`
- `artifacts/hypre_pmat_pivot/p4_step1_regularized_v1/data/petsc_run.npz`

Both runs use:

- rank `2`
- `P4`
- `rows`
- `overlap`
- `block_metis`
- `pc_backend=hypre`
- `pc_hypre_coarsen_type=HMIS`
- `pc_hypre_interp_type=ext+i`
- `pc_hypre_boomeramg_max_iter=1`
- `recycle_preconditioner=false`

| Metric | Tangent baseline | Regularized |
| --- | ---: | ---: |
| runtime [s] | 514.318 | 523.763 |
| step_count | 3 | 3 |
| final lambda | 1.1603609648336928 | 1.1603609648337303 |
| final omega | 6244975.721899381 | 6244975.721899383 |
| init linear iterations | 84 | 84 |
| init linear solve [s] | 87.978 | 89.048 |
| init linear preconditioner [s] | 193.779 | 198.393 |
| attempt linear iterations total | 66 | 66 |
| attempt linear solve total [s] | 70.080 | 70.323 |
| attempt linear preconditioner total [s] | 90.180 | 91.928 |
| total preconditioner setup [s] | 283.959 | 290.320 |
| total preconditioner apply [s] | 139.968 | 140.308 |

Interpretation:

- the reusable `regularized` preconditioner is correct
- it preserves the short continuation trajectory to solver precision
- it does not improve runtime
- it does not improve linear iteration count
- it slightly increases preconditioner setup time

## Implementation Notes

Two real probe/harness bugs were found and fixed during this pivot:

1. the native PETSc frozen-state helper returned only the local MPI slice of the solution vector and then treated it as a global vector in postprocessing
2. the frozen-state probe and batch runner were calling `KSPView()` from rank `0` only on MPI KSP objects; this parked runs inside PETSc `MatView/KSPView` collectives instead of actual solver work

Relevant code paths:

- `benchmarks/3d_hetero_ssr_default/probe_hypre_frozen.py`
- `src/slope_stability/linear/solver.py`
- `src/slope_stability/cli/run_3D_hetero_SSR_capture.py`
- `src/slope_stability/core/config.py`
- `src/slope_stability/core/run_config.py`

One remaining harness issue still exists:

- the multi-state batch runner segfaulted when transitioning from the completed easy-state cases into the hard-state cases
- isolated hard-state probes work correctly, so this is a batch lifecycle bug, not a solver-path bug

## Conclusion

The Hypre explicit-`Pmat` pivot is working, but it did not produce a better production setting on this branch.

Decision:

- keep `P = tangent` as the production default for Hypre
- do not promote `regularized` beyond research status for this `P4` path
- keep `elastic` off the track for Hypre reuse on nonlinear `P4`

The next worthwhile branch is not more `tangent / regularized / elastic` tuning. It is either:

1. a lower-order surrogate `Pmat` branch
2. a global `pc_type gamg` branch

BDDC remains research-only until the PETSc face-classification issue is resolved.
