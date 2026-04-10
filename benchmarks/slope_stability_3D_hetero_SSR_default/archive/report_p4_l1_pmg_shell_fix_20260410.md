# P4(L1) PMG-Shell Fix Continuation Report

## Executive Summary

- Scope: `P4(L1)` on `SSR_hetero_ada_L1.msh`, SSR indirect continuation, `mpi_ranks = 8`.
- The original artifact compared here is the pre-fix benchmark-local bundle that was snapshotted before rerun.
- The permanent PMG-shell fix is the parallel smoother switch to `chebyshev + jacobi` for shell hierarchies with level orders `(1, 2, 4)` or `(1, 1, 2)` at MPI size `> 1`.
- On this benchmark, the post-fix source-default run reduced full runtime from `1182.090 s` to `814.854 s` (31.1% lower wall time).
- The Armijo residual variant finished on essentially the same continuation path in `816.578 s` (30.9% lower wall time vs baseline), so it remains an experimental option rather than a new default.
- Only `8`-rank runs are compared here. The earlier `32`-rank evidence is context for the fix, not a rerun in this report.

## Configuration Summary

- Baseline and post-fix default share the same benchmark case: `solver_type = PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE`, `pc_backend = pmg_shell`, `elem_type = P4`, `omega_max_stop = 6.7e6`.
- The fixed default differs from the original artifact only in the PMG-shell smoother selection for the robust parallel shell case.
- The Armijo variant keeps the same fixed PMG-shell path and overrides only:
  - `newton.line_search = "armijo_residual"`
  - `newton.armijo_max_ls = 10`
  - `newton.armijo_rescale_trial_to_omega = true`
  - `newton.armijo_fallback_to_alg5 = true`
- In this implementation, the Armijo path is applied to the continuation Newton corrections; init remains on the existing path.

## What Changed In PMG-Shell

- Before the fix, the benchmark-local artifact used `richardson + sor` on the shell levels even for the parallel `P4` hierarchy with orders `(1, 2, 4)`.
- The permanent driver change now detects that robust parallel shell case and switches those levels to `chebyshev + jacobi` while keeping the coarse HYPRE solve and the rest of the continuation stack unchanged.

## Overall Metrics

| Variant | Runtime [s] | Vs baseline | Init lin | Cont. Newton | Cont. linear | LS total | Fallbacks | Max defl. basis | Final lambda | Final omega |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline artifact | 1182.090 | +0.0% | 95 | 70 | 1097 | 166 | 0 | 46 | 1.568634 | 6700000.0 |
| Post-fix source default | 814.854 | -31.1% | 58 | 59 | 733 | 118 | 0 | 46 | 1.568403 | 6700000.0 |
| Post-fix Armijo residual | 816.578 | -30.9% | 58 | 59 | 733 | 118 | 0 | 46 | 1.568403 | 6700000.0 |

## Accepted Continuation Steps

| Step | Omega [e6] | Lambda baseline | Lambda fixed | Lambda Armijo | Linear baseline | Linear fixed | Linear Armijo | Wall baseline [s] | Wall fixed [s] | Wall Armijo [s] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 6.245 | 1.159969 | 1.160467 | 1.160467 | 69 | 19 | 19 | 64.561 | 17.318 | 16.746 |
| 4 | 6.273 | 1.245518 | 1.245949 | 1.245949 | 21 | 42 | 42 | 23.035 | 41.735 | 41.796 |
| 5 | 6.302 | 1.311694 | 1.312334 | 1.312334 | 35 | 23 | 23 | 36.129 | 23.752 | 24.237 |
| 6 | 6.358 | 1.417284 | 1.418009 | 1.418009 | 73 | 68 | 68 | 79.138 | 62.074 | 61.563 |
| 7 | 6.415 | 1.502407 | 1.503175 | 1.503175 | 68 | 92 | 92 | 76.681 | 75.121 | 75.357 |
| 8 | 6.528 | 1.565674 | 1.565537 | 1.565537 | 535 | 345 | 345 | 479.882 | 324.204 | 325.327 |
| 9 | 6.700 | 1.568634 | 1.568403 | 1.568403 | 296 | 144 | 144 | 278.257 | 161.173 | 162.323 |

## Late-Step Focus

| Step | Metric | Baseline | Fixed default | Armijo |
| ---: | --- | ---: | ---: | ---: |
| 8 | Newton | 24 | 22 | 22 |
| 8 | Linear | 535 | 345 | 345 |
| 8 | Wall [s] | 479.882 | 324.204 | 325.327 |
| 9 | Newton | 17 | 15 | 15 |
| 9 | Linear | 296 | 144 | 144 |
| 9 | Wall [s] | 278.257 | 161.173 | 162.323 |

## Line Search And Deflation Diagnostics

| Variant | Continuation LS total | Fallback count | Max deflation basis | Final deflation basis |
| --- | ---: | ---: | ---: | ---: |
| Post-fix source default | 118 | 0 | 46 | 33 |
| Post-fix Armijo residual | 118 | 0 | 46 | 33 |

## Comparison Plots

### Accepted trajectory overlay

![Accepted trajectory overlay](../../../artifacts/comparisons/p4_l1_pmg_shell_fix_20260410/plots/trajectory_overlay.png)

### Step Newton iteration overlay

![Step Newton overlay](../../../artifacts/comparisons/p4_l1_pmg_shell_fix_20260410/plots/step_newton_overlay.png)

### Step linear iteration overlay

![Step linear overlay](../../../artifacts/comparisons/p4_l1_pmg_shell_fix_20260410/plots/step_linear_overlay.png)

### Step wall-time overlay

![Step wall overlay](../../../artifacts/comparisons/p4_l1_pmg_shell_fix_20260410/plots/step_wall_overlay.png)

## Recommendation

- Keep the PMG-shell smoother correction permanent for parallel `P4(L1)` source-default runs.
- Keep the rest of the source-default continuation path unchanged.
- Keep `armijo_residual` as an optional debug and experiment mode. On this benchmark it matches the fixed default path closely, but it does not buy a material runtime or iteration reduction.

## Artifact Locations

- Baseline snapshot: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/p4_l1_pmg_shell_fix_20260410/original_baseline`
- Post-fix source-default benchmark-local artifact: `/home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/artifacts/simulation`
- Post-fix Armijo run: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/p4_l1_pmg_shell_fix_20260410/armijo_run`
- Comparison root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/p4_l1_pmg_shell_fix_20260410`
