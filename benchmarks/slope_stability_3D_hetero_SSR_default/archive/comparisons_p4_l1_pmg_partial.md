# P4(L1) PMG Newton-Stop Comparison (Partial Live)

This partial report is built from the live `progress.jsonl` stream of the current PMG `P4(L1)` rerun. No PMG less-precise variant has finished yet, so this report currently contains the in-progress `default` case only.

- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6`
- Progress source: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/default/data/progress.jsonl`
- Solver family: `P4(L1)`, `secant`, `pc_backend = "pmg_shell"`
- Requested comparison order: default, 100x less precision, relative correction `1e-2`, `|Δlambda| < 1e-2`, `|Δlambda| < 1e-3`, `|Δlambda| < 1e-3` + initial-segment step-length cap
- Last progress event: `2026-03-31T04:39:09.796647+00:00`

## Live Status

| Case | Status | Runtime so far [s] | Accepted states | Accepted continuation steps | Last accepted lambda | Last accepted omega | Current target step | Current lambda | Current omega | Current relres | Current `ΔU/U` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | running | 1570.720 | 8 | 6 | 1.565464 | 6527843.2 | 9 | 1.568211 | 6700000.0 | 1.671e-03 | 1.275e-03 |

## Accepted-Step Summary

| Case | Residual tol | Stop criterion | Stop tol | Init Newton | Accepted continuation Newton | Init linear | Accepted continuation linear | Linear / Newton |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | `1.0e-04` | relative residual | `1.0e-04` | 20 | 82 | 159 | 1217 | 14.841 |

### Accepted-Step Lambda

| Step | Lambda |
| --- | ---: |
| 3 | 1.160363 |
| 4 | 1.245960 |
| 5 | 1.312221 |
| 6 | 1.417962 |
| 7 | 1.503115 |
| 8 | 1.565464 |

### Accepted-Step Omega

| Step | Omega |
| --- | ---: |
| 3 | 6244976.2 |
| 4 | 6273262.9 |
| 5 | 6301549.6 |
| 6 | 6358123.0 |
| 7 | 6414696.4 |
| 8 | 6527843.2 |

### Accepted-Step Newton Iterations

| Step | Newton | Linear | Step wall [s] | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 8 | 57 | 50.036 | 2.378e-05 | 4.248e-05 |
| 4 | 8 | 59 | 51.245 | 6.882e-05 | 1.454e-04 |
| 5 | 11 | 87 | 78.655 | 1.449e-05 | 3.186e-05 |
| 6 | 9 | 78 | 66.223 | 6.581e-05 | 3.768e-04 |
| 7 | 14 | 130 | 117.100 | 4.840e-05 | 5.489e-04 |
| 8 | 32 | 806 | 579.713 | 6.859e-05 | 9.481e-04 |

## Comparison Plots

### Lambda vs omega

![Lambda vs omega](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/lambda_omega_overlay.png)

### Lambda by accepted state

![Lambda by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/lambda_vs_state.png)

### Omega by accepted state

![Omega by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/omega_vs_state.png)

### Runtime by case

![Runtime by case](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/runtime_by_case.png)

### Timing breakdown

![Timing breakdown](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/timing_breakdown_stacked.png)

### Newton iterations per step

![Newton iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/step_newton_iterations.png)

### Linear iterations per step

![Linear iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/step_linear_iterations.png)

### Linear per Newton

![Linear per Newton](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/step_linear_per_newton.png)

### Wall time per step

![Wall time per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/step_wall_time.png)

### Final relative residual per step

![Final relative residual per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/step_relres_end.png)

### Final relative correction per step

![Final relative correction per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/step_relcorr_end.png)

## Per-Run Plot Availability

The usual final per-run images are not available yet because the PMG rerun has not finished writing `petsc_run.npz` and the final plot bundle.

- Available now: [Continuation curve](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/lambda_omega_overlay.png)
- Not available until completion: `petsc_displacements_3D.png`, `petsc_deviatoric_strain_3D.png`, `petsc_step_displacement.png`

## Newton Solves

Accepted steps below use the successful Newton solve that produced the accepted continuation step. The last section is the current live Newton trace of the in-progress step.

### Accepted Continuation Step 3

| Case | Attempt in step | Newton iterations | Step wall [s] | Lambda | Omega | Relres | `ΔU/U` | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 1 | 8 | 50.036 | 1.160363 | 6244976.2 | 2.378e-05 | 4.248e-05 | accepted |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_03/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 4

| Case | Attempt in step | Newton iterations | Step wall [s] | Lambda | Omega | Relres | `ΔU/U` | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 1 | 8 | 51.245 | 1.245960 | 6273262.9 | 6.882e-05 | 1.454e-04 | accepted |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_04/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 5

| Case | Attempt in step | Newton iterations | Step wall [s] | Lambda | Omega | Relres | `ΔU/U` | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 1 | 11 | 78.655 | 1.312221 | 6301549.6 | 1.449e-05 | 3.186e-05 | accepted |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_05/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 6

| Case | Attempt in step | Newton iterations | Step wall [s] | Lambda | Omega | Relres | `ΔU/U` | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 1 | 9 | 66.223 | 1.417962 | 6358123.0 | 6.581e-05 | 3.768e-04 | accepted |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_06/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 7

| Case | Attempt in step | Newton iterations | Step wall [s] | Lambda | Omega | Relres | `ΔU/U` | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 1 | 14 | 117.100 | 1.503115 | 6414696.4 | 4.840e-05 | 5.489e-04 | accepted |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_07/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 8

| Case | Attempt in step | Newton iterations | Step wall [s] | Lambda | Omega | Relres | `ΔU/U` | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 1 | 32 | 579.713 | 1.565464 | 6527843.2 | 6.859e-05 | 9.481e-04 | accepted |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_08/relative_increment_vs_criterion.png) |  |

### Continuation Step 9 (In Progress)

| Case | Attempt in step | Newton iterations | Step wall [s] | Lambda | Omega | Relres | `ΔU/U` | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default | 1 | 27 | 460.463 | 1.568211 | 6700000.0 | 1.671e-03 | 1.275e-03 | in_progress |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report_partial_live/plots/newton_by_step/step_09/relative_increment_vs_criterion.png) |  |

## Artifacts

- Live default run: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/default`
- Live progress file: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/default/data/progress.jsonl`
