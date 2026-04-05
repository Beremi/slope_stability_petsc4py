# P4(L1) Less-Precise Secant Newton-Stopping Comparison

This compares the current default `slope_stability_3D_hetero_SSR_default` case with `elem_type = "P4"` and all other settings left unchanged using the standard secant predictor with the continuation stop forced to `omega = 6.7e+06`.

- Base case: `benchmarks/slope_stability_3D_hetero_SSR_default/case.toml`
- Problem override: `elem_type = "P4"`
- Predictor: `secant`
- Continuation stop: `omega_max = 6.7e+06`
- Residual-tolerance sweep: looser by `100x` (`1.0e-02`), looser by `10x` (`1.0e-03`), with `r_min` fixed at `1.0e-04`.
- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6`

## Summary

| Case | Residual tol | Stop criterion | Stop tol | Runtime [s] | Speedup | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Init linear | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | `1.0e-02` | relative residual | `1.0e-02` | 3315.504 | 1.000 | 9 | 7 | 1.569503 | 6700000.0 | 10 | 39 | 58 | 1663 | 42.641 | 9.987e-03 | 8.750e-04 |
| Less precise x10 | `1.0e-03` | relative residual | `1.0e-03` | 7807.955 | 1.000 | 9 | 7 | 1.568077 | 6700000.0 | 13 | 98 | 82 | 3903 | 39.827 | 9.597e-04 | 1.004e-03 |

## Accepted-Step Lambda

| Step | Less precise x100 lambda | Less precise x10 lambda |
| --- | ---: | ---: |
| 3 | 1.161540 | 1.160372 |
| 4 | 1.246084 | 1.245974 |
| 5 | 1.312194 | 1.312253 |
| 6 | 1.417718 | 1.418011 |
| 7 | 1.503226 | 1.503203 |
| 8 | 1.568435 | 1.565509 |
| 9 | 1.569503 | 1.568077 |

## Accepted-Step Omega

| Step | Less precise x100 omega | Less precise x10 omega |
| --- | ---: | ---: |
| 3 | 6244907.3 | 6244977.5 |
| 4 | 6273133.4 | 6273268.5 |
| 5 | 6301359.4 | 6301559.4 |
| 6 | 6357811.5 | 6358141.3 |
| 7 | 6414263.5 | 6414723.2 |
| 8 | 6527167.6 | 6527887.0 |
| 9 | 6700000.0 | 6700000.0 |

## Accepted-Step Newton Iterations

| Step | Less precise x100 Newton | Less precise x10 Newton |
| --- | ---: | ---: |
| 3 | 2 | 5 |
| 4 | 3 | 6 |
| 5 | 3 | 8 |
| 6 | 5 | 8 |
| 7 | 5 | 10 |
| 8 | 10 | 32 |
| 9 | 11 | 29 |

## Accepted-Step Linear Iterations

| Step | Less precise x100 linear | Less precise x10 linear |
| --- | ---: | ---: |
| 3 | 15 | 105 |
| 4 | 46 | 145 |
| 5 | 53 | 148 |
| 6 | 125 | 224 |
| 7 | 149 | 371 |
| 8 | 842 | 1744 |
| 9 | 433 | 1166 |

## Accepted-Step Final Relative Correction

| Step | Less precise x100 ΔU/U | Less precise x10 ΔU/U |
| --- | ---: | ---: |
| 3 | 0.006773 | 0.000266 |
| 4 | 0.003211 | 0.000327 |
| 5 | 0.004389 | 0.001070 |
| 6 | 0.006756 | 0.003218 |
| 7 | 0.003669 | 0.002025 |
| 8 | 0.010708 | 0.000989 |
| 9 | 0.000875 | 0.001004 |

## Comparison Plots

### Lambda vs omega

![Lambda vs omega](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/lambda_omega_overlay.png)

### Lambda by accepted state

![Lambda by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/lambda_vs_state.png)

### Omega by accepted state

![Omega by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/omega_vs_state.png)

### Runtime by case

![Runtime by case](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/runtime_by_case.png)

### Timing breakdown

![Timing breakdown](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/timing_breakdown_stacked.png)

### Newton iterations per step

![Newton iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/step_newton_iterations.png)

### Linear iterations per step

![Linear iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/step_linear_iterations.png)

### Linear per Newton

![Linear per Newton](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/step_linear_per_newton.png)

### Wall time per step

![Wall time per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/step_wall_time.png)

### Final relative residual per step

![Final relative residual per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/step_relres_end.png)

### Final relative correction per step

![Final relative correction per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/step_relcorr_end.png)

## Existing Per-Run Plots

### Continuation Curve

| Less precise x100 | Less precise x10 |
| --- | --- |
| ![Less precise x100 omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x100/plots/petsc_omega_lambda.png) | ![Less precise x10 omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x10/plots/petsc_omega_lambda.png) |

### Displacements

| Less precise x100 | Less precise x10 |
| --- | --- |
| ![Less precise x100 displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x100/plots/petsc_displacements_3D.png) | ![Less precise x10 displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x10/plots/petsc_displacements_3D.png) |

### Deviatoric Strain

| Less precise x100 | Less precise x10 |
| --- | --- |
| ![Less precise x100 deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x100/plots/petsc_deviatoric_strain_3D.png) | ![Less precise x10 deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x10/plots/petsc_deviatoric_strain_3D.png) |

### Step Displacement History

| Less precise x100 | Less precise x10 |
| --- | --- |
| ![Less precise x100 step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x100/plots/petsc_step_displacement.png) | ![Less precise x10 step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x10/plots/petsc_step_displacement.png) |

## Accepted-Step Newton Solves

Each section below overlays the successful Newton solve that produced the accepted continuation step for every case that reached that step.

### Accepted Continuation Step 3

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | 1 | 2 | 51.568 | 1.161540 | 6244907.3 | 9.145e-03 | 6.773e-03 |
| Less precise x10 | 1 | 5 | 220.492 | 1.160372 | 6244977.5 | 9.821e-04 | 2.656e-04 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_03/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 4

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | 1 | 3 | 87.668 | 1.246084 | 6273133.4 | 7.404e-03 | 3.211e-03 |
| Less precise x10 | 1 | 6 | 275.658 | 1.245974 | 6273268.5 | 3.490e-04 | 3.271e-04 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_04/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 5

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | 1 | 3 | 100.736 | 1.312194 | 6301359.4 | 5.976e-03 | 4.389e-03 |
| Less precise x10 | 1 | 8 | 290.284 | 1.312253 | 6301559.4 | 5.299e-04 | 1.070e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_05/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 6

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | 1 | 5 | 235.021 | 1.417718 | 6357811.5 | 8.920e-03 | 6.756e-03 |
| Less precise x10 | 1 | 8 | 432.350 | 1.418011 | 6358141.3 | 8.614e-04 | 3.218e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_06/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 7

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | 1 | 5 | 279.504 | 1.503226 | 6414263.5 | 7.205e-03 | 3.669e-03 |
| Less precise x10 | 1 | 10 | 713.116 | 1.503203 | 6414723.2 | 6.189e-04 | 2.025e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_07/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 8

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | 1 | 10 | 1580.294 | 1.568435 | 6527167.6 | 8.737e-03 | 1.071e-02 |
| Less precise x10 | 1 | 32 | 3411.940 | 1.565509 | 6527887.0 | 6.041e-04 | 9.891e-04 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_08/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 9

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | 1 | 11 | 826.339 | 1.569503 | 6700000.0 | 9.987e-03 | 8.750e-04 |
| Less precise x10 | 1 | 29 | 2255.973 | 1.568077 | 6700000.0 | 9.597e-04 | 1.004e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/report/plots/newton_by_step/step_09/relative_increment_vs_criterion.png) |  |

## Run Artifacts

- Less precise x100: config `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/configs/less_precise_x100.toml`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x100`
- Less precise x10: config `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/configs/less_precise_x10.toml`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x10`
