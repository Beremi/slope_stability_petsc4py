# P4(L1) Secant Newton-Stopping Comparison

This compares the current default `slope_stability_3D_hetero_SSR_default` case with `elem_type = "P4"` and all other settings left unchanged using the standard secant predictor with the continuation stop forced to `omega = 6.7e+06`.

- Base case: `benchmarks/slope_stability_3D_hetero_SSR_default/case.toml`
- Problem override: `elem_type = "P4"`
- Predictor: `secant`
- Continuation stop: `omega_max = 6.7e+06`
- Residual-tolerance sweep: looser by `100x` (`1.0e-02`), looser by `10x` (`1.0e-03`), default (`1.0e-04`), with `r_min` fixed at `1.0e-04`.
- Additional case: stop on relative Newton correction `||alpha ΔU|| / ||U|| <= 1e-2` with residual `tol` kept at the default `1e-4`.
- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6`

## Summary

| Case | Residual tol | Stop criterion | Stop tol | Runtime [s] | Speedup vs default | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Init linear | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | `1.0e-02` | relative residual | `1.0e-02` | 3315.504 | 3.709 | 9 | 7 | 1.569503 | 6700000.0 | 10 | 39 | 58 | 1663 | 42.641 | 9.987e-03 | 8.750e-04 |
| Less precise x10 | `1.0e-03` | relative residual | `1.0e-03` | 7807.955 | 1.575 | 9 | 7 | 1.568077 | 6700000.0 | 13 | 98 | 82 | 3903 | 39.827 | 9.597e-04 | 1.004e-03 |
| Default | `1.0e-04` | relative residual | `1.0e-04` | 12297.639 | 1.000 | 9 | 7 | 1.567999 | 6700000.0 | 18 | 140 | 140 | 6274 | 44.814 | 9.657e-05 | 3.524e-04 |
| Relative correction 1e-2 | `1.0e-04` | relative correction | `1.0e-02` | 3062.527 | 4.016 | 9 | 7 | 1.572337 | 6700000.0 | 6 | 27 | 45 | 1585 | 58.704 | 2.274e-02 | 2.537e-03 |

## Accepted-Step Lambda

| Step | Less precise x100 lambda | Less precise x10 lambda | Default lambda | Relative correction 1e-2 lambda |
| --- | ---: | ---: | ---: | ---: |
| 3 | 1.161540 | 1.160372 | 1.160364 | 1.161997 |
| 4 | 1.246084 | 1.245974 | 1.245959 | 1.247952 |
| 5 | 1.312194 | 1.312253 | 1.312221 | 1.315436 |
| 6 | 1.417718 | 1.418011 | 1.417961 | 1.422168 |
| 7 | 1.503226 | 1.503203 | 1.503114 | 1.508930 |
| 8 | 1.568435 | 1.565509 | 1.565470 | 1.569128 |
| 9 | 1.569503 | 1.568077 | 1.567999 | 1.572337 |

## Accepted-Step Omega

| Step | Less precise x100 omega | Less precise x10 omega | Default omega | Relative correction 1e-2 omega |
| --- | ---: | ---: | ---: | ---: |
| 3 | 6244907.3 | 6244977.5 | 6244976.1 | 6245110.7 |
| 4 | 6273133.4 | 6273268.5 | 6273262.6 | 6273846.6 |
| 5 | 6301359.4 | 6301559.4 | 6301549.2 | 6302582.5 |
| 6 | 6357811.5 | 6358141.3 | 6358122.3 | 6360054.3 |
| 7 | 6414263.5 | 6414723.2 | 6414695.4 | 6417526.1 |
| 8 | 6527167.6 | 6527887.0 | 6527841.6 | 6532469.7 |
| 9 | 6700000.0 | 6700000.0 | 6700000.0 | 6700000.0 |

## Accepted-Step Newton Iterations

| Step | Less precise x100 Newton | Less precise x10 Newton | Default Newton | Relative correction 1e-2 Newton |
| --- | ---: | ---: | ---: | ---: |
| 3 | 2 | 5 | 7 | 1 |
| 4 | 3 | 6 | 10 | 2 |
| 5 | 3 | 8 | 11 | 1 |
| 6 | 5 | 8 | 11 | 2 |
| 7 | 5 | 10 | 15 | 4 |
| 8 | 10 | 32 | 35 | 13 |
| 9 | 11 | 29 | 51 | 4 |

## Accepted-Step Linear Iterations

| Step | Less precise x100 linear | Less precise x10 linear | Default linear | Relative correction 1e-2 linear |
| --- | ---: | ---: | ---: | ---: |
| 3 | 15 | 105 | 166 | 15 |
| 4 | 46 | 145 | 226 | 46 |
| 5 | 53 | 148 | 200 | 20 |
| 6 | 125 | 224 | 289 | 67 |
| 7 | 149 | 371 | 682 | 189 |
| 8 | 842 | 1744 | 2171 | 1035 |
| 9 | 433 | 1166 | 2540 | 213 |

## Accepted-Step Final Relative Correction

| Step | Less precise x100 ΔU/U | Less precise x10 ΔU/U | Default ΔU/U | Relative correction 1e-2 ΔU/U |
| --- | ---: | ---: | ---: | ---: |
| 3 | 0.006773 | 0.000266 | 0.000039 | 0.009536 |
| 4 | 0.003211 | 0.000327 | 0.000060 | 0.002998 |
| 5 | 0.004389 | 0.001070 | 0.000283 | 0.008820 |
| 6 | 0.006756 | 0.003218 | 0.000464 | 0.002486 |
| 7 | 0.003669 | 0.002025 | 0.000723 | 0.005325 |
| 8 | 0.010708 | 0.000989 | 0.000829 | 0.004751 |
| 9 | 0.000875 | 0.001004 | 0.000352 | 0.002537 |

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

| Less precise x100 | Less precise x10 | Default | Relative correction 1e-2 |
| --- | --- | --- | --- |
| ![Less precise x100 omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x100/plots/petsc_omega_lambda.png) | ![Less precise x10 omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x10/plots/petsc_omega_lambda.png) | ![Default omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/default/plots/petsc_omega_lambda.png) | ![Relative correction 1e-2 omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_omega_lambda.png) |

### Displacements

| Less precise x100 | Less precise x10 | Default | Relative correction 1e-2 |
| --- | --- | --- | --- |
| ![Less precise x100 displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x100/plots/petsc_displacements_3D.png) | ![Less precise x10 displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x10/plots/petsc_displacements_3D.png) | ![Default displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/default/plots/petsc_displacements_3D.png) | ![Relative correction 1e-2 displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_displacements_3D.png) |

### Deviatoric Strain

| Less precise x100 | Less precise x10 | Default | Relative correction 1e-2 |
| --- | --- | --- | --- |
| ![Less precise x100 deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x100/plots/petsc_deviatoric_strain_3D.png) | ![Less precise x10 deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x10/plots/petsc_deviatoric_strain_3D.png) | ![Default deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/default/plots/petsc_deviatoric_strain_3D.png) | ![Relative correction 1e-2 deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_deviatoric_strain_3D.png) |

### Step Displacement History

| Less precise x100 | Less precise x10 | Default | Relative correction 1e-2 |
| --- | --- | --- | --- |
| ![Less precise x100 step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x100/plots/petsc_step_displacement.png) | ![Less precise x10 step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/less_precise_x10/plots/petsc_step_displacement.png) | ![Default step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/default/plots/petsc_step_displacement.png) | ![Relative correction 1e-2 step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_step_displacement.png) |

## Accepted-Step Newton Solves

Each section below overlays the successful Newton solve that produced the accepted continuation step for every case that reached that step.

### Accepted Continuation Step 3

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | 1 | 2 | 51.568 | 1.161540 | 6244907.3 | 9.145e-03 | 6.773e-03 |
| Less precise x10 | 1 | 5 | 220.492 | 1.160372 | 6244977.5 | 9.821e-04 | 2.656e-04 |
| Default | 1 | 7 | 320.087 | 1.160364 | 6244976.1 | 4.746e-05 | 3.874e-05 |
| Relative correction 1e-2 | 1 | 1 | 47.398 | 1.161997 | 6245110.7 | 5.541e-02 | 9.536e-03 |

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
| Default | 1 | 10 | 418.080 | 1.245959 | 6273262.6 | 3.768e-05 | 6.016e-05 |
| Relative correction 1e-2 | 1 | 2 | 81.490 | 1.247952 | 6273846.6 | 1.993e-02 | 2.998e-03 |

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
| Default | 1 | 11 | 381.336 | 1.312221 | 6301549.2 | 9.293e-05 | 2.835e-04 |
| Relative correction 1e-2 | 1 | 1 | 36.012 | 1.315436 | 6302582.5 | 5.876e-02 | 8.820e-03 |

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
| Default | 1 | 11 | 535.827 | 1.417961 | 6358122.3 | 8.993e-05 | 4.636e-04 |
| Relative correction 1e-2 | 1 | 2 | 120.218 | 1.422168 | 6360054.3 | 2.543e-02 | 2.486e-03 |

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
| Default | 1 | 15 | 1252.342 | 1.503114 | 6414695.4 | 9.763e-05 | 7.231e-04 |
| Relative correction 1e-2 | 1 | 4 | 350.711 | 1.508930 | 6417526.1 | 1.484e-02 | 5.325e-03 |

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
| Default | 1 | 35 | 4085.154 | 1.565470 | 6527841.6 | 8.976e-05 | 8.289e-04 |
| Relative correction 1e-2 | 1 | 13 | 1928.995 | 1.569128 | 6532469.7 | 7.644e-03 | 4.751e-03 |

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
| Default | 1 | 51 | 4990.948 | 1.567999 | 6700000.0 | 9.657e-05 | 3.524e-04 |
| Relative correction 1e-2 | 1 | 4 | 381.245 | 1.572337 | 6700000.0 | 2.274e-02 | 2.537e-03 |

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
- Default: config `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/configs/default.toml`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/default`
- Relative correction 1e-2: config `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/configs/relative_correction_1e_2.toml`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/secant_newton_precision_p4_l1_omega6p7e6/runs/relative_correction_1e_2`
