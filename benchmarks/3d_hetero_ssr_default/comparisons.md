# Secant Newton-Precision Comparison

This compares the current default `3d_hetero_ssr_default` case using the standard secant predictor with the continuation stop forced to `omega = 6.7e6`.

- Base case: `benchmarks/3d_hetero_ssr_default/case.toml`
- Predictor: `secant`
- Continuation stop: `omega_max = 6.7e6`
- Residual-tolerance sweep: looser by `100x` and `10x` (`1.0e-02`, `1.0e-03`), default (`1.0e-04`), and tighter by `10x` and `100x` (`1.0e-05`, `1.0e-06`), with `r_min` fixed at `1.0e-04`.
- Additional case: stop on relative Newton correction `||alpha ΔU|| / ||U|| <= 1e-2` with residual `tol` kept at the default `1e-4`.
- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6`

## Summary

| Case | Residual tol | Stop criterion | Stop tol | Runtime [s] | Speedup vs default | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Init linear | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Less precise x100 | `1.0e-02` | relative residual | `1.0e-02` | 59.315 | 2.965 | 10 | 8 | 1.637656 | 6700000.0 | 7 | 22 | 20 | 448 | 20.364 | 8.209e-03 | 6.986e-02 |
| Less precise x10 | `1.0e-03` | relative residual | `1.0e-03` | 110.843 | 1.587 | 10 | 8 | 1.630338 | 6700000.0 | 11 | 39 | 44 | 861 | 22.077 | 7.127e-04 | 2.002e-03 |
| Default | `1.0e-04` | relative residual | `1.0e-04` | 175.877 | 1.000 | 10 | 8 | 1.629970 | 6700000.0 | 13 | 61 | 61 | 1353 | 22.180 | 5.162e-05 | 9.596e-04 |
| More precise x10 | `1.0e-05` | relative residual | `1.0e-05` | 238.312 | 0.738 | 10 | 8 | 1.629963 | 6700000.0 | 15 | 75 | 76 | 1861 | 24.813 | 3.670e-06 | 1.441e-03 |
| More precise x100 | `1.0e-06` | relative residual | `1.0e-06` | 293.074 | 0.600 | 10 | 8 | 1.629960 | 6700000.0 | 17 | 86 | 93 | 2305 | 26.802 | 5.838e-07 | 1.820e-04 |
| Relative correction 1e-2 | `1.0e-04` | relative correction | `1.0e-02` | 89.844 | 1.958 | 10 | 8 | 1.631459 | 6700000.0 | 7 | 22 | 33 | 710 | 32.273 | 1.591e-03 | 2.906e-03 |

## Accepted-Step Lambda

| Step | Less precise x100 lambda | Less precise x10 lambda | Default lambda | More precise x10 lambda | More precise x100 lambda | Relative correction 1e-2 lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.161674 | 1.159894 | 1.159862 | 1.159862 | 1.159862 | 1.160668 |
| 4 | 1.246624 | 1.245010 | 1.244961 | 1.244960 | 1.244960 | 1.245313 |
| 5 | 1.314316 | 1.311467 | 1.311412 | 1.311411 | 1.311411 | 1.312483 |
| 6 | 1.422474 | 1.418875 | 1.418746 | 1.418743 | 1.418743 | 1.419014 |
| 7 | 1.510811 | 1.505854 | 1.505724 | 1.505719 | 1.505719 | 1.506617 |
| 8 | 1.639524 | 1.609228 | 1.608521 | 1.608514 | 1.608512 | 1.608903 |
| 9 | 1.648030 | 1.625872 | 1.625521 | 1.625518 | 1.625513 | 1.626842 |
| 10 | 1.637656 | 1.630338 | 1.629970 | 1.629963 | 1.629960 | 1.631459 |

## Accepted-Step Omega

| Step | Less precise x100 omega | Less precise x10 omega | Default omega | More precise x10 omega | More precise x100 omega | Relative correction 1e-2 omega |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 6243553.3 | 6243530.1 | 6243526.8 | 6243526.6 | 6243526.6 | 6243505.0 |
| 4 | 6272497.9 | 6272119.4 | 6272111.3 | 6272110.9 | 6272111.0 | 6272101.2 |
| 5 | 6301442.5 | 6300708.7 | 6300695.9 | 6300695.2 | 6300695.3 | 6300697.4 |
| 6 | 6359331.8 | 6357887.3 | 6357865.0 | 6357863.8 | 6357863.9 | 6357889.9 |
| 7 | 6417221.0 | 6415066.0 | 6415034.1 | 6415032.3 | 6415032.5 | 6415082.4 |
| 8 | 6532999.5 | 6529423.2 | 6529372.2 | 6529369.5 | 6529369.8 | 6529467.3 |
| 9 | 6648778.0 | 6643780.4 | 6643710.4 | 6643706.6 | 6643707.1 | 6643852.2 |
| 10 | 6700000.0 | 6700000.0 | 6700000.0 | 6700000.0 | 6700000.0 | 6700000.0 |

## Accepted-Step Newton Iterations

| Step | Less precise x100 Newton | Less precise x10 Newton | Default Newton | More precise x10 Newton | More precise x100 Newton | Relative correction 1e-2 Newton |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | 4 | 6 | 7 | 7 | 1 |
| 4 | 2 | 4 | 6 | 7 | 8 | 2 |
| 5 | 2 | 4 | 6 | 7 | 8 | 1 |
| 6 | 3 | 4 | 9 | 11 | 13 | 4 |
| 7 | 2 | 5 | 7 | 8 | 9 | 1 |
| 8 | 4 | 7 | 11 | 13 | 14 | 8 |
| 9 | 5 | 8 | 12 | 15 | 18 | 3 |
| 10 | 2 | 3 | 4 | 7 | 9 | 2 |

## Accepted-Step Linear Iterations

| Step | Less precise x100 linear | Less precise x10 linear | Default linear | More precise x10 linear | More precise x100 linear | Relative correction 1e-2 linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 6 | 33 | 55 | 65 | 64 | 7 |
| 4 | 9 | 34 | 55 | 63 | 77 | 20 |
| 5 | 11 | 41 | 73 | 81 | 92 | 9 |
| 6 | 35 | 61 | 131 | 153 | 173 | 71 |
| 7 | 14 | 81 | 138 | 163 | 185 | 17 |
| 8 | 98 | 268 | 387 | 487 | 566 | 357 |
| 9 | 230 | 276 | 404 | 508 | 678 | 143 |
| 10 | 45 | 67 | 110 | 341 | 470 | 86 |

## Accepted-Step Final Relative Correction

| Step | Less precise x100 ΔU/U | Less precise x10 ΔU/U | Default ΔU/U | More precise x10 ΔU/U | More precise x100 ΔU/U | Relative correction 1e-2 ΔU/U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.008855 | 0.001112 | 0.000044 | 0.000004 | 0.000004 | 0.007053 |
| 4 | 0.019437 | 0.002457 | 0.000085 | 0.000020 | 0.000006 | 0.003948 |
| 5 | 0.015269 | 0.002681 | 0.000260 | 0.000050 | 0.000005 | 0.009874 |
| 6 | 0.015157 | 0.007044 | 0.000228 | 0.000091 | 0.000012 | 0.006923 |
| 7 | 0.007924 | 0.003210 | 0.002279 | 0.000494 | 0.000084 | 0.008668 |
| 8 | 0.116887 | 0.024118 | 0.000551 | 0.002707 | 0.000253 | 0.006696 |
| 9 | 0.072556 | 0.003516 | 0.000922 | 0.000652 | 0.000558 | 0.008906 |
| 10 | 0.069860 | 0.002002 | 0.000960 | 0.001441 | 0.000182 | 0.002906 |

## Comparison Plots

### Lambda vs omega

![Lambda vs omega](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/lambda_omega_overlay.png)

### Lambda by accepted state

![Lambda by accepted state](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/lambda_vs_state.png)

### Omega by accepted state

![Omega by accepted state](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/omega_vs_state.png)

### Runtime by case

![Runtime by case](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/runtime_by_case.png)

### Timing breakdown

![Timing breakdown](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/timing_breakdown_stacked.png)

### Newton iterations per step

![Newton iterations per step](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/step_newton_iterations.png)

### Linear iterations per step

![Linear iterations per step](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/step_linear_iterations.png)

### Linear per Newton

![Linear per Newton](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/step_linear_per_newton.png)

### Wall time per step

![Wall time per step](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/step_wall_time.png)

### Final relative residual per step

![Final relative residual per step](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/step_relres_end.png)

### Final relative correction per step

![Final relative correction per step](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/report/plots/step_relcorr_end.png)

## Existing Per-Run Plots

### Continuation Curve

| Less precise x100 | Less precise x10 | Default | More precise x10 | More precise x100 | Relative correction 1e-2 |
| --- | --- | --- | --- | --- | --- |
| ![Less precise x100 omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x100/plots/petsc_omega_lambda.png) | ![Less precise x10 omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x10/plots/petsc_omega_lambda.png) | ![Default omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/default/plots/petsc_omega_lambda.png) | ![More precise x10 omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x10/plots/petsc_omega_lambda.png) | ![More precise x100 omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x100/plots/petsc_omega_lambda.png) | ![Relative correction 1e-2 omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_omega_lambda.png) |

### Displacements

| Less precise x100 | Less precise x10 | Default | More precise x10 | More precise x100 | Relative correction 1e-2 |
| --- | --- | --- | --- | --- | --- |
| ![Less precise x100 displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x100/plots/petsc_displacements_3D.png) | ![Less precise x10 displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x10/plots/petsc_displacements_3D.png) | ![Default displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/default/plots/petsc_displacements_3D.png) | ![More precise x10 displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x10/plots/petsc_displacements_3D.png) | ![More precise x100 displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x100/plots/petsc_displacements_3D.png) | ![Relative correction 1e-2 displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_displacements_3D.png) |

### Deviatoric Strain

| Less precise x100 | Less precise x10 | Default | More precise x10 | More precise x100 | Relative correction 1e-2 |
| --- | --- | --- | --- | --- | --- |
| ![Less precise x100 deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x100/plots/petsc_deviatoric_strain_3D.png) | ![Less precise x10 deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x10/plots/petsc_deviatoric_strain_3D.png) | ![Default deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/default/plots/petsc_deviatoric_strain_3D.png) | ![More precise x10 deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x10/plots/petsc_deviatoric_strain_3D.png) | ![More precise x100 deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x100/plots/petsc_deviatoric_strain_3D.png) | ![Relative correction 1e-2 deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_deviatoric_strain_3D.png) |

### Step Displacement History

| Less precise x100 | Less precise x10 | Default | More precise x10 | More precise x100 | Relative correction 1e-2 |
| --- | --- | --- | --- | --- | --- |
| ![Less precise x100 step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x100/plots/petsc_step_displacement.png) | ![Less precise x10 step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x10/plots/petsc_step_displacement.png) | ![Default step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/default/plots/petsc_step_displacement.png) | ![More precise x10 step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x10/plots/petsc_step_displacement.png) | ![More precise x100 step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x100/plots/petsc_step_displacement.png) | ![Relative correction 1e-2 step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_step_displacement.png) |

## Run Artifacts

- Less precise x100: config `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/configs/less_precise_x100.toml`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x100`
- Less precise x10: config `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/configs/less_precise_x10.toml`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/less_precise_x10`
- Default: config `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/configs/default.toml`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/default`
- More precise x10: config `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/configs/precision_x10.toml`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x10`
- More precise x100: config `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/configs/precision_x100.toml`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/precision_x100`
- Relative correction 1e-2: config `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/configs/relative_correction_1e_2.toml`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/secant_newton_precision_omega6p7e6/runs/relative_correction_1e_2`
