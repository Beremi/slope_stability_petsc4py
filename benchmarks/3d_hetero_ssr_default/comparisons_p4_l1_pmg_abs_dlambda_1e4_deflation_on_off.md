# P4(L1) PMG Abs Delta Lambda 1e-4: Deflation On vs Off

This compares `P4(L1)` with the PMG backend using the standard secant predictor, stopping at `omega = 6.7e+06`.

- Base benchmark: `benchmarks/3d_hetero_ssr_default/case.toml`
- Overrides: `elem_type = "P4"`, `mesh_path = "/home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh"`, `pc_backend = "pmg_shell"`, `node_ordering = "block_metis"`
- MPI ranks: `8`
- PMG PETSc opts: `pc_hypre_boomeramg_max_iter=4`, `pc_hypre_boomeramg_tol=0.0`
- Cases: Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1), Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off)
- History-box step-length cap: affine-rescale the full current `lambda-omega` history into `[0,1]^2`, measure the first segment (`lambda 1.0 -> 1.1`) there, and limit the next step so the projected last-segment direction has at most the same normalized length.

- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6`

## Summary

| Case | Residual tol | Stop criterion | Stop tol | Runtime [s] | Speedup | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Init linear | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | `1.0e-04` | |Δlambda| | `1.0e-04` | 1064.022 | 1.000 | 9 | 7 | 1.568533 | 6700000.0 | 9 | 69 | 66 | 1057 | 15.319 | 9.499e-03 | 3.470e-04 |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | `1.0e-04` | |Δlambda| | `1.0e-04` | 2002.291 | 1.000 | 11 | 9 | 1.576156 | 6700000.0 | 8 | 121 | 47 | 2708 | 22.380 | 6.019e-03 | 2.080e-04 |

## Accepted-Step Lambda

| Step | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) lambda | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) lambda |
| --- | ---: | ---: |
| 3 | 1.158393 | 1.110792 |
| 4 | 1.243261 | 1.174129 |
| 5 | 1.308950 | 1.223186 |
| 6 | 1.413847 | 1.301714 |
| 7 | 1.498371 | 1.365972 |
| 8 | 1.565367 | 1.471467 |
| 9 | 1.568533 | 1.552299 |
| 10 | n/a | 1.567187 |
| 11 | n/a | 1.576156 |

## Accepted-Step Omega

| Step | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) omega | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) omega |
| --- | ---: | ---: |
| 3 | 6244441.3 | 6232952.1 |
| 4 | 6272225.7 | 6248887.4 |
| 5 | 6300010.2 | 6264822.7 |
| 6 | 6355579.1 | 6296693.3 |
| 7 | 6411148.0 | 6328563.9 |
| 8 | 6522285.7 | 6392305.1 |
| 9 | 6700000.0 | 6456046.3 |
| 10 | n/a | 6583528.7 |
| 11 | n/a | 6700000.0 |

## Accepted-Step Newton Iterations

| Step | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) Newton | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) Newton |
| --- | ---: | ---: |
| 3 | 5 | 12 |
| 4 | 7 | 11 |
| 5 | 8 | 9 |
| 6 | 5 | 6 |
| 7 | 8 | 9 |
| 8 | 21 | 8 |
| 9 | 15 | 18 |
| 10 | n/a | 42 |
| 11 | n/a | 6 |

## Accepted-Step Linear Iterations

| Step | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) linear | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) linear |
| --- | ---: | ---: |
| 3 | 77 | 156 |
| 4 | 87 | 136 |
| 5 | 76 | 101 |
| 6 | 56 | 73 |
| 7 | 69 | 99 |
| 8 | 461 | 91 |
| 9 | 231 | 325 |
| 10 | n/a | 1669 |
| 11 | n/a | 58 |

## Accepted-Step Final Relative Correction

| Step | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) ΔU/U | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) ΔU/U |
| --- | ---: | ---: |
| 3 | 0.000608 | 0.000042 |
| 4 | 0.000114 | 0.000116 |
| 5 | 0.000094 | 0.000062 |
| 6 | 0.000247 | 0.000924 |
| 7 | 0.000125 | 0.000830 |
| 8 | 0.002026 | 0.000126 |
| 9 | 0.000347 | 0.000218 |
| 10 | n/a | 0.003041 |
| 11 | n/a | 0.000208 |

## Comparison Plots

### Lambda vs omega

![Lambda vs omega](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/lambda_omega_overlay.png)

### Lambda by accepted state

![Lambda by accepted state](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/lambda_vs_state.png)

### Omega by accepted state

![Omega by accepted state](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/omega_vs_state.png)

### Runtime by case

![Runtime by case](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/runtime_by_case.png)

### Timing breakdown

![Timing breakdown](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/timing_breakdown_stacked.png)

### Newton iterations per step

![Newton iterations per step](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_newton_iterations.png)

### Linear iterations per step

![Linear iterations per step](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_linear_iterations.png)

### Linear per Newton

![Linear per Newton](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_linear_per_newton.png)

### Wall time per step

![Wall time per step](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_wall_time.png)

### Final relative residual per step

![Final relative residual per step](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_relres_end.png)

### Final relative correction per step

![Final relative correction per step](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_relcorr_end.png)

## Existing Per-Run Plots

### Continuation Curve

| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) |
| --- | --- |
| ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1_no_deflation/plots/petsc_omega_lambda.png) |

### Displacements

| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) |
| --- | --- |
| ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1_no_deflation/plots/petsc_displacements_3D.png) |

### Deviatoric Strain

| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) |
| --- | --- |
| ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1_no_deflation/plots/petsc_deviatoric_strain_3D.png) |

### Step Displacement History

| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) |
| --- | --- |
| ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1_no_deflation/plots/petsc_step_displacement.png) |

## Accepted-Step Newton Solves

These sections overlay the successful Newton solve that produced each accepted continuation step for the main PMG cases without the step-length cap.

### Accepted Continuation Step 3

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 5 | 65.200 | 1.158393 | 6244441.3 | 8.200e-03 | 6.083e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | 1 | base | |Δlambda| | 1.000e-04 | 12 | 125.916 | 1.110792 | 6232952.1 | 5.886e-06 | 4.175e-05 | n/a | n/a | n/a | 2 | no |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 4

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 7 | 81.817 | 1.243261 | 6272225.7 | 6.452e-03 | 1.139e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | 1 | base | |Δlambda| | 1.000e-04 | 11 | 112.187 | 1.174129 | 6248887.4 | 2.036e-04 | 1.161e-04 | n/a | n/a | n/a | 2 | no |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 5

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 8 | 80.963 | 1.308950 | 6300010.2 | 2.911e-03 | 9.448e-05 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | 1 | base | |Δlambda| | 1.000e-04 | 9 | 86.164 | 1.223186 | 6264822.7 | 5.035e-06 | 6.194e-05 | n/a | n/a | n/a | 2 | no |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 6

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 5 | 53.744 | 1.413847 | 6355579.1 | 7.084e-03 | 2.475e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | 1 | base | |Δlambda| | 1.000e-04 | 6 | 60.746 | 1.301714 | 6296693.3 | 4.710e-04 | 9.240e-04 | n/a | n/a | n/a | 2 | no |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 7

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 8 | 77.694 | 1.498371 | 6411148.0 | 2.062e-03 | 1.252e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | 1 | base | |Δlambda| | 1.000e-04 | 9 | 85.214 | 1.365972 | 6328563.9 | 2.686e-04 | 8.302e-04 | n/a | n/a | n/a | 2 | no |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 8

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 21 | 406.837 | 1.565367 | 6522285.7 | 2.114e-03 | 2.026e-03 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | 1 | base | |Δlambda| | 1.000e-04 | 8 | 77.042 | 1.471467 | 6392305.1 | 7.795e-04 | 1.264e-04 | n/a | n/a | n/a | 2 | no |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 9

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 15 | 202.282 | 1.568533 | 6700000.0 | 9.499e-03 | 3.470e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | 1 | base | |Δlambda| | 1.000e-04 | 18 | 247.148 | 1.552299 | 6456046.3 | 6.056e-05 | 2.184e-04 | n/a | n/a | n/a | 2 | no |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 10

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | 1 | base | |Δlambda| | 1.000e-04 | 42 | 1079.690 | 1.567187 | 6583528.7 | 9.858e-05 | 3.041e-03 | n/a | n/a | n/a | 2 | no |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_10/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 11

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off) | 1 | base | |Δlambda| | 1.000e-04 | 6 | 47.990 | 1.576156 | 6700000.0 | 6.019e-03 | 2.080e-04 | n/a | n/a | n/a | 2 | no |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_11/relative_increment_vs_criterion.png) |  |

## Run Artifacts

- Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1): command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1`
- Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off): command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1_no_deflation.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1_no_deflation`
