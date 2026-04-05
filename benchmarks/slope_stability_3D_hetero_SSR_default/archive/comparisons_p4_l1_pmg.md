# P4(L1) PMG Newton-Stop Comparison

This compares `P4(L1)` with the PMG backend using the standard secant predictor, stopping at `omega = 6.7e+06`.

- Base benchmark: `benchmarks/slope_stability_3D_hetero_SSR_default/case.toml`
- Overrides: `elem_type = "P4"`, `mesh_path = "/home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh"`, `pc_backend = "pmg_shell"`, `node_ordering = "block_metis"`
- MPI ranks: `8`
- PMG PETSc opts: `pc_hypre_boomeramg_max_iter=4`, `pc_hypre_boomeramg_tol=0.0`
- Case order: default, 100x less precision, relative correction `1e-2`, `|Δlambda| < 1e-2`, `|Δlambda| < 1e-3`, `|Δlambda| < 1e-3` with the moving history-box step-length cap
- History-box step-length cap: affine-rescale the full current `lambda-omega` history into `[0,1]^2`, measure the first segment (`lambda 1.0 -> 1.1`) there, and limit the next step so the projected last-segment direction has at most the same normalized length.
- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6`

## Summary

| Case | Residual tol | Stop criterion | Stop tol | Runtime [s] | Speedup vs default | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Init linear | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | `1.0e-04` | relative residual | `1.0e-04` | 2699.961 | 1.000 | 9 | 7 | 1.567954 | 6700000.0 | 20 | 144 | 159 | 3395 | 23.576 | 7.444e-05 | 9.374e-04 |
| 100x less precision | `1.0e-02` | relative residual | `1.0e-02` | 567.696 | 4.756 | 9 | 7 | 1.569520 | 6700000.0 | 12 | 52 | 77 | 634 | 12.192 | 9.067e-03 | 2.157e-03 |
| Relative correction 1e-2 | `1.0e-04` | relative correction | `1.0e-02` | 358.071 | 7.540 | 9 | 7 | 1.573196 | 6700000.0 | 9 | 27 | 66 | 378 | 14.000 | 1.125e-01 | 9.523e-04 |
| Abs Delta Lambda 1e-2 | `1.0e-04` | |Δlambda| | `1.0e-02` | 305.348 | 8.842 | 9 | 7 | 1.589603 | 6640990.0 | 20 | 17 | 159 | 191 | 11.235 | 8.980e-02 | 2.322e-02 |
| Abs Delta Lambda 1e-3 | `1.0e-04` | |Δlambda| | `1.0e-03` | 461.921 | 5.845 | 9 | 7 | 1.570530 | 6700000.0 | 20 | 30 | 159 | 397 | 13.233 | 3.449e-02 | 2.338e-03 |
| Abs Delta Lambda 1e-3 + History-Box Cap | `1.0e-04` | |Δlambda| | `1.0e-03` | 618.713 | 4.364 | 16 | 14 | 1.570160 | 6700000.0 | 20 | 38 | 159 | 404 | 10.632 | 1.136e-01 | 3.919e-03 |

## Accepted-Step Lambda

| Step | Default lambda | 100x less precision lambda | Relative correction 1e-2 lambda | Abs Delta Lambda 1e-2 lambda | Abs Delta Lambda 1e-3 lambda | Abs Delta Lambda 1e-3 + History-Box Cap lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.160363 | 1.162439 | 1.168997 | 1.160511 | 1.160511 | 1.160511 |
| 4 | 1.245960 | 1.245555 | 1.243909 | 1.246285 | 1.246285 | 1.218281 |
| 5 | 1.312221 | 1.311917 | 1.309466 | 1.312453 | 1.312453 | 1.265697 |
| 6 | 1.417962 | 1.417392 | 1.415035 | 1.418503 | 1.418503 | 1.319074 |
| 7 | 1.503115 | 1.502583 | 1.501143 | 1.504571 | 1.504571 | 1.365753 |
| 8 | 1.565464 | 1.573338 | 1.568645 | 1.635808 | 1.566183 | 1.416590 |
| 9 | 1.567954 | 1.569520 | 1.573196 | 1.589603 | 1.570530 | 1.462473 |
| 10 | n/a | n/a | n/a | n/a | n/a | 1.504139 |
| 11 | n/a | n/a | n/a | n/a | n/a | 1.540856 |
| 12 | n/a | n/a | n/a | n/a | n/a | 1.563288 |
| 13 | n/a | n/a | n/a | n/a | n/a | 1.566244 |
| 14 | n/a | n/a | n/a | n/a | n/a | 1.567791 |
| 15 | n/a | n/a | n/a | n/a | n/a | 1.569761 |
| 16 | n/a | n/a | n/a | n/a | n/a | 1.570160 |

## Accepted-Step Omega

| Step | Default omega | 100x less precision omega | Relative correction 1e-2 omega | Abs Delta Lambda 1e-2 omega | Abs Delta Lambda 1e-3 omega | Abs Delta Lambda 1e-3 + History-Box Cap omega |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 6244976.2 | 6244871.8 | 6244441.3 | 6244976.2 | 6244976.2 | 6244976.2 |
| 4 | 6273262.9 | 6273086.8 | 6272225.7 | 6273262.9 | 6273262.9 | 6263018.9 |
| 5 | 6301549.6 | 6301301.8 | 6300010.2 | 6301549.6 | 6301549.6 | 6281061.5 |
| 6 | 6358123.0 | 6357731.8 | 6355579.1 | 6358123.0 | 6358123.0 | 6304725.9 |
| 7 | 6414696.4 | 6414161.7 | 6411148.0 | 6414696.4 | 6414696.4 | 6328390.3 |
| 8 | 6527843.2 | 6527021.6 | 6522285.7 | 6527843.2 | 6527843.2 | 6357219.1 |
| 9 | 6700000.0 | 6700000.0 | 6700000.0 | 6640990.0 | 6700000.0 | 6386048.0 |
| 10 | n/a | n/a | n/a | n/a | n/a | 6414876.8 |
| 11 | n/a | n/a | n/a | n/a | n/a | 6443705.7 |
| 12 | n/a | n/a | n/a | n/a | n/a | 6482766.8 |
| 13 | n/a | n/a | n/a | n/a | n/a | 6530355.3 |
| 14 | n/a | n/a | n/a | n/a | n/a | 6587492.6 |
| 15 | n/a | n/a | n/a | n/a | n/a | 6654302.5 |
| 16 | n/a | n/a | n/a | n/a | n/a | 6700000.0 |

## Accepted-Step Newton Iterations

| Step | Default Newton | 100x less precision Newton | Relative correction 1e-2 Newton | Abs Delta Lambda 1e-2 Newton | Abs Delta Lambda 1e-3 Newton | Abs Delta Lambda 1e-3 + History-Box Cap Newton |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 8 | 3 | 2 | 2 | 2 | 2 |
| 4 | 8 | 6 | 3 | 2 | 2 | 2 |
| 5 | 11 | 3 | 2 | 2 | 2 | 2 |
| 6 | 9 | 9 | 2 | 2 | 2 | 2 |
| 7 | 14 | 7 | 1 | 2 | 2 | 2 |
| 8 | 32 | 10 | 12 | 2 | 13 | 2 |
| 9 | 62 | 14 | 5 | 5 | 7 | 2 |
| 10 | n/a | n/a | n/a | n/a | n/a | 2 |
| 11 | n/a | n/a | n/a | n/a | n/a | 2 |
| 12 | n/a | n/a | n/a | n/a | n/a | 10 |
| 13 | n/a | n/a | n/a | n/a | n/a | 5 |
| 14 | n/a | n/a | n/a | n/a | n/a | 2 |
| 15 | n/a | n/a | n/a | n/a | n/a | 2 |
| 16 | n/a | n/a | n/a | n/a | n/a | 1 |

## Accepted-Step Linear Iterations

| Step | Default linear | 100x less precision linear | Relative correction 1e-2 linear | Abs Delta Lambda 1e-2 linear | Abs Delta Lambda 1e-3 linear | Abs Delta Lambda 1e-3 + History-Box Cap linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 57 | 22 | 40 | 11 | 11 | 11 |
| 4 | 59 | 54 | 22 | 10 | 10 | 11 |
| 5 | 87 | 9 | 9 | 11 | 11 | 11 |
| 6 | 78 | 59 | 12 | 12 | 12 | 10 |
| 7 | 130 | 43 | 4 | 13 | 13 | 15 |
| 8 | 806 | 190 | 224 | 20 | 256 | 20 |
| 9 | 2178 | 257 | 67 | 114 | 84 | 14 |
| 10 | n/a | n/a | n/a | n/a | n/a | 18 |
| 11 | n/a | n/a | n/a | n/a | n/a | 16 |
| 12 | n/a | n/a | n/a | n/a | n/a | 157 |
| 13 | n/a | n/a | n/a | n/a | n/a | 77 |
| 14 | n/a | n/a | n/a | n/a | n/a | 22 |
| 15 | n/a | n/a | n/a | n/a | n/a | 12 |
| 16 | n/a | n/a | n/a | n/a | n/a | 10 |

## Accepted-Step Final Relative Correction

| Step | Default ΔU/U | 100x less precision ΔU/U | Relative correction 1e-2 ΔU/U | Abs Delta Lambda 1e-2 ΔU/U | Abs Delta Lambda 1e-3 ΔU/U | Abs Delta Lambda 1e-3 + History-Box Cap ΔU/U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.000042 | 0.004250 | 0.009032 | 0.004200 | 0.004200 | 0.004200 |
| 4 | 0.000145 | 0.002234 | 0.003042 | 0.003117 | 0.003117 | 0.001880 |
| 5 | 0.000032 | 0.001554 | 0.001279 | 0.003601 | 0.003601 | 0.002336 |
| 6 | 0.000377 | 0.000412 | 0.003224 | 0.005073 | 0.005073 | 0.001504 |
| 7 | 0.000549 | 0.005433 | 0.008613 | 0.006891 | 0.006891 | 0.003357 |
| 8 | 0.000948 | 0.035071 | 0.009311 | 0.063488 | 0.011775 | 0.002607 |
| 9 | 0.000937 | 0.002157 | 0.000952 | 0.023221 | 0.002338 | 0.000692 |
| 10 | n/a | n/a | n/a | n/a | n/a | 0.004770 |
| 11 | n/a | n/a | n/a | n/a | n/a | 0.011772 |
| 12 | n/a | n/a | n/a | n/a | n/a | 0.001428 |
| 13 | n/a | n/a | n/a | n/a | n/a | 0.000662 |
| 14 | n/a | n/a | n/a | n/a | n/a | 0.002896 |
| 15 | n/a | n/a | n/a | n/a | n/a | 0.001981 |
| 16 | n/a | n/a | n/a | n/a | n/a | 0.003919 |

## Comparison Plots

### Lambda vs omega

![Lambda vs omega](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/lambda_omega_overlay.png)

### Lambda by accepted state

![Lambda by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/lambda_vs_state.png)

### Omega by accepted state

![Omega by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/omega_vs_state.png)

### Runtime by case

![Runtime by case](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/runtime_by_case.png)

### Timing breakdown

![Timing breakdown](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/timing_breakdown_stacked.png)

### Newton iterations per step

![Newton iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_newton_iterations.png)

### Linear iterations per step

![Linear iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_linear_iterations.png)

### Linear per Newton

![Linear per Newton](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_linear_per_newton.png)

### Wall time per step

![Wall time per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_wall_time.png)

### Final relative residual per step

![Final relative residual per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_relres_end.png)

### Final relative correction per step

![Final relative correction per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/plots/step_relcorr_end.png)

## Existing Per-Run Plots

### Continuation Curve

| Default | 100x less precision | Relative correction 1e-2 | Abs Delta Lambda 1e-2 | Abs Delta Lambda 1e-3 | Abs Delta Lambda 1e-3 + History-Box Cap |
| --- | --- | --- | --- | --- | --- |
| ![Default omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/default/plots/petsc_omega_lambda.png) | ![100x less precision omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/less_precise_x100/plots/petsc_omega_lambda.png) | ![Relative correction 1e-2 omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-2 omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-3 omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_omega_lambda.png) |

### Displacements

| Default | 100x less precision | Relative correction 1e-2 | Abs Delta Lambda 1e-2 | Abs Delta Lambda 1e-3 | Abs Delta Lambda 1e-3 + History-Box Cap |
| --- | --- | --- | --- | --- | --- |
| ![Default displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/default/plots/petsc_displacements_3D.png) | ![100x less precision displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/less_precise_x100/plots/petsc_displacements_3D.png) | ![Relative correction 1e-2 displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-2 displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-3 displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_displacements_3D.png) |

### Deviatoric Strain

| Default | 100x less precision | Relative correction 1e-2 | Abs Delta Lambda 1e-2 | Abs Delta Lambda 1e-3 | Abs Delta Lambda 1e-3 + History-Box Cap |
| --- | --- | --- | --- | --- | --- |
| ![Default deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/default/plots/petsc_deviatoric_strain_3D.png) | ![100x less precision deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/less_precise_x100/plots/petsc_deviatoric_strain_3D.png) | ![Relative correction 1e-2 deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-2 deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-3 deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_deviatoric_strain_3D.png) |

### Step Displacement History

| Default | 100x less precision | Relative correction 1e-2 | Abs Delta Lambda 1e-2 | Abs Delta Lambda 1e-3 | Abs Delta Lambda 1e-3 + History-Box Cap |
| --- | --- | --- | --- | --- | --- |
| ![Default step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/default/plots/petsc_step_displacement.png) | ![100x less precision step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/less_precise_x100/plots/petsc_step_displacement.png) | ![Relative correction 1e-2 step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-2 step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-3 step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_step_displacement.png) |

## Accepted-Step Newton Solves

These sections overlay the successful Newton solve that produced each accepted continuation step for the main PMG cases without the step-length cap.

### Accepted Continuation Step 3

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 1 | 8 | 50.036 | 1.160363 | 6244976.2 | 2.378e-05 | 4.248e-05 |
| 100x less precision | 1 | 3 | 16.653 | 1.162439 | 6244871.8 | 7.583e-03 | 4.250e-03 |
| Relative correction 1e-2 | 1 | 2 | 24.861 | 1.168997 | 6244441.3 | 1.953e-02 | 9.032e-03 |
| Abs Delta Lambda 1e-2 | 1 | 2 | 10.521 | 1.160511 | 6244976.2 | 8.378e-03 | 4.200e-03 |
| Abs Delta Lambda 1e-3 | 1 | 2 | 10.521 | 1.160511 | 6244976.2 | 8.378e-03 | 4.200e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_03/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 4

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 1 | 8 | 51.245 | 1.245960 | 6273262.9 | 6.882e-05 | 1.454e-04 |
| 100x less precision | 1 | 6 | 41.864 | 1.245555 | 6273086.8 | 8.875e-03 | 2.234e-03 |
| Relative correction 1e-2 | 1 | 3 | 19.318 | 1.243909 | 6272225.7 | 2.176e-02 | 3.042e-03 |
| Abs Delta Lambda 1e-2 | 1 | 2 | 10.315 | 1.246285 | 6273262.9 | 1.425e-02 | 3.117e-03 |
| Abs Delta Lambda 1e-3 | 1 | 2 | 10.596 | 1.246285 | 6273262.9 | 1.425e-02 | 3.117e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_04/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 5

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 1 | 11 | 78.655 | 1.312221 | 6301549.6 | 1.449e-05 | 3.186e-05 |
| 100x less precision | 1 | 3 | 10.951 | 1.311917 | 6301301.8 | 6.984e-03 | 1.554e-03 |
| Relative correction 1e-2 | 1 | 2 | 10.355 | 1.309466 | 6300010.2 | 3.886e-02 | 1.279e-03 |
| Abs Delta Lambda 1e-2 | 1 | 2 | 11.181 | 1.312453 | 6301549.6 | 1.058e-02 | 3.601e-03 |
| Abs Delta Lambda 1e-3 | 1 | 2 | 10.813 | 1.312453 | 6301549.6 | 1.058e-02 | 3.601e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_05/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 6

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 1 | 9 | 66.223 | 1.417962 | 6358123.0 | 6.581e-05 | 3.768e-04 |
| 100x less precision | 1 | 9 | 57.602 | 1.417392 | 6357731.8 | 8.567e-03 | 4.115e-04 |
| Relative correction 1e-2 | 1 | 2 | 11.681 | 1.415035 | 6355579.1 | 6.689e-02 | 3.224e-03 |
| Abs Delta Lambda 1e-2 | 1 | 2 | 11.919 | 1.418503 | 6358123.0 | 2.031e-02 | 5.073e-03 |
| Abs Delta Lambda 1e-3 | 1 | 2 | 11.495 | 1.418503 | 6358123.0 | 2.031e-02 | 5.073e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_06/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 7

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 1 | 14 | 117.100 | 1.503115 | 6414696.4 | 4.840e-05 | 5.489e-04 |
| 100x less precision | 1 | 7 | 41.477 | 1.502583 | 6414161.7 | 8.627e-03 | 5.433e-03 |
| Relative correction 1e-2 | 1 | 1 | 5.050 | 1.501143 | 6411148.0 | 1.268e-01 | 8.613e-03 |
| Abs Delta Lambda 1e-2 | 1 | 2 | 12.407 | 1.504571 | 6414696.4 | 2.430e-02 | 6.891e-03 |
| Abs Delta Lambda 1e-3 | 1 | 2 | 12.290 | 1.504571 | 6414696.4 | 2.430e-02 | 6.891e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_07/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 8

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 1 | 32 | 579.713 | 1.565464 | 6527843.2 | 6.859e-05 | 9.481e-04 |
| 100x less precision | 1 | 10 | 126.759 | 1.573338 | 6527021.6 | 8.661e-03 | 3.507e-02 |
| Relative correction 1e-2 | 1 | 12 | 158.866 | 1.568645 | 6522285.7 | 1.739e-02 | 9.311e-03 |
| Abs Delta Lambda 1e-2 | 1 | 2 | 15.893 | 1.635808 | 6527843.2 | 4.822e-02 | 6.349e-02 |
| Abs Delta Lambda 1e-3 | 1 | 13 | 180.291 | 1.566183 | 6527843.2 | 7.889e-03 | 1.177e-02 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_08/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 9

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 1 | 62 | 1587.138 | 1.567954 | 6700000.0 | 7.444e-05 | 9.374e-04 |
| 100x less precision | 1 | 14 | 184.654 | 1.569520 | 6700000.0 | 9.067e-03 | 2.157e-03 |
| Relative correction 1e-2 | 1 | 5 | 50.023 | 1.573196 | 6700000.0 | 1.125e-01 | 9.523e-04 |
| Abs Delta Lambda 1e-2 | 1 | 5 | 73.956 | 1.589603 | 6640990.0 | 8.980e-02 | 2.322e-02 |
| Abs Delta Lambda 1e-3 | 1 | 7 | 66.748 | 1.570530 | 6700000.0 | 3.449e-02 | 2.338e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_09/relative_increment_vs_criterion.png) |  |

## Accepted-Step Newton Solves With Step-Length Cap

These sections show the separate Newton convergence history for the `|Δlambda| < 1e-3` case with the moving history-box step-length cap enabled.

### Accepted Continuation Step 3

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 12.507 | 1.160511 | 6244976.2 | 8.378e-03 | 4.200e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 4

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 13.048 | 1.218281 | 6263018.9 | 9.530e-03 | 1.880e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 5

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 13.244 | 1.265697 | 6281061.5 | 6.449e-03 | 2.336e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 6

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 12.693 | 1.319074 | 6304725.9 | 8.164e-03 | 1.504e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 7

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 15.694 | 1.365753 | 6328390.3 | 7.865e-03 | 3.357e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 8

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 18.865 | 1.416590 | 6357219.1 | 8.375e-03 | 2.607e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 9

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 15.446 | 1.462473 | 6386048.0 | 1.030e-02 | 6.915e-04 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 10

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 18.119 | 1.504139 | 6414876.8 | 1.131e-02 | 4.770e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 11

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 17.206 | 1.540856 | 6443705.7 | 2.356e-02 | 1.177e-02 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 12

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 10 | 144.694 | 1.563288 | 6482766.8 | 3.949e-03 | 1.428e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 13

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 5 | 69.520 | 1.566244 | 6530355.3 | 1.329e-02 | 6.615e-04 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 14

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 21.672 | 1.567791 | 6587492.6 | 2.951e-02 | 2.896e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 15

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 2 | 15.979 | 1.569761 | 6654302.5 | 1.180e-01 | 1.981e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 16

| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | 1 | 10.510 | 1.570160 | 6700000.0 | 1.136e-01 | 3.919e-03 |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/criterion.png) | ![Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/delta_lambda.png) | ![Delta U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/relative_increment_vs_criterion.png) |  |

## Run Artifacts

- Default: command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/default.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/default`
- 100x less precision: command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/less_precise_x100.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/less_precise_x100`
- Relative correction 1e-2: command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/relative_correction_1e_2.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2`
- Abs Delta Lambda 1e-2: command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_2.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2`
- Abs Delta Lambda 1e-3: command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_3.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3`
- Abs Delta Lambda 1e-3 + History-Box Cap: command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_3_cap_initial_segment.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment`
