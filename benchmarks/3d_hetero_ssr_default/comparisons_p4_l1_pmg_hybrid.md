# P4(L1) PMG Hybrid Rough/Fine Comparison

This compares `P4(L1)` with the PMG backend using the standard secant predictor, stopping at `omega = 6.7e+06`.

- Base benchmark: `benchmarks/3d_hetero_ssr_default/case.toml`
- Overrides: `elem_type = "P4"`, `mesh_path = "/home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh"`, `pc_backend = "pmg_shell"`, `node_ordering = "block_metis"`
- MPI ranks: `8`
- PMG PETSc opts: `pc_hypre_boomeramg_max_iter=4`, `pc_hypre_boomeramg_tol=0.0`
- Cases: Relative correction 1e-2, Abs Delta Lambda 1e-2, Abs Delta Lambda 1e-3 + History-Box Cap, Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off)
- History-box step-length cap: affine-rescale the full current `lambda-omega` history into `[0,1]^2`, measure the first segment (`lambda 1.0 -> 1.1`) there, and limit the next step so the projected last-segment direction has at most the same normalized length.
- Hybrid rough/fine trigger: use `|Δlambda| < 1e-2` by default, then switch the crossing step to `|Δlambda| < 1e-3` once the cumulative accepted rough-path distance plus the current capped projected step exceeds `2x` the initial fine-segment length.
- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6`

## Summary

| Case | Residual tol | Stop criterion | Stop tol | Runtime [s] | Speedup | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Init linear | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Relative correction 1e-2 | `1.0e-04` | relative correction | `1.0e-02` | 358.071 | 1.000 | 9 | 7 | 1.573196 | 6700000.0 | 9 | 27 | 66 | 378 | 14.000 | 1.125e-01 | 9.523e-04 |
| Abs Delta Lambda 1e-2 | `1.0e-04` | |Δlambda| | `1.0e-02` | 305.348 | 1.000 | 9 | 7 | 1.589603 | 6640990.0 | 20 | 17 | 159 | 191 | 11.235 | 8.980e-02 | 2.322e-02 |
| Abs Delta Lambda 1e-3 + History-Box Cap | `1.0e-04` | |Δlambda| | `1.0e-03` | 618.713 | 1.000 | 16 | 14 | 1.570160 | 6700000.0 | 20 | 38 | 159 | 404 | 10.632 | 1.136e-01 | 3.919e-03 |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | `1.0e-04` | |Δlambda| | `1.0e-02` | 708.395 | 1.000 | 16 | 14 | 1.572559 | 6700000.0 | 9 | 46 | 66 | 660 | 14.348 | 6.698e-01 | 3.873e-02 |

## Accepted-Step Lambda

| Step | Relative correction 1e-2 lambda | Abs Delta Lambda 1e-2 lambda | Abs Delta Lambda 1e-3 + History-Box Cap lambda | Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) lambda |
| --- | ---: | ---: | ---: | ---: |
| 3 | 1.168997 | 1.160511 | 1.160511 | 1.158363 |
| 4 | 1.243909 | 1.246285 | 1.218281 | 1.217256 |
| 5 | 1.309466 | 1.312453 | 1.265697 | 1.264685 |
| 6 | 1.415035 | 1.418503 | 1.319074 | 1.317884 |
| 7 | 1.501143 | 1.504571 | 1.365753 | 1.364250 |
| 8 | 1.568645 | 1.635808 | 1.416590 | 1.415196 |
| 9 | 1.573196 | 1.589603 | 1.462473 | 1.460924 |
| 10 | n/a | n/a | 1.504139 | 1.509437 |
| 11 | n/a | n/a | 1.540856 | 1.550025 |
| 12 | n/a | n/a | 1.563288 | 1.572782 |
| 13 | n/a | n/a | 1.566244 | 1.581516 |
| 14 | n/a | n/a | 1.567791 | 1.568891 |
| 15 | n/a | n/a | 1.569761 | 1.568598 |
| 16 | n/a | n/a | 1.570160 | 1.572559 |

## Accepted-Step Omega

| Step | Relative correction 1e-2 omega | Abs Delta Lambda 1e-2 omega | Abs Delta Lambda 1e-3 + History-Box Cap omega | Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) omega |
| --- | ---: | ---: | ---: | ---: |
| 3 | 6244441.3 | 6244976.2 | 6244976.2 | 6244441.3 |
| 4 | 6272225.7 | 6273262.9 | 6263018.9 | 6262455.4 |
| 5 | 6300010.2 | 6301549.6 | 6281061.5 | 6280469.6 |
| 6 | 6355579.1 | 6358123.0 | 6304725.9 | 6303961.6 |
| 7 | 6411148.0 | 6414696.4 | 6328390.3 | 6327453.6 |
| 8 | 6522285.7 | 6527843.2 | 6357219.1 | 6356123.7 |
| 9 | 6700000.0 | 6640990.0 | 6386048.0 | 6384793.8 |
| 10 | n/a | n/a | 6414876.8 | 6418532.8 |
| 11 | n/a | n/a | 6443705.7 | 6452271.7 |
| 12 | n/a | n/a | 6482766.8 | 6492302.4 |
| 13 | n/a | n/a | 6530355.3 | 6540616.0 |
| 14 | n/a | n/a | 6587492.6 | 6597742.4 |
| 15 | n/a | n/a | 6654302.5 | 6664040.2 |
| 16 | n/a | n/a | 6700000.0 | 6700000.0 |

## Accepted-Step Newton Iterations

| Step | Relative correction 1e-2 Newton | Abs Delta Lambda 1e-2 Newton | Abs Delta Lambda 1e-3 + History-Box Cap Newton | Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) Newton |
| --- | ---: | ---: | ---: | ---: |
| 3 | 2 | 2 | 2 | 4 |
| 4 | 3 | 2 | 2 | 2 |
| 5 | 2 | 2 | 2 | 2 |
| 6 | 2 | 2 | 2 | 2 |
| 7 | 1 | 2 | 2 | 3 |
| 8 | 12 | 2 | 2 | 2 |
| 9 | 5 | 5 | 2 | 2 |
| 10 | n/a | n/a | 2 | 2 |
| 11 | n/a | n/a | 2 | 6 |
| 12 | n/a | n/a | 10 | 2 |
| 13 | n/a | n/a | 5 | 1 |
| 14 | n/a | n/a | 2 | 16 |
| 15 | n/a | n/a | 2 | 1 |
| 16 | n/a | n/a | 1 | 1 |

## Accepted-Step Linear Iterations

| Step | Relative correction 1e-2 linear | Abs Delta Lambda 1e-2 linear | Abs Delta Lambda 1e-3 + History-Box Cap linear | Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) linear |
| --- | ---: | ---: | ---: | ---: |
| 3 | 40 | 11 | 11 | 62 |
| 4 | 22 | 10 | 11 | 21 |
| 5 | 9 | 11 | 11 | 22 |
| 6 | 12 | 12 | 10 | 12 |
| 7 | 4 | 13 | 15 | 27 |
| 8 | 224 | 20 | 20 | 8 |
| 9 | 67 | 114 | 14 | 14 |
| 10 | n/a | n/a | 18 | 14 |
| 11 | n/a | n/a | 16 | 55 |
| 12 | n/a | n/a | 157 | 32 |
| 13 | n/a | n/a | 77 | 27 |
| 14 | n/a | n/a | 22 | 348 |
| 15 | n/a | n/a | 12 | 7 |
| 16 | n/a | n/a | 10 | 11 |

## Accepted-Step Final Relative Correction

| Step | Relative correction 1e-2 ΔU/U | Abs Delta Lambda 1e-2 ΔU/U | Abs Delta Lambda 1e-3 + History-Box Cap ΔU/U | Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) ΔU/U |
| --- | ---: | ---: | ---: | ---: |
| 3 | 0.009032 | 0.004200 | 0.004200 | 0.001165 |
| 4 | 0.003042 | 0.003117 | 0.001880 | 0.000635 |
| 5 | 0.001279 | 0.003601 | 0.002336 | 0.001559 |
| 6 | 0.003224 | 0.005073 | 0.001504 | 0.001456 |
| 7 | 0.008613 | 0.006891 | 0.003357 | 0.001090 |
| 8 | 0.009311 | 0.063488 | 0.002607 | 0.001312 |
| 9 | 0.000952 | 0.023221 | 0.000692 | 0.004223 |
| 10 | n/a | n/a | 0.004770 | 0.003031 |
| 11 | n/a | n/a | 0.011772 | 0.003317 |
| 12 | n/a | n/a | 0.001428 | 0.083566 |
| 13 | n/a | n/a | 0.000662 | 0.133357 |
| 14 | n/a | n/a | 0.002896 | 0.004200 |
| 15 | n/a | n/a | 0.001981 | 0.019239 |
| 16 | n/a | n/a | 0.003919 | 0.038727 |

## Hybrid Trigger Summary

| Accepted step | Precision mode | Stop criterion | Stop tol | Cum rough dist | Current length | Dist + current | Threshold | Reference step | Triggered |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | rough | |Δlambda| | 1.000e-02 | 0.000000 | 1.414214 | 1.414214 | 2.828427 | 2 | no |
| 4 | rough | |Δlambda| | 1.000e-02 | 0.621146 | 0.805444 | 1.426590 | 1.610888 | 2 | no |
| 5 | fine | |Δlambda| | 1.000e-03 | 0.882885 | 0.477696 | 1.360581 | 1.102495 | 2 | yes |
| 6 | rough | |Δlambda| | 1.000e-02 | 0.000000 | 0.436042 | 0.436042 | 0.872085 | 5 | no |
| 7 | rough | |Δlambda| | 1.000e-02 | 0.316877 | 0.316877 | 0.633754 | 0.705070 | 5 | no |
| 8 | fine | |Δlambda| | 1.000e-03 | 0.504765 | 0.301814 | 0.806579 | 0.603628 | 5 | yes |
| 9 | rough | |Δlambda| | 1.000e-02 | 0.000000 | 0.239405 | 0.239405 | 0.521271 | 8 | no |
| 10 | rough | |Δlambda| | 1.000e-02 | 0.197278 | 0.232156 | 0.429434 | 0.464313 | 8 | no |
| 11 | fine | |Δlambda| | 1.000e-03 | 0.360361 | 0.192354 | 0.552715 | 0.416016 | 8 | yes |
| 12 | rough | |Δlambda| | 1.000e-02 | 0.000000 | 0.191132 | 0.191132 | 0.382263 | 11 | no |
| 13 | rough | |Δlambda| | 1.000e-02 | 0.150562 | 0.181715 | 0.332277 | 0.363431 | 11 | no |
| 14 | fine | |Δlambda| | 1.000e-03 | 0.279505 | 0.177231 | 0.456736 | 0.354461 | 11 | yes |
| 15 | rough | |Δlambda| | 1.000e-02 | 0.000000 | 0.175786 | 0.175786 | 0.351572 | 14 | no |
| 16 | rough | |Δlambda| | 1.000e-02 | 0.148191 | 0.080379 | 0.228569 | 0.349491 | 14 | no |

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

| Relative correction 1e-2 | Abs Delta Lambda 1e-2 | Abs Delta Lambda 1e-3 + History-Box Cap | Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) |
| --- | --- | --- | --- |
| ![Relative correction 1e-2 omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-2 omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_omega_lambda.png) | ![Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop/plots/petsc_omega_lambda.png) |

### Displacements

| Relative correction 1e-2 | Abs Delta Lambda 1e-2 | Abs Delta Lambda 1e-3 + History-Box Cap | Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) |
| --- | --- | --- | --- |
| ![Relative correction 1e-2 displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-2 displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_displacements_3D.png) | ![Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop/plots/petsc_displacements_3D.png) |

### Deviatoric Strain

| Relative correction 1e-2 | Abs Delta Lambda 1e-2 | Abs Delta Lambda 1e-3 + History-Box Cap | Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) |
| --- | --- | --- | --- |
| ![Relative correction 1e-2 deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-2 deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_deviatoric_strain_3D.png) | ![Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop/plots/petsc_deviatoric_strain_3D.png) |

### Step Displacement History

| Relative correction 1e-2 | Abs Delta Lambda 1e-2 | Abs Delta Lambda 1e-3 + History-Box Cap | Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) |
| --- | --- | --- | --- |
| ![Relative correction 1e-2 step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-2 step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_step_displacement.png) | ![Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop/plots/petsc_step_displacement.png) |

## Accepted-Step Newton Solves

These sections overlay the successful Newton solve that produced each accepted continuation step for the main PMG cases without the step-length cap.

### Accepted Continuation Step 3

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Relative correction 1e-2 | 1 | base | relative correction | 1.000e-02 | 2 | 24.861 | 1.168997 | 6244441.3 | 1.953e-02 | 9.032e-03 | n/a | n/a | n/a | n/a | no |
| Abs Delta Lambda 1e-2 | 1 | base | |Δlambda| | 1.000e-02 | 2 | 10.521 | 1.160511 | 6244976.2 | 8.378e-03 | 4.200e-03 | n/a | n/a | n/a | n/a | no |

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
| Relative correction 1e-2 | 1 | base | relative correction | 1.000e-02 | 3 | 19.318 | 1.243909 | 6272225.7 | 2.176e-02 | 3.042e-03 | n/a | n/a | n/a | n/a | no |
| Abs Delta Lambda 1e-2 | 1 | base | |Δlambda| | 1.000e-02 | 2 | 10.315 | 1.246285 | 6273262.9 | 1.425e-02 | 3.117e-03 | n/a | n/a | n/a | n/a | no |

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
| Relative correction 1e-2 | 1 | base | relative correction | 1.000e-02 | 2 | 10.355 | 1.309466 | 6300010.2 | 3.886e-02 | 1.279e-03 | n/a | n/a | n/a | n/a | no |
| Abs Delta Lambda 1e-2 | 1 | base | |Δlambda| | 1.000e-02 | 2 | 11.181 | 1.312453 | 6301549.6 | 1.058e-02 | 3.601e-03 | n/a | n/a | n/a | n/a | no |

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
| Relative correction 1e-2 | 1 | base | relative correction | 1.000e-02 | 2 | 11.681 | 1.415035 | 6355579.1 | 6.689e-02 | 3.224e-03 | n/a | n/a | n/a | n/a | no |
| Abs Delta Lambda 1e-2 | 1 | base | |Δlambda| | 1.000e-02 | 2 | 11.919 | 1.418503 | 6358123.0 | 2.031e-02 | 5.073e-03 | n/a | n/a | n/a | n/a | no |

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
| Relative correction 1e-2 | 1 | base | relative correction | 1.000e-02 | 1 | 5.050 | 1.501143 | 6411148.0 | 1.268e-01 | 8.613e-03 | n/a | n/a | n/a | n/a | no |
| Abs Delta Lambda 1e-2 | 1 | base | |Δlambda| | 1.000e-02 | 2 | 12.407 | 1.504571 | 6414696.4 | 2.430e-02 | 6.891e-03 | n/a | n/a | n/a | n/a | no |

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
| Relative correction 1e-2 | 1 | base | relative correction | 1.000e-02 | 12 | 158.866 | 1.568645 | 6522285.7 | 1.739e-02 | 9.311e-03 | n/a | n/a | n/a | n/a | no |
| Abs Delta Lambda 1e-2 | 1 | base | |Δlambda| | 1.000e-02 | 2 | 15.893 | 1.635808 | 6527843.2 | 4.822e-02 | 6.349e-02 | n/a | n/a | n/a | n/a | no |

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
| Relative correction 1e-2 | 1 | base | relative correction | 1.000e-02 | 5 | 50.023 | 1.573196 | 6700000.0 | 1.125e-01 | 9.523e-04 | n/a | n/a | n/a | n/a | no |
| Abs Delta Lambda 1e-2 | 1 | base | |Δlambda| | 1.000e-02 | 5 | 73.956 | 1.589603 | 6640990.0 | 8.980e-02 | 2.322e-02 | n/a | n/a | n/a | n/a | no |

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
| Relative correction 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Abs Delta Lambda 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

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
| Relative correction 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Abs Delta Lambda 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

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

### Accepted Continuation Step 12

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Relative correction 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Abs Delta Lambda 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_12/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 13

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Relative correction 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Abs Delta Lambda 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_13/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 14

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Relative correction 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Abs Delta Lambda 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_14/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 15

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Relative correction 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Abs Delta Lambda 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_15/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 16

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Relative correction 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Abs Delta Lambda 1e-2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/main_cases/report/plots/newton_by_step/step_16/relative_increment_vs_criterion.png) |  |

## Accepted-Step Newton Solves With Step-Length Cap

These sections show the separate Newton convergence history for the cases that use the moving history-box step-length cap, including the hybrid rough/fine run when present.

### Accepted Continuation Step 3

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 12.507 | 1.160511 | 6244976.2 | 8.378e-03 | 4.200e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_03/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 4

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 13.048 | 1.218281 | 6263018.9 | 9.530e-03 | 1.880e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_04/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 5

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 13.244 | 1.265697 | 6281061.5 | 6.449e-03 | 2.336e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_05/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 6

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 12.693 | 1.319074 | 6304725.9 | 8.164e-03 | 1.504e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_06/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 7

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 15.694 | 1.365753 | 6328390.3 | 7.865e-03 | 3.357e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_07/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 8

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 18.865 | 1.416590 | 6357219.1 | 8.375e-03 | 2.607e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_08/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 9

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 15.446 | 1.462473 | 6386048.0 | 1.030e-02 | 6.915e-04 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_09/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 10

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 18.119 | 1.504139 | 6414876.8 | 1.131e-02 | 4.770e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_10/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 11

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 17.206 | 1.540856 | 6443705.7 | 2.356e-02 | 1.177e-02 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_11/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 12

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 10 | 144.694 | 1.563288 | 6482766.8 | 3.949e-03 | 1.428e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_12/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 13

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 5 | 69.520 | 1.566244 | 6530355.3 | 1.329e-02 | 6.615e-04 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_13/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 14

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 21.672 | 1.567791 | 6587492.6 | 2.951e-02 | 2.896e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_14/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 15

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 15.979 | 1.569761 | 6654302.5 | 1.180e-01 | 1.981e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_15/relative_increment_vs_criterion.png) |  |

### Accepted Continuation Step 16

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 1 | 10.510 | 1.570160 | 6700000.0 | 1.136e-01 | 3.919e-03 | n/a | n/a | n/a | n/a | no |
| Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

| Criterion | Lambda |
| --- | --- |
| ![Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/criterion.png) | ![Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/lambda.png) |

| Abs Delta Lambda | Delta U |
| --- | --- |
| ![Abs Delta Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/delta_lambda.png) | ![Delta U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/delta_u.png) |

| Delta U / U |  |
| --- | --- |
| ![Delta U / U](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/delta_u_over_u.png) |  |

| Delta U vs Lambda | Delta U vs Criterion | Lambda vs Criterion |
| --- | --- | --- |
| ![Delta U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/correction_norm_vs_lambda.png) | ![Delta U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/correction_norm_vs_criterion.png) | ![Lambda vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/lambda_vs_criterion.png) |

| Delta U / U vs Lambda | Delta U / U vs Criterion |  |
| --- | --- | --- |
| ![Delta U / U vs Lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/relative_increment_vs_lambda.png) | ![Delta U / U vs Criterion](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/report/step_length_cap_case/report/plots/newton_by_step/step_16/relative_increment_vs_criterion.png) |  |

## Run Artifacts

- Relative correction 1e-2: command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/relative_correction_1e_2.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2`
- Abs Delta Lambda 1e-2: command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_2.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_2`
- Abs Delta Lambda 1e-3 + History-Box Cap: command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_3_cap_initial_segment.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment`
- Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off): command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop`
