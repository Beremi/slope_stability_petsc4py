# P4(L1) PMG Comparison

This compares `P4(L1)` with the PMG backend using the standard secant predictor, stopping at `omega = 6.7e+06`.

- Base benchmark: `benchmarks/3d_hetero_ssr_default/case.toml`
- Overrides: `elem_type = "P4"`, `mesh_path = "/home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh"`, `pc_backend = "pmg_shell"`, `node_ordering = "block_metis"`
- MPI ranks: `8`
- PMG PETSc opts: `pc_hypre_boomeramg_max_iter=4`, `pc_hypre_boomeramg_tol=0.0`
- Cases: Relative correction 1e-2, Relative correction 1e-2 (rerun current code), Abs Delta Lambda 1e-3 + History-Box Cap, Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1), Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1), Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1), Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1)
- History-box step-length cap: affine-rescale the full current `lambda-omega` history into `[0,1]^2`, measure the first segment (`lambda 1.0 -> 1.1`) there, and limit the next step so the projected last-segment direction has at most the same normalized length.

- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6`

## Summary

| Case | Residual tol | Stop criterion | Stop tol | Runtime [s] | Speedup | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Init linear | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Relative correction 1e-2 | `1.0e-04` | relative correction | `1.0e-02` | 358.071 | 1.000 | 9 | 7 | 1.573196 | 6700000.0 | 9 | 27 | 66 | 378 | 14.000 | 1.125e-01 | 9.523e-04 |
| Relative correction 1e-2 (rerun current code) | `1.0e-04` | relative correction | `1.0e-02` | 415.104 | 1.000 | 9 | 7 | 1.573196 | 6700000.0 | 9 | 27 | 66 | 378 | 14.000 | 1.125e-01 | 9.523e-04 |
| Abs Delta Lambda 1e-3 + History-Box Cap | `1.0e-04` | |Δlambda| | `1.0e-03` | 618.713 | 1.000 | 16 | 14 | 1.570160 | 6700000.0 | 20 | 38 | 159 | 404 | 10.632 | 1.136e-01 | 3.919e-03 |
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | `1.0e-04` | |Δlambda| | `1.0e-03` | 600.770 | 1.000 | 16 | 14 | 1.569866 | 6700000.0 | 9 | 43 | 66 | 486 | 11.302 | 5.384e-02 | 5.499e-04 |
| Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | `1.0e-04` | |Δlambda| | `1.0e-03` | 539.390 | 1.000 | 9 | 7 | 1.571281 | 6700000.0 | 9 | 33 | 66 | 506 | 15.333 | 9.741e-02 | 7.792e-04 |
| Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | `1.0e-04` | relative correction | `1.0e-03` | 770.720 | 1.000 | 9 | 7 | 1.589106 | 6700000.0 | 9 | 50 | 66 | 740 | 14.800 | 6.557e-02 | 3.112e-04 |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | `1.0e-04` | |Δlambda| | `1.0e-04` | 1064.022 | 1.000 | 9 | 7 | 1.568533 | 6700000.0 | 9 | 69 | 66 | 1057 | 15.319 | 9.499e-03 | 3.470e-04 |

## Accepted-Step Lambda

| Step | Relative correction 1e-2 lambda | Relative correction 1e-2 (rerun current code) lambda | Abs Delta Lambda 1e-3 + History-Box Cap lambda | Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) lambda | Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) lambda | Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) lambda | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.168997 | 1.168997 | 1.160511 | 1.158363 | 1.158363 | 1.158393 | 1.158393 |
| 4 | 1.243909 | 1.243909 | 1.218281 | 1.217256 | 1.243835 | 1.243695 | 1.243261 |
| 5 | 1.309466 | 1.309466 | 1.265697 | 1.264685 | 1.309426 | 1.309042 | 1.308950 |
| 6 | 1.415035 | 1.415035 | 1.319074 | 1.317884 | 1.415023 | 1.413777 | 1.413847 |
| 7 | 1.501143 | 1.501143 | 1.365753 | 1.364250 | 1.500866 | 1.498487 | 1.498371 |
| 8 | 1.568645 | 1.568645 | 1.416590 | 1.415196 | 1.566380 | 1.565619 | 1.565367 |
| 9 | 1.573196 | 1.573196 | 1.462473 | 1.460924 | 1.571281 | 1.589106 | 1.568533 |
| 10 | n/a | n/a | 1.504139 | 1.509437 | n/a | n/a | n/a |
| 11 | n/a | n/a | 1.540856 | 1.550025 | n/a | n/a | n/a |
| 12 | n/a | n/a | 1.563288 | 1.564651 | n/a | n/a | n/a |
| 13 | n/a | n/a | 1.566244 | 1.567124 | n/a | n/a | n/a |
| 14 | n/a | n/a | 1.567791 | 1.568443 | n/a | n/a | n/a |
| 15 | n/a | n/a | 1.569761 | 1.569889 | n/a | n/a | n/a |
| 16 | n/a | n/a | 1.570160 | 1.569866 | n/a | n/a | n/a |

## Accepted-Step Omega

| Step | Relative correction 1e-2 omega | Relative correction 1e-2 (rerun current code) omega | Abs Delta Lambda 1e-3 + History-Box Cap omega | Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) omega | Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) omega | Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) omega | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) omega |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 6244441.3 | 6244441.3 | 6244976.2 | 6244441.3 | 6244441.3 | 6244441.3 | 6244441.3 |
| 4 | 6272225.7 | 6272225.7 | 6263018.9 | 6262455.4 | 6272225.7 | 6272225.7 | 6272225.7 |
| 5 | 6300010.2 | 6300010.2 | 6281061.5 | 6280469.6 | 6300010.2 | 6300010.2 | 6300010.2 |
| 6 | 6355579.1 | 6355579.1 | 6304725.9 | 6303961.6 | 6355579.1 | 6355579.1 | 6355579.1 |
| 7 | 6411148.0 | 6411148.0 | 6328390.3 | 6327453.6 | 6411148.0 | 6411148.0 | 6411148.0 |
| 8 | 6522285.7 | 6522285.7 | 6357219.1 | 6356123.7 | 6522285.7 | 6522285.7 | 6522285.7 |
| 9 | 6700000.0 | 6700000.0 | 6386048.0 | 6384793.8 | 6700000.0 | 6700000.0 | 6700000.0 |
| 10 | n/a | n/a | 6414876.8 | 6418532.8 | n/a | n/a | n/a |
| 11 | n/a | n/a | 6443705.7 | 6452271.7 | n/a | n/a | n/a |
| 12 | n/a | n/a | 6482766.8 | 6492302.4 | n/a | n/a | n/a |
| 13 | n/a | n/a | 6530355.3 | 6542269.0 | n/a | n/a | n/a |
| 14 | n/a | n/a | 6587492.6 | 6601316.6 | n/a | n/a | n/a |
| 15 | n/a | n/a | 6654302.5 | 6670389.0 | n/a | n/a | n/a |
| 16 | n/a | n/a | 6700000.0 | 6700000.0 | n/a | n/a | n/a |

## Accepted-Step Newton Iterations

| Step | Relative correction 1e-2 Newton | Relative correction 1e-2 (rerun current code) Newton | Abs Delta Lambda 1e-3 + History-Box Cap Newton | Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) Newton | Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) Newton | Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) Newton | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) Newton |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | 2 | 2 | 4 | 4 | 5 | 5 |
| 4 | 3 | 3 | 2 | 2 | 3 | 3 | 7 |
| 5 | 2 | 2 | 2 | 2 | 2 | 6 | 8 |
| 6 | 2 | 2 | 2 | 2 | 2 | 7 | 5 |
| 7 | 1 | 1 | 2 | 3 | 2 | 7 | 8 |
| 8 | 12 | 12 | 2 | 2 | 13 | 20 | 21 |
| 9 | 5 | 5 | 2 | 2 | 7 | 2 | 15 |
| 10 | n/a | n/a | 2 | 2 | n/a | n/a | n/a |
| 11 | n/a | n/a | 2 | 6 | n/a | n/a | n/a |
| 12 | n/a | n/a | 10 | 9 | n/a | n/a | n/a |
| 13 | n/a | n/a | 5 | 3 | n/a | n/a | n/a |
| 14 | n/a | n/a | 2 | 2 | n/a | n/a | n/a |
| 15 | n/a | n/a | 2 | 3 | n/a | n/a | n/a |
| 16 | n/a | n/a | 1 | 1 | n/a | n/a | n/a |

## Accepted-Step Linear Iterations

| Step | Relative correction 1e-2 linear | Relative correction 1e-2 (rerun current code) linear | Abs Delta Lambda 1e-3 + History-Box Cap linear | Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) linear | Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) linear | Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) linear | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 40 | 40 | 11 | 62 | 62 | 77 | 77 |
| 4 | 22 | 22 | 11 | 21 | 33 | 33 | 87 |
| 5 | 9 | 9 | 11 | 22 | 19 | 58 | 76 |
| 6 | 12 | 12 | 10 | 12 | 9 | 51 | 56 |
| 7 | 4 | 4 | 15 | 27 | 15 | 50 | 69 |
| 8 | 224 | 224 | 20 | 8 | 258 | 438 | 461 |
| 9 | 67 | 67 | 14 | 14 | 110 | 33 | 231 |
| 10 | n/a | n/a | 18 | 14 | n/a | n/a | n/a |
| 11 | n/a | n/a | 16 | 55 | n/a | n/a | n/a |
| 12 | n/a | n/a | 157 | 136 | n/a | n/a | n/a |
| 13 | n/a | n/a | 77 | 47 | n/a | n/a | n/a |
| 14 | n/a | n/a | 22 | 21 | n/a | n/a | n/a |
| 15 | n/a | n/a | 12 | 30 | n/a | n/a | n/a |
| 16 | n/a | n/a | 10 | 17 | n/a | n/a | n/a |

## Accepted-Step Final Relative Correction

| Step | Relative correction 1e-2 ΔU/U | Relative correction 1e-2 (rerun current code) ΔU/U | Abs Delta Lambda 1e-3 + History-Box Cap ΔU/U | Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) ΔU/U | Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) ΔU/U | Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) ΔU/U | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) ΔU/U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.009032 | 0.009032 | 0.004200 | 0.001165 | 0.001165 | 0.000608 | 0.000608 |
| 4 | 0.003042 | 0.003042 | 0.001880 | 0.000635 | 0.000579 | 0.000993 | 0.000114 |
| 5 | 0.001279 | 0.001279 | 0.002336 | 0.001559 | 0.001887 | 0.000999 | 0.000094 |
| 6 | 0.003224 | 0.003224 | 0.001504 | 0.001456 | 0.003029 | 0.000891 | 0.000247 |
| 7 | 0.008613 | 0.008613 | 0.003357 | 0.001090 | 0.008711 | 0.000272 | 0.000125 |
| 8 | 0.009311 | 0.009311 | 0.002607 | 0.001312 | 0.001771 | 0.000166 | 0.002026 |
| 9 | 0.000952 | 0.000952 | 0.000692 | 0.004223 | 0.000779 | 0.000311 | 0.000347 |
| 10 | n/a | n/a | 0.004770 | 0.003031 | n/a | n/a | n/a |
| 11 | n/a | n/a | 0.011772 | 0.003317 | n/a | n/a | n/a |
| 12 | n/a | n/a | 0.001428 | 0.000893 | n/a | n/a | n/a |
| 13 | n/a | n/a | 0.000662 | 0.001886 | n/a | n/a | n/a |
| 14 | n/a | n/a | 0.002896 | 0.002917 | n/a | n/a | n/a |
| 15 | n/a | n/a | 0.001981 | 0.000296 | n/a | n/a | n/a |
| 16 | n/a | n/a | 0.003919 | 0.000550 | n/a | n/a | n/a |

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

| Relative correction 1e-2 | Relative correction 1e-2 (rerun current code) | Abs Delta Lambda 1e-3 + History-Box Cap | Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) |
| --- | --- | --- | --- | --- | --- | --- |
| ![Relative correction 1e-2 omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_omega_lambda.png) | ![Relative correction 1e-2 (rerun current code) omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2_rerun/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment_initrelcorr_dlambda0p1/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_initrelcorr_dlambda0p1/plots/petsc_omega_lambda.png) | ![Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_3_initrelcorr_dlambda0p1/plots/petsc_omega_lambda.png) | ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) omega-lambda](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1/plots/petsc_omega_lambda.png) |

### Displacements

| Relative correction 1e-2 | Relative correction 1e-2 (rerun current code) | Abs Delta Lambda 1e-3 + History-Box Cap | Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) |
| --- | --- | --- | --- | --- | --- | --- |
| ![Relative correction 1e-2 displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_displacements_3D.png) | ![Relative correction 1e-2 (rerun current code) displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2_rerun/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment_initrelcorr_dlambda0p1/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_initrelcorr_dlambda0p1/plots/petsc_displacements_3D.png) | ![Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_3_initrelcorr_dlambda0p1/plots/petsc_displacements_3D.png) | ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1/plots/petsc_displacements_3D.png) |

### Deviatoric Strain

| Relative correction 1e-2 | Relative correction 1e-2 (rerun current code) | Abs Delta Lambda 1e-3 + History-Box Cap | Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) |
| --- | --- | --- | --- | --- | --- | --- |
| ![Relative correction 1e-2 deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_deviatoric_strain_3D.png) | ![Relative correction 1e-2 (rerun current code) deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2_rerun/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment_initrelcorr_dlambda0p1/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_initrelcorr_dlambda0p1/plots/petsc_deviatoric_strain_3D.png) | ![Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_3_initrelcorr_dlambda0p1/plots/petsc_deviatoric_strain_3D.png) | ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) deviatoric strain](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1/plots/petsc_deviatoric_strain_3D.png) |

### Step Displacement History

| Relative correction 1e-2 | Relative correction 1e-2 (rerun current code) | Abs Delta Lambda 1e-3 + History-Box Cap | Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) |
| --- | --- | --- | --- | --- | --- | --- |
| ![Relative correction 1e-2 step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2/plots/petsc_step_displacement.png) | ![Relative correction 1e-2 (rerun current code) step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2_rerun/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment_initrelcorr_dlambda0p1/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_initrelcorr_dlambda0p1/plots/petsc_step_displacement.png) | ![Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_3_initrelcorr_dlambda0p1/plots/petsc_step_displacement.png) | ![Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) step displacement](../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1/plots/petsc_step_displacement.png) |

## Accepted-Step Newton Solves

These sections overlay the successful Newton solve that produced each accepted continuation step for the main PMG cases without the step-length cap.

### Accepted Continuation Step 3

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Relative correction 1e-2 | 1 | base | relative correction | 1.000e-02 | 2 | 24.861 | 1.168997 | 6244441.3 | 1.953e-02 | 9.032e-03 | n/a | n/a | n/a | n/a | no |
| Relative correction 1e-2 (rerun current code) | 1 | base | relative correction | 1.000e-02 | 2 | 28.575 | 1.168997 | 6244441.3 | 1.953e-02 | 9.032e-03 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 4 | 50.578 | 1.158363 | 6244441.3 | 1.536e-02 | 1.165e-03 | n/a | n/a | n/a | 2 | no |
| Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | relative correction | 1.000e-03 | 5 | 62.954 | 1.158393 | 6244441.3 | 8.200e-03 | 6.083e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 5 | 65.200 | 1.158393 | 6244441.3 | 8.200e-03 | 6.083e-04 | n/a | n/a | n/a | 2 | no |

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
| Relative correction 1e-2 (rerun current code) | 1 | base | relative correction | 1.000e-02 | 3 | 21.798 | 1.243909 | 6272225.7 | 2.176e-02 | 3.042e-03 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 3 | 29.660 | 1.243835 | 6272225.7 | 1.342e-02 | 5.789e-04 | n/a | n/a | n/a | 2 | no |
| Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | relative correction | 1.000e-03 | 3 | 29.918 | 1.243695 | 6272225.7 | 1.862e-02 | 9.932e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 7 | 81.817 | 1.243261 | 6272225.7 | 6.452e-03 | 1.139e-04 | n/a | n/a | n/a | 2 | no |

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
| Relative correction 1e-2 (rerun current code) | 1 | base | relative correction | 1.000e-02 | 2 | 11.318 | 1.309466 | 6300010.2 | 3.886e-02 | 1.279e-03 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 17.909 | 1.309426 | 6300010.2 | 1.677e-02 | 1.887e-03 | n/a | n/a | n/a | 2 | no |
| Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | relative correction | 1.000e-03 | 6 | 57.239 | 1.309042 | 6300010.2 | 9.747e-03 | 9.994e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 8 | 80.963 | 1.308950 | 6300010.2 | 2.911e-03 | 9.448e-05 | n/a | n/a | n/a | 2 | no |

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
| Relative correction 1e-2 (rerun current code) | 1 | base | relative correction | 1.000e-02 | 2 | 13.142 | 1.415035 | 6355579.1 | 6.689e-02 | 3.224e-03 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 12.262 | 1.415023 | 6355579.1 | 6.053e-02 | 3.029e-03 | n/a | n/a | n/a | 2 | no |
| Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | relative correction | 1.000e-03 | 7 | 58.409 | 1.413777 | 6355579.1 | 7.520e-03 | 8.906e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 5 | 53.744 | 1.413847 | 6355579.1 | 7.084e-03 | 2.475e-04 | n/a | n/a | n/a | 2 | no |

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
| Relative correction 1e-2 (rerun current code) | 1 | base | relative correction | 1.000e-02 | 1 | 5.290 | 1.501143 | 6411148.0 | 1.268e-01 | 8.613e-03 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 15.844 | 1.500866 | 6411148.0 | 2.886e-02 | 8.711e-03 | n/a | n/a | n/a | 2 | no |
| Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | relative correction | 1.000e-03 | 7 | 58.948 | 1.498487 | 6411148.0 | 5.463e-03 | 2.720e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 8 | 77.694 | 1.498371 | 6411148.0 | 2.062e-03 | 1.252e-04 | n/a | n/a | n/a | 2 | no |

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
| Relative correction 1e-2 (rerun current code) | 1 | base | relative correction | 1.000e-02 | 12 | 185.025 | 1.568645 | 6522285.7 | 1.739e-02 | 9.311e-03 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 13 | 221.045 | 1.566380 | 6522285.7 | 9.651e-03 | 1.771e-03 | n/a | n/a | n/a | 2 | no |
| Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | relative correction | 1.000e-03 | 20 | 381.520 | 1.565619 | 6522285.7 | 1.482e-03 | 1.665e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 21 | 406.837 | 1.565367 | 6522285.7 | 2.114e-03 | 2.026e-03 | n/a | n/a | n/a | 2 | no |

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
| Relative correction 1e-2 (rerun current code) | 1 | base | relative correction | 1.000e-02 | 5 | 58.568 | 1.573196 | 6700000.0 | 1.125e-01 | 9.523e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 7 | 97.922 | 1.571281 | 6700000.0 | 9.741e-02 | 7.792e-04 | n/a | n/a | n/a | 2 | no |
| Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | relative correction | 1.000e-03 | 2 | 28.254 | 1.589106 | 6700000.0 | 6.557e-02 | 3.112e-04 | n/a | n/a | n/a | 2 | no |
| Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-04 | 15 | 202.282 | 1.568533 | 6700000.0 | 9.499e-03 | 3.470e-04 | n/a | n/a | n/a | 2 | no |

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

## Accepted-Step Newton Solves With Step-Length Cap

These sections show the separate Newton convergence history for the cases that use the moving history-box step-length cap, including the hybrid rough/fine run when present.

### Accepted Continuation Step 3

| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Abs Delta Lambda 1e-3 + History-Box Cap | 1 | base | |Δlambda| | 1.000e-03 | 2 | 12.507 | 1.160511 | 6244976.2 | 8.378e-03 | 4.200e-03 | n/a | n/a | n/a | n/a | no |
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 4 | 52.030 | 1.158363 | 6244441.3 | 1.536e-02 | 1.165e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 19.501 | 1.217256 | 6262455.4 | 1.220e-02 | 6.350e-04 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 20.295 | 1.264685 | 6280469.6 | 1.662e-02 | 1.559e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 14.594 | 1.317884 | 6303961.6 | 1.873e-02 | 1.456e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 3 | 28.122 | 1.364250 | 6327453.6 | 1.318e-02 | 1.090e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 12.187 | 1.415196 | 6356123.7 | 3.770e-02 | 1.312e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 16.291 | 1.460924 | 6384793.8 | 2.752e-02 | 4.223e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 16.443 | 1.509437 | 6418532.8 | 2.786e-02 | 3.031e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 6 | 61.884 | 1.550025 | 6452271.7 | 1.072e-02 | 3.317e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 9 | 130.972 | 1.564651 | 6492302.4 | 1.434e-02 | 8.927e-04 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 3 | 42.338 | 1.567124 | 6542269.0 | 2.967e-02 | 1.886e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 2 | 19.573 | 1.568443 | 6601316.6 | 5.370e-02 | 2.917e-03 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 3 | 30.045 | 1.569889 | 6670389.0 | 2.725e-02 | 2.963e-04 | n/a | n/a | n/a | 2 | no |

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
| Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1) | 1 | base | |Δlambda| | 1.000e-03 | 1 | 14.561 | 1.569866 | 6700000.0 | 5.384e-02 | 5.499e-04 | n/a | n/a | n/a | 2 | no |

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
- Relative correction 1e-2 (rerun current code): command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/relative_correction_1e_2_rerun.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_2_rerun`
- Abs Delta Lambda 1e-3 + History-Box Cap: command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_3_cap_initial_segment.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment`
- Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1): command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_3_cap_initial_segment_initrelcorr_dlambda0p1.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_cap_initial_segment_initrelcorr_dlambda0p1`
- Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1): command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_3_initrelcorr_dlambda0p1.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_3_initrelcorr_dlambda0p1`
- Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1): command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/relative_correction_1e_3_initrelcorr_dlambda0p1.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/relative_correction_1e_3_initrelcorr_dlambda0p1`
- Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1): command `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/commands/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1.json`, artifact `../../artifacts/comparisons/3d_hetero_ssr_default/p4_l1_pmg_newton_stops_omega6p7e6/runs/absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1`
