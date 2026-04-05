# P4(L2) PMG Comparison: Current vs P2-Intermediate Hierarchy

This compares the existing mixed `P4(L2)` PMG hierarchy against a rerun that inserts a same-mesh `P2(L2)` intermediate level.

- Common benchmark base: `benchmarks/slope_stability_3D_hetero_SSR_default/case.toml`
- Common stop target: `omega = 6.7e+06`
- Common continuation Newton stop: `|Δlambda| < 1e-4`
- Common init Newton stop: `relative correction < 1e-3`
- Common `d_lambda_init = 0.1`
- Deflation: on (`max_deflation_basis_vectors = 48`)
- MPI ranks: `8`
- PETSc opts: `pc_hypre_boomeramg_max_iter=4`, `pc_hypre_boomeramg_tol=0.0`
- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6`

## Headline

- Runtime ratio (`new/current`): `0.929x`
- Final lambda shift (`new - current`): `+0.000098`
- Continuation linear-iteration shift (`new - current`): `-314`

## Summary

| Case | Hierarchy | Unknowns | Runtime [s] | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current P4(L2) | P4(L2) -> P1(L2) -> P1(L1) | 1124746 | 3492.921 | 9 | 7 | 1.560756 | 6700000.0 | 11 | 111 | 2506 | 22.577 | 5.489e-03 | 1.053e-04 |
| P4(L2) with P2(L2) Intermediate | P4(L2) -> P2(L2) -> P1(L2) -> P1(L1) | 1124746 | 3245.219 | 9 | 7 | 1.560854 | 6700000.0 | 11 | 100 | 2192 | 21.920 | 4.434e-03 | 3.199e-04 |

## Accepted-Step Lambda

| Step | Current P4(L2) lambda | P4(L2) with P2(L2) Intermediate lambda |
| --- | ---: | ---: |
| 3 | 1.160419 | 1.160390 |
| 4 | 1.245834 | 1.245684 |
| 5 | 1.312015 | 1.312021 |
| 6 | 1.417712 | 1.417777 |
| 7 | 1.502722 | 1.502774 |
| 8 | 1.559072 | 1.559053 |
| 9 | 1.560756 | 1.560854 |

## Accepted-Step Omega

| Step | Current P4(L2) omega | P4(L2) with P2(L2) Intermediate omega |
| --- | ---: | ---: |
| 3 | 6244963.6 | 6244947.7 |
| 4 | 6273222.6 | 6273173.4 |
| 5 | 6301481.6 | 6301399.1 |
| 6 | 6357999.5 | 6357850.5 |
| 7 | 6414517.5 | 6414301.9 |
| 8 | 6527553.4 | 6527204.7 |
| 9 | 6700000.0 | 6700000.0 |

## Accepted-Step Newton Iterations

| Step | Current P4(L2) Newton | P4(L2) with P2(L2) Intermediate Newton |
| --- | ---: | ---: |
| 3 | 3 | 2 |
| 4 | 6 | 6 |
| 5 | 6 | 2 |
| 6 | 8 | 4 |
| 7 | 11 | 7 |
| 8 | 50 | 53 |
| 9 | 27 | 26 |

## Accepted-Step Linear Iterations

| Step | Current P4(L2) linear | P4(L2) with P2(L2) Intermediate linear |
| --- | ---: | ---: |
| 3 | 16 | 8 |
| 4 | 31 | 22 |
| 5 | 28 | 7 |
| 6 | 41 | 16 |
| 7 | 64 | 29 |
| 8 | 1610 | 1510 |
| 9 | 716 | 600 |

## Accepted-Step Final Relative Correction

| Step | Current P4(L2) ΔU/U | P4(L2) with P2(L2) Intermediate ΔU/U |
| --- | ---: | ---: |
| 3 | 0.000849 | 0.004553 |
| 4 | 0.000145 | 0.001342 |
| 5 | 0.001657 | 0.000866 |
| 6 | 0.000261 | 0.003554 |
| 7 | 0.000054 | 0.001348 |
| 8 | 0.000453 | 0.000703 |
| 9 | 0.000105 | 0.000320 |

## Timing Totals

| Case | Constitutive [s] | Linear solve [s] | PC apply [s] | PC setup [s] | Orthogonalization [s] | Other [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current P4(L2) | 340.056 | 2364.393 | 68.974 | 68.974 | 538.605 | 111.919 |
| P4(L2) with P2(L2) Intermediate | 306.525 | 2124.046 | 117.263 | 117.263 | 527.886 | 52.236 |

## PMG Layout

| Case | ManualMG levels | Level orders | Level global sizes | Coarse operator | Coarse KSP/PC/Hypre | Fine smoother | Mid smoother |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Current P4(L2) | 3 | [1, 1, 4] | [10835, 19282, 1124746] | galerkin_free | preonly/hypre/boomeramg | richardson/sor | richardson/sor |
| P4(L2) with P2(L2) Intermediate | 4 | [1, 1, 2, 4] | [10835, 19282, 145298, 1124746] | galerkin_free | preonly/hypre/boomeramg | richardson/sor | richardson/sor |

## Comparison Plots

### Lambda vs omega

![Lambda vs omega](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/lambda_omega_overlay.png)

### Lambda by accepted state

![Lambda by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/lambda_vs_state.png)

### Omega by accepted state

![Omega by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/omega_vs_state.png)

### Runtime by case

![Runtime by case](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/runtime_by_case.png)

### Timing breakdown

![Timing breakdown](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/timing_breakdown_stacked.png)

### Newton iterations per step

![Newton iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_newton_iterations.png)

### Linear iterations per step

![Linear iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_linear_iterations.png)

### Linear per Newton

![Linear per Newton](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_linear_per_newton.png)

### Wall time per step

![Wall time per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_wall_time.png)

### Final relative residual per step

![Final relative residual per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_relres_end.png)

### Final relative correction per step

![Final relative correction per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_relcorr_end.png)

## Existing Per-Run Plots

### Continuation Curve

| Current P4(L2) | P4(L2) with P2(L2) Intermediate |
| --- | --- |
| ![Current P4(L2) omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2/plots/petsc_omega_lambda.png) | ![P4(L2) with P2(L2) Intermediate omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2_p2_intermediate/plots/petsc_omega_lambda.png) |

### Displacements

| Current P4(L2) | P4(L2) with P2(L2) Intermediate |
| --- | --- |
| ![Current P4(L2) displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2/plots/petsc_displacements_3D.png) | ![P4(L2) with P2(L2) Intermediate displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2_p2_intermediate/plots/petsc_displacements_3D.png) |

### Deviatoric Strain

| Current P4(L2) | P4(L2) with P2(L2) Intermediate |
| --- | --- |
| ![Current P4(L2) deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2/plots/petsc_deviatoric_strain_3D.png) | ![P4(L2) with P2(L2) Intermediate deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2_p2_intermediate/plots/petsc_deviatoric_strain_3D.png) |

### Step Displacement History

| Current P4(L2) | P4(L2) with P2(L2) Intermediate |
| --- | --- |
| ![Current P4(L2) step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2/plots/petsc_step_displacement.png) | ![P4(L2) with P2(L2) Intermediate step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2_p2_intermediate/plots/petsc_step_displacement.png) |

## Run Artifacts

- Current P4(L2): command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/commands/p4_l2.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2`
- P4(L2) with P2(L2) Intermediate: command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/commands/p4_l2_p2_intermediate.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2_p2_intermediate`
