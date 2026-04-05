# PMG Level Comparison: |Δlambda| < 1e-4, Init Relative Correction < 1e-3

This compares PMG runs of the 3D heterogeneous SSR benchmark with common Newton settings across mesh/element levels.

- Benchmark base: `benchmarks/slope_stability_3D_hetero_SSR_default/case.toml`
- Common stop target: `omega = 6.7e+06`
- Common continuation predictor: `secant`
- Common continuation Newton stop: `|Δlambda| < 1e-4`
- Common init Newton stop: `relative correction < 1e-3`
- Common `d_lambda_init = 0.1`
- Deflation: on (`max_deflation_basis_vectors = 48`)
- MPI ranks: `8`
- `P2` PETSc opts: `manualmg_coarse_operator_source=direct_elastic_full_system`, `mg_levels_ksp_type=chebyshev`, `mg_levels_ksp_max_it=3`, `mg_levels_pc_type=jacobi`, `mg_coarse_ksp_type=cg`, `mg_coarse_max_it=4`, `mg_coarse_rtol=0.0`, `pc_hypre_boomeramg_numfunctions=3`, `pc_hypre_boomeramg_nodal_coarsen=6`, `pc_hypre_boomeramg_nodal_coarsen_diag=1`, `pc_hypre_boomeramg_vec_interp_variant=3`, `pc_hypre_boomeramg_vec_interp_qmax=4`, `pc_hypre_boomeramg_vec_interp_smooth=true`, `pc_hypre_boomeramg_coarsen_type=HMIS`, `pc_hypre_boomeramg_interp_type=ext+i`, `pc_hypre_boomeramg_P_max=4`, `pc_hypre_boomeramg_strong_threshold=0.5`, `pc_hypre_boomeramg_max_iter=4`, `pc_hypre_boomeramg_tol=0.0`, `pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi`
- `P4` PETSc opts: `pc_hypre_boomeramg_max_iter=4`, `pc_hypre_boomeramg_tol=0.0`
- Artifact root: `/home/beremi/repos/slope_stability-1/artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6`

## Summary

| Case | Hierarchy | Unknowns | Runtime [s] | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Linear / Newton | Final relres | Final `ΔU/U` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2(L1) | P2(L1) -> P1(L1) | 80362 | 109.262 | 10 | 8 | 1.630064 | 6700000.0 | 9 | 45 | 9.800 | 1.995e-04 | 1.033e-03 |
| P2(L2) | P2(L2) -> P1(L2) -> P1(L1) | 145298 | 165.535 | 10 | 8 | 1.613993 | 6700000.0 | 9 | 50 | 10.160 | 3.332e-04 | 1.627e-03 |
| P2(L3) | P2(L3) -> P1(L3) -> P1(L1) | 265739 | 242.992 | 10 | 8 | 1.600485 | 6700000.0 | 9 | 49 | 10.367 | 5.594e-04 | 1.988e-03 |
| P2(L4) | P2(L4) -> P1(L4) -> P1(L1) | 490015 | 403.573 | 9 | 7 | 1.588718 | 6700000.0 | 9 | 48 | 10.396 | 1.277e-03 | 1.137e-03 |
| P2(L5) | P2(L5) -> P1(L5) -> P1(L1) | 913231 | 756.997 | 9 | 7 | 1.579168 | 6700000.0 | 9 | 49 | 11.122 | 3.424e-03 | 9.035e-04 |
| P4(L1) | P4(L1) -> P2(L1) -> P1(L1) | 616322 | 1016.493 | 9 | 7 | 1.568267 | 6700000.0 | 12 | 70 | 17.200 | 2.193e-03 | 1.757e-03 |
| P4(L2) | P4(L2) -> P1(L2) -> P1(L1) | 1124746 | 3492.921 | 9 | 7 | 1.560756 | 6700000.0 | 11 | 111 | 22.577 | 5.489e-03 | 1.053e-04 |

## Accepted-Step Lambda

| Step | P2(L1) lambda | P2(L2) lambda | P2(L3) lambda | P2(L4) lambda | P2(L5) lambda | P4(L1) lambda | P4(L2) lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.159851 | 1.160142 | 1.160167 | 1.160148 | 1.160186 | 1.160132 | 1.160419 |
| 4 | 1.244966 | 1.245594 | 1.245636 | 1.245440 | 1.245467 | 1.245680 | 1.245834 |
| 5 | 1.311414 | 1.312251 | 1.312242 | 1.311672 | 1.311155 | 1.311846 | 1.312015 |
| 6 | 1.418739 | 1.419547 | 1.419112 | 1.417709 | 1.416250 | 1.417552 | 1.417712 |
| 7 | 1.505755 | 1.506244 | 1.505218 | 1.503185 | 1.501140 | 1.502678 | 1.502722 |
| 8 | 1.608523 | 1.598021 | 1.587846 | 1.578284 | 1.570926 | 1.565523 | 1.559072 |
| 9 | 1.625608 | 1.610822 | 1.597923 | 1.588718 | 1.579168 | 1.568267 | 1.560756 |
| 10 | 1.630064 | 1.613993 | 1.600485 | n/a | n/a | n/a | n/a |

## Accepted-Step Omega

| Step | P2(L1) omega | P2(L2) omega | P2(L3) omega | P2(L4) omega | P2(L5) omega | P4(L1) omega | P4(L2) omega |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 6243523.3 | 6243992.0 | 6244284.5 | 6244501.9 | 6244827.3 | 6244903.5 | 6244963.6 |
| 4 | 6272104.9 | 6272646.0 | 6272836.7 | 6272826.5 | 6272869.4 | 6273128.7 | 6273222.6 |
| 5 | 6300686.6 | 6301300.1 | 6301388.9 | 6301151.1 | 6300911.4 | 6301353.9 | 6301481.6 |
| 6 | 6357849.9 | 6358608.2 | 6358493.3 | 6357800.3 | 6356995.5 | 6357804.3 | 6357999.5 |
| 7 | 6415013.2 | 6415916.4 | 6415597.6 | 6414449.5 | 6413079.6 | 6414254.6 | 6414517.5 |
| 8 | 6529339.9 | 6530532.6 | 6529806.4 | 6527747.9 | 6525247.9 | 6527155.4 | 6527553.4 |
| 9 | 6643666.6 | 6645148.9 | 6644015.1 | 6700000.0 | 6700000.0 | 6700000.0 | 6700000.0 |
| 10 | 6700000.0 | 6700000.0 | 6700000.0 | n/a | n/a | n/a | n/a |

## Accepted-Step Newton Iterations

| Step | P2(L1) Newton | P2(L2) Newton | P2(L3) Newton | P2(L4) Newton | P2(L5) Newton | P4(L1) Newton | P4(L2) Newton |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 4 | 3 | 4 | 4 | 4 | 4 | 3 |
| 4 | 4 | 4 | 4 | 4 | 4 | 5 | 6 |
| 5 | 4 | 5 | 4 | 4 | 4 | 6 | 6 |
| 6 | 5 | 4 | 4 | 4 | 4 | 5 | 8 |
| 7 | 5 | 5 | 5 | 5 | 5 | 5 | 11 |
| 8 | 10 | 13 | 12 | 13 | 18 | 23 | 50 |
| 9 | 9 | 9 | 11 | 14 | 10 | 22 | 27 |
| 10 | 4 | 7 | 5 | n/a | n/a | n/a | n/a |

## Accepted-Step Linear Iterations

| Step | P2(L1) linear | P2(L2) linear | P2(L3) linear | P2(L4) linear | P2(L5) linear | P4(L1) linear | P4(L2) linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 28 | 21 | 28 | 31 | 30 | 39 | 16 |
| 4 | 27 | 27 | 26 | 28 | 29 | 31 | 31 |
| 5 | 27 | 34 | 25 | 29 | 32 | 52 | 28 |
| 6 | 39 | 30 | 33 | 35 | 35 | 35 | 41 |
| 7 | 36 | 38 | 43 | 47 | 54 | 38 | 64 |
| 8 | 136 | 165 | 163 | 190 | 255 | 503 | 1610 |
| 9 | 96 | 85 | 119 | 139 | 110 | 506 | 716 |
| 10 | 52 | 108 | 71 | n/a | n/a | n/a | n/a |

## Accepted-Step Final Relative Correction

| Step | P2(L1) ΔU/U | P2(L2) ΔU/U | P2(L3) ΔU/U | P2(L4) ΔU/U | P2(L5) ΔU/U | P4(L1) ΔU/U | P4(L2) ΔU/U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.000897 | 0.001260 | 0.000463 | 0.000571 | 0.000740 | 0.000317 | 0.000849 |
| 4 | 0.001333 | 0.001117 | 0.001081 | 0.001464 | 0.000927 | 0.000555 | 0.000145 |
| 5 | 0.001271 | 0.001011 | 0.002366 | 0.001636 | 0.001802 | 0.002306 | 0.001657 |
| 6 | 0.005643 | 0.003637 | 0.002764 | 0.004485 | 0.002890 | 0.001649 | 0.000261 |
| 7 | 0.001454 | 0.002598 | 0.003128 | 0.005374 | 0.004518 | 0.002122 | 0.000054 |
| 8 | 0.002898 | 0.003660 | 0.002644 | 0.000529 | 0.003184 | 0.002616 | 0.000453 |
| 9 | 0.001829 | 0.002346 | 0.000224 | 0.001137 | 0.000903 | 0.001757 | 0.000105 |
| 10 | 0.001033 | 0.001627 | 0.001988 | n/a | n/a | n/a | n/a |

## Timing Totals

| Case | Constitutive [s] | Linear solve [s] | PC apply [s] | PC setup [s] | Orthogonalization [s] | Other [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2(L1) | 33.362 | 55.467 | 31.270 | 31.270 | 2.871 | 0.000 |
| P2(L2) | 71.765 | 76.671 | 40.158 | 40.158 | 7.535 | 0.000 |
| P2(L3) | 143.107 | 100.982 | 47.959 | 47.959 | 14.477 | 0.000 |
| P2(L4) | 289.923 | 153.223 | 64.431 | 64.431 | 28.973 | 0.000 |
| P2(L5) | 556.414 | 281.651 | 97.715 | 97.715 | 68.306 | 0.000 |
| P4(L1) | 102.698 | 664.446 | 44.524 | 44.524 | 129.695 | 30.607 |
| P4(L2) | 340.056 | 2364.393 | 68.974 | 68.974 | 538.605 | 111.919 |

## PMG Layout

| Case | ManualMG levels | Level orders | Level global sizes | Coarse operator | Coarse KSP/PC/Hypre | Fine smoother | Mid smoother |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| P2(L1) | 2 | [1, 2] | [10859, 80362] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | n/a |
| P2(L2) | 3 | [1, 1, 2] | [10835, 19282, 145298] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | chebyshev/jacobi |
| P2(L3) | 3 | [1, 1, 2] | [10847, 34810, 265739] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | chebyshev/jacobi |
| P2(L4) | 3 | [1, 1, 2] | [10850, 63261, 490015] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | chebyshev/jacobi |
| P2(L5) | 3 | [1, 1, 2] | [10859, 116561, 913231] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | chebyshev/jacobi |
| P4(L1) | 3 | [1, 2, 4] | [10859, 80362, 616322] | galerkin_free | preonly/hypre/boomeramg | richardson/sor | richardson/sor |
| P4(L2) | 3 | [1, 1, 4] | [10835, 19282, 1124746] | galerkin_free | preonly/hypre/boomeramg | richardson/sor | richardson/sor |

## Comparison Plots

### Lambda vs omega

![Lambda vs omega](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/lambda_omega_overlay.png)

### Lambda by accepted state

![Lambda by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/lambda_vs_state.png)

### Omega by accepted state

![Omega by accepted state](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/omega_vs_state.png)

### Runtime by case

![Runtime by case](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/runtime_by_case.png)

### Timing breakdown

![Timing breakdown](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/timing_breakdown_stacked.png)

### Newton iterations per step

![Newton iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_newton_iterations.png)

### Linear iterations per step

![Linear iterations per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_linear_iterations.png)

### Linear per Newton

![Linear per Newton](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_linear_per_newton.png)

### Wall time per step

![Wall time per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_wall_time.png)

### Final relative residual per step

![Final relative residual per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_relres_end.png)

### Final relative correction per step

![Final relative correction per step](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/report/plots/step_relcorr_end.png)

## Existing Per-Run Plots

### Continuation Curve

| P2(L1) | P2(L2) | P2(L3) | P2(L4) | P2(L5) | P4(L1) | P4(L2) |
| --- | --- | --- | --- | --- | --- | --- |
| ![P2(L1) omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l1/plots/petsc_omega_lambda.png) | ![P2(L2) omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l2/plots/petsc_omega_lambda.png) | ![P2(L3) omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l3/plots/petsc_omega_lambda.png) | ![P2(L4) omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l4/plots/petsc_omega_lambda.png) | ![P2(L5) omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l5/plots/petsc_omega_lambda.png) | ![P4(L1) omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l1/plots/petsc_omega_lambda.png) | ![P4(L2) omega-lambda](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2/plots/petsc_omega_lambda.png) |

### Displacements

| P2(L1) | P2(L2) | P2(L3) | P2(L4) | P2(L5) | P4(L1) | P4(L2) |
| --- | --- | --- | --- | --- | --- | --- |
| ![P2(L1) displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l1/plots/petsc_displacements_3D.png) | ![P2(L2) displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l2/plots/petsc_displacements_3D.png) | ![P2(L3) displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l3/plots/petsc_displacements_3D.png) | ![P2(L4) displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l4/plots/petsc_displacements_3D.png) | ![P2(L5) displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l5/plots/petsc_displacements_3D.png) | ![P4(L1) displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l1/plots/petsc_displacements_3D.png) | ![P4(L2) displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2/plots/petsc_displacements_3D.png) |

### Deviatoric Strain

| P2(L1) | P2(L2) | P2(L3) | P2(L4) | P2(L5) | P4(L1) | P4(L2) |
| --- | --- | --- | --- | --- | --- | --- |
| ![P2(L1) deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l1/plots/petsc_deviatoric_strain_3D.png) | ![P2(L2) deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l2/plots/petsc_deviatoric_strain_3D.png) | ![P2(L3) deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l3/plots/petsc_deviatoric_strain_3D.png) | ![P2(L4) deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l4/plots/petsc_deviatoric_strain_3D.png) | ![P2(L5) deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l5/plots/petsc_deviatoric_strain_3D.png) | ![P4(L1) deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l1/plots/petsc_deviatoric_strain_3D.png) | ![P4(L2) deviatoric strain](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2/plots/petsc_deviatoric_strain_3D.png) |

### Step Displacement History

| P2(L1) | P2(L2) | P2(L3) | P2(L4) | P2(L5) | P4(L1) | P4(L2) |
| --- | --- | --- | --- | --- | --- | --- |
| ![P2(L1) step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l1/plots/petsc_step_displacement.png) | ![P2(L2) step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l2/plots/petsc_step_displacement.png) | ![P2(L3) step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l3/plots/petsc_step_displacement.png) | ![P2(L4) step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l4/plots/petsc_step_displacement.png) | ![P2(L5) step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l5/plots/petsc_step_displacement.png) | ![P4(L1) step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l1/plots/petsc_step_displacement.png) | ![P4(L2) step displacement](../../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2/plots/petsc_step_displacement.png) |

## Run Artifacts

- P2(L1): command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/commands/p2_l1.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l1`
- P2(L2): command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/commands/p2_l2.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l2`
- P2(L3): command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/commands/p2_l3.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l3`
- P2(L4): command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/commands/p2_l4.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l4`
- P2(L5): command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/commands/p2_l5.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p2_l5`
- P4(L1): command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/commands/p4_l1.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l1`
- P4(L2): command `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/commands/p4_l2.json`, artifact `../../artifacts/comparisons/slope_stability_3D_hetero_SSR_default/pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6/runs/p4_l2`
