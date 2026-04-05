# P2(L2) Rank-8: Hypre Default vs Mixed PMG-Shell

## Configuration

- Fine mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L2.msh`
- Mixed PMG coarse mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- MPI ranks: `8`
- Outer solver for both runs: `PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE`
- Requested continuation advances after init: `10`
- Actual runner `step_max`: `12`
- Note: `step_max` counts accepted states including the 2-state initialization, so `step_max = continuation_steps + 2`.

## Commands

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/hypre_default_rank8_step12 --mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L2.msh --elem_type P2 --node_ordering original --step_max 12 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear_tolerance 0.1 --linear_max_iter 100 --no-store_step_u --pc_backend hypre --preconditioner_matrix_source tangent
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/pmg_shell_mixed_rank8_step12 --mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L2.msh --elem_type P2 --node_ordering original --step_max 12 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear_tolerance 0.1 --linear_max_iter 100 --no-store_step_u --pc_backend pmg_shell --pmg_coarse_mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --preconditioner_matrix_source tangent --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
```

## Headline Metrics

| Metric | Hypre default | Mixed PMG-shell | PMG / Hypre |
| --- | ---: | ---: | ---: |
| Runtime [s] | 826.419 | 408.703 | 0.495x |
| Continuation wall time [s] | 826.419 | 408.703 | 0.495x |
| Init linear iterations | 190 | 38 | 0.200x |
| Continuation linear iterations total | 5806 | 1091 | 0.188x |
| Accepted-step Newton iterations total | 112 | 117 | 1.045x |
| Total linear / Newton | 51.839 | 9.325 | 0.180x |
| Init solve [s] | 21.801 | 6.587 | 0.302x |
| Init preconditioner collector [s] | 5.072 | 7.962 | 1.570x |
| Continuation solve [s] | 664.965 | 189.794 | 0.285x |
| Continuation preconditioner collector [s] | 4.798 | 78.406 | 16.342x |
| Continuation orthogonalization [s] | 27.946 | 28.004 | 1.002x |
| Preconditioner setup diagnostic [s] | 9.870 | 86.368 | 8.751x |
| Preconditioner apply diagnostic [s] | 581.377 | 174.300 | 0.300x |
| Preconditioner rebuild count | 115 | 118 | 1.026x |
| build_tangent_local [s] | 15.554 | 21.134 | 1.359x |
| build_F [s] | 22.720 | 44.024 | 1.938x |
| local_strain [s] | 12.772 | 8.206 | 0.643x |
| local_constitutive [s] | 21.884 | 15.439 | 0.705x |
| stress [s] | 20.901 | 13.367 | 0.640x |
| stress_tangent [s] | 13.958 | 10.401 | 0.745x |

## Accepted Continuation State Evolution

| Cont. step | Hypre lambda | PMG lambda | Hypre omega | PMG omega | Hypre Umax | PMG Umax |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.160140307 | 1.160145614 | 6243994.516 | 6243995.714 | 0.829826 | 0.829827 |
| 2 | 1.245601390 | 1.245611879 | 6272652.012 | 6272655.641 | 0.853630 | 0.853625 |
| 3 | 1.312259884 | 1.312273631 | 6301309.508 | 6301315.568 | 0.872646 | 0.872641 |
| 4 | 1.419527204 | 1.419544647 | 6358624.499 | 6358635.422 | 0.905350 | 0.905348 |
| 5 | 1.506208611 | 1.506239423 | 6415939.491 | 6415955.276 | 0.935561 | 0.934786 |
| 6 | 1.598016607 | 1.598010276 | 6530569.474 | 6530594.984 | 1.595743 | 1.598529 |
| 7 | 1.610628435 | 1.610643315 | 6645199.457 | 6645234.691 | 3.837421 | 3.830952 |
| 8 | 1.620833886 | 1.620845815 | 6874459.423 | 6874514.107 | 8.680785 | 8.673454 |
| 9 | 1.629414139 | 1.629406451 | 7332979.355 | 7333072.938 | 18.712545 | 18.702721 |
| 10 | 1.636221319 | 1.636203372 | 8250019.219 | 8250190.600 | 39.186642 | 39.161682 |

## Accepted Continuation Iterations And Step Times

| Cont. step | Hypre wall [s] | PMG wall [s] | Hypre attempts | PMG attempts | Hypre Newton | PMG Newton | Hypre linear | PMG linear | Hypre lin/Newton | PMG lin/Newton |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17.895 | 13.545 | 1 | 1 | 5 | 6 | 90 | 32 | 18.000 | 5.333 |
| 2 | 21.031 | 14.064 | 1 | 1 | 6 | 6 | 144 | 34 | 24.000 | 5.667 |
| 3 | 19.342 | 17.902 | 1 | 1 | 6 | 7 | 126 | 45 | 21.000 | 6.429 |
| 4 | 38.409 | 27.138 | 1 | 1 | 8 | 10 | 273 | 69 | 34.125 | 6.900 |
| 5 | 63.968 | 31.581 | 1 | 1 | 10 | 10 | 467 | 90 | 46.700 | 9.000 |
| 6 | 138.431 | 60.796 | 1 | 1 | 14 | 16 | 1047 | 187 | 74.786 | 11.688 |
| 7 | 129.121 | 54.983 | 1 | 1 | 16 | 15 | 947 | 167 | 59.188 | 11.133 |
| 8 | 96.921 | 58.533 | 1 | 1 | 14 | 15 | 731 | 181 | 52.214 | 12.067 |
| 9 | 131.372 | 46.252 | 1 | 1 | 17 | 14 | 986 | 127 | 58.000 | 9.071 |
| 10 | 131.446 | 60.475 | 1 | 1 | 16 | 18 | 995 | 159 | 62.188 | 8.833 |

## Plots

[continuation.png](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/plots/continuation.png)

![Continuation evolution](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/plots/continuation.png)

[step_metrics.png](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/plots/step_metrics.png)

![Per-step metrics](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/plots/step_metrics.png)

[linear_timing.png](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/plots/linear_timing.png)

![Per-step linear timing](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/plots/linear_timing.png)

[total_timing.png](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/plots/total_timing.png)

![Total timing breakdown](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/plots/total_timing.png)

## Raw Artifacts

- Hypre run info: [run_info.json](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/hypre_default_rank8_step12/data/run_info.json)
- Hypre history: [petsc_run.npz](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/hypre_default_rank8_step12/data/petsc_run.npz)
- PMG run info: [run_info.json](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/pmg_shell_mixed_rank8_step12/data/run_info.json)
- PMG history: [petsc_run.npz](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/pmg_shell_mixed_rank8_step12/data/petsc_run.npz)

## Notes

- The Hypre control uses the repo's default direct elasticity-Hypre path for `P2(L2)` with the same outer `DFGMRES` solver.
- The PMG candidate is the mixed hierarchy `P1(L1) -> P1(L2) -> P2(L2)` shell V-cycle with `chebyshev + jacobi` smoothers and `cg + hypre(boomeramg)` on the direct elastic coarse operator.
- The per-step iteration tables use accepted continuation steps only. Newton counts are the total across all attempts that led to that accepted step.
- `* collector [s]` rows come from the solver iteration collector; `* diagnostic [s]` rows come from the preconditioner backend diagnostics.
