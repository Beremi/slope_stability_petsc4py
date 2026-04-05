# P2(L5) Rank-8 Mixed PMG-Shell

## Configuration

- Fine mesh: `meshes/3d_hetero_ssr/SSR_hetero_ada_L5.msh`
- Continuation backend: `pmg_shell`
- Mixed hierarchy family: `P2(L5) -> P1(L5)` with optional deeper `P1(Lk)` h-tail
- MPI ranks: `8`
- Outer solver: `PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE`
- Frozen smoke state: synthetic zero displacement on `P2(L5)` with `lambda=1.0`, `omega=0.0`.
- Frozen smoke tolerance: `1e-3`
- Requested continuation advances after init: `10`
- Actual runner `step_max`: `12`
- `step_max` counts accepted states including the 2-state initialization, so `step_max = continuation_steps + 2`.

## Commands

```bash
synthetic_zero_state /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/petsc_run.npz
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/smoke/hypre_frozen --state-npz /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/run_info.json --state-selector final --mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L5.msh --elem-type P2 --node-ordering original --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc-backend hypre --pmat-source tangent --linear-tolerance 0.001 --linear-max-iter 100
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/smoke/pmg_shell_L4 --state-npz /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/run_info.json --state-selector final --mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L5.msh --elem-type P2 --node-ordering original --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc-backend pmg_shell --pmat-source tangent --linear-tolerance 0.001 --linear-max-iter 100 --pmg-coarse-mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L4.msh --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/smoke/pmg_shell_L3 --state-npz /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/run_info.json --state-selector final --mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L5.msh --elem-type P2 --node-ordering original --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc-backend pmg_shell --pmat-source tangent --linear-tolerance 0.001 --linear-max-iter 100 --pmg-coarse-mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L3.msh --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/smoke/pmg_shell_L2 --state-npz /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/run_info.json --state-selector final --mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L5.msh --elem-type P2 --node-ordering original --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc-backend pmg_shell --pmat-source tangent --linear-tolerance 0.001 --linear-max-iter 100 --pmg-coarse-mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L2.msh --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/smoke/pmg_shell_L1 --state-npz /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/run_info.json --state-selector final --mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L5.msh --elem-type P2 --node-ordering original --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc-backend pmg_shell --pmat-source tangent --linear-tolerance 0.001 --linear-max-iter 100 --pmg-coarse-mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/smoke/pmg_shell_L4_L3_L2_L1_tail --state-npz /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/run_info.json --state-selector final --mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L5.msh --elem-type P2 --node-ordering original --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc-backend pmg_shell --pmat-source tangent --linear-tolerance 0.001 --linear-max-iter 100 --pmg-coarse-mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L4.msh --pmg-coarse-mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L3.msh --pmg-coarse-mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L2.msh --pmg-coarse-mesh-path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/pmg_shell_rank8_step12 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L5.msh --elem_type P2 --node_ordering original --step_max 12 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --pmg_coarse_mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --no-store_step_u --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
```

## Smoke Sweep

- State artifact for frozen probes: [`/home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/run_info.json`](../../../artifacts/p2_l5_rank8_mixed_pmg_step10/state_hypre_rank8_step1/data/run_info.json)
- Best smoke candidate by converged frozen setup+solve wall time: `P2(L5) -> P1(L5) -> P1(L1)`
- Existing continuation artifact shown below still uses: `P2(L5) -> P1(L5) -> P1(L1)`

| Case | Hierarchy | Iterations | Final rel. residual | Setup max [s] | Solve max [s] | Setup+solve max [s] | Converged @ 1e-3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Hypre frozen | direct P2(L5) | 34 | 7.054e-04 | 40.081 | 31.032 | 71.111 | yes |
| PMG-shell `L4` | P2(L5) -> P1(L5) -> P1(L4) | 5 | 2.403e-04 | 5.944 | 8.322 | 14.266 | yes |
| PMG-shell `L3` | P2(L5) -> P1(L5) -> P1(L3) | 4 | 8.537e-04 | 3.802 | 4.180 | 7.982 | yes |
| PMG-shell `L2` | P2(L5) -> P1(L5) -> P1(L2) | 4 | 8.608e-04 | 2.576 | 2.875 | 5.451 | yes |
| PMG-shell `L1` (best) | P2(L5) -> P1(L5) -> P1(L1) | 5 | 2.598e-04 | 2.182 | 2.751 | 4.933 | yes |
| PMG-shell `L4_L3_L2_L1_tail` | P2(L5) -> P1(L5) -> P1(L4) -> P1(L3) -> P1(L2) -> P1(L1) | 5 | 2.333e-04 | 3.985 | 3.208 | 7.192 | yes |

![Frozen smoke sweep](../../../artifacts/p2_l5_rank8_mixed_pmg_step10/plots/smoke_sweep.png)

## Continuation Result

- Continuation artifact: [`/home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/pmg_shell_rank8_step12/data/run_info.json`](../../../artifacts/p2_l5_rank8_mixed_pmg_step10/pmg_shell_rank8_step12/data/run_info.json)
- History artifact: [`/home/beremi/repos/slope_stability-1/artifacts/p2_l5_rank8_mixed_pmg_step10/pmg_shell_rank8_step12/data/petsc_run.npz`](../../../artifacts/p2_l5_rank8_mixed_pmg_step10/pmg_shell_rank8_step12/data/petsc_run.npz)
- Final accepted states: `12`
- Accepted continuation advances: `10`
- Final lambda: `1.593622920`
- Final omega: `9891009.437`

| Metric | Value |
| --- | ---: |
| Runtime [s] | 4268.279 |
| Continuation wall time [s] | 4268.276 |
| Unknowns | 913231 |
| Init linear iterations | 57 |
| Continuation linear iterations total | 2116 |
| Total linear / Newton | 9.200 |
| Init solve collector [s] | 34.856 |
| Init preconditioner collector [s] | 29.825 |
| Continuation solve collector [s] | 1306.841 |
| Continuation preconditioner collector [s] | 435.152 |
| Continuation orthogonalization collector [s] | 800.674 |
| Preconditioner setup diagnostic [s] | 464.976 |
| Preconditioner apply diagnostic [s] | 993.037 |
| Preconditioner rebuild count | 235 |
| build_tangent_local [s] | 297.675 |
| build_F [s] | 788.316 |

## Reference Curves Included

- `P2(L1) reference` from [`artifacts/p2_p4_compare_rank8_final_guarded80_v2/p2_rank8_step100/data/petsc_run.npz`](../../../artifacts/p2_p4_compare_rank8_final_guarded80_v2/p2_rank8_step100/data/petsc_run.npz)
- `P2(L2) PMG-shell` from [`artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/pmg_shell_mixed_rank8_step12/data/petsc_run.npz`](../../../artifacts/p2_l2_rank8_hypre_vs_mixed_pmg_step10/pmg_shell_mixed_rank8_step12/data/petsc_run.npz)
- `P4(L1) PMG-shell` from [`artifacts/p4_pmg_shell_best_rank8_full/p4_rank8_step100/data/petsc_run.npz`](../../../artifacts/p4_pmg_shell_best_rank8_full/p4_rank8_step100/data/petsc_run.npz)

![Lambda-omega continuation](../../../artifacts/p2_l5_rank8_mixed_pmg_step10/plots/lambda_omega.png)

## Accepted-Step Metrics

| Cont. step | Lambda | Omega | Wall [s] | Attempts | Newton | Linear | Linear/Newton |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.160196339 | 6244836.291 | 93.749 | 1 | 8 | 55 | 6.875 |
| 2 | 1.245422118 | 6272883.777 | 94.170 | 1 | 8 | 56 | 7.000 |
| 3 | 1.311174870 | 6300931.262 | 96.211 | 1 | 8 | 54 | 6.750 |
| 4 | 1.416230585 | 6357026.234 | 151.970 | 1 | 11 | 94 | 8.545 |
| 5 | 1.501038922 | 6413121.205 | 159.365 | 1 | 11 | 101 | 9.182 |
| 6 | 1.570885683 | 6525311.148 | 516.331 | 1 | 25 | 360 | 14.400 |
| 7 | 1.580347732 | 6749691.034 | 835.572 | 1 | 43 | 394 | 9.163 |
| 8 | 1.586299093 | 7198450.806 | 729.309 | 1 | 39 | 339 | 8.692 |
| 9 | 1.590624329 | 8095970.350 | 684.321 | 1 | 37 | 320 | 8.649 |
| 10 | 1.593622920 | 9891009.437 | 743.570 | 1 | 40 | 343 | 8.575 |

![Step metrics](../../../artifacts/p2_l5_rank8_mixed_pmg_step10/plots/step_metrics.png)

## Frozen PMG Timing Breakdown

- The breakdown below is for the best frozen smoke case (`P2(L5) -> P1(L5) -> P1(L1)`), not the full nonlinear continuation.

| PMG-shell component | Total wall time [s] | Per V-cycle [ms] |
| --- | ---: | ---: |
| Fine pre smoother | 0.595 | 119.011 |
| Fine post smoother | 0.820 | 163.956 |
| Mid pre smoother | 0.037 | 7.316 |
| Mid post smoother | 0.043 | 8.594 |
| Fine residual | 0.208 | 41.525 |
| Mid residual | 0.007 | 1.483 |
| Restrict fine->mid | 0.005 | 1.036 |
| Restrict mid->coarse | 0.003 | 0.596 |
| Prolong coarse->mid | 0.002 | 0.307 |
| Prolong mid->fine | 0.008 | 1.637 |
| Vector sum | 0.002 | 0.340 |
| Coarse Hypre | 0.538 | 107.641 |

![Total timing components](../../../artifacts/p2_l5_rank8_mixed_pmg_step10/plots/timing_breakdown.png)
