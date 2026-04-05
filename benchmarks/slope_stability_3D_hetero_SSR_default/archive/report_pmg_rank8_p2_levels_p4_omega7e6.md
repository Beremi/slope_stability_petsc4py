# Rank-8 PMG Comparison: P2(L1..L5) and P4(L1) at `omega_max = 7e6`

## Setup

- MPI ranks: `8`
- `omega_max_stop`: `7000000.0`
- Runner `step_max`: `12`
- `step_max` counts accepted states including the 2-state initialization, so `step_max = continuation_advances + 2`.
- `P2(LN)` runs use `pmg_shell` with `P2(LN) -> P1(LN) -> P1(L1)` for `N=2..5`, and same-mesh `P2(L1) -> P1(L1)` for `L1`.
- `P4(L1)` is shown four times: the existing baseline, a capped rerun with `step_max = 12`, an uncapped capped-controller rerun with `step_max = 100`, and the newer smart-controller rerun with `step_max = 100`.
- The Newton-based omega caps are `no increase if accepted-step Newton total > 10` and `halve next d_omega if > 20`.
- `P2` cases use the mixed-hierarchy direct-elastic coarse-Hypre configuration; `P4` uses the previously best working same-mesh shell PMG coarse-Hypre configuration.

## Commands

### P2(L1)

```bash
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l1_rank8_step12 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P2 --node_ordering original --step_max 12 --omega_max_stop 7000000.0 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --no-store_step_u --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
```

### P2(L2)

```bash
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l2_rank8_step12 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L2.msh --elem_type P2 --node_ordering original --step_max 12 --omega_max_stop 7000000.0 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --no-store_step_u --pmg_coarse_mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
```

### P2(L3)

```bash
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l3_rank8_step12 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L3.msh --elem_type P2 --node_ordering original --step_max 12 --omega_max_stop 7000000.0 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --no-store_step_u --pmg_coarse_mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
```

### P2(L4)

```bash
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l4_rank8_step12 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L4.msh --elem_type P2 --node_ordering original --step_max 12 --omega_max_stop 7000000.0 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --no-store_step_u --pmg_coarse_mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
```

### P2(L5)

```bash
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l5_rank8_step12 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L5.msh --elem_type P2 --node_ordering original --step_max 12 --omega_max_stop 7000000.0 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --no-store_step_u --pmg_coarse_mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --petsc-opt manualmg_coarse_operator_source=direct_elastic_full_system --petsc-opt mg_levels_ksp_type=chebyshev --petsc-opt mg_levels_ksp_max_it=3 --petsc-opt mg_levels_pc_type=jacobi --petsc-opt mg_coarse_ksp_type=cg --petsc-opt mg_coarse_max_it=4 --petsc-opt mg_coarse_rtol=0.0 --petsc-opt pc_hypre_boomeramg_numfunctions=3 --petsc-opt pc_hypre_boomeramg_nodal_coarsen=6 --petsc-opt pc_hypre_boomeramg_nodal_coarsen_diag=1 --petsc-opt pc_hypre_boomeramg_vec_interp_variant=3 --petsc-opt pc_hypre_boomeramg_vec_interp_qmax=4 --petsc-opt pc_hypre_boomeramg_vec_interp_smooth=true --petsc-opt pc_hypre_boomeramg_coarsen_type=HMIS --petsc-opt pc_hypre_boomeramg_interp_type=ext+i --petsc-opt pc_hypre_boomeramg_P_max=4 --petsc-opt pc_hypre_boomeramg_strong_threshold=0.5 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0 --petsc-opt pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi
```

### P4(L1) baseline

```bash
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_rank8_step12 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --node_ordering block_metis --step_max 12 --omega_max_stop 7000000.0 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --no-store_step_u --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0
```

### P4(L1) + Newton omega caps

```bash
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_newton_caps_rank8_step12 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --node_ordering block_metis --step_max 12 --omega_max_stop 7000000.0 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --no-store_step_u --omega_no_increase_newton_threshold 10 --omega_half_newton_threshold 20 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0
```

### P4(L1) + Newton omega caps, uncapped

```bash
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_newton_caps_rank8_step100 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --node_ordering block_metis --step_max 100 --omega_max_stop 7000000.0 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --no-store_step_u --omega_no_increase_newton_threshold 10 --omega_half_newton_threshold 20 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0
```

### P4(L1) + smart omega controller

```bash
mpirun -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100 --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --node_ordering block_metis --step_max 100 --omega_max_stop 7000000.0 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --pc_backend pmg_shell --preconditioner_matrix_source tangent --no-store_step_u --omega_no_increase_newton_threshold 10 --omega_half_newton_threshold 20 --omega_target_newton_iterations 12 --omega_adapt_min_scale 0.7 --omega_adapt_max_scale 1.25 --omega_hard_newton_threshold 18 --omega_hard_linear_threshold 250 --omega_efficiency_drop_ratio 0.5 --omega_efficiency_window 3 --omega_hard_shrink_scale 0.85 --petsc-opt pc_hypre_boomeramg_max_iter=4 --petsc-opt pc_hypre_boomeramg_tol=0.0
```

## Summary

| Case | Hierarchy | Unknowns | Accepted advances | Final lambda | Final omega | Runtime [s] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2(L1) | P2(L1) -> P1(L1) | 80362 | 9 | 1.642606486 | 7000000.000 | 197.742 |
| P2(L2) | P2(L2) -> P1(L2) -> P1(L1) | 145298 | 9 | 1.624031891 | 7000000.000 | 324.963 |
| P2(L3) | P2(L3) -> P1(L3) -> P1(L1) | 265739 | 9 | 1.608320112 | 7000000.000 | 594.359 |
| P2(L4) | P2(L4) -> P1(L4) -> P1(L1) | 490015 | 8 | 1.594952599 | 7000000.000 | 1218.233 |
| P2(L5) | P2(L5) -> P1(L5) -> P1(L1) | 913231 | 8 | 1.584383668 | 7000000.000 | 2894.568 |
| P4(L1) baseline | P4(L1) -> P2(L1) -> P1(L1) | 616322 | 8 | 1.569076398 | 7000000.000 | 4082.876 |
| P4(L1) + Newton omega caps | P4(L1) -> P2(L1) -> P1(L1) | 616322 | 10 | 1.566538452 | 6570273.222 | 2017.487 |
| P4(L1) + Newton omega caps, uncapped | P4(L1) -> P2(L1) -> P1(L1) | 616322 | 21 | 1.569051958 | 7000000.000 | 4682.647 |
| P4(L1) + smart omega controller | P4(L1) -> P2(L1) -> P1(L1) | 616322 | 34 | 1.569058819 | 7000000.000 | 4329.135 |

## P4 Delta

| Metric | P4(L1) baseline | P4(L1) + Newton omega caps | Ratio new/base |
| --- | ---: | ---: | ---: |
| Runtime [s] | 4082.876 | 2017.487 | 0.494 |
| Accepted continuation advances | 8 | 10 | - |
| Final omega | 7000000.000 | 6570273.222 | - |
| Final lambda | 1.569076398 | 1.566538452 | - |
| Continuation Newton iterations | 192 | 143 | 0.745 |
| Continuation linear iterations | 3706 | 2059 | 0.556 |
| Preconditioner apply total [s] | 2098.945 | 1016.216 | 0.484 |

## P4 Delta To Uncapped Rerun

| Metric | P4(L1) baseline | P4(L1) + Newton omega caps, uncapped | Ratio new/base |
| --- | ---: | ---: | ---: |
| Runtime [s] | 4082.876 | 4682.647 | 1.147 |
| Accepted continuation advances | 8 | 21 | - |
| Final omega | 7000000.000 | 7000000.000 | - |
| Final lambda | 1.569076398 | 1.569051958 | - |
| Continuation Newton iterations | 192 | 301 | 1.568 |
| Continuation linear iterations | 3706 | 5038 | 1.359 |
| Preconditioner apply total [s] | 2098.945 | 2395.694 | 1.141 |

## P4 Delta To Smart Controller

| Metric | P4(L1) baseline | P4(L1) + smart omega controller | Ratio new/base |
| --- | ---: | ---: | ---: |
| Runtime [s] | 4082.876 | 4329.135 | 1.060 |
| Accepted continuation advances | 8 | 34 | - |
| Final omega | 7000000.000 | 7000000.000 | - |
| Final lambda | 1.569076398 | 1.569058819 | - |
| Continuation Newton iterations | 192 | 310 | 1.615 |
| Continuation linear iterations | 3706 | 4704 | 1.269 |
| Preconditioner apply total [s] | 2098.945 | 2153.197 | 1.026 |

## Iteration Totals

| Case | Attempts | Successful attempts | Continuation Newton iters | Continuation linear iters | Linear/Newton | Init linear iters | Preconditioner rebuilds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2(L1) | 9 | 9 | 78 | 716 | 9.179 | 31 | 79 |
| P2(L2) | 9 | 9 | 90 | 836 | 9.289 | 38 | 92 |
| P2(L3) | 9 | 9 | 104 | 998 | 9.596 | 42 | 107 |
| P2(L4) | 8 | 8 | 120 | 1144 | 9.533 | 42 | 124 |
| P2(L5) | 8 | 8 | 150 | 1415 | 9.433 | 57 | 157 |
| P4(L1) baseline | 8 | 8 | 192 | 3706 | 19.302 | 159 | 202 |
| P4(L1) + Newton omega caps | 10 | 10 | 143 | 2059 | 14.399 | 159 | 151 |
| P4(L1) + Newton omega caps, uncapped | 21 | 21 | 301 | 5038 | 16.738 | 159 | 298 |
| P4(L1) + smart omega controller | 34 | 34 | 310 | 4704 | 15.174 | 159 | 294 |

## Linear Timing Totals

| Case | Init solve [s] | Init PC [s] | Init orthog [s] | Continuation solve [s] | Continuation PC [s] | Continuation orthog [s] | PC setup total [s] | PC apply total [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2(L1) | 4.503 | 6.175 | 0.300 | 104.788 | 42.651 | 6.282 | 48.826 | 101.876 |
| P2(L2) | 6.886 | 8.333 | 0.771 | 152.695 | 61.163 | 19.033 | 69.496 | 142.548 |
| P2(L3) | 10.677 | 11.409 | 1.842 | 247.867 | 89.205 | 57.818 | 100.614 | 217.873 |
| P2(L4) | 16.069 | 15.789 | 3.617 | 440.661 | 146.181 | 163.133 | 161.970 | 359.011 |
| P2(L5) | 36.756 | 30.409 | 11.934 | 928.908 | 288.061 | 493.888 | 318.470 | 715.355 |
| P4(L1) baseline | 109.113 | 11.959 | 17.316 | 2597.388 | 119.849 | 698.047 | 131.809 | 2098.945 |
| P4(L1) + Newton omega caps | 92.016 | 10.613 | 14.999 | 1199.262 | 77.383 | 289.479 | 87.995 | 1016.216 |
| P4(L1) + Newton omega caps, uncapped | 91.768 | 10.582 | 14.589 | 2970.202 | 161.695 | 708.684 | 172.277 | 2395.694 |
| P4(L1) + smart omega controller | 82.000 | 9.918 | 13.510 | 2675.767 | 155.600 | 694.296 | 165.518 | 2153.197 |

## Constitutive Timing Totals

| Case | build_tangent_local [s] | build_F [s] | local_strain [s] | local_constitutive [s] | stress [s] | stress_tangent [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2(L1) | 7.639 | 14.606 | 3.292 | 5.440 | 4.670 | 4.125 |
| P2(L2) | 17.400 | 34.701 | 6.291 | 11.607 | 9.952 | 8.012 |
| P2(L3) | 39.056 | 80.489 | 13.810 | 25.528 | 21.460 | 18.097 |
| P2(L4) | 85.356 | 208.881 | 40.744 | 60.690 | 61.002 | 41.107 |
| P2(L5) | 208.815 | 522.141 | 109.891 | 145.940 | 157.556 | 99.692 |
| P4(L1) baseline | 132.742 | 67.019 | 20.120 | 8.194 | 20.967 | 7.192 |
| P4(L1) + Newton omega caps | 86.711 | 40.691 | 12.186 | 5.234 | 12.387 | 4.847 |
| P4(L1) + Newton omega caps, uncapped | 200.510 | 109.131 | 26.192 | 12.823 | 28.433 | 10.514 |
| P4(L1) + smart omega controller | 197.056 | 96.056 | 23.257 | 12.085 | 24.771 | 10.442 |

## PMG Layout

| Case | ManualMG levels | Level orders | Level global sizes | Coarse operator | Coarse KSP/PC | Fine smoother | Mid smoother |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| P2(L1) | 2 | [1, 2] | [10859, 80362] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | - |
| P2(L2) | 3 | [1, 1, 2] | [10835, 19282, 145298] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | chebyshev/jacobi |
| P2(L3) | 3 | [1, 1, 2] | [10847, 34810, 265739] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | chebyshev/jacobi |
| P2(L4) | 3 | [1, 1, 2] | [10850, 63261, 490015] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | chebyshev/jacobi |
| P2(L5) | 3 | [1, 1, 2] | [10859, 116561, 913231] | direct_elastic_full_system | cg/hypre/boomeramg | chebyshev/jacobi | chebyshev/jacobi |
| P4(L1) baseline | 3 | [1, 2, 4] | [10859, 80362, 616322] | galerkin_free | preonly/hypre/boomeramg | richardson/sor | richardson/sor |
| P4(L1) + Newton omega caps | 3 | [1, 2, 4] | [10859, 80362, 616322] | galerkin_free | preonly/hypre/boomeramg | richardson/sor | richardson/sor |
| P4(L1) + Newton omega caps, uncapped | 3 | [1, 2, 4] | [10859, 80362, 616322] | galerkin_free | preonly/hypre/boomeramg | richardson/sor | richardson/sor |
| P4(L1) + smart omega controller | 3 | [1, 2, 4] | [10859, 80362, 616322] | galerkin_free | preonly/hypre/boomeramg | richardson/sor | richardson/sor |

## Plots

![Lambda Omega](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/plots/lambda_omega.png)

![Step Wall Time](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/plots/step_wall_time.png)

![Step Newton Iterations](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/plots/step_newton_iterations.png)

![Step Linear Iterations](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/plots/step_linear_iterations.png)

![Step Linear Per Newton](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/plots/step_linear_per_newton.png)

![Final Lambda Vs Time](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/plots/final_lambda_vs_time.png)

## Raw Artifacts

- Summary JSON: [summary.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/summary.json)
- P2(L1): [run_info.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l1_rank8_step12/data/run_info.json), [petsc_run.npz](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l1_rank8_step12/data/petsc_run.npz)
- P2(L2): [run_info.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l2_rank8_step12/data/run_info.json), [petsc_run.npz](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l2_rank8_step12/data/petsc_run.npz)
- P2(L3): [run_info.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l3_rank8_step12/data/run_info.json), [petsc_run.npz](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l3_rank8_step12/data/petsc_run.npz)
- P2(L4): [run_info.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l4_rank8_step12/data/run_info.json), [petsc_run.npz](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l4_rank8_step12/data/petsc_run.npz)
- P2(L5): [run_info.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l5_rank8_step12/data/run_info.json), [petsc_run.npz](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p2_l5_rank8_step12/data/petsc_run.npz)
- P4(L1) baseline: [run_info.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_rank8_step12/data/run_info.json), [petsc_run.npz](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_rank8_step12/data/petsc_run.npz)
- P4(L1) + Newton omega caps: [run_info.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_newton_caps_rank8_step12/data/run_info.json), [petsc_run.npz](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_newton_caps_rank8_step12/data/petsc_run.npz)
- P4(L1) + Newton omega caps, uncapped: [run_info.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_newton_caps_rank8_step100/data/run_info.json), [petsc_run.npz](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_newton_caps_rank8_step100/data/petsc_run.npz)
- P4(L1) + smart omega controller: [run_info.json](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data/run_info.json), [petsc_run.npz](../../../artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data/petsc_run.npz)
