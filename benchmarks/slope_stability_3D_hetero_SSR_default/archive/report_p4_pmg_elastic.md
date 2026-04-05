# P4 Elastic PMG Report

- Summary JSON: `/home/beremi/repos/slope_stability-1/artifacts/p4_pmg_elastic/summary.json`
- Linear gate tolerance: `1.0e-03`

## Linear Gate

| Run | Status | Converged | Backend | Setup [s] | Solve [s] | Total [s] | Iterations | Final relative residual |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| rank1_hypre_current | completed | yes | hypre | 21.189 | 47.940 | 69.129 | 27 | 8.965e-04 |
| rank1_pmg_elastic | completed | no | pmg | 0.932 | 44.497 | 45.429 | 80 | 4.220e-01 |
| rank8_hypre_current | completed | yes | hypre | 6.830 | 20.895 | 27.725 | 27 | 8.458e-04 |
| rank8_pmg_elastic | completed | no | pmg | 0.599 | 25.251 | 25.850 | 80 | 4.080e-01 |

## Nonlinear Gate

| Run | Status | Backend | Total [s] | Steps | Final lambda | Final omega |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| rank8_hypre_current_step2 | skipped | - | 0.000 | - | - | - |
| rank8_pmg_elastic_step2 | skipped | - | 0.000 | - | - | - |

## Commands

### rank1_hypre_current

```bash
/home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p4_pmg_elastic/linear/rank1_hypre_current --state-npz /home/beremi/repos/slope_stability-1/artifacts/p4_scaling_step2/rank1/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p4_scaling_step2/rank1/data/run_info.json --pc-backend hypre --pmat-source tangent --state-selector hard --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear-tolerance 0.001 --linear-max-iter 80
```

### rank1_pmg_elastic

```bash
/home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p4_pmg_elastic/linear/rank1_pmg_elastic --state-npz /home/beremi/repos/slope_stability-1/artifacts/p4_scaling_step2/rank1/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p4_scaling_step2/rank1/data/run_info.json --pc-backend pmg --pmat-source elastic --state-selector hard --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear-tolerance 0.001 --linear-max-iter 80
```

### rank8_hypre_current

```bash
/usr/bin/mpiexec -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p4_pmg_elastic/linear/rank8_hypre_current --state-npz /home/beremi/repos/slope_stability-1/artifacts/p4_scaling_step2/rank8/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p4_scaling_step2/rank8/data/run_info.json --pc-backend hypre --pmat-source tangent --state-selector hard --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear-tolerance 0.001 --linear-max-iter 80
```

### rank8_pmg_elastic

```bash
/usr/bin/mpiexec -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python /home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir /home/beremi/repos/slope_stability-1/artifacts/p4_pmg_elastic/linear/rank8_pmg_elastic --state-npz /home/beremi/repos/slope_stability-1/artifacts/p4_scaling_step2/rank8/data/petsc_run.npz --state-run-info /home/beremi/repos/slope_stability-1/artifacts/p4_scaling_step2/rank8/data/run_info.json --pc-backend pmg --pmat-source elastic --state-selector hard --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear-tolerance 0.001 --linear-max-iter 80
```

### rank8_hypre_current_step2

```bash
/usr/bin/mpiexec -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/p4_pmg_elastic/nonlinear/rank8_hypre_current_step2 --pc_backend hypre --preconditioner_matrix_source tangent --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --step_max 2 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear_tolerance 1e-1 --linear_max_iter 100 --no-store_step_u
```

### rank8_pmg_elastic_step2

```bash
/usr/bin/mpiexec -n 8 /home/beremi/repos/slope_stability-1/.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir /home/beremi/repos/slope_stability-1/artifacts/p4_pmg_elastic/nonlinear/rank8_pmg_elastic_step2 --pc_backend pmg --preconditioner_matrix_source elastic --mesh_path /home/beremi/repos/slope_stability-1/meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --step_max 2 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear_tolerance 1e-1 --linear_max_iter 100 --no-store_step_u
```

## Notes

- `rank8_hypre_current_step2`: PMG linear gate did not converge to <= 1.0e-03.
- `rank8_pmg_elastic_step2`: PMG linear gate did not converge to <= 1.0e-03.
