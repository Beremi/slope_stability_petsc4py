# P4 Tangent/Galerkin PMG Report

- Summary JSON: `/home/beremi/repos/slope_stability-1/artifacts/p4_pmg_galerkin/summary.json`
- Benchmark case: `meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh`
- Solver path: `PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE` with `pc_backend=pmg` or `hypre`
- Status: rank-8 PMG frozen and nonlinear gates both completed successfully after fixing PMG/nonlinear matrix lifetimes.

## PMG Microbenchmark

| Run | Status | PCApply [s] | Coarse PC | Coarse subcomm | Coarse iters |
| --- | --- | ---: | --- | ---: | ---: |
| rank8_pmg_galerkin_micro (from frozen probe) | completed | 0.477 | redundant | 1 | 1 |

## Linear Gate

| Run | Backend | Setup [s] | Solve [s] | Total [s] | Iterations | Final relative residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rank1_hypre_current | hypre | 21.106 | 45.186 | 66.291 | 27 | 8.965e-04 |
| rank1_pmg_galerkin | pmg | 2.965 | 9.772 | 12.737 | 9 | 8.325e-04 |
| rank8_hypre_current | hypre | 6.483 | 18.776 | 25.259 | 27 | 8.458e-04 |
| rank8_pmg_galerkin | pmg | 4.551 | 10.160 | 14.711 | 19 | 8.068e-04 |

## Nonlinear Gate

| Run | Backend | Runtime [s] | Steps | Final lambda | Final omega | Init linear iters | Continuation linear iters | PC setup total [s] | PC apply total [s] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rank1_hypre_current_step2 | hypre | 700.514 | 3 | 1.160364 | 6244976.095602 | 140 | 166 | 42.987 | 479.566 |
| rank1_pmg_galerkin_step2 | pmg | 573.288 | 3 | 1.160360 | 6244975.127283 | 117 | 258 | 9.252 | 366.758 |
| rank8_hypre_current_step2 | hypre | 294.936 | 3 | 1.160364 | 6244976.078749 | 139 | 159 | 22.683 | 195.249 |
| rank8_pmg_galerkin_step2 | pmg | 346.691 | 3 | 1.160359 | 6244974.946505 | 217 | 321 | 11.346 | 247.668 |

## Notes

- The PMG crash on rank 8 was caused by nonlinear cleanup destroying constitutive-builder cached PETSc tangent/regularized matrices after each Newton iteration.
- The PMG solver path now reuses the PMG hierarchy/KSP state across Newton steps and owns PETSc operator copies internally, matching the intended assembled/Galerkin workflow more closely.
- On rank 1, PMG completed the nonlinear step-2 gate faster than Hypre (`573.288 s` vs `700.514 s`) with much lower setup time (`9.252 s` vs `42.987 s`), but it required more continuation linear iterations (`258` vs `166`).
- In the rank-8 nonlinear step-2 comparison, PMG used more continuation linear iterations than Hypre (`321` vs `159`), but much lower preconditioner setup time (`11.346 s` vs `22.683 s`) and higher total preconditioner apply time (`247.668 s` vs `195.249 s`).
- The final rank-1 load points are again nearly identical between backends: PMG `lambda=1.160360`, `omega=6244975.127283` versus Hypre `lambda=1.160364`, `omega=6244976.095602`.
- PMG reached the same step count and nearly the same final load point as Hypre: `lambda=1.160359`, `omega=6244974.946505` versus Hypre `lambda=1.160364`, `omega=6244976.078749`.

## Commands

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src ./.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir artifacts/p4_pmg_galerkin/nonlinear/rank1_hypre_current_step2 --mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --step_max 2 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear_tolerance 1e-1 --linear_max_iter 100 --no-store_step_u --pc_backend hypre --preconditioner_matrix_source tangent
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src ./.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir artifacts/p4_pmg_galerkin/nonlinear/rank1_pmg_galerkin_step2 --mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --step_max 2 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear_tolerance 1e-1 --linear_max_iter 100 --no-store_step_u --pc_backend pmg --preconditioner_matrix_source tangent
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /usr/bin/mpiexec -n 8 ./.venv/bin/python benchmarks/slope_stability_3D_hetero_SSR_default/archive/probe_hypre_frozen.py --out-dir artifacts/p4_pmg_galerkin/linear/rank8_pmg_galerkin --state-npz artifacts/p4_scaling_step2/rank8/data/petsc_run.npz --state-run-info artifacts/p4_scaling_step2/rank8/data/run_info.json --state-selector hard --outer-solver-family repo --solver-type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear-tolerance 1e-3 --linear-max-iter 80 --pc-backend pmg --pmat-source tangent
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /usr/bin/mpiexec -n 8 ./.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir artifacts/p4_pmg_galerkin/nonlinear/rank8_hypre_current_step2 --mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --step_max 2 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear_tolerance 1e-1 --linear_max_iter 100 --no-store_step_u --pc_backend hypre --preconditioner_matrix_source tangent
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /usr/bin/mpiexec -n 8 ./.venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture --out_dir artifacts/p4_pmg_galerkin/nonlinear/rank8_pmg_galerkin_step2 --mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh --elem_type P4 --step_max 2 --solver_type PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE --linear_tolerance 1e-1 --linear_max_iter 100 --no-store_step_u --pc_backend pmg --preconditioner_matrix_source tangent
```
