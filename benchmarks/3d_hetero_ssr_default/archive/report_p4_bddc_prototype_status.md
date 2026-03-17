# P4 BDDC Prototype Status

- Date: `2026-03-16`
- Target path: `P4 / rows / overlap / block_metis`
- Solver branch: `pc_backend=bddc`, `preconditioner_matrix_policy=current`, `preconditioner_rebuild_policy=every_newton`, outer `FGMRES`

## Implemented

- `MATIS + PCBDDC` path is wired through the production solver/config/CLI path.
- `BDDCSubdomainPattern` now reports local subdomain bytes and explicit primal-vertex stats.
- BDDC explicit primal vertices are now populated from interface DOFs instead of staying empty.
- `run_info.json` now records `bddc_subdomain_pattern` stats and timings.
- `compare_preconditioners.py` now supports staged execution:
  - `smoke`
  - `screen`
  - `bddc_gate`
  - `full_compare`
- The benchmark harness now includes dedicated BDDC runtime smokes and a 3-way final-report path.

## Automated Validation

- Command:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_petsc_matis_bddc_helpers.py tests/test_preconditioner_mpi.py tests/test_compare_preconditioners.py`
- Result:
  - `8 passed`

Covered by tests:

- `MATIS` creation/update
- BDDC metadata wiring for local field splits, Dirichlet rows, adjacency, and explicit primal vertices
- BDDC solver-policy handling
- MPI linear-solve smoke including `bddc`
- staged benchmark-report generation

## Real Runtime Checks

### Rank-1 production smoke

Command attempted:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
mpiexec -n 1 .venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture \
  --mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh \
  --elem_type P2 \
  --step_max 1 \
  --node_ordering block_metis \
  --tangent_kernel rows \
  --constitutive_mode overlap \
  --no-recycle_preconditioner \
  --max_deflation_basis_vectors 16 \
  --solver_type PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE \
  --pc_backend bddc \
  --preconditioner_matrix_policy current \
  --preconditioner_rebuild_policy every_newton \
  --no-pc_bddc_symmetric \
  --no-store_step_u
```

Observed before stopping:

- elapsed: about `120 s`
- RSS: about `5.71 GiB`
- no `progress.jsonl` yet

Native stack snapshot showed the process inside PETSc:

- `PCBDDCSetUpLocalSolvers()`
- `PCSetUp_BDDC()`
- `MatLUFactorNumeric_SeqAIJ_Inode()`

Interpretation:

- the code is not stuck in Python-side tangent/constitutive assembly
- on `rank=1`, BDDC is effectively spending its time factoring the full local problem during local-solver setup

### Rank-2 production smoke

Command attempted:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
mpiexec -n 2 .venv/bin/python -m slope_stability.cli.run_3D_hetero_SSR_capture \
  --mesh_path meshes/3d_hetero_ssr/SSR_hetero_ada_L1.msh \
  --elem_type P2 \
  --step_max 1 \
  --node_ordering block_metis \
  --tangent_kernel rows \
  --constitutive_mode overlap \
  --no-recycle_preconditioner \
  --max_deflation_basis_vectors 16 \
  --solver_type PETSC_MATLAB_DFGMRES_GAMG_NULLSPACE \
  --pc_backend bddc \
  --preconditioner_matrix_policy current \
  --preconditioner_rebuild_policy every_newton \
  --no-pc_bddc_symmetric \
  --no-store_step_u
```

Observed before stopping:

- elapsed: about `53 s`
- rank-local RSS: about `2.89 GiB` and `3.41 GiB`
- no `progress.jsonl` yet

Native stack snapshot again showed PETSc BDDC local-solver setup:

- `PCBDDCSetUpLocalSolvers()`

Interpretation:

- the distributed path is also reaching `PCBDDC`
- the present bottleneck is PETSc BDDC local solver setup, not the repo’s owned-row tangent path

## Current Conclusion

- BDDC is implemented and reaches PETSc `PCBDDC` on the production runner.
- The prototype is functionally much further along than the previous smoke-only helper state.
- The remaining blocker for the planned gate/full-comparison workflow is runtime cost in `PCBDDCSetUpLocalSolvers`, not missing solver plumbing.

## Immediate Next Step

- Benchmark and/or prototype cheaper local BDDC setup choices on the same branch before retrying the full `rank-8 step_max=2` gate.
- Until then, the staged harness and final 3-way report path are implemented, but the production BDDC benchmark sequence is not yet complete.
