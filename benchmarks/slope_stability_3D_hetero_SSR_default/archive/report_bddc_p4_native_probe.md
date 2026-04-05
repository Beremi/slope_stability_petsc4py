# BDDC P4 Native Elastic Probe

## Scope

This note records the current `P4` status after switching the local BDDC elastic path away from the old full local `K_elast` CSR build and toward direct elastic-value assembly on the fixed local pattern, and after adding a native PETSc outer-KSP probe path.

Relevant code paths:

- [distributed_tangent.py](/home/beremi/repos/slope_stability-1/src/slope_stability/fem/distributed_tangent.py)
- [probe_bddc_elastic.py](/home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/archive/probe_bddc_elastic.py)
- [compare_preconditioners.py](/home/beremi/repos/slope_stability-1/benchmarks/slope_stability_3D_hetero_SSR_default/archive/archive/compare_preconditioners.py)

## What Changed

- `prepare_bddc_subdomain_pattern(...)` now uses `assemble_strain_geometry(...)` and assembles `elastic_values` directly on the fixed local square pattern.
- The old local path that built a full local `K_elast_local` CSR and then projected it back onto the pattern is no longer used in the BDDC subdomain builder.
- The elastic probe now supports a native PETSc outer solver path:
  - `--outer_solver_family native_petsc`
  - `--native_ksp_type cg|fgmres|gmres`
- Elastic-first BDDC probe variants now disable:
  - `pc_bddc_use_change_of_basis`
  - `pc_bddc_use_change_on_faces`

These changes were validated by:

- [test_petsc_matis_bddc_helpers.py](/home/beremi/repos/slope_stability-1/tests/test_petsc_matis_bddc_helpers.py)
- [test_probe_bddc_elastic.py](/home/beremi/repos/slope_stability-1/tests/test_probe_bddc_elastic.py)
- [test_compare_preconditioners.py](/home/beremi/repos/slope_stability-1/tests/test_compare_preconditioners.py)

## Measured Results

### Rank 1, P4, Native PETSc CG + BDDC

Command completed successfully:

- artifact: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/bddc_elastic_probe/rank1_p4_native_cg/data/run_info.json)
- progress: [progress.jsonl](/home/beremi/repos/slope_stability-1/artifacts/bddc_elastic_probe/rank1_p4_native_cg/data/progress.jsonl)

Key numbers:

- `build_problem`: about `109.5 s`
- `elastic MATIS build`: about `0.8 s`
- `KSP/PC setup`: about `10.9 s`
- solve time: about `92.8 s`
- total runtime: about `214.1 s`
- iterations: `144`
- converged reason: `2`
- relative residual: about `8.85e-8`
- local BDDC bytes: about `3.84 GiB`

This is the first clean `P4` elastic-only BDDC success on the native PETSc path in this repo.

### Rank 2, P4, Native PETSc CG + BDDC, Approximate ILU Local Solvers

This path does **not** work.

- progress: [progress.jsonl](/home/beremi/repos/slope_stability-1/artifacts/bddc_elastic_probe/rank2_p4_native_cg_dbg/data/progress.jsonl)
- backtrace: [rank0_setup.bt.txt](/home/beremi/repos/slope_stability-1/artifacts/bddc_elastic_probe/rank2_p4_native_cg_dbg/rank0_setup.bt.txt)

Observed behavior:

- gets through `elastic_problem_built`
- gets through `elastic_pmat_built`
- crashes during `KSPSetUp()`

Backtrace shows rank 0 in:

- `PCBDDCSetUpCorrection()`
- `KSPSolve_PREONLY()`
- `PCApply_ILU()`
- `MatSolve_SeqAIJ_Inode()`

Interpretation:

- the distributed `P4` crash is no longer in the Python/SciPy local elastic assembly path
- the failing branch is the **approximate ILU local-solver** path inside PETSc BDDC setup

### Rank 2, P4, Native PETSc CG + BDDC, Exact Local LU Solvers

This path appears **stable but slow** so far.

- progress: [progress.jsonl](/home/beremi/repos/slope_stability-1/artifacts/bddc_elastic_probe/rank2_p4_native_cg_exact/data/progress.jsonl)
- backtrace: [rank0_setup.bt.txt](/home/beremi/repos/slope_stability-1/artifacts/bddc_elastic_probe/rank2_p4_native_cg_exact/rank0_setup.bt.txt)

Observed behavior before the run was stopped:

- gets through `elastic_problem_built`
- gets through `elastic_pmat_built`
- remains in `KSPSetUp()` for several minutes

Backtrace shows rank 0 in:

- `PCBDDCSetUpLocalSolvers()`
- `PCSetUp_LU()`
- `MatLUFactorNumeric_SeqAIJ_Inode()`

Interpretation:

- exact local solves do not show the ILU crash
- but the current exact setup cost is too high to treat as a practical distributed baseline yet

## Conclusion

`P4` BDDC has materially progressed:

- the local BDDC elastic assembly bottleneck that previously stalled in SciPy CSR sorting is fixed
- native PETSc `CG + PCBDDC` now works on rank-1 `P4`
- distributed `P4` now reaches PETSc BDDC setup instead of failing earlier in Python-side assembly

Current blocker hierarchy:

1. approximate ILU local BDDC setup is unstable at rank 2
2. exact LU local BDDC setup is stable but too expensive at rank 2

## Recommended Next Step

Based on the measured failures and the PETSc guidance used for this cycle, the next contained `P4` experiment should be:

- keep `outer_solver_family=native_petsc`
- keep `native_ksp_type=cg`
- keep `A = P = K_elast` for the elastic probe
- stop using ILU as the distributed approximate-local baseline
- try PETSc’s documented approximate-local path next:
  - `pc_bddc_dirichlet_approximate`
  - `pc_bddc_neumann_approximate`
  - local `pc_type=gamg`
  - and `pc_bddc_switch_static`

That is the most defensible next move because the current data says the repo-side `P4` matrix build is now good enough, while the remaining distributed failure is inside PETSc’s local BDDC solver choice.
