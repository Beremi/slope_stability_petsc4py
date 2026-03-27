# Mixed `P2(L2) -> P1(L2) -> P1(L1)` PMG-Shell With Elasticity-Style Coarse Hypre

## Summary

The original failure was real and structural. There were actually two separate issues:

1. The shell MG coarse solve was feeding Hypre BoomerAMG a **free-DOF Galerkin coarse matrix** of size `10859`, while the requested elasticity-style BoomerAMG options (`numfunctions`, `nodal_coarsen`, `vec_interp_*`) expect a **full 3-component nodal operator**. That mismatch caused the earlier PETSc/Hypre SIGSEGV.
2. The mixed `P1(L1) -> P1(L2)` transfer itself had **24 zero coarse columns**, which produced **24 zero coarse diagonal entries** in the Galerkin `P1(L1)` coarse matrix. That made the mixed coarse space rank-deficient before Hypre even entered the solve.

The fix is now in code:

- when vector/nodal BoomerAMG options are requested on the shell coarse level,
  - build a full coarse-system `P1(L1)` matrix with constrained DOFs restored as identity rows,
  - set block size `3`,
  - attach the full rigid-body near-nullspace,
  - lift coarse RHS from free-space to full-space before the coarse solve,
  - extract the free part back after the coarse solve.
- in the mixed hierarchy builder,
  - detect coarse free DOFs whose `P1(L1) -> P1(L2)` prolongation columns are globally zero,
  - prune those disconnected coarse DOFs from the mixed coarse free space,
  - rebuild the coarse-level metadata and the cross-mesh prolongation with the reduced active coarse set.

Files changed:

- [solver.py](/home/beremi/repos/slope_stability-1/src/slope_stability/linear/solver.py)
- [test_pmg_hierarchy.py](/home/beremi/repos/slope_stability-1/tests/test_pmg_hierarchy.py)

Focused verification:

- `PYTHONPATH=src ./.venv/bin/pytest -q tests/test_pmg_hierarchy.py -k 'pmg_shell_can_opt_in_to_vector_hypre_coarse_options or pmg_shell_full_system_coarse_handles_constrained_dofs_with_vector_hypre or pmg_shell_backend_builds_manual_vcycle_with_hypre_coarse or pmg_shell_richardson_coarse_hypre_reports_inner_iterations'`
- result: `4 passed`

## Requested Coarse Recipe

Tested coarse configuration:

- coarse KSP: `cg` first, then `gmres` as MPI-safe fallback check
- coarse PC: `hypre(boomeramg)`
- `pc_hypre_boomeramg_numfunctions=3`
- `pc_hypre_boomeramg_nodal_coarsen=6`
- `pc_hypre_boomeramg_vec_interp_variant=3`
- `pc_hypre_boomeramg_strong_threshold=0.5`
- `pc_hypre_boomeramg_coarsen_type=HMIS`
- `pc_hypre_boomeramg_max_iter=4`
- `pc_hypre_boomeramg_tol=0.0`
- `pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi`
- plus `interp_type=ext+i`, `P_max=4`, `nodal_coarsen_diag=1`, `vec_interp_qmax=4`, `vec_interp_smooth=true`

## Rank-1 Result

Artifact:

- [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank1/pmg_shell_cg_vec6_k4_cap20_pruned/data/run_info.json)

Outcome:

- converged
- outer iterations: `6`
- final relative residual: `7.241e-04`
- setup: `0.6740 s`
- solve: `4.3926 s`
- coarse KSP: `cg`
- coarse PC: `hypre(boomeramg)`
- coarse matrix layout:
  - free coarse size: `10835`
  - full coarse solve size: `11535`
  - block size: `3`

Important note:

- coarse `cg` did **not** converge internally in 4 iterations on rank 1 either
- last coarse converged reason: `-3` (`max_it` reached)
- despite that, the outer frozen solve still converged quickly

## Mixed-Hierarchy Transfer / Matrix Diagnostics

Before pruning, the mixed hierarchy had:

- `P21` coarse columns with zero support: `24`
- coarse free size: `10859`
- coarse diagonal nonpositive count: `24`
- coarse diagonal minimum: `0.0`

After pruning, the mixed hierarchy has:

- `P21` coarse columns with zero support: `0`
- coarse free size: `10835`
- coarse diagonal nonpositive count: `0`
- coarse free diagonal minimum: `315.37`
- coarse full diagonal minimum: `1.0`

Artifacts:

- rank-1 coarse inspection before/after:
  - [coarse_matrix_inspect.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank1/coarse_matrix_inspect.json)
- rank-8 coarse inspection before/after:
  - [coarse_matrix_inspect.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/coarse_matrix_inspect.json)

Important conclusion from the matrix inspection:

- the repaired rank-1 and rank-8 coarse matrices match to roundoff
- the remaining coarse-solver problem is therefore **not** a rank-8-only assembly corruption

## Rank-8 Result With Requested `cg + hypre`

Artifact:

- [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_cg_vec6_k4_cap20/data/run_info.json)

Outcome:

- no crash
- no setup failure
- full coarse-system path active
- outer frozen probe remained stable, but did not converge in `20` iterations
- final relative residual after `20` iterations: `1.389e-01`
- after the coarse-DOF pruning fix, the result is materially the same:
  - [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_cg_vec6_k4_cap20_pruned/data/run_info.json)
  - final relative residual after `20` iterations: `1.388e-01`

Failure mode:

- coarse KSP type: `cg`
- coarse converged reason: `-8`
- PETSc meaning: `DIVERGED_INDEFINITE_PC`

Interpretation:

- the implementation bugs are fixed
- even with the repaired mixed hierarchy, the rank-8 coarse `cg + hypre` path still reports `DIVERGED_INDEFINITE_PC`
- since the repaired rank-1 and rank-8 coarse matrices now match, the remaining issue is with using `cg` on this **tangent** coarse operator / BoomerAMG pair, not with a broken MPI transfer or a malformed coarse matrix layout

## Rank-8 MPI-Safe Variant: `gmres + hypre`

Artifacts:

- capped run: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_gmres_vec6_k4_cap20/data/run_info.json)
- longer run: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_gmres_vec6_k4/data/run_info.json)

Results:

- `20` outer iterations:
  - final relative residual: `1.387e-01`
  - setup: `0.5579 s`
  - solve: `4.7453 s`
- `80` outer iterations:
  - final relative residual: `5.866e-02`
  - setup: `0.5553 s`
  - solve: `18.9645 s`

Interpretation:

- `gmres + hypre` avoids the `cg` breakdown and is the working MPI-safe coarse wrapper here
- however, with the requested BoomerAMG limits (`max_iter=4`) the overall mixed shell-MG hierarchy is still too weak on 8 ranks to reach the frozen `1e-3` target in a reasonable number of outer iterations

## Bottom Line

- The **crash** is fixed.
- The **mixed-hierarchy rank-deficiency bug** is fixed.
- The requested elasticity-style coarse-Hypre setup now runs on the mixed `P2(L2) -> P1(L2) -> P1(L1)` shell hierarchy with a valid full-system coarse operator.
- Rank 1 converges cleanly.
- On 8 ranks:
  - `cg + hypre` still breaks with `DIVERGED_INDEFINITE_PC`, even after the hierarchy repair.
  - this remaining problem is not a bad MPI assembly anymore; it is a coarse-solver/operator compatibility issue.

The key distinction is:

- if the goal is to keep **this tangent/Galerkin coarse operator**, then `cg` is still not justified by the observed coarse-solver behavior
- if the goal is to use the user’s direct-elasticity `cg + hypre` recipe exactly, the next implementation step is to switch the shell coarse solve from the current tangent Galerkin `P1(L1)` operator to an explicitly assembled **elastic** `P1(L1)` operator
