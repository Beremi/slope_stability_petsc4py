# Rank-8 Mixed PMG Coarse-Hypre Deep Dive

## Scope

This note narrows the remaining rank-8 failure for the mixed hierarchy

- fine: `P2(L2)`
- mid: `P1(L2)`
- coarse: `P1(L1)`

using the custom shell V-cycle backend (`pc_backend=pmg_shell`) with the requested coarse Hypre recipe:

- coarse KSP: `cg`
- coarse PC: `hypre boomeramg`
- `numfunctions = 3`
- `nodal_coarsen = 6`
- `vec_interp_variant = 3`
- `strong_threshold = 0.5`
- `coarsen_type = HMIS`
- `max_iter = 4`
- `tol = 0.0`
- `relax_type_all = symmetric-SOR/Jacobi`

## What Was Actually Broken

The first real implementation bug was in the mixed `P1(L1) -> P1(L2)` transfer:

- `24` coarse free columns were globally disconnected.
- Those `24` disconnected coarse DOFs produced `24` zero/nonpositive diagonal entries in the coarse Galerkin `P1(L1)` operator.

That is fixed in:

- [pmg.py](/home/beremi/repos/slope_stability-1/src/slope_stability/linear/pmg.py#L148)
- [pmg.py](/home/beremi/repos/slope_stability-1/src/slope_stability/linear/pmg.py#L559)

The repair prunes inactive coarse free DOFs before the coarse operator is built.

## Post-Fix Matrix Facts

After the pruning fix:

- rank-1 and rank-8 coarse matrices match to roundoff
- coarse free size is `10835`
- coarse full-system size is `11535`
- zero coarse transfer columns: `0`
- nonpositive coarse diagonal count: `0`
- coarse symmetry defect is at roundoff only

Artifacts:

- rank 1: [coarse_matrix_inspect.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank1/coarse_matrix_inspect.json)
- rank 8: [coarse_matrix_inspect.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/coarse_matrix_inspect.json)

So the remaining rank-8 problem is **not** a bad MPI transfer assembly anymore.

## Coarse-Operator Fix

The next real bug was in the shell coarse-operator path itself.

The earlier rank-8 failure was coming from using vector/nodal BoomerAMG on a **lifted full-system Galerkin** coarse matrix rather than on a directly assembled full `P1(L1)` elasticity operator. That path is now fixed in:

- [solver.py](/home/beremi/repos/slope_stability-1/src/slope_stability/linear/solver.py#L334)
- [solver.py](/home/beremi/repos/slope_stability-1/src/slope_stability/linear/solver.py#L474)

The shell backend now builds a direct full-system `P1(L1)` coarse matrix for the mixed vector-Hypre case, attaches the rigid-body near-nullspace there, and uses that as the coarse Hypre operator.

That change removed the actual rank-8 coarse breakdown:

- old lifted-Galerkin path, rank 8: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_cg_vec6_k4_cap20_pruned/data/run_info.json)
  - coarse reason `DIVERGED_INDEFINITE_PC`
- direct-elastic coarse path, rank 8: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_cg_vec6_k4_cap20_directelastic/data/run_info.json)
  - coarse reason `DIVERGED_ITS`

So the bad-SPD coarse-preconditioner failure is fixed.

## What Was Still Wrong After That

Once the direct coarse operator was in place, increasing the coarse `cg` cap from `4` to `8` changed the amount of coarse work but left the outer frozen convergence essentially unchanged:

- `cg max_it = 4`: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_cg_vec6_k4_directelastic80/data/run_info.json)
  - `80` iterations
  - final relative residual `6.425e-02`
- `cg max_it = 8`: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_cg_vec6_k4_directelastic80_cg8/data/run_info.json)
  - `80` iterations
  - final relative residual `6.425e-02`

That ruled out “under-solving the coarse problem” as the main remaining rank-8 issue.

## Actual Remaining Root Cause

The real remaining weakness was the **parallel smoother choice** on the fine and mid levels.

Using the old shell defaults on rank 8:

- fine/mid smoother: `richardson + sor`
- direct elastic coarse operator: yes
- coarse solver: `cg + hypre`

still stalled badly:

- [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_cg_vec6_k4_directelastic80/data/run_info.json)
  - `80` iterations
  - final relative residual `6.425e-02`

Switching only the fine and mid smoothers to a standard distributed MG choice:

- fine/mid smoother: `chebyshev + jacobi`
- same direct elastic coarse operator
- same coarse `cg + hypre`

made the same frozen rank-8 probe converge cleanly:

- override run: [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_cg_vec6_k4_directelastic80_cheb3jacobi/data/run_info.json)
  - `12` iterations
  - final relative residual `8.149e-04`
  - setup+solve `2.450 s`

So the remaining rank-8 issue was not coarse-Hypre symmetry anymore. It was that `richardson + sor` is a poor distributed smoother for this mixed `P1(L1) -> P1(L2) -> P2(L2)` hierarchy.

## Implemented Fix

The shell backend now defaults to:

- fine/mid: `chebyshev + jacobi`, `3` steps

for the specific mixed distributed hierarchy

- `P1(L1) -> P1(L2) -> P2(L2)`

while leaving the earlier `richardson + sor` behavior unchanged elsewhere.

Code:

- solver-side mixed-MPI default: [solver.py](/home/beremi/repos/slope_stability-1/src/slope_stability/linear/solver.py#L146)
- capture default propagation: [run_3D_hetero_SSR_capture.py](/home/beremi/repos/slope_stability-1/src/slope_stability/cli/run_3D_hetero_SSR_capture.py#L582)
- frozen probe default propagation: [probe_hypre_frozen.py](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/archive/probe_hypre_frozen.py#L477)
- regression: [test_pmg_hierarchy.py](/home/beremi/repos/slope_stability-1/tests/test_pmg_hierarchy.py#L350)

Verified default fixed run:

- [run_info.json](/home/beremi/repos/slope_stability-1/artifacts/l2_p2_mixed_pmg_probe_rank8/pmg_shell_cg_vec6_k4_directelastic80_defaultfixed_v2/data/run_info.json)
  - `12` iterations
  - final relative residual `8.149e-04`
  - fine/mid smoother recorded as `chebyshev + jacobi`

## Current Conclusion

The mixed rank-8 shell path now has a concrete fix:

- prune disconnected coarse DOFs
- use a direct full-system elastic `P1(L1)` coarse operator for vector-Hypre
- use `chebyshev + jacobi` as the mixed distributed smoother

With those three pieces in place, the frozen rank-8 mixed hierarchy converges with the requested coarse `cg + hypre boomeramg` recipe.
