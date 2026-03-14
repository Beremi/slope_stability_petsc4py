# GAMG Setup For Elastic-Like Systems In This Repo

Date: 2026-03-11

This note compiles the working PETSc `GAMG` setup for the elastic-like systems
used in this repository, focusing on:

- 3D hyperelasticity in the FEniCS custom Newton path,
- 3D hyperelasticity in the JAX + PETSc path,
- the current trust-region benchmark defaults,
- the main pitfalls we hit before the setup became reliable,
- the 2D topology mechanics benchmark as a smaller elastic-like sanity check.

The goal is practical: what to set, why it matters, and what not to repeat.

## 1. Executive Summary

For the 3D hyperelasticity problems here, the working `GAMG` recipe is:

- matrix block size `3`,
- six rigid-body near-nullspace modes,
- `pc_gamg_threshold = 0.05`,
- `pc_gamg_agg_nsmooths = 1`,
- coordinates passed to GAMG when the local DOF ordering preserves `x,y,z`
  triplets,
- block-aware node/DOF ordering.

Two linear-solver profiles are used in practice:

### A. Classic custom-Newton performance profile

- `ksp_type = gmres`
- `pc_type = gamg`
- `ksp_rtol = 1e-1`
- `ksp_max_it = 30`
- `pc_setup_on_ksp_cap = True`
- `pc_gamg_threshold = 0.05`
- `pc_gamg_agg_nsmooths = 1`
- near-nullspace ON
- GAMG coordinates ON

This was the first robust and fast GAMG setup for the 3D HE problem.

### B. Current final benchmark trust-region profile

- `ksp_type = stcg`
- `pc_type = gamg`
- `ksp_rtol = 1e-1`
- `ksp_max_it = 30`
- `pc_setup_on_ksp_cap = False`
- `pc_gamg_threshold = 0.05`
- `pc_gamg_agg_nsmooths = 1`
- near-nullspace ON
- GAMG coordinates ON

This is the current campaign default because the outer nonlinear method is now
a trust-region method and the subproblem solver is PETSc `stcg`.

## 2. Why GAMG Needs Extra Elasticity Metadata

Scalar AMG defaults are not enough for these elasticity-like systems.

The important extra metadata is:

1. Block structure

- The matrix must advertise that there are `3` displacement DOFs per node in
  3D elasticity.
- Without `A.setBlockSize(3)`, GAMG treats the operator like a scalar problem
  and coarse interpolation quality is poor.

2. Rigid-body near-nullspace

- 3D elasticity needs six rigid-body modes:
  - three translations,
  - three rotations.
- These modes tell GAMG what low-energy error looks like.

3. Coordinates

- `pc.setCoordinates(...)` gives GAMG geometric information for aggregation.
- This helps, but only if the local DOF ordering still corresponds to clean
  nodal triplets.

4. Strength threshold

- `pc_gamg_threshold = 0.05` is the critical setting in this repo.
- Leaving the threshold at PETSc's default `-1` caused wrong minima on the
  3D hyperelasticity problem even when block size, nullspace, and coordinates
  were already correct.

## 3. FEniCS Hyperelasticity Path

The main implementation is in `HyperElasticity3D_fenics/solver_custom_newton.py`.

### Matrix and KSP setup

The FEniCS custom Newton path does the following:

- creates the Hessian matrix once,
- sets block size `3` when using GAMG,
- builds a six-vector rigid-body near-nullspace,
- attaches the near-nullspace to the matrix,
- creates PETSc KSP,
- sets the KSP type and PC type,
- for GAMG:
  - sets `pc_gamg_threshold`,
  - sets `pc_gamg_agg_nsmooths`,
  - optionally stores owned coordinates for a later `pc.setCoordinates(...)`.

During each Hessian assembly:

- the matrix is reassembled,
- the near-nullspace is reattached to the freshly assembled matrix,
- `ksp.setOperators(A)` is called,
- `pc.setCoordinates(...)` is called once after operators are known,
- the PC is either rebuilt every Newton iteration or reused based on
  `pc_setup_on_ksp_cap`.

### Near-nullspace construction

The FEniCS path builds the six modes explicitly:

- translations:
  - `tx = [1, 0, 0]`,
  - `ty = [0, 1, 0]`,
  - `tz = [0, 0, 1]`,
- rotations:
  - about `x`,
  - about `y`,
  - about `z`,
- coordinates are taken only from owned nodes,
- constrained local DOFs are zeroed before creating the PETSc nullspace.

This last point matters: the nullspace must respect Dirichlet constraints.

### Working FEniCS GAMG settings

For the historical custom-Newton GMRES profile:

```text
ksp_type = gmres
pc_type = gamg
ksp_rtol = 1e-1
ksp_max_it = 30
pc_setup_on_ksp_cap = True
pc_gamg_threshold = 0.05
pc_gamg_agg_nsmooths = 1
use_near_nullspace = True
gamg_set_coordinates = True
```

For the current trust-region campaign:

```text
ksp_type = stcg
pc_type = gamg
ksp_rtol = 1e-1
ksp_max_it = 30
pc_setup_on_ksp_cap = False
pc_gamg_threshold = 0.05
pc_gamg_agg_nsmooths = 1
use_near_nullspace = True
gamg_set_coordinates = True
```

## 4. JAX + PETSc Hyperelasticity Path

The main logic is split between:

- `HyperElasticity3D_petsc_support/mesh.py`
- `HyperElasticity3D_jax_petsc/solver.py`
- `HyperElasticity3D_jax_petsc/reordered_element_assembler.py`

### Nullspace construction in JAX + PETSc

The mesh helper first builds rigid-body modes in the full DOF space, then
restricts them to free DOFs:

- translations in `x`, `y`, `z`,
- rotations derived from `nodes2coord`,
- final `elastic_kernel = rigid_modes[freedofs, :]`.

In the reordered production element path, those modes are then permuted into
the PETSc ordering before the PETSc `NullSpace` is created.

That is important:

- the physical modes are defined in original free-DOF ordering,
- PETSc owns rows in reordered free-DOF ordering,
- the nullspace vectors must therefore be reordered the same way as the matrix.

### Ordering and ownership

This was a major issue in the early JAX + PETSc path.

The production element assembler now:

- reorders free DOFs first,
- then defines PETSc ownership ranges on the reordered numbering,
- then builds overlap subdomains so each rank can assemble all owned rows
  locally.

The default reorder for the production HE element path is:

- `element_reorder_mode = block_xyz`

Alternative block-aware reorderings exist:

- `none`
- `block_rcm`
- `block_xyz`
- `block_metis`

All of these preserve `x,y,z` triplets at block size `3`.

### Why block-aware ordering matters

The key result from the distribution study was:

- reordering before PETSc ownership split matters a lot,
- natural free-DOF order gave much worse locality,
- block-aware reordered ownership reduced `pc_setup` and solve time strongly,
- the reordered overlap path removed most of the old JAX solve-side penalty.

### GAMG coordinates in the JAX path

The JAX solver has an explicit safety check before building GAMG coordinates.

It verifies that the owned reordered DOFs:

- are divisible by `3`,
- appear as contiguous triples,
- correspond to one node per triple.

If that is not true, the solver raises an error and tells you that the ordering
does not preserve `x,y,z` triplets.

That check is correct and should stay.

### Working JAX + PETSc GAMG settings

For the historical GMRES performance profile:

```text
ksp_type = gmres
pc_type = gamg
ksp_rtol = 1e-1
ksp_max_it = 30
pc_setup_on_ksp_cap = True
pc_gamg_threshold = 0.05
pc_gamg_agg_nsmooths = 1
use_near_nullspace = True
gamg_set_coordinates = True
assembly_mode = element
element_reorder_mode = block_xyz
local_hessian_mode = element
```

For the current final trust-region profile:

```text
ksp_type = stcg
pc_type = gamg
ksp_rtol = 1e-1
ksp_max_it = 30
pc_setup_on_ksp_cap = False
pc_gamg_threshold = 0.05
pc_gamg_agg_nsmooths = 1
use_near_nullspace = True
gamg_set_coordinates = True
assembly_mode = element
element_reorder_mode = block_xyz
local_hessian_mode = element
```

## 5. What Actually Failed Earlier

This is the main caution section.

### 5.1 The threshold was the real make-or-break parameter

This was the single most important GAMG lesson.

What happened:

- naive GAMG with default threshold did not give correct HE solutions,
- adding block size helped but was still wrong,
- adding near-nullspace helped but was still wrong,
- adding coordinates helped but was still wrong,
- only after setting `pc_gamg_threshold = 0.05` did the correct solution
  appear reliably.

So, in this repo:

- do not leave `pc_gamg_threshold` at `-1` for the 3D HE problem,
- treat `0.05` as required unless a new sweep proves otherwise.

### 5.2 Tight GAMG plus PC reuse was unstable

We also learned that:

- loose GAMG:
  - `ksp_rtol = 1e-1`,
  - `ksp_max_it = 30`,
  - `pc_setup_on_ksp_cap = True`
  worked well and was fast,
- tight GAMG:
  - `ksp_rtol = 1e-6`,
  - `ksp_max_it = 500`
  needed a fresh PC every Newton iteration.

Using tight GAMG together with `pc_setup_on_ksp_cap` caused divergence in the
earlier tests.

Rule of thumb:

- loose solve -> PC reuse is fine,
- tight solve -> rebuild every Newton iteration.

### 5.3 Nullspace formatting can silently poison AMG

Earlier nullspace problems came from:

- inconsistent vector layout,
- wrong ownership range,
- missing matrix-compatible vector allocation,
- missing synchronization / bad local fill assumptions.

The robust rule is:

1. allocate nullspace vectors with matrix-compatible PETSc layout,
2. fill them from owned coordinates,
3. enforce BC compatibility,
4. attach them to the matrix after assembly.

In the JAX reordered path, also reorder the modes before slicing owned ranges.

### 5.4 Node ordering is not cosmetic

For elasticity-like problems, ordering changes:

- PETSc row ownership,
- overlap volume,
- `pc_setup` time,
- KSP time,
- whether coordinates can even be supplied safely.

The important point is not just "reorder", but:

- use a block-aware reorder,
- preserve `x,y,z` triplets,
- assign PETSc ownership after that reorder.

### 5.5 PETSc COO preallocation had a subtle JAX-side bug

In the reordered element assembler work, one bug came from
`MatSetPreallocationCOO` for `MPIAIJ` matrices:

- PETSc remaps off-process column indices in place,
- so the element-to-COO position map cannot be built from the mutated arrays,
- it must be built from the original adjacency-derived indices.

This is not a GAMG option issue, but it directly affected the correctness and
performance of the JAX matrix assembly that GAMG sees.

### 5.6 Layout alone does not reproduce FEniCS KSP behavior

One exact-mapping experiment showed:

- FEniCS layout plus JAX values still did not reproduce FEniCS KSP counts,
- when both layout and values matched, the KSP behavior matched.

So if the JAX path and FEniCS path behave differently:

- do not assume the problem is only ordering or partitioning,
- operator values matter too.

### 5.7 HYPRE-related failures taught us what to avoid

Even though this note is about GAMG, several earlier HYPRE failures were useful
warnings:

- `vec_interp_variant = 3` made the AMG non-symmetric,
- that caused CG breakdown inside standard SNES linear-solve checks,
- on some runs the `nodal_coarsen = 6`, `vec_interp_variant = 3` combination
  even segfaulted.

That history is one reason the repo moved toward:

- GMRES for the classic robust profile,
- and later the trust-region `stcg` campaign for the final benchmark.

## 6. 2D Elastic-Like Topology Sanity Check

The topology mechanics benchmark is smaller but confirms the same structural
lessons.

For that 2D elasticity-like linear system:

- PETSc GAMG without rigid-body nullspace needed many more iterations,
- PETSc GAMG with the 2D rigid-body modes `(t_x, t_y, r_z)` was much better,
- block size was `2`,
- `pc_gamg_threshold = 0.05`,
- `pc_gamg_agg_nsmooths = 1`,
- the tested serial PETSc build segfaulted on `pc.setCoordinates(...)`, so
  coordinates were deliberately disabled there.

This does not contradict the 3D HE setup.

It reinforces two points:

- rigid-body near-nullspace matters a lot,
- coordinate injection is useful but still PETSc-build-sensitive and should be
  verified on the actual environment.

## 7. Recommended Option Blocks

### Historical robust/faster GMRES + GAMG profile

```text
--ksp_type gmres
--pc_type gamg
--ksp_rtol 1e-1
--ksp_max_it 30
--gamg_threshold 0.05
--gamg_agg_nsmooths 1
--pc_setup_on_ksp_cap
```

Plus:

- block size `3`,
- six rigid-body near-nullspace modes,
- coordinates ON,
- block-aware ordering,
- for JAX element path: `element_reorder_mode = block_xyz`.

### Current final trust-region GAMG profile

```text
--ksp_type stcg
--pc_type gamg
--ksp_rtol 1e-1
--ksp_max_it 30
--gamg_threshold 0.05
--gamg_agg_nsmooths 1
```

Plus:

- rebuild PC every Newton iteration,
- near-nullspace ON,
- coordinates ON,
- trust region ON,
- post trust-subproblem line search ON.

## 8. Practical Rules

If we want GAMG to work reliably on elasticity-like systems in this repo, the
practical rules are:

1. Always set the correct block size.
2. Always provide the rigid-body near-nullspace.
3. Use `pc_gamg_threshold = 0.05` for 3D HE unless a new sweep proves
   something better.
4. Only pass coordinates if ordering preserves nodal block triplets.
5. Reorder in a block-aware way before PETSc ownership split.
6. Reuse the PC only in the loose GMRES profile, not in the tight-solve or
   final trust-region setup.
7. Treat nullspace formatting and PETSc ownership layout as correctness issues,
   not just performance issues.

## 9. Bottom Line

The final answer for this repo is not "just turn on GAMG".

It is:

- GAMG + block structure,
- GAMG + rigid-body near-nullspace,
- GAMG + correct threshold,
- GAMG + safe coordinate injection,
- GAMG + block-aware ordering before PETSc ownership,
- and a KSP policy matched to the nonlinear method.

That combination is what made the elastic-like systems behave correctly and
predictably here.
