# Standalone Pure PETSc P4 Plasticity Case

This directory is an extracted C/PETSc-only test case for the heterogenous 3D
P4(L1) slope mesh. It intentionally avoids petsc4py, Python package imports,
continuation logic, NUMA/GASM experiments, plotting, and repo-local solver
abstractions. The only multigrid hierarchy kept here is the self-contained
PETSc `PCMG` P4 -> P2 -> P1 preconditioner used by `-pc_variant pmg`.

The program reads `data/adaptive_family_a_l1.msh` with `DMPlexCreateFromFile`,
adds a 3-component degree-4 Lagrange FE space, creates PETSc-owned matrices and
vectors with `DMCreateMatrix()` / `DMCreateGlobalVector()`, and assembles
directly into PETSc objects. Element vectors are added to DM local vectors with
`DMPlexVecSetClosure()` and then accumulated with `DMLocalToGlobal()`. Element
matrices use `DMPlexGetClosureIndices()` to obtain PETSc's closure ordering, then
insert unconstrained rows/columns explicitly with PETSc global `MatSetValues()`
for both AIJ and MATIS. Essential boundary conditions are represented in the
PETSc section with `DMAddBoundary()`, so constrained dofs are removed from the
algebraic problem instead of being kept as artificial unit rows.

## Build

From the repository root:

```bash
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt \
  make -C standalone_petsc_p4_plasticity
```

## Smoke Runs

```bash
cd standalone_petsc_p4_plasticity
mpiexec -n 1 ./p4_plasticity -pc_variant gamg -refine_levels 0
mpiexec -n 4 ./p4_plasticity -pc_variant gamg -refine_levels 0
mpiexec -n 4 ./p4_plasticity -pc_variant pmg -refine_levels 0
mpiexec -n 4 ./p4_plasticity -pc_variant bddc -refine_levels 0
mpiexec -n 4 ./p4_plasticity -pc_variant fetidp -refine_levels 0
```

The helper script uses short Newton limits so it can be used as a quick build
and PETSc-interface check. It uses `data/tiny_box.msh` to exercise the same P4
assembly/solver path on a tiny tetra mesh before running the larger slope mesh
manually. The smoke is intentionally serial because the tiny 6-cell mesh is too
small to be a useful distributed DMPlex partition:

```bash
./run_local_smoke.sh
```

The smallest distributed assembly gate intentionally uses the tiny mesh and a
direct linear solve:

```bash
mpiexec -n 2 ./p4_plasticity \
  -mesh data/tiny_box.msh \
  -petscpartitioner_type simple \
  -pc_variant none \
  -ksp_type preonly \
  -pc_type lu \
  -newton_max_it 0 \
  -malloc_debug
```

## Runtime Switches

- `-lambda 1.2`
- `-refine_levels N`
- `-newton_rtol 1e-4`
- `-newton_max_it 20`
- `-linear_rtol 1e-8`
- `-mesh_bc_mode rollers|base_only|full_sides`
- `-pc_variant gamg|pmg|bddc|fetidp|none`
- `-pmg_coarse_pc_type auto|hypre|gamg|lu`
- `-pmg_coarse_lu_max_dofs 50000`
- `-pmg_coarse_redundant_group_size 16`
- `-pmg_coarse_gamg_aggressive_square_graph false`
- `-pmg_coarse_telescope_active_ranks 0`
- `-pmg_coarse_telescope_subcomm_type interlaced`
- `-pmg_coarse_telescope_ksp_type fgmres`
- `-pmg_coarse_telescope_ksp_rtol 1e-3`
- `-pmg_coarse_telescope_ksp_max_it 100`
- `-pmg_coarse_telescope_pc_type gamg`
- `-pmg_p2_telescope_active_ranks 0`
- `-pmg_p2_telescope_subcomm_type interlaced`
- `-pmg_p2_telescope_ksp_type fgmres`
- `-pmg_p2_telescope_ksp_rtol 1e-3`
- `-pmg_p2_telescope_ksp_max_it 50`
- `-pmg_p2_telescope_pc_type jacobi`
- `-pmg_smoother_ksp_type chebyshev`
- `-pmg_smoother_pc_type jacobi`
- `-pmg_smoother_max_it 2`
- `-bddc_graph petsc|topology`
- `-bddc_coordinates scalar|blocked|none`
- `-bddc_collapse_shared true|false`
- `-bddc_local_solver_auto false`
- `-bddc_use_local_dirichlet false`
- `-bddc_exact_local_max_dofs 8000`
- `-debug_bddc_dirichlet_rows`
- `-inspect_partition` to print DMPlex/MATIS partition diagnostics and exit
- `-reuse_linear_solver true` to keep one KSP/PC hierarchy and refresh operators
  across the elastic solve and Newton corrections
- `-deflation true` to enable an explicit Newton-solve deflation experiment
- `-deflation_solver fgmres|cg`
- `-deflation_basis_tol 1e-3`
- `-deflation_max_it 0` to use `-ksp_max_it` with a safe fallback
- `-deflation_max_vectors 0` to keep all collected elastic/Newton corrections
- `-deflation_monitor false`
- `-use_box_mesh` for a tiny generated DMPlex tetra mesh smoke test
- `-check_matrix_symmetry` to print an elastic `MatIsSymmetric()` check
- `-ksp_view`
- `-log_view`

Option files are provided in `options/`:

```bash
mpiexec -n 4 ./p4_plasticity -options_file options/gamg.opts
mpiexec -n 4 ./p4_plasticity -options_file options/pmg.opts
mpiexec -n 64 ./p4_plasticity -options_file options/pmg_telescope_final.opts
mpiexec -n 4 ./p4_plasticity -options_file options/bddc.opts
mpiexec -n 4 ./p4_plasticity -options_file options/fetidp.opts
mpiexec -n 4 ./p4_plasticity -options_file options/bddc_approx_local.opts
mpiexec -n 4 ./p4_plasticity -options_file options/fetidp_approx_local.opts
```

With `-deflation true`, the elastic solve is still solved normally. Its solution
is then stored as the first deflation vector, each Newton correction appends one
more vector, and every Newton tangent rebuilds an A-orthonormal basis before the
explicit deflated outer solve. The projected preconditioned vector follows the
same core rule as the Python DFGMRES path: `z <- z - W (W^T A z)`, with
`W^T A W = I`, so the coarse initial correction is `W (W^T b)`.
Deflation runs print `DEFLATION_ORTHO`, `DEFLATION_COARSE_INITIAL`, and a final
`DEFLATION_TIMING` line; the `RESULT` line also includes accumulated
orthonormalization, coarse-initial-correction, PC-apply, and projection timings
plus call counts.

For full comparisons, use the guarded runner so failed PETSc setup paths do not
consume the workstation:

```bash
MEM_LIMIT_GB=120 TIME_LIMIT_SEC=600 OMP_NUM_THREADS=1 \
  ./run_guarded_preconditioners.sh
```

The guarded runner does not force a partitioner. PETSc's MPI default is a graph
partitioner when available; on the local PETSc build this is ParMETIS, with
PT-Scotch also available. Use `PARTITIONER=parmetis` or `PARTITIONER=ptscotch`
when reproducible layout comparisons are needed. Its default variant list is
`gamg pmg bddc fetidp`; `none` is still available as a manual negative control
but is not part of the comparison sweep. Avoid `PARTITIONER=simple` for
full BDDC/FETI-DP runs; on the P4(L1) mesh it creates much larger duplicated
MATIS interfaces than graph partitioners. The driver verifies every linear solve
with an explicit true residual check, and GAMG defaults to
`KSP_NORM_UNPRECONDITIONED` so the reported tolerance is not just the
left-preconditioned residual. A layout-only check is cheap and safe:

```bash
mpiexec -n 16 ./p4_plasticity -pc_variant bddc -inspect_partition \
  -petscpartitioner_type parmetis
```

The runner can also reproduce the elasticity PMG telescope profile without
putting long option strings into the shell history:

```bash
MEM_LIMIT_GB=120 VARIANTS=pmg RANKS="64 128 256" PARTITIONER=parmetis \
  PLEX_PARTITION_BALANCE=true \
  PMG_COARSE_TELESCOPE_ACTIVE_RANKS=32 \
  PMG_P2_TELESCOPE_ACTIVE_RANKS=32 \
  ./run_guarded_preconditioners.sh
```

## Model Notes

The material table is copied from the heterogenous 3D slope asset:

- physical region 1: general foundation
- physical region 2: weak foundation
- physical region 3: slope mass
- physical region 4: cover layer

The strength reduction is Davis-B at a fixed `lambda`. The Newton initial guess
is the elastic gravity solution. The 24-point tetrahedral quadrature rule is
stored in unit-simplex coordinates and converted to PETSc's biunit simplex
before tabulating the PETSc P4 basis.

The imported L1 mesh uses PETSc's Gmsh `Face Sets` label. The driver copies the
physical face ids into `boundary_marker`, completes the label, prints
`BOUNDARY_COUNT ...` diagnostics, and applies essential constraints through
`DMAddBoundary()`. The default `-mesh_bc_mode rollers` matches the L1 elasticity
runner:

- `u_x = 0` on physical faces `x_max` and `x_min`
- `u_y = 0` on physical face `base`
- `u_z = 0` on physical faces `z_min` and `z_max`

`base_only` keeps only the glued base, and `full_sides` clamps all components on
the side faces. The old manual constrained-dof list is now empty by design:
PETSc's constrained local/global sections provide the negative closure indices
used to skip eliminated rows and columns during element insertion.

The GAMG, PMG, BDDC, and FETI-DP paths attach PETSc's rigid-body
near-nullspace. The PMG variant builds same-mesh P1, P2, and P4 DMs with the
same boundary constraints, evaluates coarse FE basis functions at fine dual
points to form P1 -> P2 and P2 -> P4 interpolation, and lets PETSc build
Galerkin operators inside `PCMG`. Its P1 bottom solve now
matches the L1 elasticity experiments: GAMG by default, aggressive square-graph
coarsening disabled, optional `PCREDUNDANT` grouping through
`-pmg_coarse_redundant_group_size`, and optional PETSc `PCTELESCOPE` activation
through `-pmg_coarse_telescope_active_ranks`. A second optional PETSc
`PCTELESCOPE` can be attached to the P2-level smoother PC through
`-pmg_p2_telescope_active_ranks`; this is the PETSc-native way to telescope the
P2-level work in the current 3-level same-communicator `PCMG`. It does not
redistribute the P2/P1 DMs or the P2->P4 / P1->P2 interpolation matrices onto
smaller communicators. For example, on high-rank L1 runs the elasticity
telescope campaign used DMPlex partition-boundary balancing, group size 16, and
32 to 64 active coarse ranks:

```bash
mpiexec -n 256 ./p4_plasticity -pc_variant pmg \
  -dm_plex_partition_balance true \
  -pmg_coarse_telescope_active_ranks 32 \
  -pmg_p2_telescope_active_ranks 32 \
  -pmg_coarse_telescope_subcomm_type interlaced
```

Use `options/pmg_telescope_final.opts` for this combined profile. On the local
32-rank L1 probe, adding the P2-level telescope converged but was slower than
the P1-only telescope; it is kept as an explicit high-rank scaling option rather
than a hard-coded default.

By default the linear solver is now persistent across the elastic predictor and
Newton corrections. This reuses the KSP/PCMG object, same-mesh P1/P2/P4 DMs, and
interpolation matrices while still calling `KSPSetOperators()` and refreshing the
preconditioner for each newly assembled tangent. Use `-reuse_linear_solver false`
to recover the old fresh-KSP-per-solve path for A/B comparisons.

The BDDC path supplies PETSc's MATIS solver with component-wise local dof
splitting recovered from MATIS local-to-global rows, global and local
rigid-body near-nullspace data, and optional scalable subsolvers for large
local MATIS matrices. FETI-DP configures its internal BDDC object explicitly;
PETSc 3.24 expects the inner
BDDC options as `-fetidp_bddc_pc_bddc_*`. Although public `PCSetCoordinates()`
documentation describes blocked vector coordinates, PETSc 3.24 BDDC still has a
local import check marked `TODO: support for blocked`, so BDDC/FETI-DP must use
scalar-equation coordinates in this build.

For BDDC/FETI-DP on more than one rank, if the user has not explicitly supplied
`-petscpartitioner_type`, the program asks PETSc for ParMETIS when compiled in,
falling back to PT-Scotch when available. The previous validation runner forced
`simple`, which balanced cell counts but produced pathological high-order
interfaces: on the P4(L1) mesh at 16 ranks, `simple` duplicated MATIS rows by
about 1.92x with a max rank interface of 83k rows, while ParMETIS/PT-Scotch were
about 1.06x with max interfaces near 7k rows.

The default BDDC/FETI-DP comparison path keeps PETSc's exact/default local
subsolves, matching the L1 elasticity driver that converged. For P4 L1, vertex
constraints alone are too weak and can give misleading one-iteration
preconditioned-residual convergence, so the default BDDC/FETI-DP setup uses
vertices, edges, and change-of-basis, with faces disabled to keep the coarse
problem smaller. Top-level BDDC uses flexible GMRES with an unpreconditioned
residual norm so the explicit primal true-residual check agrees with the KSP
stopping test. FETI-DP defaults its inner multiplier KSP to GMRES and applies a
stricter multiplier tolerance than the requested primal verification tolerance.

As in the elasticity repair, local Dirichlet metadata is not sent to BDDC by
default. The constrained `PetscSection` removes those rows already, and the old
full local-section offsets are not valid MATIS local matrix row ids. The debug
checker remains available through `-debug_bddc_dirichlet_rows`; forcing
`-bddc_use_local_dirichlet true` is intentionally rejected until that path is
rewritten in MATIS local-row space. When `-bddc_local_solver_auto true` is
requested and local MATIS matrices exceed
`-bddc_exact_local_max_dofs`, the automatic large-subdomain path marks
Dirichlet/Neumann subsolves as approximate and uses HYPRE BoomerAMG when
available, otherwise GAMG.

Current BDDC/FETI-DP status after the elasticity-style boundary repair:

```text
ranks variant  partitioner status peak_GiB elastic_its newton_its newton_linear_its elastic_solve_s newton_solve_s
16    bddc     parmetis    pass   31.293   27          0          0                 94.2148         0
16    fetidp   parmetis    pass   31.416   36          0          0                 108.633         0
32    bddc     parmetis    pass   28.650   19          0          0                 26.6822         0
32    fetidp   parmetis    pass   28.789   24          0          0                 32.5585         0
32    bddc     parmetis    pass   29.153   19          1          25                26.9916         27.6169
32    fetidp   parmetis    pass   29.234   24          1          25                32.6922         32.7397
```

These are guarded local workstation runs at `-linear_rtol 1e-3` on the full
P4(L1) mesh with the default `rollers` boundary mode and a 120 GiB aggregate RSS
limit. Full nonlinear sweeps should still be run through the guarded runner or
on Karolina, because BDDC/FETI-DP exact local/coarse setup is memory-heavy.
