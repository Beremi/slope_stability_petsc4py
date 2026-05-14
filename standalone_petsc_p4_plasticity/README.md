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
insert unconstrained rows/columns explicitly with `MatSetValues()` for AIJ and
`MatSetValuesLocal()` for MATIS.

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
- `-pc_variant gamg|pmg|bddc|fetidp|none`
- `-pmg_coarse_pc_type auto|hypre|gamg|lu`
- `-pmg_coarse_lu_max_dofs 50000`
- `-pmg_smoother_ksp_type chebyshev`
- `-pmg_smoother_pc_type jacobi`
- `-pmg_smoother_max_it 2`
- `-bddc_graph petsc|topology`
- `-bddc_coordinates scalar|blocked|none`
- `-bddc_collapse_shared true|false`
- `-bddc_local_solver_auto true`
- `-bddc_exact_local_max_dofs 8000`
- `-debug_bddc_dirichlet_rows`
- `-use_box_mesh` for a tiny generated DMPlex tetra mesh smoke test
- `-check_matrix_symmetry` to print an elastic `MatIsSymmetric()` check
- `-ksp_view`
- `-log_view`

Option files are provided in `options/`:

```bash
mpiexec -n 4 ./p4_plasticity -options_file options/gamg.opts
mpiexec -n 4 ./p4_plasticity -options_file options/pmg.opts
mpiexec -n 4 ./p4_plasticity -options_file options/bddc.opts
mpiexec -n 4 ./p4_plasticity -options_file options/fetidp.opts
mpiexec -n 4 ./p4_plasticity -options_file options/bddc_approx_local.opts
mpiexec -n 4 ./p4_plasticity -options_file options/fetidp_approx_local.opts
```

For full comparisons, use the guarded runner so failed PETSc setup paths do not
consume the workstation:

```bash
MEM_LIMIT_GB=120 TIME_LIMIT_SEC=600 OMP_NUM_THREADS=1 \
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
before tabulating the PETSc P4 basis. Boundary elimination is symmetric:

- `u_x = 0` on physical faces `x_max` and `x_min`
- `u_y = 0` on physical face `base`
- `u_z = 0` on physical faces `z_min` and `z_max`

The constrained global dofs are reconstructed after mesh creation from the
distributed mesh geometry, all-gathered, sorted, and used during element matrix
insertion. Constrained rows and columns are skipped during assembly and unit
diagonal entries are inserted afterward: owned global diagonals for AIJ, local
subdomain diagonals for MATIS. The residual/RHS path still zeroes owned
constrained vector entries, but the solve path does not call
`MatZeroRowsColumnsIS()`.

The GAMG variant attaches six projected rigid-body-like near-nullspace vectors
with constrained entries zeroed and supplies owned P4 block coordinates to
PETSc. The PMG variant builds same-mesh P1, P2, and P4 DMs, evaluates coarse FE
basis functions at fine dual points to form P1 -> P2 and P2 -> P4 interpolation,
and lets PETSc build Galerkin operators inside `PCMG`.

The BDDC path supplies PETSc's MATIS solver with local Dirichlet boundaries,
component-wise local dof splitting, global and local rigid-body near-nullspace
data, and scalable default subsolvers for large local MATIS matrices. FETI-DP
configures its internal BDDC object explicitly; PETSc 3.24 expects the inner
BDDC options as `-fetidp_bddc_pc_bddc_*`. Although public `PCSetCoordinates()`
documentation describes blocked vector coordinates, PETSc 3.24 BDDC still has a
local import check marked `TODO: support for blocked`, so BDDC/FETI-DP must use
scalar-equation coordinates in this build. GAMG continues to use blocked
coordinates.

Current BDDC status: tiny distributed MATIS smoke tests converge with PETSc's
local matrix graph and vertex-only defaults. The note3-style strict
`-pc_bddc_use_local_mat_graph false` presets are kept in
`options/bddc_safe.opts`, `options/bddc_edges.opts`, and the matching FETI-DP
files, but they still expose singular local Neumann solves on the tiny mesh.
The usable tiny BDDC/FETI-DP configuration is the topology graph plus
approximate GAMG local Dirichlet/Neumann solvers in
`options/bddc_approx_local.opts` and `options/fetidp_approx_local.opts`.

On the full P4(L1) mesh, 16- and 32-rank BDDC/FETI-DP setup still remains too
large under the 120 GiB guarded validation runs when edge/change-of-basis
constraints are enabled. `-pc_bddc_graph_maxcount 2` bounds memory by producing
no coarse problem, but then BDDC fails with `DIVERGED_PC_FAILED` and FETI-DP
with `DIVERGED_NANORINF`. Oversubscribed 64-rank BDDC reduced peak memory to
about 22 GiB but timed out before reaching an elastic result on this workstation,
so it is only evidence that smaller subdomains help memory, not a valid
convergence result. These PETSc failures are left visible rather than hidden
behind fallbacks.
