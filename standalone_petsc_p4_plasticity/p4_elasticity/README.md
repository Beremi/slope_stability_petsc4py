# P4 Elasticity Comparison Drivers

This folder contains two pure C/PETSc P4 tetrahedral linear-elasticity drivers
that deliberately share the same implementation:

- `cube_elasticity.c`: tiny main for a generated tetrahedral cube.
- `l1_elasticity.c`: tiny main for the copied L1 Gmsh mesh.
- `p4_elasticity_common.c`: all mesh labeling, FE setup, assembly through
  `PetscDS`/`DMPlexSetSNESLocalFEM`, solver setup, BDDC/FETI-DP metadata,
  layout diagnostics, solve reporting, and result printing.

The runners should stay boring. If a change affects assembly, FE degree,
boundary treatment, KSP/PC setup, or diagnostics, it belongs in the common file
so the cube and L1 mesh remain a useful apples-to-apples comparison.

## Build

From the repository root:

```bash
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt \
  make -C standalone_petsc_p4_plasticity/p4_elasticity
```

Or from this directory:

```bash
make
```

## Runs

Small cube solve:

```bash
mpiexec -n 2 ./cube_elasticity \
  -cube_faces 1,1,1 \
  -pc_variant gamg \
  -ksp_converged_reason
```

Cube BDDC/MATIS layout:

```bash
mpiexec -n 2 ./cube_elasticity \
  -cube_faces 1,1,1 \
  -pc_variant bddc \
  -inspect_layout \
  -petscpartitioner_type simple
```

L1 mesh BDDC/MATIS layout:

```bash
mpiexec -n 2 ./l1_elasticity \
  -pc_variant bddc \
  -inspect_layout \
  -petscpartitioner_type parmetis
```

The L1 runner defaults to `../data/adaptive_family_a_l1.msh`, gravity loading,
glued bottom, and roller side constraints. Override with `-mesh` or
`-mesh_bc_mode rollers|base_only|full_sides`.

## What Is The Same

Both programs use the same:

- PETSc P4 Lagrange vector FE space by default;
- linear isotropic elasticity residual/Jacobian callbacks;
- `SNESKSPONLY` solve path;
- GAMG, BDDC, FETI-DP, and `none` variant selection;
- rigid-body near-nullspace attachment;
- BDDC component splitting and optional local Dirichlet metadata;
- `LAYOUT` and `MATIS_LAYOUT` diagnostic lines.

That means solver differences should come from geometry, constraints, mesh
partitioning, and PETSc MATIS/BDDC behavior, not duplicated example code.

## Current Comparison

Recent smoke checks:

```text
cube_tet_p4 faces=1,1,1 ranks=2 global_dofs=300 variant=gamg
RESULT ... ksp_its=27 ksp_reason=2

LAYOUT cube ranks=2 mat_global_rows=300 partial_constraint_points_sum=0
MATIS_LAYOUT cube map_ltog_bs=3 component_rows_sum=120,120,120

LAYOUT L1 ranks=2 mat_global_rows=610964 partial_constraint_points_sum=2380
MATIS_LAYOUT L1 map_ltog_bs=1 component_rows_sum=207022,208432,203468
```

The cube has regular topology, synthetic coordinate-aligned boundaries, and only
full-vector clamping on the bottom face. Its MATIS local-to-global map keeps a
block size of 3 in this tiny BDDC layout check.

The L1 mesh has irregular Gmsh tetrahedra, reconstructed bounding-box face
labels, a glued base plus side rollers, and many partial constraints. PETSc's
constrained section therefore becomes scalar from the MATIS/BDDC point of view:
`map_ltog_bs=1`. This is expected for the roller case and is why we provide
component splitting manually.

## Why The L1 Mesh Is Harder

The most important discovered issue was partitioning. For the P4(L1) mesh,
forcing `-petscpartitioner_type simple` balances cell counts but creates very
large duplicated high-order interfaces:

```text
16 ranks simple:   MATIS duplication 1.91869, max interface rows 83457
16 ranks parmetis: MATIS duplication 1.06537, max interface rows 6921
16 ranks ptscotch: MATIS duplication 1.06280, max interface rows 7098

32 ranks simple:   MATIS duplication 2.22397, max interface rows 49137
32 ranks parmetis: MATIS duplication 1.09672, max interface rows 5424
32 ranks ptscotch: MATIS duplication 1.09607, max interface rows 5361
```

So `simple` is not a valid BDDC/FETI-DP test for the full mesh. Use ParMETIS or
PT-Scotch for L1 runs.

Even with graph partitioning, the full P4(L1) BDDC/FETI-DP setup is still much
more demanding than the cube because:

- P4 tetrahedra put many high-order nodes on every subdomain interface;
- the L1 boundary constraints are mixed full-vector and component-only rollers;
- BDDC sees scalar local rows after constrained-section elimination;
- local MATIS subdomain sizes are still tens of thousands of rows at 16 ranks;
- edge/change-of-basis BDDC constraints can create large dense local correction
  objects in PETSc setup.

With ParMETIS, a guarded 16-rank L1 BDDC elastic-only run no longer hit the OOM
failure seen with bad layouts. It peaked around 37 GiB. The CG path failed with
`DIVERGED_INDEFINITE_PC`, and after marking large local subsolves approximate
and using flexible GMRES, the run reached a 300 second guard before producing an
elastic result. That is progress from “bad partition explodes memory”, but it is
not yet a successful BDDC/FETI-DP full solve.

## Practical Rules

- Use this folder to compare mesh/constraint effects, not plasticity behavior.
- Keep solver and FE changes in `p4_elasticity_common.c`.
- Use `-inspect_layout` before any expensive BDDC/FETI-DP solve.
- Avoid `-petscpartitioner_type simple` for the L1 mesh except as a diagnostic
  demonstration of a bad partition.
- Treat BDDC/FETI-DP failures on L1 as visible PETSc/setup data, not something
  to hide behind a fallback.
