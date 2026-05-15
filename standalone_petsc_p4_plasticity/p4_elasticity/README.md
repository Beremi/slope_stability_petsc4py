# P4 Elasticity Comparison Drivers

This folder contains two pure C/PETSc P4 tetrahedral linear-elasticity drivers
that deliberately share the same implementation:

- `cube_elasticity.c`: tiny main for a generated tetrahedral cube.
- `l1_elasticity.c`: tiny main for the copied L1 Gmsh mesh.
- `p4_elasticity_common.c`: all FE setup, assembly through
  `PetscDS`/`DMPlexSetSNESLocalFEM`, solver setup, PMG hierarchy setup,
  BDDC/FETI-DP component metadata, boundary-label handling, layout diagnostics,
  solve reporting, and result printing.

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

L1 mesh P4 -> P2 -> P1 PMG solve:

```bash
mpiexec -n 2 ./l1_elasticity \
  -pc_variant pmg \
  -petscpartitioner_type simple \
  -ksp_rtol 1e-3 \
  -ksp_converged_reason
```

The L1 runner defaults to `../data/adaptive_family_a_l1.msh`, gravity loading,
glued bottom, and roller side constraints. Override with `-mesh` or
`-mesh_bc_mode rollers|base_only|full_sides`.

The L1 path uses PETSc's imported Gmsh `Face Sets` label and copies the physical
surface ids into `boundary_marker`. The generated cube still uses synthetic
coordinate-derived labels.

## What Is The Same

Both programs use the same:

- PETSc P4 Lagrange vector FE space by default;
- linear isotropic elasticity residual/Jacobian callbacks;
- `SNESKSPONLY` solve path;
- GAMG, PMG, BDDC, FETI-DP, and `none` variant selection;
- rigid-body near-nullspace attachment;
- BDDC component splitting;
- `BOUNDARY_COUNT`, `LAYOUT`, `MATIS_LAYOUT`, and `MATIS_DUPLICATION`
  diagnostic lines.

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

The L1 mesh has irregular Gmsh tetrahedra, imported physical surface labels, a
glued base plus side rollers, and many partial constraints. PETSc's constrained
section therefore becomes scalar from the MATIS/BDDC point of view:
`map_ltog_bs=1`. This is expected for the roller case and is why we provide
component splitting manually.

## PMG Variant

`-pc_variant pmg` is a pure PETSc `PCMG` p-multigrid path for the default P4
space. It builds same-topology DMPlex levels with degree progression P4 -> P2
-> P1, supplies explicit DMPlex interpolation and transpose restriction
matrices, uses Galerkin coarse operators, and sets the P1 bottom solve to GAMG.
The P4 and P2 levels use two Chebyshev/Jacobi smoothing steps by default.

This variant intentionally avoids MATIS, BDDC, and FETI-DP interface
duplication. It currently requires `-degree 4`; very small toy meshes can have
an empty constrained P1 space, in which case the driver asks for a larger mesh.

Recent local PMG checks:

```text
cube faces=2,2,2 ranks=2: P2=300 P1=54, 16 KSP iterations
L1 ranks=2 inspect_layout: P2=79024 P1=10526
L1 ranks=2 rtol=1e-3: 3 KSP iterations, solve_time=87.5441 s
```

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

So `simple` is not a valid BDDC/FETI-DP solve test for the full mesh. It is still
allowed with `-inspect_layout` as a negative diagnostic, but full L1 BDDC/FETI-DP
runs abort before solve when `MATIS_DUPLICATION >= 1.25`. Use ParMETIS or
PT-Scotch for L1 runs. If no partitioner is provided for L1 BDDC/FETI-DP on more
than one rank, the driver defaults to ParMETIS when available, otherwise
PT-Scotch.

Even with graph partitioning, the full P4(L1) BDDC/FETI-DP setup is still much
more demanding than the cube because:

- P4 tetrahedra put many high-order nodes on every subdomain interface;
- the L1 boundary constraints are mixed full-vector and component-only rollers;
- BDDC sees scalar local rows after constrained-section elimination;
- local MATIS subdomain sizes are still tens of thousands of rows at 16 ranks;
- edge/change-of-basis BDDC constraints can create large dense local correction
  objects in PETSc setup.

With imported `Face Sets`, graph partitioning, and mandatory component
splitting, guarded 16-rank L1 elastic-only checks now converge:

```text
base_only bddc  parmetis fgmres rtol=1e-8: 32 iterations, CONVERGED_RTOL
rollers   bddc  parmetis fgmres rtol=1e-8: 32 iterations, CONVERGED_RTOL
rollers   fetidp parmetis gmres rtol=1e-8: 28 iterations, CONVERGED_RTOL
```

The roller BDDC/FETI-DP checks each peaked around 30 GiB aggregate RSS on the
local workstation. These are still expensive P4/MATIS runs, but they are no
longer blocked by bad labels, bad partitions, or invalid local-Dirichlet
metadata.

`-configure_bddc_metadata` is intentionally unsupported in this driver for now.
The current optional local-Dirichlet builder uses full DMPlex local-section
offsets, while `PCBDDCSetDirichletBoundariesLocal()` needs MATIS local matrix row
numbering after constrained-section elimination. Component splitting remains
mandatory and safe; local Dirichlet metadata needs a separate MATIS-row-space
rewrite before it can be enabled.

## Practical Rules

- Use this folder to compare mesh/constraint effects, not plasticity behavior.
- Keep solver and FE changes in `p4_elasticity_common.c`.
- Use `-inspect_layout` before any expensive BDDC/FETI-DP solve.
- Avoid `-petscpartitioner_type simple` for the L1 mesh except as a diagnostic
  demonstration of a bad partition.
- Check `BOUNDARY_COUNT` first: `base`, `x_min`, `x_max`, `z_min`, and `z_max`
  must be nonzero for the default L1 rollers case.
- Treat `MATIS_DUPLICATION >= 1.25` as a bad partition for full L1 BDDC/FETI-DP
  solves.
- Treat BDDC/FETI-DP failures on L1 as visible PETSc/setup data, not something
  to hide behind a fallback.
