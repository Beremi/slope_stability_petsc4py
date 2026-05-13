# Standalone Pure PETSc P4 Plasticity Case

This directory is an extracted C/PETSc-only test case for the heterogenous 3D
P4(L1) slope mesh. It intentionally avoids petsc4py, Python package imports,
continuation logic, PMG, NUMA/GASM experiments, plotting, and repo-local solver
abstractions.

The program reads `data/adaptive_family_a_l1.msh` with `DMPlexCreateFromFile`,
adds a 3-component degree-4 Lagrange FE space, creates PETSc-owned matrices and
vectors with `DMCreateMatrix()` / `DMCreateGlobalVector()`, and assembles element
closures with `DMPlexMatSetClosure()` and `DMPlexVecSetClosure()`.

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

## Runtime Switches

- `-lambda 1.2`
- `-refine_levels N`
- `-newton_rtol 1e-4`
- `-newton_max_it 20`
- `-linear_rtol 1e-8`
- `-pc_variant gamg|bddc|fetidp|none`
- `-use_box_mesh` for a tiny generated DMPlex tetra mesh smoke test
- `-ksp_view`
- `-log_view`

Option files are provided in `options/`:

```bash
mpiexec -n 4 ./p4_plasticity -options_file options/gamg.opts
mpiexec -n 4 ./p4_plasticity -options_file options/bddc.opts
mpiexec -n 4 ./p4_plasticity -options_file options/fetidp.opts
```

## Model Notes

The material table is copied from the heterogenous 3D slope asset:

- physical region 1: general foundation
- physical region 2: weak foundation
- physical region 3: slope mass
- physical region 4: cover layer

The strength reduction is Davis-B at a fixed `lambda`. The Newton initial guess
is the elastic gravity solution. Boundary elimination is symmetric:

- `u_x = 0` on physical faces `x_max` and `x_min`
- `u_y = 0` on physical face `base`
- `u_z = 0` on physical faces `z_min` and `z_max`

The GAMG variant attaches six projected rigid-body-like near-nullspace vectors
with constrained entries zeroed. BDDC and FETI-DP are exposed as PETSc-native
runtime variants over MATIS; if this PETSc build rejects a setup, the PETSc
error is left visible so the small case can be discussed with PETSc experts.
