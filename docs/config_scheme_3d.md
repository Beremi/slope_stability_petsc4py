# 3D Configuration Scheme

This note defines the first PETSc-side configuration scheme for the 3D drivers.

The design goal is the same split the MATLAB repository already uses:

- problem specification
- continuation / Newton controls
- linear-solver controls
- output / runner controls

The first implemented scope was intentionally narrow:

- 3D SSR
- non-seepage only
- indirect continuation only
- a unified `elem_type` interface, with current production support still concentrated on `P2`

That is deliberate. The MATLAB drivers show that seepage adds a second physics block and different preprocessing, so it should be an explicit extension of the schema instead of being mixed into the basic 3D SSR case.

## MATLAB Boundary

The standard 3D script [slope_stability_3D_hetero_SSR.m](/home/beremi/repos/slope_stability-1/slope_stability/slope_stability_3D_hetero_SSR.m) is structured as:

1. case definition
   - `elem_type`
   - `Davis_type`
   - material table
   - mesh path
2. FE preprocessing
   - quadrature
   - mesh load
   - integration-point material expansion
   - `K_elast`, `B`, `f_V`
3. continuation / Newton settings
4. linear solver factory
5. constitutive builder
6. continuation call
7. postprocessing

The seepage 3D script [slope_stability_3D_hetero_seepage_SSR.m](/home/beremi/repos/slope_stability-1/slope_stability/slope_stability_3D_hetero_seepage_SSR.m) adds one extra block between mesh load and mechanics:

1. seepage conductivity + seepage boundary conditions
2. seepage solve
3. `pw`, `grad_p`, `mater_sat`
4. mechanical `gamma` / `f_V` derived from seepage state

So the clean PETSc split is:

- `problem`
  - mesh
  - material set
  - constitutive choice
  - optional seepage block later
- `execution`
  - reordering / distributed assembly mode
- `continuation`
- `newton`
- `linear_solver`

## TOML Sections

The implemented config file sections are:

- `[problem]`
  - case identity and mechanical model
- `[execution]`
  - PETSc-side mesh ordering / distributed assembly mode
- `[continuation]`
  - SSR continuation settings
- `[newton]`
  - Newton and damping settings
- `[linear_solver]`
  - Krylov / preconditioner settings
- `[[materials]]`
  - material rows in the same logical order as the MATLAB material table

The example file is:

- [3d_hetero_ssr_default/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/3d_hetero_ssr_default/case.toml)
- [3d_homo_ssr_default/case.toml](/home/beremi/repos/slope_stability-1/benchmarks/3d_homo_ssr_default/case.toml)

## Implemented Interface

The config objects live in:

- [config.py](/home/beremi/repos/slope_stability-1/src/slope_stability/core/config.py)

The generic config-driven runner is:

- [run_case_from_config.py](/home/beremi/repos/slope_stability-1/src/slope_stability/cli/run_case_from_config.py)

It feeds the existing capture backend:

- [run_3D_hetero_SSR_capture.py](/home/beremi/repos/slope_stability-1/src/slope_stability/cli/run_3D_hetero_SSR_capture.py)

The capture backend now accepts:

- `elem_type`
- `davis_type`
- `material_rows`

so the problem specification is no longer hardcoded into the runner.

The config layer now accepts:

- `2D`: `P1`, `P2`, `P4`
- `3D`: `P1`, `P2`, `P4`

That is an interface guarantee, not a blanket promise that every case family is already numerically implemented for every order. Current production status is:

- `3D P2`: production path
- `3D P1`: FE path and generic HDF5 loader are wired; actual benchmark availability depends on having matching `P1` meshes
- `3D P4`: config and mesh-order plumbing are wired, but the mechanics/seepage benchmark paths still raise an explicit `NotImplementedError`

## Current Best Default

The checked-in default config mirrors the current best working PETSc setting:

- `PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE`
- `block_metis`
- `mpi_distribute_by_nodes = true`
- `constitutive_mode = overlap`
- HYPRE `HMIS + ext+i`
- recycle enabled

This is also the current CLI default in:

- [run_3D_hetero_SSR_capture.py](/home/beremi/repos/slope_stability-1/src/slope_stability/cli/run_3D_hetero_SSR_capture.py)

## Not Yet Implemented

The following still remain outside the currently implemented 3D config path:

- generic 3D `P4` benchmark execution

For seepage specifically, the future extension should be an explicit `[seepage]` section with:

- conductivity by material
- seepage mesh loader variant
- seepage boundary model
- seepage preprocessing options

That keeps the non-seepage 3D SSR path simple and avoids mixing hydraulic parameters into standard cases.
