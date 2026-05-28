# Architecture

The engine is organized as a PETSc-first HPC application with a small Python
control surface.

Python responsibilities:

- parse compact case TOMLs and named solver profiles;
- expose CLI commands for running, validating, and inspecting cases;
- generate benchmark notebooks and read result artifacts;
- keep mesh/problem definitions scriptable through `meshes/<asset>/definition.py`.

C/PETSc responsibilities:

- own all distributed DMPlex meshes, labels, sections, matrices, and vectors;
- assemble residuals, tangents, elastic operators, loads, and lambda derivatives;
- run PMG, deflation, Krylov solves, Newton methods, continuation, and profiling;
- write continuation curves, summaries, PETSc binary vectors, and VTU outputs.

The Python-to-C boundary is a serialized PETSc option string. Python does not
copy global matrices or vectors into NumPy for the maintained solve path.

## Native Layout

`src/petsc_ssr/native/` is divided by solver subsystem:

- `include/`: public C API for the Cython bridge.
- `core/`: context, option parsing, lifecycle, and CLI glue.
- `mesh/`: PETSc seepage/DMPlex mesh-facing code.
- `materials/`: Mohr-Coulomb material routines.
- `assembly/`: element basis and mechanics assembly kernels.
- `linear/`: PMG shell V-cycle, deflation, and Krylov routines.
- `nonlinear/`: fixed-load and indirect Newton methods.
- `continuation/`: indirect SSR, direct SSR, and limit-load continuation.
- `reporting/`: CSV/JSON summaries and PETSc log-facing timing helpers.
- `replay/`: debug/replay-only comparison helpers.
- `cython/`: thin C API implementation exposed to `_core.pyx`.

The implementation still builds the mechanics engine as one translation unit to
avoid numerical or performance changes from symbol visibility or call-boundary
refactors.

## Design References

The organization follows these external patterns:

- PETSc DMPlex/DM labels for mesh topology, geometry, distribution, and boundary
  marking.
- PETSc profiling stages/events for authoritative phase timing.
- libCEED-style separation of global parallel layout, element restriction,
  basis evaluation, and pointwise physics.
- PyLith/MOOSE-style named mesh/material/boundary entities.
- ASPECT-style parameter files that specify only case-relevant deviations from
  documented defaults.
