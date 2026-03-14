# Implementation Notes and MATLAB Parity

## Architecture

The package mirrors the MATLAB organization into three core blocks:

1. `nonlinear`
   - `damping.py` implements line-search routines.
   - `newton.py` implements plain and nested Newton drivers.
2. `continuation`
   - `direct.py` + `omega.py`: direct continuation in strength reduction `lambda`.
   - `indirect.py`: implicit continuation in work measure `omega`.
   - `limit_load.py`: indirect limit-load for `t`-type loading.
3. `linear`
   - `solver.py` and `deflated_fgmres.py` for iterative linear solves.
   - `preconditioners.py` and `orthogonalize.py` for GAMG/Jacobi and A-orthogonalization.

All linear systems use masked/free-DOF extraction (`q_to_free_indices`) to align with MATLAB's
`Q` masking in column-major memory layout.

## MPI and nullspace details

- `linear/preconditioners.py` builds elastic nullspaces through nodal rigid modes (`make_near_nullspace_elasticity`).
- In GAMG setup, the nullspace is attached when coordinates are available.
- PETSc-native objects are accepted and preferred when present (`PETSc.Mat`, `PETSc.Vec`).

## MATLAB behavior differences

- **Deflation basis indexing**: MATLAB logical indexing `Q` is column-major; python helpers flatten with `order='F'` before masking.
- **Solver state snapshots**: iteration counters/stats are collected in `IterationCollector` and reported through continuation metadata.
- **Factory side effects**: preconditioner builders are instantiated per-captured matrix in `setup_preconditioner`, not at solver construction.

## Cython boundary

- `slope_stability/_kernels.py` is pure Python fallback.
- If built, Cython extension `slope_stability._kernels` provides fast `dot`/`norm2` used in orthogonalization and GMRES internals.
