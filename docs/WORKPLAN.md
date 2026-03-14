# WORKPLAN

## Phase 1 — Structural parity pass
- [x] Recreate `NEWTON` API (`newton`, `newton_ind_ssr`, `newton_ind_ll`, `damping`, `damping_ALG5`).
- [x] Recreate `CONTINUATION` API (`direct` and `indirect` workflows, limit-load).
- [x] Keep boolean mask handling in MATLAB column-major order.

## Phase 2 — Linear solver backbone
- [x] Add deflated FGMRES + GMRES/Jacobi/GAMG factory wiring.
- [x] Implement A-orthogonalization and nullspace-aware GAMG preconditioner hooks.
- [x] Fix solver factory configuration and preconditioner creation lifecycle.

## Phase 3 — Performance and portability
- [x] Add optional Cython kernels and Python fallback implementation.
- [ ] Add deeper micro-kernels for tangent assembly/matvec once profiling is available.

## Phase 4 — Validation
- [x] Add parity scaffold and fixture format for MATLAB-generated in/out pairs.
- [ ] Add generated reference fixtures for core solvers.

## Phase 5 — MPI hardening
- [x] Preserve PETSc-native pathways for PETSc matrices/vectors.
- [ ] Add explicit collective checks and communicator-specific tests.
