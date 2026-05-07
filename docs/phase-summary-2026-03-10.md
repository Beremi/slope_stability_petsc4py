# Phase Summary 2026-03-10

This historical note records an early investigation before the current asset-first
repository layout. Paths and local artifacts in this document are provenance notes, not
current runnable entrypoints.

Date: 2026-03-10

## 1) What was done

### A. MATLAB reference baseline captured
- Added/used the local MATLAB capture script `run_3D_hetero_SSR_capture.m`.
- Executed on the historical `SSR_hetero_ada_L1.h5` input and produced local MATLAB
  artifacts under a local temporary output directory and the then-local
  `slope_stability/results/matlab`
  tree.
- MATLAB reference statistics observed in full run:
  - Final `lambda = 1.6660984712183886`
  - Final `omega = 12,000,000`
  - `14` converged states (`lambda_hist`, `omega_hist`, `step_U` all length 14)
  - Initial Newton attempts: `[11, 6]`

### B. PETSc implementation scaffold / run path prepared
- Maintained a structured PETSc reimplementation that has since moved to the current
  `src/slope_stability` package layout.
- Mechanics runner implemented and tuned for parity workflow:
  - [src/slope_stability/execution/asset_case/mechanics_3d.py](../src/slope_stability/execution/asset_case/mechanics_3d.py)
- Run driver now supports MATLAB-like parameterization through asset-first `case.toml` sections for continuation, Newton, and linear-solver controls.
- Environment bootstrap now lives under `bootstrap.sh` and `build_scripts/`.
- Result artifacts were collected locally during the investigation; they are not a current
  public artifact contract.

### C. Key bugfixes made during debug
- Fixed orthogonalization shape handling in the deflated solver path:
  - [src/slope_stability/linear/orthogonalize.py](../src/slope_stability/linear/orthogonalize.py)
- Result: avoids mismatch crash for multi-column projection operations in `A`-orthogonalization.

## 2) Core design choices so far

1. Keep MATLAB API-level parity for high-level algorithms
- Nonlinear continuation is implemented in Python with direct analogs to MATLAB routines rather than relying on PETSc nonlinear solvers.
- Newton / continuation modules stay explicit and traceable.

2. Keep outer solver stack hand-rolled, linear solve as swappable backend
- Continuation + Newton + damping are custom Python code paths.
- Linear system solving was done through a custom deflated GMRES path with configurable preconditioner strategy.

3. Data/path parity with MATLAB
- The early investigation used the same historical MATLAB mesh and material setup.
- Current config-driven runs express that setup through `asset`, `mesh_variant`, and
  optional `profile` instead of raw mesh paths.

4. Maintain optional performance path for kernels
- Kept fallback pure-NumPy/Python versions and optional Cython hooks to evolve performance later.

## 3) Problems encountered (current blockers)

### A) PETSc full execution not yet achieved on full 3D mesh
- Repeated full-run attempts on `SSR_hetero_ada_L1.h5` did not complete in-session.
- Observed failures/stop conditions:
  - long-running solve loops (multiple-hour wall time with no completion)
  - `RuntimeError: Initial choice of lambda seems to be too large.` at initialization

### B) Environment/runtime setup friction
- Initial runs lacked PETSc dependencies in interpreter path.
- After installing `petsc4py`, additional runtime modules were needed (`matplotlib`, `h5py`, `scipy`).
- Cython build previously failed (`dot` signature issue), so NumPy fallback behavior was used.

### C) Solver robustness/performance gap
- Even with smaller Newton/iteration budgets, full-mesh continuation remained too slow to complete robustly in the environment.
- This leaves no completed PETSc `.npz/.json` artifact for the exact full-setup parity run yet.

## 4) What the next phase should do

### Priority: move linear solve to PETSc-native KSP and test stiffness solve quality
- Replace/augment current custom linear path with PETSc-native
  - `KSPFGMRES` + `PCGAMG`
  - with explicit elastic near-nullspace support.
- Use the nullspace construction from elasticity modes already discussed in `matlab-parity-notes.md` to help GAMG convergence.
- Keep outer Newton + damping + indirect continuation and deflation logic as explicit code (do not use PETSc high-level Newton wrappers).

### Concrete next-phase plan
1. Implement `KSP`-backed FGMRES solver adapter in PETSc linear layer.
2. Attach near-nullspace vectors for displacement block coordinates before GAMG setup.
3. Add a small-mesh smoke test first.
4. Run the same capture script with step-limited settings and then scale to `step_max=100` once stable.
5. On first successful PETSc run, produce:
   - `data/petsc_run.npz`
   - `data/run_info.json`
   - `petsc_displacements_3D.png`
   - `petsc_deviatoric_strain_3D.png`
   - `petsc_omega_lambda.png`
   - `petsc_step_displacement.png`

### Explicit next action statement
Next phase will try **PETSc `KSP` FGMRES + GAMG with near nullspace from elasticity** (elastic rigid-body modes) before attempting another full 3D parity solve.
