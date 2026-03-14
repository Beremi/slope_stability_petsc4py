# slope_stability: Phase 0 Wrap-up and Starting Point for Next Investigation

Date: 2026-03-10

## 1) What was done

### A. MATLAB reference baseline captured
- Added/used MATLAB capture script:
  - [slope_stability/scripts/run_3D_hetero_SSR_capture.m](/home/beremi/repos/slope_stability-1/slope_stability/scripts/run_3D_hetero_SSR_capture.m)
- Executed on `SSR_hetero_ada_L1.h5` and produced:
  - [matlab_run4.mat](/tmp/slope_run/matlab_run4.mat)
  - [matlab_displacements_3D.png](/home/beremi/repos/slope_stability-1/slope_stability/results/matlab/matlab_displacements_3D.png)
  - [matlab_deviatoric_strain_3D.png](/home/beremi/repos/slope_stability-1/slope_stability/results/matlab/matlab_deviatoric_strain_3D.png)
  - [matlab_omega_lambda.png](/home/beremi/repos/slope_stability-1/slope_stability/results/matlab/matlab_omega_lambda.png)
- MATLAB reference statistics observed in full run:
  - Final `lambda = 1.6660984712183886`
  - Final `omega = 12,000,000`
  - `14` converged states (`lambda_hist`, `omega_hist`, `step_U` all length 14)
  - Initial Newton attempts: `[11, 6]`

### B. PETSc implementation scaffold / run path prepared
- Maintained a structured PETSc reimplementation under:
  - [slope_stability](/home/beremi/repos/slope_stability-1/slope_stability)
- Capture driver implemented and tuned for parity workflow:
  - [slope_stability/src/slope_stability/cli/run_3D_hetero_SSR_capture.py](/home/beremi/repos/slope_stability-1/slope_stability/src/slope_stability/cli/run_3D_hetero_SSR_capture.py)
- Run driver now supports MATLAB-like parameterization and CLI options (e.g. `--lambda_init`, `--d_lambda_init`, `--it_newt_max`, `--it_damp_max`, `--tol`, `--r_min`, `--linear_tolerance`, `--linear_max_iter`, `--solver_type`).
- Environment bootstrap exists:
  - [slope_stability/build_scripts/bootstrap_petsc4py_venv.sh](/home/beremi/repos/slope_stability-1/slope_stability/build_scripts/bootstrap_petsc4py_venv.sh)
- Result artifacts and docs already collected for MATLAB, plus this handoff summary:
  - [slope_stability/results/matlab](/home/beremi/repos/slope_stability-1/slope_stability/results/matlab)
  - [slope_stability/results/matlab_vs_petsc_run_3D_hetero_SSR.md](/home/beremi/repos/slope_stability-1/slope_stability/results/matlab_vs_petsc_run_3D_hetero_SSR.md)

### C. Key bugfixes made during debug
- Fixed orthogonalization shape handling in deflated solver path:
  - [slope_stability/src/slope_stability/linear/orthogonalize.py](/home/beremi/repos/slope_stability-1/slope_stability/src/slope_stability/linear/orthogonalize.py)
- Result: avoids mismatch crash for multi-column projection operations in `A`-orthogonalization.

## 2) Core design choices so far

1. Keep MATLAB API-level parity for high-level algorithms
- Nonlinear continuation is implemented in Python with direct analogs to MATLAB routines rather than relying on PETSc nonlinear solvers.
- Newton / continuation modules stay explicit and traceable.

2. Keep outer solver stack hand-rolled, linear solve as swappable backend
- Continuation + Newton + damping are custom Python code paths.
- Linear system solving was done through a custom deflated GMRES path with configurable preconditioner strategy.

3. Data/path parity with MATLAB
- Use same mesh (`SSR_hetero_ada_L1.h5`) and same material setup.
- Use same intermediate outputs: `lambda_hist`, `omega_hist`, `step_U`, attempt/newton counters, timings, etc.

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
- Cython build previously failed (`dot` signature issue), so we proceeded with NumPy fallback behavior.

### C) Solver robustness/performance gap
- Even with smaller Newton/iteration budgets, full-mesh continuation remained too slow to complete robustly in the environment.
- This leaves no completed PETSc `.npz/.json` artifact for the exact full-setup parity run yet.

## 4) What the next phase should do

### Priority: move linear solve to PETSc-native KSP and test stiffness solve quality
- Replace/augment current custom linear path with PETSc-native
  - `KSPFGMRES` + `PCGAMG`
  - with explicit elastic near-nullspace support.
- Use the nullspace construction from elasticity modes already discussed in `docs.md` to help GAMG convergence.
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
