# CHECKOUTS

Use this checklist before each experimental run.

- [ ] venv created and `petsc4py` imports without error.
- [ ] `./build_scripts/bootstrap_petsc4py_venv.sh` completed successfully.
- [ ] `python - <<` quick import check executes for:
  - `from slope_stability.nonlinear import newton`
  - `from slope_stability.nonlinear import newton_ind_ssr`
  - `from slope_stability.continuation import SSR_direct_continuation`
- [ ] Continuation solvers run one dry-step with a small manufactured fixture.
- [ ] No `NameError` from linear preconditioner factory for all `solver_type` variants.
- [ ] Cython extension builds (optional):
  - `python setup.py build_ext --inplace`
  - confirm `slope_stability._kernels` import works.

- [ ] Execute parity tests (optional) if fixtures are available:
  - `pytest tests_local/validation/parity/test_parity.py`
