# INSTRUCTIONS

## Environment and venv
1. Create an isolated venv and install build/runtime deps.

```bash
./bootstrap.sh
```

This is the default full setup path. It prepares `./.venv`, builds PETSc with `HYPRE` under `./.build`, installs `petsc4py`, and then installs the package in editable mode with the partitioning extras needed by the benchmark defaults.

If you explicitly want a lighter wheel-based setup that may not support the HYPRE benchmark defaults:

```bash
BOOTSTRAP_MODE=wheel ./bootstrap.sh
```

2. Optionally build a local PETSc tree under `./.build`.

```bash
./build_scripts/build_local_petsc_opt.sh
source ./build_scripts/activate_local_petsc_env.sh
```

Or, if you only want the default wheel-based setup, use the active venv directly:

```bash
source .venv/bin/activate
pip install petsc4py
pip install -e .[test,cython]
```

3. Build optional Cython extension (used for hot kernels):

```bash
python setup.py build_ext --inplace
```

If the extension is not built, fallback kernels in `src/slope_stability/_kernels.py` are used.

## Running in MPI
Use `mpiexec` with the packaged CLI entrypoints:

```bash
mpiexec -n 4 python -m slope_stability.cli.run_case_from_config \
  benchmarks/run_3D_hetero_SSR_capture/case.toml \
  --out_dir /tmp/ssr_run
```

The linear solvers and context are MPI-aware through PETSc communicators and support distributed matrix/vector handles.

## Quick MPI smoke-test

```bash
mpiexec -n 4 python - <<'PY'
from slope_stability.mpi.context import MPIContext
print('MPI ranks:', MPIContext().size)
PY
```

## Project entry structure

- `src/slope_stability/nonlinear`: Newton and damping logic.
- `src/slope_stability/continuation`: direct/indirect continuation flows.
- `src/slope_stability/linear`: linear and deflated GMRES wrappers.
- `src/slope_stability/fem`: finite-element assembly and quadrature.
- `src/slope_stability/constitutive`: constitutive operators and reductions.
- `src/slope_stability/cython`: optional Cython implementation points.
- `tests_local/validation`: local smoke/parity scripts and fixtures.
