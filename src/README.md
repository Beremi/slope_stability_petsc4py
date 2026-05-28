# Source Layout

All importable code lives under `src/petsc_ssr`.

- `context.py`, `problem.py`, `options.py`: public Python configuration API.
- `case_config.py`, `hydro_cases.py`: case TOML translation and outputs.
- `assets/`, `mesh/`, `postprocess/`, `seepage/`: engine-owned support code
  for benchmark assets, visualisation, and seepage prepasses.
- `native/`: Cython extension plus the PETSc-owned C implementation.
- `runners/`: command-line entry points.

The default production path is full C through `petsc_ssr.native._core`.
