# Repository Layout

The standalone engine is organized as a small solver repository:

- `src/petsc_ssr`: Python package and native backend.
- `meshes`: mesh assets and asset definitions.
- `benchmarks/cases`: runnable case TOML benchmarks.
- `benchmarks/suites`: local/HPC sweep definitions.
- `benchmarks/tools`: notebook and benchmark helper scripts.
- `benchmarks/reports`: curated historical reports.
- `benchmarks/targets`: committed performance targets.
- `configs/petsc`: PETSc options.
- `configs/solver_profiles`: named solver profiles referenced by case TOMLs.
- `cluster/karolina`: Slurm scripts.
- `tools`: local developer scripts.

Generated run outputs stay under `.local/tmp` or per-case `artifacts/`, both of
which are ignored.
