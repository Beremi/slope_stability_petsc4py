# Repository Layout

The standalone engine is organized as a small solver repository:

- `src/petsc_ssr`: Python package and native backend.
- `src/petsc_ssr/cli/commands`: small command modules for run, case, mesh,
  asset, benchmark, suite, doctor, and target workflows; `cli/main.py` should
  stay thin dispatch glue as commands are split out.
- `src/petsc_ssr/config`: public case/profile schemas, shared validators,
  case/profile/asset resolution, and resolved run/environment manifests. Use
  `config/schema.py` for TOML dataclasses and `config/resolver.py` for the
  resolved public run model.
- `src/petsc_ssr/assets`: mesh-asset contracts split into `api.py`,
  `registry.py`, `gmsh.py`, `bcs.py`, and `curved.py` public facades.
- `src/petsc_ssr/runtime`: runtime option, runner, result, and environment
  facades for Python orchestration around native PETSc execution.
- `src/petsc_ssr/benchmarks`: benchmark registry, generation, suite expansion,
  report, and comparison facades for local and HPC sweeps.
- `meshes`: mesh assets and asset definitions.
- `benchmarks/cases`: runnable case TOML benchmarks.
- `benchmarks/suites`: local/HPC sweep definitions.
- `benchmarks/tools`: notebook and benchmark helper scripts.
- `benchmarks/reports`: curated historical reports.
- `benchmarks/targets`: committed performance targets.
- `configs/petsc`: PETSc options.
- `configs/continuation_profiles`, `configs/newton_profiles`,
  `configs/seepage_profiles`, and `configs/solver_profiles`: named
  algorithm/runtime/PETSc profiles referenced by case TOMLs.
- `cluster/karolina`: Slurm scripts.
- `tools`: local developer scripts.

Generated run outputs stay under `.local/tmp` or per-case `artifacts/`, both of
which are ignored.

Public model documentation lives in:

- `docs/quickstart.md`
- `docs/create-a-benchmark.md`
- `docs/case-schema.md`
- `docs/assets.md`
- `docs/curved-boundaries.md`
- `docs/neumann-bcs.md`
- `docs/solver-profiles.md`
- `docs/suite-runs.md`
- `docs/local-32-testing.md`
- `docs/hpc.md`
- `docs/profiling.md`
- `docs/architecture.md`
