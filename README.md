# PETSc SSR Engine

`petsc-ssr` is a self-contained PETSc/DMPlex slope-stability
engine. Python owns compact case TOMLs, benchmark metadata, launch commands, and
notebook/reporting helpers. C/PETSc owns distributed meshes, matrices, vectors,
assembly, PMG, deflation, Krylov solves, Newton iterations, continuation, and
profiling.

The maintained benchmark path is the full-C PETSc implementation reached through
`petsc-ssr run`.

## Layout

- `src/petsc_ssr/`: importable Python package, CLI, case schema, and helpers.
- `src/petsc_ssr/native/`: Cython extension and subsystem-organized C runtime.
- `meshes/`: canonical mesh assets and mesh definition modules.
- `benchmarks/cases/`: runnable case TOML folders and generated notebooks.
- `benchmarks/suites/`: local/HPC sweep definitions.
- `benchmarks/targets/`: committed performance targets.
- `benchmarks/reports/`: curated historical validation reports.
- `benchmarks/tools/`: notebook and benchmark maintenance helpers.
- `configs/petsc/`: PETSc option profiles.
- `configs/solver_profiles/`: named solver profiles used by case TOMLs.
- `cluster/karolina/`: Slurm submission and collection scripts.
- `tools/`: local validation and memory-sampling helpers.
- `docs/`: architecture, layout, benchmark, mesh, and validation notes.

See [docs/layout.md](docs/layout.md) and
[docs/architecture.md](docs/architecture.md) for more detail.

## Build

```bash
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 \
PETSC_ARCH=linux-c-opt \
make
```

The build creates `petsc_ssr.native._core` in-place under `src/`.

## Run

Tiny smoke solve:

```bash
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 \
PETSC_ARCH=linux-c-opt \
make smoke
```

Local L1 benchmark:

```bash
RANKS=32 ./tools/run_local_l1_benchmark.sh
```

Case TOML runner:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main run \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml \
  --output .local/tmp/3d-heterogeneous-ssr-p4 \
  --continuation-step-max 3
```

Inspect a case before solving:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main case validate \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main mesh-only \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
```

## Benchmarks And Notebooks

Cases live in `benchmarks/cases/<slug>/`. Each case owns:

- `case.toml`
- `README.md`
- `simulation.ipynb`
- `visualisation.ipynb`

Active `case.toml` files use compact sections: `[case]`, `[mesh]`,
`[physics]`, `[continuation]`, `[newton]`, `[linear]`, `[output]`, and optional
`[notebook]`/`[seepage]`. MPI ranks and node counts belong to launcher flags or
`benchmarks/suites/*.toml`, not case TOMLs.

Regenerate notebook shells and per-case READMEs with:

```bash
PYTHONPATH=$PWD/src .venv/bin/python benchmarks/tools/generate_benchmark_readmes.py
PYTHONPATH=$PWD/src .venv/bin/python benchmarks/tools/generate_benchmark_notebooks.py
```

Old benchmark names are mapped to canonical slugs in
[docs/benchmark-migration.md](docs/benchmark-migration.md).

## Karolina

```bash
cd cluster/karolina
NODE_COUNTS="1 2" TIME_LIMIT=00:30:00 ./submit_scaling.sh
```

Use Karolina for these jobs. The scripts default to the maintained full-C
baseline profile.

## Validation

The normal cleanup gate is:

```bash
python -m compileall -q src tools benchmarks/tools
bash -n tools/*.sh benchmarks/tools/*.sh cluster/karolina/*.sh cluster/karolina/*.sbatch
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt make
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt make smoke
```

Performance targets for the maintained L1 baseline are stored in
`benchmarks/targets/`.
