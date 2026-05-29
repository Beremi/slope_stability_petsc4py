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
- `src/petsc_ssr/runtime/`: lightweight run-environment and artifact readers.
- `src/petsc_ssr/native/`: Cython extension and subsystem-organized C runtime.
- `meshes/`: canonical mesh assets and mesh definition modules.
- `benchmarks/cases/`: runnable case TOML folders and generated notebooks.
- `benchmarks/suites/`: local/HPC sweep definitions.
- `benchmarks/targets/`: committed performance targets.
- `benchmarks/reports/`: curated historical validation reports.
- `benchmarks/tools/`: notebook and benchmark maintenance helpers.
- `configs/petsc/`: PETSc option profiles.
- `configs/continuation_profiles/`, `configs/newton_profiles/`,
  `configs/seepage_profiles/`, and `configs/solver_profiles/`: named
  algorithm/runtime/PETSc profiles used by case TOMLs.
- `cluster/karolina/`: Slurm submission and collection scripts.
- `tools/`: local validation and memory-sampling helpers.
- `docs/`: architecture, layout, benchmark, mesh, and validation notes.

See [docs/layout.md](docs/layout.md) and
[docs/architecture.md](docs/architecture.md) for more detail.

## Build

Minimal HPC/runtime install keeps plotting, notebooks, mesh conversion/HDF5 mesh
readers, and seepage-only Python solvers out of the base environment:

```bash
pip install .
pip install .[mesh]
pip install .[reports]
pip install .[seepage]
pip install .[notebooks]
pip install .[dev]
```

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
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main case explain \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main asset validate 3d_hetero_slope
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main doctor
```

List registered benchmark cases, suites, and targets:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark list
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark list --kind suites
```

Resolve a local 32-core suite manifest without launching the sweep:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main suite expand \
  benchmarks/suites/local-32-smoke.toml \
  --output .local/runs/local-32-smoke/manifest.json
```

## Benchmarks And Notebooks

Cases live in `benchmarks/cases/<slug>/`. Each case owns:

- `case.toml`
- `README.md`
- `simulation.ipynb`
- `visualisation.ipynb`

Active `case.toml` files use compact sections: `[case]`, `[mesh]`,
`[physics]`, `[continuation]`, `[newton]`, `[linear]`, and optional
`[seepage]`/`[output]`. MPI ranks, node counts, partitioners, refinement sweeps,
and generated output paths belong to launcher flags or `benchmarks/suites/*.toml`,
not case TOMLs. Notebook display settings live in per-case `notebook.toml`
sidecars. Continuation, Newton, seepage runtime, and linear solver policy live
in `configs/continuation_profiles/`, `configs/newton_profiles/`,
`configs/seepage_profiles/`, and `configs/solver_profiles/`; normal runs do
not force the C baseline unless `--force-c-baseline` is passed explicitly for
debugging.

Regenerate notebook shells and per-case READMEs with:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark init \
  3d-new-slope-ssr-p4 --asset 3d_hetero_slope --element P4 --analysis ssr
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark init --check
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
python -m pytest tests/config tests/benchmark -q
PYTHONPATH=$PWD/src python -m petsc_ssr.cli.main benchmark init --check
bash -n tools/*.sh benchmarks/tools/*.sh cluster/karolina/*.sh cluster/karolina/*.sbatch
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt make
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt make smoke
```

Performance targets for the maintained L1 baseline are stored in
`benchmarks/targets/`.
