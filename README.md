# slope_stability

PETSc-based Python reimplementation of the slope-stability workflows, organized around config-driven runs and benchmark parity against the local MATLAB reference tree.

## Repository layout

- `src/`: Python/PETSc implementation, including runtime CLI entrypoints under `src/slope_stability/cli/`
- `build_scripts/`: tracked environment/bootstrap/build helpers
- `scripts_local/`: ignored ad hoc developer scripts and one-off investigations
- `benchmarks/`: unified case registry; canonical benchmarks and extra runnable cases both live here
- `meshes/`: temporary problem-family mesh sorting plus setup metadata
- `docs/`: notes worth keeping for future work
- `artifacts/`: ignored generated outputs
- `archives/`: ignored archived experiments and legacy outputs
- `.build/`: ignored build workspace
- `.venv/`: ignored local Python environment
- `slope_stability_matlab/`: local-only MATLAB reference tree, intentionally not tracked here

## Main entry points

Bootstrap the local environment:

```bash
./bootstrap.sh
```

By default this performs the full local build needed by the benchmark stack:

- creates `./.venv`
- builds PETSc with `HYPRE` under `./.build`
- installs `petsc4py`
- installs `slope_stability` in editable mode with the benchmark partitioning extras

The first run is intentionally heavier because it produces the real benchmark-capable environment. For a lighter wheel-based setup that may lack `HYPRE`, use:

```bash
BOOTSTRAP_MODE=wheel ./bootstrap.sh
```

Run one case:

```bash
./.venv/bin/python -m slope_stability.cli.run_case_from_config \
  benchmarks/run_3D_hetero_SSR_capture/case.toml \
  --out_dir /tmp/ssr_run
```

Run one canonical benchmark:

```bash
./benchmarks/run_3D_hetero_SSR_capture/run.sh
```

Run the whole benchmark suite:

```bash
./.venv/bin/python -m slope_stability.cli.run_benchmark_suite
```

## Benchmark contract

Each case folder under `benchmarks/` contains:

- `case.toml`
- `run.sh`
- `README.md`

Canonical MATLAB-parity benchmark folders additionally contain `report.md`.

Generated outputs go under `artifacts/...` and stay out of git.

## Exports

Config-driven runs export:

- `exports/run_debug.h5`
- `exports/continuation_history.json`
- `exports/final_solution.vtu`
- `exports/resolved_config.toml`

The intent is straightforward postprocessing with PyVista, meshio, or ParaView.

## Mesh organization

`meshes/` is currently a temporary problem-family sorting layer. The intended direction is:

- one canonical mesh file per mesh variant, in a standard triangle/tetrahedral format
- separate tags for materials and boundary-condition assignment
- any extra derived setup handled by Python setup code, not embedded in the mesh file format

The temporary setup API lives in `meshes/*/definition.py`.

## Notes / TODO

- Benchmarks are currently live MATLAB-vs-PETSc comparisons, not frozen regression snapshots yet.
- TODO: once benchmark parity is stable, freeze compact reference snapshots for regression-style testing.
- `tests_local/` is intentionally ignored and reserved for local smoke/regression scripts during development.
- `scripts_local/` is intentionally ignored and holds exploratory utilities that are not part of the benchmark-replication surface.
- Mesh storage is not in its final standardized form yet; current family folders are a transition step.
- The MATLAB tree is expected at `./slope_stability_matlab` locally for benchmark runs.

## Supporting docs

- `benchmarks/README.md`
- `docs/config_case_matrix.md`
- `docs/config_scheme_3d.md`
