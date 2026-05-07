# Slope Stability

PETSc-based Python reimplementation of the slope-stability workflows, organized around config-driven runs and benchmark parity against the local MATLAB reference tree.

## Repository layout

- `src/`: Python/PETSc implementation, including runtime CLI entrypoints under `src/slope_stability/cli/`
- `build_scripts/`: tracked environment/bootstrap/build helpers
- `scripts_local/`: ignored ad hoc developer scripts and one-off investigations
- `benchmarks/`: unified asset-first case registry; canonical benchmarks and extra runnable cases both live here
- `meshes/`: canonical mesh assets; each asset owns mesh variants, materials, hydraulics, BCs, and profiles
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
  --out_dir artifacts/examples/ssr-run
```

Run one canonical benchmark:

```bash
./benchmarks/run_3D_hetero_SSR_capture/run.sh
```

Run the whole benchmark suite:

```bash
./.venv/bin/python -m slope_stability.cli.run_benchmark_suite
```

## Codespaces And Devcontainer

The repository ships a prebuild-friendly devcontainer under [`.devcontainer`](.devcontainer/).

- slow setup runs in `onCreateCommand` and `updateContentCommand`, so Codespaces prebuilds can absorb:
  - local PETSc compilation under `./.build`
  - `petsc4py` installation against that PETSc
  - editable project install with `test`, `viz`, `cython`, and `partition` extras
  - Jupyter kernel registration for `Slope Stability (.venv)`
- the attached editor then opens against the ready `.venv` interpreter with the PETSc runtime environment already exported
- validation entrypoint:

```bash
bash .devcontainer/validate.sh --imports-only
```

## Benchmark contract

Each case folder under `benchmarks/` contains:

- `case.toml`
- `run.sh`
- `README.md`

Generated benchmark reports and archived comparison material live under `archive/`.

Reusable full-run plotting artifacts live under:

- `artifacts/simulation/generated_case.toml`
- `artifacts/simulation/data/run_info.json`
- `artifacts/simulation/data/petsc_run.npz`
- `artifacts/simulation/exports/final_solution.vtu`

Generated outputs go under `artifacts/...` and stay out of git.

Benchmark configs are intentionally thin. `case.toml` selects an asset, mesh variant,
optional profile, analysis type, element order, solver settings, and export settings.
Problem physics belongs in `meshes/<asset>/definition.py`, not in benchmark TOML files or
`src/` runtime modules.

## Exports

Config-driven runs export:

- `exports/run_debug.h5`
- `exports/continuation_history.json`
- `exports/final_solution.vtu`
- `exports/resolved_config.toml`

The intent is straightforward postprocessing with PyVista, meshio, or ParaView.

## Mesh organization

`meshes/` is the source of truth for problem assets:

- `meshes/<asset>/definition.py` exports `ASSET`
- `meshes/<asset>/*.msh` are canonical linear Gmsh `MSH 4.1` variants
- `definition.py` declares materials, hydraulic conductivity, water unit weight, mechanics
  BCs, seepage head BCs, hydraulic state, profiles, and region assignments
- runtime code in `src/` stays problem-agnostic

## Notes

- Benchmarks are currently live MATLAB-vs-PETSc comparisons, not frozen regression snapshots yet.
- Once benchmark parity is stable, freeze compact reference snapshots for regression-style testing.
- `tests_local/` is intentionally ignored and reserved for local smoke/regression scripts during development.
- `scripts_local/` is intentionally ignored and holds exploratory utilities that are not part of the benchmark-replication surface.
- The MATLAB tree is expected at `./slope_stability_matlab` locally for benchmark runs.

## Supporting docs

- `benchmarks/README.md`
- `docs/new-benchmark-new-geometry-guide.md`
- `docs/config-case-matrix.md`
- `docs/config-scheme-3d.md`
- `docs/computational-path.md`
