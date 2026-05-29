# Benchmarks

Active benchmark cases live under `benchmarks/cases/<slug>/`. Use lower-kebab
slugs for new cases, for example `3d-heterogeneous-ssr-p4`.

The case runner accepts a compact `case.toml` and writes case artifacts such as
`data/summary.json`, `data/continuation_curve.csv`, `data/run_info.json`,
`data/petsc_run.npz`, `data/native_problem_manifest.json`,
`data/resolved_run_manifest.json`, `data/environment.json`,
`data/mechanics_bc_labels.csv`, optional `data/mechanics_neumann_labels.csv`,
optional `data/seepage_boundary_labels.csv`, and
`exports/final_solution.vtu` where applicable.

## Case TOML

Case TOMLs describe the mathematical case:

- `[case]`: canonical `id`, title, and tags.
- `[mesh]`: asset, mesh variant, and element order.
- `[physics]`: mechanics/seepage model names.
- `[continuation]`, `[newton]`, `[linear]`: profile names and mathematical
  controls only. Reusable continuation, Newton, linear solver, PMG, and raw
  PETSc policy live in `configs/continuation_profiles/`,
  `configs/newton_profiles/`, `configs/seepage_profiles/`, and
  `configs/solver_profiles/`, or explicit CLI/debug overrides.
- `[output]`: optional named output preset only.

MPI ranks, wall time, node counts, and sweeps belong to launcher flags or
`benchmarks/suites/*.toml`, not case TOMLs. Refinement levels, partitioners,
and generated output paths follow the same rule.

Tags must not duplicate structured state such as `2d`, `3d`, `p2`, `p4`,
`mechanics`, `ssr`, or `limit-load`. Use tags for orthogonal labels such as
`regression`, `scaling`, `validation`, `slow`, `nightly`, or `experimental`.

Notebook visualization settings live in `benchmarks/cases/<slug>/notebook.toml`
so the canonical `case.toml` remains a mathematical case description.

## Creating A Benchmark

1. Add or reuse a mesh asset under `meshes/<asset>/`.
2. Validate the asset: `petsc-ssr asset validate <asset>`.
3. Create the case skeleton from the asset:

   ```bash
   PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark init \
     3d-new-slope-ssr-p4 \
     --asset 3d_hetero_slope \
     --element P4 \
     --analysis ssr
   ```

   The command writes a compact `case.toml`, `notebook.toml` sidecar, `run.sh`,
   README, and notebooks. Use `--no-notebooks` when notebook extras are not
   installed.

4. Edit only mathematical case values in `case.toml`; geometry/material/BC
   supports stay in the asset, and solver policy stays in the selected profile.
5. Add the case to a suite only if it should be part of a repeated sweep.

Check that generated benchmark scaffolding is still reproducible without
rewriting files:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark init --check
```

Use `--no-notebooks` with the check in minimal environments that intentionally
skip notebook extras.

## Suites And Targets

Discover cases, suites, and targets without importing PETSc or mesh readers:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark list
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark list --kind cases
```

Validate target files before using them in reports:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main targets validate \
  benchmarks/targets
```

Local suite manifests can be generated without running the expensive solves:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main suite validate \
  benchmarks/suites/local-32-smoke.toml
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main suite expand \
  benchmarks/suites/local-32-smoke.toml \
  --output .local/runs/local-32-smoke/manifest.json
```

`suite validate` performs the same suite schema, case-reference, and
profile-reference checks as expansion, but only prints a JSON preflight summary.

Each manifest entry records the concrete PMG active-rank choices resolved for
that rank count, the profile-owned PMG coarse/telescope/smoother policy, and
the requested and concrete PC variant for the case element order. Individual
runs also write `data/resolved_run_manifest.json` with the concrete profile, PC
variant, PMG policy, and artifact paths actually used.
Suite resources live in `[resources.<target>]` tables, for example
`[resources.local]` with `machine`, `cores`, `max_ranks`, `launcher`, and
`time_limit`, so machine policy stays out of case TOMLs and inside suite
manifests. Expansion picks the first declared resource that supports each rank
count, uses that resource's `launcher` when building the command, and records
the concrete `resource` and `launcher` on every run manifest entry.
Suite run environments live in `[environment]`; local scaling suites pin
`OMP_NUM_THREADS = "1"` there and record that value in the planned manifest.
Committed suites use only the modern suite schema: `[suite].id`,
`[suite].profiles`, and `[suite].ranks`; top-level `[solver]` compatibility
tables are rejected.
Suite TOMLs may add a `[sweeps]` section for benchmark-owned axes such as
`refine_levels`, `linear_rtol`, and `continuation_step_max`; those values are
expanded into concrete CLI overrides per run and recorded in the manifest/report.
One-off suite overrides are limited to `[overrides.continuation]`,
`[overrides.linear]`, `[overrides.mesh]`, and `[overrides.output]` so
unsupported run policy is not silently ignored.

Run roots contain `manifest.json`, and reports can be rendered from completed or
planned runs:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main suite report \
  .local/runs/local-32-smoke
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main targets compare \
  .local/runs/local-32-smoke benchmarks/targets/local-32
```

`suite run` writes `logs/stdout.txt` for each run and, when `options_left`
collection is enabled, derives `logs/options_left.txt` from that captured PETSc
finalization output. It also writes `command.json` at the run root with the
suite/run ids, concrete resource, launcher tokens, environment pins, sweep axes,
resolved profile, command, and artifact paths used for that run.
Direct `petsc-ssr run` and `case dry-run` preflight artifacts also create a
run-root `command.json` with the runner entrypoint, argv, MPI size, output path,
resolved profile, and standard artifact paths. Suite runs keep their richer
suite-owned payload; direct preflight does not overwrite it.
`suite report` always writes the run table to `report.csv`.
The run table includes the profile-level linear algorithm, concrete native
linear selector, concrete PC variant, concrete PMG active ranks, and standard
artifact paths from the resolved suite manifest, including the `command.json`
provenance path. When completed runs have
`data/summary.json` files with wall times, the report also writes
`report.scaling.csv` and renders scaling, iteration, and numerical median
sections in Markdown. Speedup and parallel efficiency are derived from the
lowest completed rank in each case/profile/sweep group, so planned manifests do
not imply performance claims.
First-class JSON targets are case-bound and live under directories such as
`benchmarks/targets/local-32/` and `benchmarks/targets/numerical/`.
Historical unstructured target JSON retained for audit context is not
advertised by the benchmark registry. `targets compare` accepts either one
target-set directory or the top-level `benchmarks/targets` directory; when the
top-level directory is used it prefers a target whose `suite` matches the suite
manifest being compared.
First-class targets are schema-checked by `targets validate`, `benchmark init
--check`, registry discovery, and target comparison. Historical JSON without a
`case` key is parsed only and reported as legacy audit context. First-class
targets may contain `case`, `profile`, `suite`, `status`, `notes`, `metrics`,
`rank_metrics`, and `groups`; they must not contain run commands, launch
settings, or scheduler policy. Metric specs use either `max` or
`expected`/`value` plus optional non-negative
`abs_tol`/`rel_tol`.
Target comparisons require clean options-left evidence for completed groups.
They report `options_left_check`, `options_left_missing`, or
`options_left_unknown` instead of a metric pass when PETSc reports unused
options or the run lacks trustworthy options-left output.
