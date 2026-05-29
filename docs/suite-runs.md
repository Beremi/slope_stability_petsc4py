# Suite Runs

Suites own benchmark sweeps. They are the place for ranks, repeats, resource
labels, time limits, and controlled overrides that would make a case TOML less
declarative.

Modern suite TOMLs use:

- `[suite]`: id, title, cases, profiles, ranks, repeats, and optional timeout.
- `[resources.<name>]`: machine/resource metadata such as local 32-core limits.
  Integer limits such as `cores`, `nodes`, `ranks_per_node`, and `max_ranks`
  must be positive. When `max_ranks` is declared, every rank in `[suite].ranks`
  must fit at least one declared resource. Expansion resolves each run to the
  first declared resource that supports its rank count, uses that resource's
  `launcher` value when building the command, and records both `resource` and
  `launcher` in the manifest.
- `[environment]`: run environment pins such as `OMP_NUM_THREADS = "1"`.
- `[sweeps]`: sweep axes such as `refine_levels`, `linear_rtol`, and
  `continuation_step_max`.
- `[overrides.continuation]`, `[overrides.linear]`, `[overrides.mesh]`: explicit
  benchmark-owned one-off overrides.
- `[overrides.output]`: optional named output preset for the suite.
- `[collect]`: TOML boolean collection policy such as `petsc_log_view`,
  `options_view`, `options_left`, `environment`, and `resolved_manifest`.
  Strings like `"false"` are rejected instead of being treated as truthy, and
  aliases such as `petsc_log_view`/`log_view` must use one spelling per suite.
  Suites marked by `preset = "performance"` or named as scaling/performance
  suites must enable `petsc_log_view` and `options_left`; otherwise expansion is
  rejected because reports would lack PETSc timing evidence or options-left
  cleanliness.

All committed suites use this public schema. Migration-era `[suite].name`,
`[suite].description`, and top-level `[solver]` fields are rejected; solver
choice belongs in `[suite].profiles`.

Expand without solving:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main suite validate \
  benchmarks/suites/local-32-smoke.toml
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main suite expand \
  benchmarks/suites/local-32-smoke.toml \
  --output .local/runs/local-32-smoke/manifest.json
```

`suite validate` parses the suite schema, checks referenced cases and profiles
through the same expansion path as manifest generation, and reports the planned
run count without writing files.

Run or dry-run a suite:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main suite run \
  benchmarks/suites/local-32-smoke.toml --dry-run
```

Render reports and target comparisons:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main suite report \
  .local/runs/local-32-smoke
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main targets validate \
  benchmarks/targets/local-32
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main targets compare \
  .local/runs/local-32-smoke benchmarks/targets/local-32
```

Reports use medians across completed repeats for scaling, iteration, numerical,
and target-comparison rows. Planned manifests do not imply performance claims.
Each planned run also records the resolved profile policy for that case and rank,
including concrete PMG active ranks and requested-vs-concrete PC variant. This
also includes the concrete native linear selector derived from the profile,
element, PC variant, and deflation policy, so P1 PMG fallbacks are visible
before the suite launches.
Each planned run also carries a standard artifact contract for
`command.json`,
`data/resolved_run_manifest.json`, `data/resolved_config.toml`,
`data/resolved_options.txt`, `data/environment.json`, `data/summary.json`,
`logs/petsc_log.txt`, `logs/options_view.txt`, and `logs/options_left.txt`, so
dry-run manifests and completed runs use the same provenance paths.
Completed suite runs write `command.json` as a versioned command provenance
record with the suite id, run id, case, profile, ranks, concrete resource,
launcher tokens, environment pins, sweep axes, resolved profile, and artifact
paths.
Standalone run and dry-run preflight paths write the same file only when it is
absent, recording the runner entrypoint, argv, MPI size, resolved profile, and
standard artifact paths. This keeps suite-owned resource/launcher provenance as
the authority for suite executions.
When `logs/petsc_log.txt` exists, reports include a `PETSc Log Events` section
and write `report.petsc-events.csv` with the top event timings per run.
Reports also include concrete native linear selectors, concrete PC/PMG choices,
and an `Artifact Paths` section listing the command provenance, resolved
manifest, resolved options, summary, PETSc log, and options-left files for each
planned or completed run.
Target comparisons read case-bound first-class JSON targets, including per-rank
metric groups when present. `targets validate` checks first-class target schemas
and reports historical JSON without `case` as parse-only legacy context. Targets
may live in nested target sets such as `benchmarks/targets/local-32/` or
`benchmarks/targets/numerical/`; comparing against the top-level
`benchmarks/targets` directory prefers the target whose `suite` matches the run
manifest. Historical unstructured target JSON can remain for audit context but
is not part of the benchmark registry.
Target comparison requires clean options-left evidence for completed repeats. If
any repeat reports `check`, `missing`, or `unknown`, the case/profile/rank group
is marked as `options_left_check`, `options_left_missing`, or
`options_left_unknown` even when timing or iteration metrics would otherwise
pass.
When `options_left` collection is enabled, suite runs capture stdout and write a
derived `logs/options_left.txt` status artifact so reports do not need to parse
the full terminal log repeatedly.

Discover registered benchmark assets with:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark list
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main benchmark list --kind suites
```

The registry is machine-readable JSON over cases, suites, and targets. It is
implemented without PETSc or mesh-reader imports so it can run on login nodes
and lightweight CI jobs.

Committed suite scaffolds include `local-32-smoke`,
`local-32-strong-scaling`, `validation`, and `hpc-strong-scaling`.
