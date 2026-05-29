# Local 32-Core Testing

The committed local suites are the reviewable path for workstation-scale
benchmark checks:

- `benchmarks/suites/local-32-smoke.toml`
- `benchmarks/suites/local-32-strong-scaling.toml`

They sweep ranks `1, 2, 4, 8, 16, 32`, pin `OMP_NUM_THREADS = "1"`, and record
resources as suite metadata instead of case metadata. Expanded manifests record
the concrete local resource and launcher command used for each planned rank.

Prepare a resolved manifest without solving:

```bash
petsc-ssr suite expand benchmarks/suites/local-32-smoke.toml \
  --output .local/runs/local-32-smoke/manifest.json
```

Run a dry suite launch:

```bash
petsc-ssr suite run benchmarks/suites/local-32-smoke.toml --dry-run
```

Run the strong-scaling suite when the local machine is available:

```bash
petsc-ssr suite run benchmarks/suites/local-32-strong-scaling.toml \
  --output .local/runs/local-32-strong-scaling
```

Create reports and compare targets:

```bash
petsc-ssr suite report .local/runs/local-32-strong-scaling
petsc-ssr targets compare .local/runs/local-32-strong-scaling benchmarks/targets/local-32
```

Suites collect PETSc log output and options-left status when requested. Reports
may compute medians, speedup, and efficiency from completed runs. Do not claim a
performance improvement unless the suite was actually run and the measurement
environment is recorded. Completed run directories include `command.json`, which
records the concrete local resource, launcher, environment pins, resolved
profile, and artifact paths for that run.
