# Benchmarks

Active benchmark cases live under `benchmarks/cases/<slug>/`. Use lower-kebab
slugs for new cases, for example `3d-heterogeneous-ssr-p4`.

The case runner accepts a compact `case.toml` and writes case artifacts such as
`data/summary.json`, `data/continuation_curve.csv`, `data/run_info.json`,
`data/petsc_run.npz`, and `exports/final_solution.vtu` where applicable.

## Case TOML

Case TOMLs describe the mathematical case:

- `[case]`: name, title, and tags.
- `[mesh]`: asset, mesh variant, element order, optional refinement, partitioner.
- `[physics]`: mechanics/seepage model names.
- `[continuation]`, `[newton]`, `[linear]`: profile names and case-specific
  overrides only.
- `[output]`: artifact families requested by notebooks and runners.

MPI ranks, wall time, node counts, and sweeps belong to launcher flags or
`benchmarks/suites/*.toml`, not case TOMLs.

## Creating A Benchmark

1. Add or reuse a mesh asset under `meshes/<asset>/`.
2. Add `benchmarks/cases/<slug>/case.toml`.
3. Run `petsc-ssr benchmark init <slug>` to generate README/notebooks.
4. Add the case to a suite only if it should be part of a repeated sweep.
