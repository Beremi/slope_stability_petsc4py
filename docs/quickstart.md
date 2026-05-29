# Quickstart

Use the installed `petsc-ssr` command in normal environments. In a development
checkout, use:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main <command>
```

The public workflow keeps responsibilities separate:

- Case TOMLs describe mathematical cases.
- Mesh assets describe geometry, materials, regions, labels, and BC supports.
- Profiles describe continuation, Newton, seepage, linear solver, and PETSc
  policy.
- Suites describe sweeps, ranks, resources, collection, and target comparisons.

Start with a runtime check:

```bash
petsc-ssr doctor
```

Validate and inspect a case before solving:

```bash
petsc-ssr case validate benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
petsc-ssr case validate --all
petsc-ssr case explain benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
```

Inspect the asset-derived mesh, labels, materials, and constrained DOFs:

```bash
petsc-ssr mesh-only benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
```

File-backed Gmsh assets require the mesh optional extra, for example
`pip install .[mesh]`. Minimal runtime installs can still validate cases,
profiles, assets, suites, and dry-run manifests without it.

Write the resolved preflight bundle without launching the solver:

```bash
petsc-ssr case dry-run benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml \
  --output .local/tmp/3d-heterogeneous-ssr-p4-dry-run
```

Run the case through the maintained PETSc/C engine:

```bash
petsc-ssr run benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml \
  --output .local/tmp/3d-heterogeneous-ssr-p4-run
```

Normal runs use the resolved profiles. `--force-c-baseline` is a debug escape
hatch, not a benchmark default.

For repeated work, use suites:

```bash
petsc-ssr suite expand benchmarks/suites/local-32-smoke.toml \
  --output .local/runs/local-32-smoke/manifest.json
petsc-ssr suite run benchmarks/suites/local-32-smoke.toml --dry-run
petsc-ssr suite report .local/runs/local-32-smoke
petsc-ssr targets compare .local/runs/local-32-smoke benchmarks/targets/local-32
```

Suite manifests and per-run manifests record concrete PMG active ranks,
requested-vs-concrete PC variants, resolved profiles, environment, and artifact
paths. Reports may summarize completed timing and iteration data, but planned
manifests are not performance measurements.
