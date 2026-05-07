# 3D Configuration Scheme

This page is retained as a short compatibility note. The original 3D-only schema has been
superseded by the asset-first `RunCaseConfig` loader and the canonical mesh asset runtime.

For new work, use:

- [new-benchmark-new-geometry-guide.md](new-benchmark-new-geometry-guide.md)
- [config-case-matrix.md](config-case-matrix.md)

## Current Contract

Config-driven runs use:

```bash
./.venv/bin/python -m slope_stability.cli.run_case_from_config <benchmarks/.../case.toml> --out_dir <dir>
```

`case.toml` may select:

- `[problem].asset`
- `[problem].mesh_variant`
- optional `[problem].profile`
- `[problem].analysis`
- `[problem].elem_type`
- solver, continuation, Newton, seepage numerical, export, and notebook settings

`case.toml` must not define raw mesh paths, boundary types, material rows, water unit
weight, or hydraulic conductivity. Those values are owned by
`meshes/<asset>/definition.py`.

## Runtime Split

- `meshes/<asset>/definition.py`: mesh variants, materials, hydraulic conductivity, water
  unit weight, mechanics BCs, seepage BCs, hydraulic state, and profiles
- `benchmarks/<case>/case.toml`: benchmark metadata and numerical controls
- `src/slope_stability`: generic asset loading, mesh elevation, assembly, solvers,
  postprocessing, and CLI dispatch

## Element Orders

- 2D configs accept `P1`, `P2`, and `P4`
- 3D configs accept `P1`, `P2`, `P3`, and `P4`

Numerical availability still depends on the selected runner and solver path, but mesh
loading and asset resolution are unified across dimensions.

## Removed Legacy Fields

The config loader rejects these fields in committed configs:

- `[problem].dimension`
- `[problem].variant`
- `[problem].seepage`
- `[problem].mesh_path`
- `[problem].mesh_boundary_type`
- `[case_data]`
- `[[materials]]`
- `[seepage].water_unit_weight`
- `[seepage].conductivity`
