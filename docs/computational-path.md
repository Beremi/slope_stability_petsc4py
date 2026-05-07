# Computational Path

Benchmark execution is asset-first. A benchmark `case.toml` selects the asset,
mesh variant, optional profile, analysis, element order, and numerical/export
settings. Problem geometry, materials, hydraulic constants, head conditions, and
mechanics boundary conditions live in `meshes/<asset>/definition.py`.

## Flow

1. Command:
   `./.venv/bin/python -m slope_stability.cli.run_case_from_config benchmarks/<case>/case.toml --out_dir <dir>`
   enters `src/slope_stability/cli/run_case_from_config.py`.
2. Config load:
   `src/slope_stability/core/run_config.py` parses TOML, derives dimension from
   the asset, and rejects legacy inputs such as raw mesh paths, `[case_data]`,
   `[[materials]]`, `[seepage].water_unit_weight`, and `[seepage].conductivity`.
3. Asset resolution:
   `src/slope_stability/problem_asset_runtime.py` loads `ASSET` through
   `src/slope_stability/assets/__init__.py`, resolves `mesh_variant` and
   `profile`, and exposes mechanics and seepage specs.
4. Mesh build:
   `src/slope_stability/assets/factories.py` calls
   `src/slope_stability/assets/support/canonical_gmsh.py` to read the canonical
   linear Gmsh mesh, map `region:*`, `boundary:*`, and `nodeset:*` names,
   promote to the requested element order, and attach material ids and masks.
5. Route selection:
   `src/slope_stability/execution/asset_case/runner.py` dispatches by resolved
   dimension, `problem.analysis`, and asset capabilities to mechanics, seepage,
   or seepage-coupled SSR execution modules.
6. Builders and solve:
   Mechanics paths build material arrays, displacement masks, elastic rows,
   tangent patterns, constitutive operators, and PETSc linear solvers. Seepage
   paths build conductivity arrays and head masks from the asset seepage spec,
   solve seepage, and pass pore pressure, gradients, and saturation into the
   mechanical SSR path when coupled.
7. Continuation and export:
   Continuation runs through the nonlinear/continuation stack. Rank 0 writes
   run data under `data/` and exports debug bundles, history, resolved config,
   and VTU files through `src/slope_stability/execution/asset_case/runner.py`
   and `src/slope_stability/postprocess/case_mesh.py`.

## Concrete 3D Seepage-Coupled SSR Example

Command:

```bash
./.venv/bin/python -m slope_stability.cli.run_case_from_config \
  benchmarks/run_3D_hetero_seepage_SSR_comsol_capture/case.toml \
  --out_dir artifacts/examples/transition-ssr
```

The config selects:

- asset: `3d_hetero_seepage_transition`
- mesh variant: `transition_default.msh`
- profile: `fixed_base`
- analysis: `ssr`
- element order: `P2`

`meshes/3d_hetero_seepage_transition/definition.py` owns the four material
models, the `fixed_base` mechanics profile, uniform conductivity, water unit
weight, and head conditions. The route is 3D plus `ssr` plus seepage capability,
so execution uses `src/slope_stability/execution/asset_case/seepage_ssr_3d.py`.
