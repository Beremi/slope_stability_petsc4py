# Docs

Active docs:

- `new_benchmark_new_geometry_guide.md`
- `config_case_matrix.md`
- `config_scheme_3d.md`
- `computational_path.md`

Start with `new_benchmark_new_geometry_guide.md` when adding a new problem, mesh, or
benchmark. It is the asset-first reference for `meshes/<asset>/definition.py`, canonical
Gmsh physical names, and benchmark `case.toml` fields.
Use `computational_path.md` when tracing a benchmark command through config loading,
asset resolution, mesh building, solver execution, and exports.

Historical notes from earlier investigation phases are kept here as reference material.
Some of those notes may still mention the pre-reorganization `slope_stability/...` path
layout; the current root layout is described in the top-level `README.md`.
