# Mesh Assets

Each subdirectory in `meshes/` is a canonical problem asset.

## Runtime Contract

- `meshes/<asset>/definition.py`
- one or more canonical Gmsh `MSH 4.1` files at `meshes/<asset>/*.msh`
- optional legacy inputs and rerun helpers under `meshes/<asset>/legacy/`

## Runtime Rules

- runtime problem definitions are loaded only from `definition.py`
- runtime geometry/topology comes only from canonical `.msh` variants
- `definition.py` owns variants, materials, hydraulic conductivity, water unit weight, hydraulic state, region-to-material assignment, mechanics BCs, seepage BCs, and named profiles
- runtime code outside `meshes/` is asset-agnostic; it does not contain case-specific geometry or BC logic

## Canonical Mesh Naming

- volume regions: `region:<name>`
- support boundaries: `boundary:<name>`
- optional node supports: `nodeset:<name>`
- optional curved geometry patches: `boundary_geom:<name>`

## Canonical Mesh Rules

- 2D volume cells are linear `triangle`
- 3D volume cells are linear `tetra`
- support boundaries are linear `line` in 2D and linear `triangle` in 3D
- higher-order solver meshes are generated on demand from the canonical linear mesh

## Migration Notes

- old text bundles, generated geometry inputs, and pre-canonical `.msh` files are kept under each asset’s `legacy/source/`
- rerun scripts live under each asset’s `legacy/`
- the converter index is tracked in [converter-index.md](converter-index.md)

## Adding A Benchmark

- add or update `meshes/<asset>/definition.py`
- add the canonical mesh variant at `meshes/<asset>/<variant>.msh`
- add `benchmarks/cases/<slug>/case.toml` with `asset`, `mesh_variant`, optional `profile`, analysis, element type, solver, export, and notebook metadata
- update `tests/test_executable_asset_definitions.py` if the asset becomes part of the canonical required set
- do not edit `src/` for new benchmark data; only add to `src/petsc_ssr/assets/evaluators.py` when the mesh needs a genuinely new generic BC or value model

For the current repository layout, see [../docs/layout.md](../docs/layout.md).
