# Mesh Assets

Each subdirectory in `meshes/` is a canonical problem asset.

Canonical runtime contract:

- `meshes/<asset>/definition.py`
- one or more canonical Gmsh `MSH 4.1` files at `meshes/<asset>/*.msh`
- optional legacy inputs and rerun helpers under `meshes/<asset>/legacy/`

Runtime rules:

- runtime problem definitions are loaded only from `definition.py`
- runtime geometry/topology comes only from canonical `.msh` variants
- `definition.py` may declare only variants, materials, region-to-material assignment, mechanics BCs, seepage BCs, and named profiles
- runtime code outside `meshes/` is asset-agnostic; it does not contain case-specific geometry or BC logic

Canonical mesh naming:

- volume regions: `region:<name>`
- support boundaries: `boundary:<name>`
- optional node supports: `nodeset:<name>`
- optional curved geometry patches: `boundary_geom:<name>`

Canonical mesh rules:

- 2D volume cells are linear `triangle`
- 3D volume cells are linear `tetra`
- support boundaries are linear `line` in 2D and linear `triangle` in 3D
- higher-order solver meshes are generated on demand from the canonical linear mesh

Migration notes:

- old text bundles, generated geometry inputs, and pre-canonical `.msh` files are kept under each asset’s `legacy/source/`
- rerun scripts live under each asset’s `legacy/`
- the converter index is tracked in [CONVERTERS.md](CONVERTERS.md)
