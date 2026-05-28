# Meshes

All input meshes live under `meshes/`.

- `meshes/3d_hetero_slope/adaptive_family_a_l1.msh` is the default L1 P4 SSR
  benchmark mesh.
- `meshes/fixtures/tiny_box.msh` is used by smoke tests.
- `meshes/3d_hetero_seepage_transition/transition_default.msh` is the COMSOL
  seepage transition fixture.

Add new benchmark meshes by creating a mesh asset directory with a
`definition.py` and referencing it from a case TOML.

Mesh asset definitions own human-readable region names, boundary names,
nodesets, material assignments, mechanics BC targets, seepage BC targets, and
future curved-boundary metadata. Case TOMLs should reference these names instead
of raw numeric mesh tags.

For future curved/high-order boundary support, add asset metadata such as
`boundary_geometry` or generated high-order mesh coordinates in the asset
definition. The solver should continue to consume DMPlex labels and coordinates,
so curved Neumann faces and high-order midpoint generation do not require a new
distributed solver I/O path.
