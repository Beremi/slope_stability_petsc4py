# Assets

Mesh assets own geometry, materials, region assignments, physical names,
boundary labels, nodesets, Dirichlet supports, Neumann supports, seepage
head/flux supports, and boundary-geometry declarations. Case TOMLs reference
asset names and support names; they should not duplicate raw mesh tags,
coordinate tables, material rows, or boundary-condition support definitions.

Validate an asset before using it in a case:

```bash
petsc-ssr asset validate 3d_hetero_slope
petsc-ssr asset validate --all
```

Validation checks every declared mesh variant that has a mesh file. For each
variant, region physical names must have material assignments, assigned regions
must exist in the mesh, Dirichlet and seepage head supports must exist as
boundary or nodeset labels, mechanics Neumann and seepage flux supports must be
boundary labels, and `boundary_geometry` supports must exist as boundary labels.
`--all` enumerates registered mesh assets only, so local
fixtures under `meshes/` are not mistaken for production assets. The command
reports `variant_supports` so missing labels are visible before a case or suite
reaches native startup.
It also dry-builds the native manifest contract for each mesh variant/profile
and reports `native_manifest_contracts` with support counts, rule counts, and
label-table columns, row counts, statuses, and deterministic row fingerprints.
This gives lightweight CI a PETSc-free check that the asset can produce the same
label/rule contract consumed by native startup.

The resolved run writes `data/native_problem_manifest.json` with DMPlex-facing
label names and support declarations. Mechanics Dirichlet supports are exported
through `data/mechanics_bc_labels.csv`; mechanics Neumann supports use
`data/mechanics_neumann_labels.csv`; seepage supports use
`data/seepage_boundary_labels.csv`. These label tables are the normal solver
contract. The manifest also records deterministic row fingerprints for these
tables, and native startup validates the files before using their paths.
Coordinate CSVs are debug compatibility artifacts only and are not
written by normal mechanics runs unless `--write-coordinate-bc-table` is used;
the native engine also requires `-debug_coordinate_bc_table true` before such a
table is consumed, including when the table path comes from manifest
`native_inputs`. Manifest-backed native startup checks label-table row counts
against manifest rule counts and row fingerprints, while Python preflight checks
full row contents.
Coupled seepage pressure is the remaining active coordinate bridge; mechanics
runs must mark it with `seepage_pressure_source =
"hydro_prepass_coordinate_bridge"` whenever a pressure CSV is present.

Assets are also the compatibility boundary for future high-order geometry work.
Curved boundary metadata should be declared as asset data and resolved into
DMPlex labels and coordinate sections, so solver code does not need case-local
geometry hacks.

See `docs/meshes.md` for the current manifest and label-table details.
