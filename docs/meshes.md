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

Resolved runs write `data/native_problem_manifest.json` with those supports as
DMPlex-facing labels (`Cell Sets`, `Face Sets`, and `Vertex Sets`) plus the
asset-declared Dirichlet, Neumann, seepage head, seepage flux, and
boundary-geometry rules. The mechanics runner also writes
`data/mechanics_bc_labels.csv` with DMPlex label/tag/component rows for
Dirichlet supports. Assets with mechanics Neumann rules also write
`data/mechanics_neumann_labels.csv` with boundary support, geometry, and
value-model metadata. Assets with seepage rules write
`data/seepage_boundary_labels.csv` with head/flux supports and value models.
These label tables intentionally do not contain coordinates; the native
mechanics path requires the Dirichlet label table for manifest-declared
Dirichlet rules. Coordinate CSVs are debug compatibility artifacts only and are
emitted only by explicit `--write-coordinate-bc-table` debug runs. The native
engine rejects a coordinate constraint table unless
`-debug_coordinate_bc_table true` is present too; manifest-provided coordinate
tables must carry the same guard in `native_inputs`.
Neumann label tables are validated by the dedicated native
`assembly/neumann.c` path. Affine mechanics `constant-traction` rows are now
assembled by native face quadrature into the external mechanics load vector;
curved geometry-patch rows still fail explicitly until curved face quadrature
lands. Coupled seepage pressure still uses the hydro
prepass pressure CSV as the active mechanics load bridge; the seepage label
table is validated and staged as native boundary metadata, but that coordinate
bridge must be marked with `seepage_pressure_source =
"hydro_prepass_coordinate_bridge"` in both PETSc options and manifest
`native_inputs`. The mechanics runner forwards these paths through the
PETSc options `-native_problem_manifest`, `-mechanics_bc_labels_csv`,
`-mechanics_neumann_labels_csv`, and `-seepage_boundary_labels_csv`. Native
startup can also resolve the artifact paths from the manifest `native_inputs`
block when the direct artifact options are omitted. It validates the manifest
label names, support sections, and declared rule counts at startup, and rejects
manifests that declare rules without a corresponding label-table path. With a
manifest active, native startup also rejects label tables whose row counts or
row fingerprints disagree with the manifest; Python preflight checks full
row-content parity before launch.
The same label-table contracts are visible before launch through
`petsc-ssr asset validate`, which reports per-variant/profile row counts,
columns, native statuses, and row fingerprints without building PETSc objects.

For future curved/high-order boundary support, add asset metadata such as
`boundary_geometry` or generated high-order mesh coordinates in the asset
definition. The solver should continue to consume DMPlex labels and coordinates,
so curved Neumann faces and high-order midpoint generation do not require a new
distributed solver I/O path.
Each `boundary_geometry` entry must name a declared boundary support, and any
Neumann or seepage flux rule that references a geometry patch must target that
same boundary. `petsc-ssr asset validate` and manifest generation both reject
mismatched geometry/support declarations before a run reaches native assembly.
Mechanics Neumann and seepage flux rules must target boundary labels even when
nodesets are available for Dirichlet or head conditions; face loads are not
defined on vertex-set supports in the native contract.
Asset validation checks those support declarations across every declared mesh
variant with a mesh file, not only the default variant.
