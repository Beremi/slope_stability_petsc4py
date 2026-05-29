# Neumann BCs

Neumann conditions are asset-declared boundary rules. A case may select physics
and profiles, but the support, value model, and geometry patch ownership belong
to the asset.
Mechanics Neumann supports must resolve to boundary/face physical labels, not
nodesets, because the native path is face quadrature over a DMPlex label.

The mechanics manifest path writes Neumann metadata to
`data/mechanics_neumann_labels.csv` when an asset declares mechanics Neumann
rules. Native startup validates that the manifest-declared rules have a matching
label table and fails explicitly on missing or inconsistent label data.

Status: affine native face quadrature is active for mechanics
`constant-traction` rules that do not reference a curved boundary-geometry
patch. The native code owns this path in
`src/petsc_ssr/native/assembly/neumann.c`: startup validates each row, resolves
the value-model registry name, stages the row as an `AssemblyNeumannRule`, and
elastic assembly adds the face-quadrature load contribution to `f_ext`. The
supported affine `constant-traction` model is evaluated through the native
Neumann value-model registry from a typed value context rather than through
case-local load code.
The phase is logged through the shared `SSR Assemble Neumann` PETSc event and
its counts/time are accumulated through `SsrNeumannStats`.
Curved/high-order geometry-patch Neumann rules still fail explicitly with
`pending_native_curved_face_quadrature`; they are not approximated with affine
loads.

The intended C/PETSc path is:

- resolve asset supports to DMPlex face labels;
- attach typed value-model contexts through the Neumann value registry;
- assemble residual contributions by face quadrature over the support label;
- add tangent contributions only for value models that depend on solution state.

Native startup validates each Neumann label row against the value-model
registry in `assembly/neumann.c`. The current registry names are
`constant-traction`, `normal-pressure`, `hydrostatic-pressure`,
`piecewise-linear-head`, `table-on-boundary`, and `function-pointer-debug`;
only `constant-traction` currently has a native evaluator and is assembled by
the affine native path.
The label table stores the `value_model` payload as quoted JSON inside CSV, so
the native reader treats CSV quoting as part of the compatibility contract
before resolving a row to a registry name. The native reader also checks the
exact nine-column row contract: kind is required, geometry patch names must carry a
positive `geometry_order`, and the current status must be
`native_face_quadrature_affine` for affine rows or
`pending_native_curved_face_quadrature` for geometry-patch rows.

Do not add case-local coordinate CSV loads for new Neumann work. They will not
survive curved high-order boundaries or distributed mesh adaptation.
